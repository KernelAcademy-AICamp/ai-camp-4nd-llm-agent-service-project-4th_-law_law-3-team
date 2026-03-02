"""OpenLineage 기반 데이터 계보 추적 (v0.3.0)

Consultant 피드백: 기사→요약→청크→벡터 인덱스의 lineage 명시 필요.
OpenLineage 이벤트 발행으로 데이터 흐름 추적 및 영향도 분석 지원.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)


class LineageEmitter:
    """OpenLineage 이벤트 발행기

    파이프라인 각 단계(수집→정제→요약→저장→청킹)를
    OpenLineage RunEvent로 기록하여 데이터 계보(lineage) 추적.

    lineage_url이 빈 문자열이면 로그 출력만 수행 (비활성화).
    """

    def __init__(self) -> None:
        self._lineage_url = getattr(settings, "NEWS_PIPELINE_LINEAGE_URL", "")
        self._client: Any = None

    @property
    def is_enabled(self) -> bool:
        return bool(self._lineage_url)

    def _get_client(self) -> Any:
        """OpenLineage 클라이언트 (lazy init)"""
        if self._client is None and self.is_enabled:
            from openlineage.client import OpenLineageClient
            from openlineage.client.transport.http import HttpConfig, HttpTransport
            transport = HttpTransport(HttpConfig(url=self._lineage_url))
            self._client = OpenLineageClient(transport=transport)
        return self._client

    def emit_run_start(
        self,
        run_id: str,
        job_name: str,
        inputs: list[dict[str, str]] | None = None,
    ) -> None:
        """파이프라인 단계 시작 이벤트 발행

        Args:
            run_id: 파이프라인 실행 ID
            job_name: 단계명 (예: "news_pipeline.collect", "news_pipeline.summarize")
            inputs: 입력 데이터셋 [{namespace, name}]
        """
        if not self.is_enabled:
            logger.debug("Lineage(disabled): START %s", job_name)
            return

        from openlineage.client.run import (
            InputDataset,
            Job,
            Run,
            RunEvent,
            RunState,
        )

        input_datasets = [
            InputDataset(namespace=ds["namespace"], name=ds["name"])
            for ds in (inputs or [])
        ]

        event = RunEvent(
            eventType=RunState.START,
            eventTime=datetime.now(timezone.utc).isoformat(),
            run=Run(runId=run_id),
            job=Job(namespace="news_pipeline", name=job_name),
            inputs=input_datasets,
            outputs=[],
        )

        try:
            self._get_client().emit(event)
        except Exception as exc:
            logger.warning("Lineage START 이벤트 발행 실패: %s", exc)

    def emit_run_complete(
        self,
        run_id: str,
        job_name: str,
        outputs: list[dict[str, str]] | None = None,
        record_count: int = 0,
    ) -> None:
        """파이프라인 단계 완료 이벤트 발행

        Args:
            run_id: 파이프라인 실행 ID
            job_name: 단계명
            outputs: 출력 데이터셋 [{namespace, name}]
            record_count: 처리된 레코드 수
        """
        if not self.is_enabled:
            logger.debug("Lineage(disabled): COMPLETE %s (records=%d)", job_name, record_count)
            return

        from openlineage.client.run import (
            Job,
            OutputDataset,
            Run,
            RunEvent,
            RunState,
        )

        output_datasets = [
            OutputDataset(namespace=ds["namespace"], name=ds["name"])
            for ds in (outputs or [])
        ]

        event = RunEvent(
            eventType=RunState.COMPLETE,
            eventTime=datetime.now(timezone.utc).isoformat(),
            run=Run(runId=run_id),
            job=Job(namespace="news_pipeline", name=job_name),
            inputs=[],
            outputs=output_datasets,
        )

        try:
            self._get_client().emit(event)
        except Exception as exc:
            logger.warning("Lineage COMPLETE 이벤트 발행 실패: %s", exc)

    def emit_run_fail(self, run_id: str, job_name: str, error: str) -> None:
        """파이프라인 단계 실패 이벤트 발행"""
        if not self.is_enabled:
            logger.debug("Lineage(disabled): FAIL %s — %s", job_name, error)
            return

        from openlineage.client.run import (
            Job,
            Run,
            RunEvent,
            RunState,
        )

        event = RunEvent(
            eventType=RunState.FAIL,
            eventTime=datetime.now(timezone.utc).isoformat(),
            run=Run(runId=run_id),
            job=Job(namespace="news_pipeline", name=job_name),
            inputs=[],
            outputs=[],
        )

        try:
            self._get_client().emit(event)
        except Exception as exc:
            logger.warning("Lineage FAIL 이벤트 발행 실패: %s", exc)
