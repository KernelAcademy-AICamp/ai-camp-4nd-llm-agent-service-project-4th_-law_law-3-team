"""파이프라인 운영 리포트 생성"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from app.tools.news_pipeline.models import PipelineResult

logger = logging.getLogger(__name__)

REPORT_DIR = Path("data/news_pipeline/reports")


class PipelineReporter:
    """일일 운영 리포트 생성기

    v0.2.0: Slack Webhook 알림 + KPI 메트릭 추가
    """

    def __init__(self, webhook_url: str | None = None) -> None:
        self._webhook_url = webhook_url

    def generate(self, result: PipelineResult) -> Path:
        """리포트 파일 생성 후 경로 반환"""
        REPORT_DIR.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = REPORT_DIR / f"report_{timestamp}.json"

        duration = (
            (result.finished_at - result.started_at).total_seconds()
            if result.finished_at else None
        )

        # v0.2.0: KPI 메트릭 계산
        kpi = self._compute_kpi(result, duration)

        report_data = {
            "run_id": result.run_id,
            "started_at": result.started_at.isoformat(),
            "finished_at": result.finished_at.isoformat() if result.finished_at else None,
            "duration_seconds": duration,
            "kpi": kpi,
            "stats": {
                "total_collected": result.total_collected,
                "total_deduplicated": result.total_deduplicated,
                "total_cleaned": result.total_cleaned,
                "total_summarized": result.total_summarized,
                "total_stored": result.total_stored,
                "total_chunked": result.total_chunked,
            },
            "source_stats": {
                name: {
                    "collected": stat.collected,
                    "deduplicated": stat.deduplicated,
                    "failed": stat.failed,
                    "errors": stat.errors,
                }
                for name, stat in result.source_stats.items()
            },
            "errors": [
                {
                    "stage": e.stage,
                    "article_url": e.article_url,
                    "error_type": e.error_type,
                    "error_message": e.error_message,
                    "timestamp": e.timestamp.isoformat(),
                }
                for e in result.errors
            ],
        }

        report_path.write_text(
            json.dumps(report_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # 콘솔 요약
        logger.info(
            "━━━ 파이프라인 리포트 ━━━\n"
            "수집: %d건 | 중복제거: -%d건 | 정제: %d건 | 요약: %d건 | 저장: %d건 | 청킹: %d건\n"
            "에러: %d건 | 소요: %s\n"
            "KPI → 수집성공률: %.1f%% | 요약성공률: %.1f%% | 중복률: %.1f%%",
            result.total_collected,
            result.total_deduplicated,
            result.total_cleaned,
            result.total_summarized,
            result.total_stored,
            result.total_chunked,
            len(result.errors),
            f"{duration:.1f}초" if duration else "진행중",
            kpi["collect_success_rate"],
            kpi["summary_success_rate"],
            kpi["duplicate_rate"],
        )

        # v0.2.0: 에러 발생 시 Slack 알림
        if result.errors and self._webhook_url:
            self._send_alert(result, kpi)

        return report_path

    @staticmethod
    def _compute_kpi(result: PipelineResult, duration: float | None) -> dict[str, float]:
        """KPI 메트릭 산출 (Consultant 피드백 반영)"""
        total = result.total_collected or 1  # ZeroDivision 방지
        return {
            "collect_success_rate": (total - len([
                e for e in result.errors if e.stage == "collect"
            ])) / total * 100,
            "summary_success_rate": result.total_summarized / max(result.total_cleaned, 1) * 100,
            "duplicate_rate": result.total_deduplicated / total * 100,
            "avg_processing_seconds": duration / max(result.total_stored, 1) if duration else 0,
        }

    def _send_alert(self, result: PipelineResult, kpi: dict[str, float]) -> None:
        """Slack Webhook으로 에러 알림 전송 (Red Team 피드백 반영)"""
        import httpx

        error_summary = "\n".join(
            f"- [{e.stage}] {e.error_type}: {e.error_message[:100]}"
            for e in result.errors[:5]  # 최대 5건
        )
        text = (
            f"⚠️ *뉴스 파이프라인 에러 알림*\n"
            f"Run: `{result.run_id}`\n"
            f"에러: {len(result.errors)}건\n"
            f"수집성공률: {kpi['collect_success_rate']:.1f}%\n"
            f"```{error_summary}```"
        )

        try:
            httpx.post(self._webhook_url, json={"text": text}, timeout=10.0)  # type: ignore[arg-type]
        except Exception as exc:
            logger.warning("Slack 알림 전송 실패: %s", exc)
