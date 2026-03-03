"""웹툰 스토리보드 파이프라인 서비스

3-Chain AI 파이프라인:
Chain 1: 대본 → 장면 분할 (Solar Pro2)
Chain 2: 장면 → 이미지 프롬프트 (텍스트 조합)
Chain 3: 이미지 생성 (Gemini 3 Pro Image)
"""

import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

from app.modules.content_marketing.schema import (
    ImageStatus,
    WebtoonGenerateRequest,
    WebtoonPanel,
    WebtoonStreamEvent,
)
from app.modules.storyboard.service.job_manager import JobManager, JobStatus
from app.tools.webtoon.image_generator import generate_webtoon_image
from app.tools.webtoon.panel_planner import split_script_to_scenes
from app.tools.webtoon.prompt_builder import build_image_prompt

logger = logging.getLogger(__name__)

# storyboard 모듈의 JobManager를 재사용 (별도 인스턴스)
webtoon_job_manager = JobManager()

# 동시 이미지 생성 제한
_semaphore = asyncio.Semaphore(3)

# SSE 이벤트 큐 (job_id → asyncio.Queue)
_event_queues: dict[str, asyncio.Queue[WebtoonStreamEvent | None]] = {}


def _emit_event(job_id: str, event: WebtoonStreamEvent) -> None:
    """SSE 이벤트 큐에 이벤트 발행"""
    queue = _event_queues.get(job_id)
    if queue:
        queue.put_nowait(event)


async def run_webtoon_pipeline(
    job_id: str,
    request: WebtoonGenerateRequest,
) -> None:
    """웹툰 스토리보드 전체 파이프라인 실행 (백그라운드)"""
    # 이벤트 큐 생성
    _event_queues[job_id] = asyncio.Queue()

    try:
        await webtoon_job_manager.update_progress(
            job_id, status=JobStatus.PROCESSING, message="장면 분할 중..."
        )

        # Chain 1: 장면 분할 시작 알림
        _emit_event(
            job_id,
            WebtoonStreamEvent(event="scene_split_start"),
        )

        # Chain 1: 대본 → 장면 분할
        panels: list[WebtoonPanel] = await split_script_to_scenes(
            topic=request.topic,
            sections=request.sections,
            target_panels=request.panel_count,
        )

        total = len(panels)

        # 장면 분할 완료 알림
        _emit_event(
            job_id,
            WebtoonStreamEvent(event="scene_split_done", total_panels=total),
        )

        await webtoon_job_manager.update_progress(
            job_id,
            message=f"장면 분할 완료 ({total}패널). 이미지 생성 시작...",
        )

        # Chain 2 + Chain 3: 순차 이미지 생성 (캐릭터 일관성)
        reference_image: bytes | None = None

        for idx, panel in enumerate(panels):
            panel.image_status = ImageStatus.GENERATING

            # 패널 시작 알림
            _emit_event(
                job_id,
                WebtoonStreamEvent(
                    event="panel_start",
                    panel_number=panel.panel_number,
                    total_panels=total,
                    section=panel.section,
                    caption=panel.script_excerpt,
                    scene_description=panel.scene_description,
                ),
            )

            # Chain 2: 이미지 프롬프트 빌드
            panel.image_prompt = build_image_prompt(
                panel, reference_image is not None
            )
            panel.prompt_version = "1.0"

            # Chain 3: 이미지 생성
            start_ms = time.monotonic_ns() // 1_000_000
            async with _semaphore:
                result = await generate_webtoon_image(
                    panel=panel,
                    reference_image=reference_image,
                    job_id=job_id,
                )
            elapsed_ms = (time.monotonic_ns() // 1_000_000) - start_ms
            panel.generation_cost_ms = elapsed_ms

            if result["success"]:
                panel.image_url = result["image_url"]
                panel.image_status = ImageStatus.COMPLETED
                panel.model_version = result.get(
                    "model_version", "gemini-3-pro-image-preview"
                )
                panel.safety_flags = result.get("safety_flags", [])
                # 첫 성공 패널을 레퍼런스로 저장
                if reference_image is None and result.get("image_data"):
                    reference_image = result["image_data"]

                _emit_event(
                    job_id,
                    WebtoonStreamEvent(
                        event="panel_complete",
                        panel_number=panel.panel_number,
                        total_panels=total,
                        section=panel.section,
                        image_url=panel.image_url,
                    ),
                )
            else:
                panel.image_status = ImageStatus.ERROR
                panel.error_message = result.get("error", "이미지 생성 실패")

                _emit_event(
                    job_id,
                    WebtoonStreamEvent(
                        event="panel_failed",
                        panel_number=panel.panel_number,
                        total_panels=total,
                        error=panel.error_message,
                        image_url=result.get("image_url"),
                    ),
                )

            await webtoon_job_manager.update_progress(
                job_id,
                current_step=idx + 1,
                message=f"이미지 생성 중... ({idx + 1}/{total})",
            )

        # 완료
        await webtoon_job_manager.complete_job(
            job_id,
            result={"panels": [p.model_dump() for p in panels]},
        )

        _emit_event(
            job_id,
            WebtoonStreamEvent(event="all_done", total_panels=total),
        )

    except Exception as e:
        logger.error("웹툰 파이프라인 오류 (job_id=%s): %s", job_id, e, exc_info=True)
        await webtoon_job_manager.fail_job(job_id, str(e))
        _emit_event(
            job_id,
            WebtoonStreamEvent(event="error", error=str(e)),
        )
    finally:
        # 종료 시그널
        queue = _event_queues.get(job_id)
        if queue:
            queue.put_nowait(None)


async def webtoon_sse_generator(
    job_id: str,
) -> AsyncGenerator[str, None]:
    """SSE 이벤트 스트리밍 제너레이터 (30초 heartbeat 포함)"""
    queue = _event_queues.get(job_id)
    if not queue:
        # 아직 큐가 생성되지 않은 경우 잠시 대기
        for _ in range(10):
            await asyncio.sleep(0.5)
            queue = _event_queues.get(job_id)
            if queue:
                break

    if not queue:
        yield f"data: {json.dumps({'event': 'error', 'error': 'Job not found'})}\n\n"
        return

    heartbeat_interval = 30.0

    while True:
        try:
            event = await asyncio.wait_for(queue.get(), timeout=heartbeat_interval)
        except asyncio.TimeoutError:
            # heartbeat 전송
            yield ":heartbeat\n\n"
            continue

        if event is None:
            # 스트림 종료
            break

        yield f"data: {event.model_dump_json()}\n\n"

        if event.event in ("all_done", "error"):
            break

    # 큐 정리
    _event_queues.pop(job_id, None)


async def regenerate_single_panel(
    job_id: str,
    panel_number: int,
) -> WebtoonStreamEvent:
    """개별 패널 이미지 재생성"""
    job = webtoon_job_manager.get_job(job_id)
    if not job or not job.result:
        msg = "Job 결과를 찾을 수 없습니다."
        raise ValueError(msg)

    panels_data: list[dict[str, Any]] = job.result.get("panels", [])
    target_data = None
    target_idx = -1
    for idx, p in enumerate(panels_data):
        if p.get("panel_number") == panel_number:
            target_data = p
            target_idx = idx
            break

    if target_data is None or target_idx < 0:
        msg = f"패널 {panel_number}을 찾을 수 없습니다."
        raise ValueError(msg)

    panel = WebtoonPanel(**target_data)
    panel.image_prompt = build_image_prompt(panel, has_reference=False)

    async with _semaphore:
        result = await generate_webtoon_image(
            panel=panel,
            reference_image=None,
            job_id=job_id,
        )

    if result["success"]:
        panel.image_url = result["image_url"]
        panel.image_status = ImageStatus.COMPLETED
        panel.error_message = None

        # Job 결과 업데이트
        panels_data[target_idx] = panel.model_dump()
        job.result["panels"] = panels_data

        return WebtoonStreamEvent(
            event="panel_complete",
            panel_number=panel_number,
            image_url=panel.image_url,
        )

    panel.image_status = ImageStatus.ERROR
    panel.error_message = result.get("error", "재생성 실패")
    panels_data[target_idx] = panel.model_dump()
    job.result["panels"] = panels_data

    return WebtoonStreamEvent(
        event="panel_failed",
        panel_number=panel_number,
        error=panel.error_message,
        image_url=result.get("image_url"),
    )
