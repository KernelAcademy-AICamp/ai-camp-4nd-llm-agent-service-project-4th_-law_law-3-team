"""비동기 작업 관리자 - SSE 기반 진행 상태 관리"""
import asyncio
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """작업 상태"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobProgress(BaseModel):
    """작업 진행 상태"""
    job_id: str
    status: JobStatus
    progress: int  # 0-100
    current_step: int
    total_steps: int
    message: str
    result: Optional[dict] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str


class JobManager:
    """비동기 작업 관리자"""

    def __init__(self) -> None:
        self._jobs: dict[str, JobProgress] = {}
        self._subscribers: dict[str, list[asyncio.Queue]] = {}

    def create_job(self, total_steps: int = 1) -> str:
        """
        새 작업 생성

        Args:
            total_steps: 전체 단계 수

        Returns:
            작업 ID
        """
        job_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        self._jobs[job_id] = JobProgress(
            job_id=job_id,
            status=JobStatus.PENDING,
            progress=0,
            current_step=0,
            total_steps=total_steps,
            message="작업 대기 중...",
            created_at=now,
            updated_at=now,
        )
        self._subscribers[job_id] = []

        return job_id

    async def update_progress(
        self,
        job_id: str,
        status: Optional[JobStatus] = None,
        current_step: Optional[int] = None,
        message: Optional[str] = None,
        result: Optional[dict] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        작업 진행 상태 업데이트

        Args:
            job_id: 작업 ID
            status: 새 상태
            current_step: 현재 단계
            message: 상태 메시지
            result: 결과 데이터
            error: 에러 메시지
        """
        if job_id not in self._jobs:
            return

        job = self._jobs[job_id]

        if status is not None:
            job.status = status
        if current_step is not None:
            job.current_step = current_step
            if job.total_steps > 0:
                job.progress = int((current_step / job.total_steps) * 100)
            else:
                job.progress = 0
        if message is not None:
            job.message = message
        if result is not None:
            job.result = result
        if error is not None:
            job.error = error

        job.updated_at = datetime.utcnow().isoformat()

        # 구독자들에게 알림
        await self._notify_subscribers(job_id)

    async def complete_job(
        self,
        job_id: str,
        result: Optional[dict] = None,
    ) -> None:
        """작업 완료 처리"""
        await self.update_progress(
            job_id,
            status=JobStatus.COMPLETED,
            current_step=self._jobs[job_id].total_steps,
            message="완료",
            result=result,
        )

    async def fail_job(
        self,
        job_id: str,
        error: str,
    ) -> None:
        """작업 실패 처리"""
        await self.update_progress(
            job_id,
            status=JobStatus.FAILED,
            message="실패",
            error=error,
        )

    def get_job(self, job_id: str) -> Optional[JobProgress]:
        """작업 상태 조회"""
        return self._jobs.get(job_id)

    async def subscribe(self, job_id: str) -> AsyncGenerator[JobProgress, None]:
        """
        작업 상태 구독 (SSE용)

        Args:
            job_id: 작업 ID

        Yields:
            작업 진행 상태
        """
        if job_id not in self._jobs:
            return

        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers[job_id].append(queue)

        try:
            # 현재 상태 먼저 전송
            yield self._jobs[job_id]

            # 완료/실패까지 업데이트 대기
            while True:
                job = await queue.get()
                yield job

                if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                    break
        finally:
            # Race condition 방지: dict에서 안전하게 제거
            try:
                subscribers = self._subscribers.get(job_id)
                if subscribers and queue in subscribers:
                    subscribers.remove(queue)
            except (KeyError, ValueError):
                pass  # 이미 제거됨

    async def _notify_subscribers(self, job_id: str) -> None:
        """구독자들에게 업데이트 알림"""
        if job_id not in self._subscribers:
            return

        job = self._jobs[job_id]
        for queue in self._subscribers[job_id]:
            await queue.put(job)

    def cleanup_job(self, job_id: str) -> None:
        """작업 정리 (메모리에서 제거)"""
        if job_id in self._jobs:
            del self._jobs[job_id]
        if job_id in self._subscribers:
            del self._subscribers[job_id]


# 전역 인스턴스
job_manager = JobManager()


async def run_batch_image_generation(
    job_id: str,
    items: list[dict],
    generate_fn: Callable,
) -> dict:
    """
    일괄 스토리보드 이미지 생성 작업 실행

    Args:
        job_id: 작업 ID
        items: 타임라인 항목 목록
        generate_fn: 이미지 생성 함수

    Returns:
        생성 결과
    """
    results: list[dict] = []
    failed: list[str] = []

    await job_manager.update_progress(
        job_id,
        status=JobStatus.PROCESSING,
        message="스토리보드 이미지 생성 시작...",
    )

    for idx, item in enumerate(items):
        await job_manager.update_progress(
            job_id,
            current_step=idx + 1,
            message=f"스토리보드 이미지 생성 중... ({idx + 1}/{len(items)})",
        )

        try:
            result = await generate_fn(
                item_id=item["id"],
                title=item["title"],
                description=item["description"],
                participants=item.get("participants", []),
            )

            if result["success"]:
                results.append({
                    "item_id": item["id"],
                    "image_url": result["image_url"],
                    "image_prompt": result["image_prompt"],
                })
            else:
                failed.append(item["id"])
        except Exception as e:
            logger.error(f"이미지 생성 실패 (item_id={item['id']}): {e}", exc_info=True)
            failed.append(item["id"])

    final_result = {
        "generated": results,
        "failed": failed,
        "total": len(items),
        "success_count": len(results),
    }

    await job_manager.complete_job(job_id, final_result)

    return final_result
