"""인메모리 Rate Limiter (사용자별 버킷 분리)

Red Team + Consultant 피드백 반영:
- collect 5회/시간, news 30회/시간 분리 버킷
- 스케일 아웃 시 Redis로 전환 가능하도록 인터페이스 분리
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


class RateLimitExceededError(Exception):
    """Rate limit 초과 예외"""

    def __init__(self, bucket: str, retry_after_seconds: int) -> None:
        self.bucket = bucket
        self.retry_after_seconds = retry_after_seconds
        super().__init__(
            f"Rate limit 초과: {bucket} 버킷. "
            f"{retry_after_seconds}초 후 재시도하세요."
        )


class InMemoryRateLimiter:
    """인메모리 슬라이딩 윈도우 Rate Limiter

    Args:
        max_requests: 윈도우 내 최대 요청 수
        window_seconds: 윈도우 크기 (초)
    """

    def __init__(self, max_requests: int, window_seconds: int = 3600) -> None:
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        # user_id -> list[timestamp]
        self._requests: dict[str, list[float]] = {}

    def _cleanup(self, user_id: str) -> None:
        """만료된 요청 타임스탬프 제거 + 빈 키 삭제 (메모리 누수 방지)"""
        now = time.time()
        cutoff = now - self._window_seconds
        if user_id in self._requests:
            self._requests[user_id] = [
                ts for ts in self._requests[user_id] if ts > cutoff
            ]
            # 빈 리스트면 키 자체를 삭제하여 메모리 누수 방지
            if not self._requests[user_id]:
                del self._requests[user_id]

    def check(self, user_id: str, bucket: str = "default") -> None:
        """Rate limit 검사. 초과 시 RateLimitExceededError 발생."""
        key = f"{user_id}:{bucket}"
        self._cleanup(key)

        timestamps = self._requests.get(key, [])
        if len(timestamps) >= self._max_requests:
            oldest = timestamps[0]
            retry_after = int(self._window_seconds - (time.time() - oldest)) + 1
            raise RateLimitExceededError(bucket, max(retry_after, 1))

    def record(self, user_id: str, bucket: str = "default") -> None:
        """요청 기록"""
        key = f"{user_id}:{bucket}"
        self._cleanup(key)
        if key not in self._requests:
            self._requests[key] = []
        self._requests[key].append(time.time())

    def check_and_record(self, user_id: str, bucket: str = "default") -> None:
        """검사 + 기록 (원자적 수행)"""
        self.check(user_id, bucket)
        self.record(user_id, bucket)
