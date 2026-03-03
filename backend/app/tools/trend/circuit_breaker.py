"""서킷 브레이커 (점증 backoff)

외부 소스 API 호출 시 연속 실패를 감지하고 일시적으로 호출을 차단하여
불필요한 요청과 에러 로그를 줄인다.

상태:
  - CLOSED: 정상 동작, 모든 요청 통과
  - OPEN: 차단 중, 요청 즉시 거부 (backoff 대기)
  - HALF_OPEN: backoff 만료 후 제한적 요청 허용 (성공 시 CLOSED, 실패 시 OPEN)

Backoff 전략: [60, 300, 900]초 점증 (1분 → 5분 → 15분)
"""

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# 점증 backoff 지속 시간 (초)
_BACKOFF_DURATIONS: list[int] = [60, 300, 900]


@dataclass
class _ServiceState:
    """서비스별 서킷 브레이커 상태"""

    failure_count: int = 0
    backoff_level: int = 0
    last_failure_time: float = 0.0
    half_open_attempts: int = 0


class CircuitBreaker:
    """점증 backoff 서킷 브레이커

    Args:
        failure_threshold: OPEN 전환에 필요한 연속 실패 횟수
        half_open_max_requests: HALF_OPEN 상태에서 허용할 최대 요청 수
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        half_open_max_requests: int = 2,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._half_open_max_requests = half_open_max_requests
        self._states: dict[str, _ServiceState] = {}

    def _get_state(self, service: str) -> _ServiceState:
        """서비스 상태 조회 (없으면 생성)"""
        if service not in self._states:
            self._states[service] = _ServiceState()
        return self._states[service]

    def _get_backoff_duration(self, level: int) -> int:
        """현재 backoff 레벨에 해당하는 대기 시간(초)"""
        idx = min(level, len(_BACKOFF_DURATIONS) - 1)
        return _BACKOFF_DURATIONS[idx]

    def is_open(self, service: str) -> bool:
        """서킷이 OPEN 상태인지 확인 (True면 요청 차단)"""
        state = self._get_state(service)

        if state.failure_count < self._failure_threshold:
            return False

        # backoff 시간이 지났으면 HALF_OPEN 전환
        elapsed = time.monotonic() - state.last_failure_time
        backoff = self._get_backoff_duration(state.backoff_level)
        if elapsed >= backoff:
            return False  # HALF_OPEN (is_half_open_allowed로 세부 제어)

        return True

    def is_half_open_allowed(self, service: str) -> bool:
        """HALF_OPEN 상태에서 추가 요청이 허용되는지 확인"""
        state = self._get_state(service)

        if state.failure_count < self._failure_threshold:
            return True  # CLOSED 상태

        elapsed = time.monotonic() - state.last_failure_time
        backoff = self._get_backoff_duration(state.backoff_level)
        if elapsed < backoff:
            return False  # 아직 OPEN (backoff 대기 중)

        # HALF_OPEN: 제한된 횟수만 허용
        return state.half_open_attempts < self._half_open_max_requests

    def record_failure(self, service: str) -> None:
        """실패 기록 — 임계치 도달 시 OPEN 전환"""
        state = self._get_state(service)
        state.failure_count += 1
        state.last_failure_time = time.monotonic()
        state.half_open_attempts = 0

        if state.failure_count >= self._failure_threshold:
            backoff = self._get_backoff_duration(state.backoff_level)
            logger.warning(
                "CircuitBreaker OPEN: service='%s', failures=%d, backoff=%ds (level=%d)",
                service,
                state.failure_count,
                backoff,
                state.backoff_level,
            )
            # 다음 OPEN 시 더 긴 backoff
            state.backoff_level = min(
                state.backoff_level + 1, len(_BACKOFF_DURATIONS) - 1,
            )

    def record_success(self, service: str) -> None:
        """성공 기록 — CLOSED로 리셋"""
        state = self._get_state(service)

        if state.failure_count >= self._failure_threshold:
            logger.info(
                "CircuitBreaker CLOSED: service='%s' (이전 failures=%d)",
                service,
                state.failure_count,
            )

        state.failure_count = 0
        state.backoff_level = 0
        state.half_open_attempts = 0

    def record_half_open_attempt(self, service: str) -> None:
        """HALF_OPEN 상태에서 시도 횟수 증가"""
        state = self._get_state(service)
        state.half_open_attempts += 1

    def get_status(self, service: str) -> str:
        """서비스의 현재 서킷 상태 문자열 반환"""
        state = self._get_state(service)

        if state.failure_count < self._failure_threshold:
            return "CLOSED"

        elapsed = time.monotonic() - state.last_failure_time
        backoff = self._get_backoff_duration(state.backoff_level)
        if elapsed >= backoff:
            return "HALF_OPEN"

        return "OPEN"
