"""
세션 저장소

에이전트 대화 상태를 유지하는 인메모리 세션 저장소
"""

import time
from functools import lru_cache
from threading import Lock
from typing import Any, Dict, Optional


class SessionStore:
    """
    인메모리 세션 저장소

    에이전트별 대화 상태를 세션 단위로 관리
    TTL 기반 자동 만료 지원
    """

    DEFAULT_TTL_SECONDS = 3600  # 1시간

    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        self._store: Dict[str, Dict[str, Any]] = {}
        self._timestamps: Dict[str, float] = {}
        self._ttl_seconds = ttl_seconds
        self._lock = Lock()

    def get(self, session_id: str) -> Dict[str, Any]:
        """
        세션 데이터 조회

        Args:
            session_id: 세션 ID

        Returns:
            세션 데이터 (없으면 빈 딕셔너리)
        """
        with self._lock:
            self._cleanup_expired()

            if session_id not in self._store:
                return {}

            # 접근 시 타임스탬프 갱신
            self._timestamps[session_id] = time.time()
            return self._store[session_id].copy()

    def set(self, session_id: str, data: Dict[str, Any]) -> None:
        """
        세션 데이터 설정

        Args:
            session_id: 세션 ID
            data: 저장할 데이터
        """
        with self._lock:
            self._store[session_id] = data.copy()
            self._timestamps[session_id] = time.time()

    def update(self, session_id: str, key: str, value: Any) -> None:
        """
        세션 데이터 부분 업데이트

        Args:
            session_id: 세션 ID
            key: 업데이트할 키
            value: 새 값
        """
        with self._lock:
            if session_id not in self._store:
                self._store[session_id] = {}

            self._store[session_id][key] = value
            self._timestamps[session_id] = time.time()

    def delete(self, session_id: str) -> bool:
        """
        세션 삭제

        Args:
            session_id: 세션 ID

        Returns:
            True if 삭제됨, False if 존재하지 않음
        """
        with self._lock:
            if session_id in self._store:
                del self._store[session_id]
                del self._timestamps[session_id]
                return True
            return False

    def clear(self) -> None:
        """모든 세션 삭제"""
        with self._lock:
            self._store.clear()
            self._timestamps.clear()

    def _cleanup_expired(self) -> None:
        """만료된 세션 정리 (락 내부에서 호출)"""
        current_time = time.time()
        expired = [
            sid for sid, ts in self._timestamps.items()
            if current_time - ts > self._ttl_seconds
        ]
        for sid in expired:
            del self._store[sid]
            del self._timestamps[sid]

    @property
    def size(self) -> int:
        """현재 세션 수"""
        return len(self._store)


_session_store: Optional[SessionStore] = None


@lru_cache(maxsize=1)
def get_session_store() -> SessionStore:
    """세션 저장소 싱글톤 인스턴스 반환"""
    return SessionStore()
