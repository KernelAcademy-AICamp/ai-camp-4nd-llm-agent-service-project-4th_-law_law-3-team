"""
요청 단위 로깅 유틸리티

trace_id 기반 요청 추적 및 성능 로깅
"""

import logging
import time
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def get_logger(name: str) -> logging.Logger:
    """
    모듈별 로거 반환

    Args:
        name: 로거 이름 (보통 __name__ 사용)

    Returns:
        설정된 Logger 인스턴스
    """
    return logging.getLogger(name)


def log_request(
    trace_id: str,
    agent: str,
    latency_ms: float,
    success: bool = True,
    error: Optional[str] = None,
) -> None:
    """
    요청 처리 로그 기록

    Args:
        trace_id: 요청 추적 ID
        agent: 처리한 에이전트 이름
        latency_ms: 처리 시간 (밀리초)
        success: 성공 여부
        error: 에러 메시지 (실패 시)
    """
    logger = get_logger("app.request")

    log_data = {
        "trace_id": trace_id,
        "agent": agent,
        "latency_ms": round(latency_ms, 2),
        "success": success,
    }

    if error:
        log_data["error"] = error

    if success:
        logger.info("Request processed: %s", log_data)
    else:
        logger.error("Request failed: %s", log_data)


def log_agent_execution(func: F) -> F:
    """
    에이전트 실행 로깅 데코레이터

    Usage:
        @log_agent_execution
        async def process(self, message: str) -> AgentResponse:
            ...
    """
    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = get_logger("app.agent")
        start_time = time.perf_counter()

        # self.name 접근 시도 (BaseAgent 상속 클래스)
        agent_name = "unknown"
        if args and hasattr(args[0], "name"):
            agent_name = args[0].name

        try:
            result = await func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                "Agent executed: agent=%s, latency_ms=%.2f",
                agent_name,
                elapsed_ms,
            )
            return result
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "Agent failed: agent=%s, latency_ms=%.2f, error=%s",
                agent_name,
                elapsed_ms,
                str(e),
            )
            raise

    return wrapper  # type: ignore[return-value]
