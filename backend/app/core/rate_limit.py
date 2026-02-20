"""
Rate Limiting 설정

slowapi 기반 IP별 요청 제한.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import settings

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[f"{settings.RATE_LIMIT_PER_MINUTE}/minute"],
    storage_uri=settings.RATE_LIMIT_STORAGE_URI,
)

# AI 엔드포인트용 더 엄격한 제한 데코레이터 문자열
AI_RATE_LIMIT = f"{settings.RATE_LIMIT_AI_PER_MINUTE}/minute"
