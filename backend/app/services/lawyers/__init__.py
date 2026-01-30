"""
변호사 서비스 모듈
"""

from app.services.lawyers.lawyer_service import (
    LawyerService,
    get_lawyer_service,
)

__all__ = [
    "LawyerService",
    "get_lawyer_service",
]
