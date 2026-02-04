"""
판례 서비스

PostgreSQL에서 판례 상세 정보 조회
chat_service.py에서 추출
"""

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import select

from app.core.database import sync_session_factory
from app.models.precedent_document import PrecedentDocument

logger = logging.getLogger(__name__)


def fetch_precedent_details(source_ids: List[str]) -> Dict[str, Dict[str, str]]:
    """
    source_id 목록으로 PostgreSQL에서 판례 상세 정보 조회

    Args:
        source_ids: 판례 serial_number 목록

    Returns:
        {serial_number: {ruling, claim, reasoning, full_reason}} 딕셔너리
    """
    if not source_ids:
        return {}

    try:
        with sync_session_factory() as session:
            result = session.execute(
                select(PrecedentDocument).where(
                    PrecedentDocument.serial_number.in_(source_ids)
                )
            )
            precedents = result.scalars().all()

            return {
                p.serial_number: {
                    "ruling": p.ruling or "",
                    "claim": p.claim or "",
                    "reasoning": p.reasoning or "",
                    "full_reason": p.full_reason or "",
                    "full_text": p.full_text or "",  # 판례내용 전문
                    "decision_date": str(p.decision_date) if p.decision_date else "",
                    "case_type": p.case_type or "",
                    "summary": p.summary or "",  # 판시사항
                    "reference_provisions": p.reference_provisions or "",  # 참조조문
                    "reference_cases": p.reference_cases or "",  # 참조판례
                    "court_name": p.court_name or "",  # 법원명
                    "case_name": p.case_name or "",  # 사건명
                    "case_number": p.case_number or "",  # 사건번호
                }
                for p in precedents
            }
    except Exception as e:
        logger.warning("판례 상세 정보 조회 실패: %s", e)
        return {}


class PrecedentService:
    """판례 서비스 클래스"""

    def get_details(
        self,
        serial_numbers: List[str],
    ) -> Dict[str, Dict[str, str]]:
        """
        serial_number로 판례 상세 정보 조회

        Args:
            serial_numbers: 판례 serial_number 목록

        Returns:
            판례 상세 정보 딕셔너리
        """
        return fetch_precedent_details(serial_numbers)

    def get_by_serial_number(
        self,
        serial_number: str,
    ) -> Optional[Dict[str, Any]]:
        """
        단일 판례 조회

        Args:
            serial_number: 판례 serial_number

        Returns:
            판례 정보 또는 None
        """
        try:
            with sync_session_factory() as session:
                result = session.execute(
                    select(PrecedentDocument).where(
                        PrecedentDocument.serial_number == serial_number
                    )
                )
                precedent = result.scalar_one_or_none()

                if precedent:
                    return {
                        "serial_number": precedent.serial_number,
                        "case_name": precedent.case_name,
                        "case_number": precedent.case_number,
                        "court_name": precedent.court_name,
                        "decision_date": str(precedent.decision_date) if precedent.decision_date else None,
                        "ruling": precedent.ruling,
                        "claim": precedent.claim,
                        "reasoning": precedent.reasoning,
                        "full_reason": precedent.full_reason,
                    }
                return None
        except Exception as e:
            logger.warning("판례 조회 실패: %s", e)
            return None

    def search_by_case_number(
        self,
        case_number: str,
    ) -> Optional[Dict[str, Any]]:
        """
        사건번호로 판례 검색

        Args:
            case_number: 사건번호 (예: "2023다12345")

        Returns:
            판례 정보 또는 None
        """
        try:
            with sync_session_factory() as session:
                result = session.execute(
                    select(PrecedentDocument).where(
                        PrecedentDocument.case_number == case_number
                    )
                )
                precedent = result.scalar_one_or_none()

                if precedent:
                    return {
                        "serial_number": precedent.serial_number,
                        "case_name": precedent.case_name,
                        "case_number": precedent.case_number,
                        "court_name": precedent.court_name,
                        "ruling": precedent.ruling,
                        "reasoning": precedent.reasoning,
                    }
                return None
        except Exception as e:
            logger.warning("사건번호 검색 실패: %s", e)
            return None


_precedent_service: Optional[PrecedentService] = None


def get_precedent_service() -> PrecedentService:
    """PrecedentService 싱글톤 인스턴스 반환"""
    global _precedent_service
    if _precedent_service is None:
        _precedent_service = PrecedentService()
    return _precedent_service
