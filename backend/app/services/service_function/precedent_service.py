"""
판례 서비스

PostgreSQL에서 판례 상세 정보 조회
"""

import datetime
import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import and_, case, func, or_, select
from sqlalchemy.exc import SQLAlchemyError

from app.core.database import async_session_factory
from app.models.precedent_document import PrecedentDocument

logger = logging.getLogger(__name__)


async def fetch_precedent_details(source_ids: List[str]) -> Dict[str, Dict[str, str]]:
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
        async with async_session_factory() as session:
            result = await session.execute(
                select(PrecedentDocument).where(
                    PrecedentDocument.serial_number.in_(source_ids)
                )
            )
            precedents = result.scalars().all()

            return {
                str(p.serial_number): {
                    "ruling": str(p.ruling or ""),
                    "claim": str(p.claim or ""),
                    "reasoning": str(p.reasoning or ""),
                    "full_reason": str(p.full_reason or ""),
                    "full_text": str(p.full_text or ""),  # 판례내용 전문
                    "decision_date": str(p.decision_date) if p.decision_date else "",
                    "case_type": str(p.case_type or ""),
                    "summary": str(p.summary or ""),  # 판시사항
                    "reference_provisions": str(p.reference_provisions or ""),  # 참조조문
                    "reference_cases": str(p.reference_cases or ""),  # 참조판례
                    "court_name": str(p.court_name or ""),  # 법원명
                    "case_name": str(p.case_name or ""),  # 사건명
                    "case_number": str(p.case_number or ""),  # 사건번호
                }
                for p in precedents
            }
    except (SQLAlchemyError, ConnectionError) as e:
        logger.warning("판례 상세 정보 조회 실패: %s", e)
        return {}


class PrecedentService:
    """판례 서비스 클래스"""

    async def get_details(
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
        return await fetch_precedent_details(serial_numbers)

    async def get_by_serial_number(
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
            async with async_session_factory() as session:
                result = await session.execute(
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
        except (SQLAlchemyError, ConnectionError) as e:
            logger.warning("판례 조회 실패: %s", e)
            return None

    async def search_by_case_number(
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
            async with async_session_factory() as session:
                result = await session.execute(
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
        except (SQLAlchemyError, ConnectionError) as e:
            logger.warning("사건번호 검색 실패: %s", e)
            return None


    async def search_by_filter(
        self,
        keyword: str = "",
        case_type: Optional[str] = None,
        date_from: Optional[datetime.date] = None,
        date_to: Optional[datetime.date] = None,
        sort: str = "relevance",
        offset: int = 0,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """
        필터 조건으로 판례 검색 (PostgreSQL 직접 쿼리)

        Args:
            keyword: 검색 키워드 (case_name, summary ILIKE)
            case_type: 사건종류명 (예: "민사", "형사")
            date_from: 선고일 시작
            date_to: 선고일 종료
            sort: 정렬 기준 ("relevance" | "latest")
            offset: 페이지 오프셋
            limit: 결과 수

        Returns:
            {"total": int, "precedents": [dict]}
        """
        try:
            async with async_session_factory() as session:
                conditions = self._build_filter_conditions(
                    keyword, case_type, date_from, date_to
                )

                count_query = (
                    select(func.count()).select_from(PrecedentDocument)
                )
                data_query = (
                    select(PrecedentDocument)
                    .offset(offset)
                    .limit(limit)
                )

                if conditions:
                    where_clause = and_(*conditions)
                    count_query = count_query.where(where_clause)
                    data_query = data_query.where(where_clause)

                if sort == "relevance" and keyword:
                    like_pattern = f"%{keyword}%"
                    relevance_score = (
                        case(
                            (PrecedentDocument.case_name.ilike(like_pattern), 3),
                            else_=0,
                        )
                        + case(
                            (PrecedentDocument.case_number.ilike(like_pattern), 2),
                            else_=0,
                        )
                        + case(
                            (PrecedentDocument.summary.ilike(like_pattern), 1),
                            else_=0,
                        )
                    )
                    data_query = data_query.order_by(
                        relevance_score.desc(),
                        PrecedentDocument.decision_date.desc(),
                    )
                else:
                    data_query = data_query.order_by(
                        PrecedentDocument.decision_date.desc()
                    )

                total = await session.scalar(count_query)
                result = await session.execute(data_query)
                precedents = [
                    self._to_filter_item(p) for p in result.scalars().all()
                ]

                return {"total": total or 0, "precedents": precedents}
        except (SQLAlchemyError, ConnectionError) as e:
            logger.warning("판례 필터 검색 실패: %s", e)
            return {"total": 0, "precedents": []}

    @staticmethod
    def _build_filter_conditions(
        keyword: str,
        case_type: Optional[str],
        date_from: Optional[datetime.date],
        date_to: Optional[datetime.date],
    ) -> list[Any]:
        """필터 조건 리스트 생성"""
        conditions: list[Any] = []
        if keyword:
            like_pattern = f"%{keyword}%"
            conditions.append(
                or_(
                    PrecedentDocument.case_name.ilike(like_pattern),
                    PrecedentDocument.summary.ilike(like_pattern),
                    PrecedentDocument.case_number.ilike(like_pattern),
                )
            )
        if case_type:
            conditions.append(PrecedentDocument.case_type == case_type)
        if date_from:
            conditions.append(PrecedentDocument.decision_date >= date_from)
        if date_to:
            conditions.append(PrecedentDocument.decision_date <= date_to)
        return conditions

    @staticmethod
    def _to_filter_item(p: PrecedentDocument) -> Dict[str, Any]:
        """PrecedentDocument → 필터 결과 아이템 변환"""
        return {
            "id": str(p.serial_number),
            "serial_number": str(p.serial_number),
            "case_name": p.case_name,
            "case_number": p.case_number,
            "case_type": p.case_type,
            "court_name": p.court_name,
            "decision_date": str(p.decision_date) if p.decision_date else None,
            "summary": (p.summary or "")[:300],
        }

    async def get_case_types(self) -> List[str]:
        """사건종류명 DISTINCT 목록 조회"""
        try:
            async with async_session_factory() as session:
                result = await session.execute(
                    select(PrecedentDocument.case_type)
                    .distinct()
                    .where(PrecedentDocument.case_type.is_not(None))
                    .order_by(PrecedentDocument.case_type)
                )
                return [row[0] for row in result if row[0]]
        except (SQLAlchemyError, ConnectionError) as e:
            logger.warning("사건종류 목록 조회 실패: %s", e)
            return []


_precedent_service: Optional[PrecedentService] = None


def get_precedent_service() -> PrecedentService:
    """PrecedentService 싱글톤 인스턴스 반환"""
    global _precedent_service
    if _precedent_service is None:
        _precedent_service = PrecedentService()
    return _precedent_service
