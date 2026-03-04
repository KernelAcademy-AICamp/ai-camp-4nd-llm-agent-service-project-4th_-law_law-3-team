"""
법령 필터 서비스

PostgreSQL law_documents 테이블에서 법령을 필터 검색.
키워드 검색은 BM25 + ILIKE 하이브리드 (판례 필터와 동일 패턴).
"""

import datetime
import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import and_, case, func, or_, select
from sqlalchemy.exc import SQLAlchemyError

from app.core.database import async_session_factory
from app.models.fts_index import FtsIndex
from app.models.law_document import LawDocument

logger = logging.getLogger(__name__)


class LawFilterService:
    """법령 필터 서비스 클래스"""

    async def search_by_filter(
        self,
        keyword: str = "",
        law_type: Optional[str] = None,
        ministry: Optional[str] = None,
        promulgation_from: Optional[str] = None,
        promulgation_to: Optional[str] = None,
        enforcement_from: Optional[datetime.date] = None,
        enforcement_to: Optional[datetime.date] = None,
        sort: str = "relevance",
        offset: int = 0,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """
        필터 조건으로 법령 검색 (BM25 + ILIKE 하이브리드)

        텍스트 필드(법령명 + 조문)는 BM25 검색,
        법령명은 ILIKE 패턴 매칭으로 검색합니다.
        BM25 불가 시 전체 ILIKE로 fallback합니다.

        Args:
            keyword: 검색 키워드
            law_type: 법령 유형 (예: "법률", "시행령")
            ministry: 소관부처 (예: "법무부")
            promulgation_from: 공포일자 시작 (YYYYMMDD 문자열)
            promulgation_to: 공포일자 종료 (YYYYMMDD 문자열)
            enforcement_from: 시행일자 시작 (Date)
            enforcement_to: 시행일자 종료 (Date)
            sort: 정렬 기준 ("relevance" | "latest")
            offset: 페이지 오프셋
            limit: 결과 수

        Returns:
            {"total": int, "laws": [dict]}
        """
        try:
            async with async_session_factory() as session:
                # BM25로 텍스트 매칭 source_id 조회
                bm25_source_ids: list[str] = []
                if keyword:
                    bm25_source_ids = await self._bm25_match_ids(
                        session, keyword
                    )

                conditions = self._build_filter_conditions(
                    keyword=keyword,
                    law_type=law_type,
                    ministry=ministry,
                    promulgation_from=promulgation_from,
                    promulgation_to=promulgation_to,
                    enforcement_from=enforcement_from,
                    enforcement_to=enforcement_to,
                    bm25_source_ids=bm25_source_ids,
                )

                count_query = (
                    select(func.count()).select_from(LawDocument)
                )
                data_query = (
                    select(LawDocument)
                    .offset(offset)
                    .limit(limit)
                )

                if conditions:
                    where_clause = and_(*conditions)
                    count_query = count_query.where(where_clause)
                    data_query = data_query.where(where_clause)

                if sort == "relevance" and keyword:
                    data_query = self._apply_relevance_order(
                        data_query, keyword, bm25_source_ids
                    )
                else:
                    data_query = data_query.order_by(
                        LawDocument.enforcement_date.desc().nullslast()
                    )

                total = await session.scalar(count_query)
                result = await session.execute(data_query)
                laws = [
                    self._to_filter_item(law)
                    for law in result.scalars().all()
                ]

                return {"total": total or 0, "laws": laws}
        except (SQLAlchemyError, ConnectionError) as e:
            logger.warning("법령 필터 검색 실패: %s", e)
            return {"total": 0, "laws": []}

    @staticmethod
    async def _bm25_match_ids(
        session: Any,
        keyword: str,
        max_results: int = 500,
    ) -> list[str]:
        """BM25로 텍스트 매칭되는 법령 source_id 목록 반환.

        BM25 인덱스 미사용이거나 실패 시 빈 리스트 반환 (ILIKE fallback 유도).
        """
        try:
            from app.services.rag.keyword_search import (
                _BM25_INDEX_NAME,
                _tokenize,
                is_fts_available,
            )

            if not await is_fts_available():
                return []

            tokens = _tokenize(keyword)
            if not tokens:
                return []

            search_query = " ".join(tokens)
            bm25_query = func.to_bm25query(search_query, _BM25_INDEX_NAME)
            score_expr = FtsIndex.search_text.op("<@>")(bm25_query)

            stmt = (
                select(FtsIndex.source_id)
                .where(FtsIndex.data_type == "법령")
                .order_by(score_expr.asc())
                .limit(max_results)
            )

            result = await session.execute(stmt)
            return [row.source_id for row in result.all()]
        except Exception as e:
            logger.warning("BM25 매칭 실패, ILIKE fallback: %s", e)
            return []

    @staticmethod
    def _build_filter_conditions(
        keyword: str,
        law_type: Optional[str],
        ministry: Optional[str],
        promulgation_from: Optional[str],
        promulgation_to: Optional[str],
        enforcement_from: Optional[datetime.date],
        enforcement_to: Optional[datetime.date],
        bm25_source_ids: Optional[list[str]] = None,
    ) -> list[Any]:
        """필터 조건 리스트 생성.

        BM25 source_id가 있으면 텍스트는 BM25, 법령명은 ILIKE.
        BM25 불가(빈 리스트)면 전체 ILIKE fallback.
        """
        conditions: list[Any] = []

        if keyword:
            like_pattern = f"%{keyword}%"
            if bm25_source_ids:
                # BM25 텍스트 매칭 OR 법령명 ILIKE
                conditions.append(
                    or_(
                        LawDocument.law_id.in_(bm25_source_ids),
                        LawDocument.law_name.ilike(like_pattern),
                    )
                )
            else:
                # BM25 불가 시 ILIKE fallback
                conditions.append(
                    or_(
                        LawDocument.law_name.ilike(like_pattern),
                        LawDocument.abbreviation.ilike(like_pattern),
                    )
                )

        if law_type:
            conditions.append(LawDocument.law_type == law_type)

        if ministry:
            conditions.append(LawDocument.ministry == ministry)

        # 공포일자 (String YYYYMMDD → 문자열 비교)
        if promulgation_from:
            conditions.append(
                LawDocument.promulgation_date >= promulgation_from
            )
        if promulgation_to:
            conditions.append(
                LawDocument.promulgation_date <= promulgation_to
            )

        # 시행일자 (Date → 날짜 비교)
        if enforcement_from:
            conditions.append(
                LawDocument.enforcement_date >= enforcement_from
            )
        if enforcement_to:
            conditions.append(
                LawDocument.enforcement_date <= enforcement_to
            )

        return conditions

    @staticmethod
    def _apply_relevance_order(
        query: Any,
        keyword: str,
        bm25_source_ids: list[str],
    ) -> Any:
        """관련성 정렬 적용.

        BM25 매칭(가중치 2) + 법령명 매칭(가중치 1) 합산 후 정렬.
        BM25 불가 시 법령명/약칭 ILIKE 가중치 방식 사용.
        """
        like_pattern = f"%{keyword}%"
        if bm25_source_ids:
            relevance_score = (
                case(
                    (LawDocument.law_id.in_(bm25_source_ids), 2),
                    else_=0,
                )
                + case(
                    (LawDocument.law_name.ilike(like_pattern), 1),
                    else_=0,
                )
            )
        else:
            # ILIKE fallback 가중치
            relevance_score = (
                case(
                    (LawDocument.law_name.ilike(like_pattern), 2),
                    else_=0,
                )
                + case(
                    (LawDocument.abbreviation.ilike(like_pattern), 1),
                    else_=0,
                )
            )
        return query.order_by(
            relevance_score.desc(),
            LawDocument.enforcement_date.desc().nullslast(),
        )

    @staticmethod
    def _to_filter_item(law: LawDocument) -> Dict[str, Any]:
        """LawDocument → 필터 결과 아이템 변환"""
        return {
            "id": str(law.law_id),
            "law_name": law.law_name,
            "law_type": law.law_type,
            "ministry": law.ministry,
            "enforcement_date": (
                str(law.enforcement_date) if law.enforcement_date else None
            ),
            "promulgation_date": law.promulgation_date,
            "abbreviation": law.abbreviation,
            "ai_summary": (law.ai_summary or "")[:300] if law.ai_summary else None,
        }

    async def get_filter_options(self) -> Dict[str, List[str]]:
        """법령 유형, 소관부처 DISTINCT 목록 조회"""
        try:
            async with async_session_factory() as session:
                law_types_result = await session.execute(
                    select(LawDocument.law_type)
                    .distinct()
                    .where(LawDocument.law_type.is_not(None))
                    .order_by(LawDocument.law_type)
                )
                law_types = [
                    row[0] for row in law_types_result if row[0]
                ]

                ministries_result = await session.execute(
                    select(LawDocument.ministry)
                    .distinct()
                    .where(LawDocument.ministry.is_not(None))
                    .order_by(LawDocument.ministry)
                )
                ministries = [
                    row[0] for row in ministries_result if row[0]
                ]

                return {
                    "law_types": law_types,
                    "ministries": ministries,
                }
        except (SQLAlchemyError, ConnectionError) as e:
            logger.warning("법령 필터 옵션 조회 실패: %s", e)
            return {"law_types": [], "ministries": []}


_law_filter_service: Optional[LawFilterService] = None


def get_law_filter_service() -> LawFilterService:
    """LawFilterService 싱글톤 인스턴스 반환"""
    global _law_filter_service
    if _law_filter_service is None:
        _law_filter_service = LawFilterService()
    return _law_filter_service
