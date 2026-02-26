"""
PostgreSQL 그래프 서비스

Neo4j GraphService와 동일한 인터페이스를 PostgreSQL로 구현.
Feature flag (USE_PG_GRAPH) 활성화 시 Neo4j 대신 사용.

테이블:
- law_documents: 법령 (abbreviation, citation_count 포함)
- statute_aliases: 비공식 약칭
- statute_hierarchy: 법령 계급 (child → parent = 시행령 → 법률)
- statute_relations: 법령 관련 관계
- case_statute_citations: 판례 → 법령 인용
- case_case_citations: 판례 → 판례 인용
"""

import logging
from typing import Any, Optional

from sqlalchemy import and_, desc, distinct, func, literal, or_, select, text, union_all
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import async_session_factory
from app.models.case_statute_citation import CaseStatuteCitation
from app.models.law_document import LawDocument
from app.models.precedent_document import PrecedentDocument
from app.models.statute_alias import StatuteAlias
from app.models.statute_hierarchy import StatuteHierarchy
from app.models.statute_relation import StatuteRelation

logger = logging.getLogger(__name__)

# 성능/안전 상수
MAX_GRAPH_DEPTH = 5
MAX_GRAPH_NODES = 200
QUERY_TIMEOUT_MS = 5000


class PgGraphService:
    """PostgreSQL 기반 그래프 서비스 (Neo4j 대체)"""

    @property
    def is_connected(self) -> bool:
        """항상 True (PostgreSQL은 앱 시작 시 연결 확인됨)"""
        return True

    async def get_cited_statutes(
        self,
        case_number: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        판례가 인용한 법령 조회

        Args:
            case_number: 사건번호 (예: "2023다12345")
            limit: 최대 반환 개수

        Returns:
            인용 법령 목록 [{name, type, abbreviation, citation_count}, ...]
        """
        try:
            async with async_session_factory() as session:
                stmt = (
                    select(
                        LawDocument.law_name.label("name"),
                        LawDocument.law_type.label("type"),
                        LawDocument.abbreviation,
                        LawDocument.citation_count,
                    )
                    .join(
                        CaseStatuteCitation,
                        CaseStatuteCitation.law_doc_id == LawDocument.id,
                    )
                    .join(
                        PrecedentDocument,
                        PrecedentDocument.id == CaseStatuteCitation.case_doc_id,
                    )
                    .where(PrecedentDocument.case_number == case_number)
                    .order_by(desc(LawDocument.citation_count))
                    .limit(limit)
                )
                result = await session.execute(stmt)
                return [dict(row._mapping) for row in result]
        except Exception as e:
            logger.warning("get_cited_statutes 실패: %s", e)
            return []

    async def get_statute_hierarchy(self, statute_name: str) -> dict[str, Any]:
        """
        법령의 상위/하위 계급 조회

        Args:
            statute_name: 법령명 (예: "도로교통법 시행령")

        Returns:
            {
                "statute": {name, type, abbreviation},
                "upper": [{name, type}, ...],
                "lower": [{name, type}, ...]
            }
        """
        try:
            async with async_session_factory() as session:
                # 법령 찾기 (이름 또는 약칭)
                base_stmt = select(LawDocument).where(
                    or_(
                        LawDocument.law_name == statute_name,
                        LawDocument.abbreviation == statute_name,
                    )
                )
                base_result = await session.execute(base_stmt)
                statute = base_result.scalar_one_or_none()

                if not statute:
                    return {"statute": None, "upper": [], "lower": []}

                # 상위 법령 (child→parent에서 child=현재 → parent=상위)
                upper_alias = LawDocument.__table__.alias("upper_law")
                upper_stmt = (
                    select(
                        upper_alias.c.law_name.label("name"),
                        upper_alias.c.law_type.label("type"),
                    )
                    .join(
                        StatuteHierarchy,
                        StatuteHierarchy.parent_id == upper_alias.c.id,
                    )
                    .where(StatuteHierarchy.child_id == statute.id)
                )
                upper_result = await session.execute(upper_stmt)
                upper = [dict(row._mapping) for row in upper_result]

                # 하위 법령 (child→parent에서 parent=현재 → child=하위)
                lower_alias = LawDocument.__table__.alias("lower_law")
                lower_stmt = (
                    select(
                        lower_alias.c.law_name.label("name"),
                        lower_alias.c.law_type.label("type"),
                    )
                    .join(
                        StatuteHierarchy,
                        StatuteHierarchy.child_id == lower_alias.c.id,
                    )
                    .where(StatuteHierarchy.parent_id == statute.id)
                )
                lower_result = await session.execute(lower_stmt)
                lower = [dict(row._mapping) for row in lower_result]

                return {
                    "statute": {
                        "name": statute.law_name,
                        "type": statute.law_type,
                        "abbreviation": statute.abbreviation,
                    },
                    "upper": upper,
                    "lower": lower,
                }
        except Exception as e:
            logger.warning("get_statute_hierarchy 실패: %s", e)
            return {"statute": None, "upper": [], "lower": []}

    async def get_similar_cases(
        self,
        case_number: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """
        같은 법령을 인용한 유사 판례 조회

        self-join 대신 2단계 접근으로 최적화:
        1) 대상 판례의 인용 법령 ID 집합 추출 (소규모, 보통 < 20건)
        2) 해당 법령을 인용한 다른 판례 검색 (idx_csc_statute 인덱스 활용)

        Args:
            case_number: 사건번호
            limit: 최대 반환 개수

        Returns:
            유사 판례 목록 [{case_number, name, common_count, common_statutes}, ...]
        """
        try:
            async with async_session_factory() as session:
                # 쿼리 타임아웃 설정 (안전장치)
                await session.execute(
                    text(f"SET LOCAL statement_timeout = '{QUERY_TIMEOUT_MS}'")
                )

                # 대상 판례 ID
                case_id_subq = (
                    select(PrecedentDocument.id)
                    .where(PrecedentDocument.case_number == case_number)
                    .scalar_subquery()
                )

                # 1단계: 대상 판례가 인용한 법령 ID 집합 (소규모)
                target_statute_ids = (
                    select(CaseStatuteCitation.law_doc_id)
                    .where(CaseStatuteCitation.case_doc_id == case_id_subq)
                    .subquery()
                )

                # 2단계: 같은 법령을 인용한 다른 판례 (인덱스 활용)
                stmt = (
                    select(
                        PrecedentDocument.case_number,
                        PrecedentDocument.case_name.label("name"),
                        func.count(
                            distinct(CaseStatuteCitation.law_doc_id)
                        ).label("common_count"),
                        func.array_agg(
                            distinct(LawDocument.law_name)
                        ).label("common_statutes_arr"),
                    )
                    .select_from(CaseStatuteCitation)
                    .join(
                        target_statute_ids,
                        target_statute_ids.c.law_doc_id
                        == CaseStatuteCitation.law_doc_id,
                    )
                    .join(
                        PrecedentDocument,
                        PrecedentDocument.id == CaseStatuteCitation.case_doc_id,
                    )
                    .join(
                        LawDocument,
                        LawDocument.id == CaseStatuteCitation.law_doc_id,
                    )
                    .where(CaseStatuteCitation.case_doc_id != case_id_subq)
                    .group_by(PrecedentDocument.id)
                    .order_by(desc("common_count"))
                    .limit(limit)
                )
                result = await session.execute(stmt)
                rows = result.all()

                return [
                    {
                        "case_number": row.case_number,
                        "name": row.name,
                        "common_count": row.common_count,
                        "common_statutes": (row.common_statutes_arr or [])[:3],
                    }
                    for row in rows
                ]
        except Exception as e:
            logger.warning("get_similar_cases 실패: %s", e)
            return []

    async def get_related_statutes(
        self,
        statute_name: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """
        관련 법령 조회 (RELATED_TO 관계, 양방향)

        Args:
            statute_name: 법령명
            limit: 최대 반환 개수

        Returns:
            관련 법령 목록 [{name, type}, ...]
        """
        try:
            async with async_session_factory() as session:
                # 법령 ID 찾기
                base_stmt = select(LawDocument.id).where(
                    or_(
                        LawDocument.law_name == statute_name,
                        LawDocument.abbreviation == statute_name,
                    )
                )
                base_result = await session.execute(base_stmt)
                statute_id = base_result.scalar_one_or_none()

                if not statute_id:
                    return []

                # 양방향 관계 조회 (id_1 또는 id_2에 있을 수 있음)
                related_alias = LawDocument.__table__.alias("related_law")

                # id_1 = 현재 법령 → id_2가 관련 법령
                stmt1 = (
                    select(
                        related_alias.c.law_name.label("name"),
                        related_alias.c.law_type.label("type"),
                    )
                    .join(
                        StatuteRelation,
                        StatuteRelation.law_doc_id_2 == related_alias.c.id,
                    )
                    .where(StatuteRelation.law_doc_id_1 == statute_id)
                )

                # id_2 = 현재 법령 → id_1이 관련 법령
                stmt2 = (
                    select(
                        related_alias.c.law_name.label("name"),
                        related_alias.c.law_type.label("type"),
                    )
                    .join(
                        StatuteRelation,
                        StatuteRelation.law_doc_id_1 == related_alias.c.id,
                    )
                    .where(StatuteRelation.law_doc_id_2 == statute_id)
                )

                combined = union_all(stmt1, stmt2).limit(limit)
                result = await session.execute(combined)
                return [dict(row._mapping) for row in result]
        except Exception as e:
            logger.warning("get_related_statutes 실패: %s", e)
            return []

    async def search_statute(self, query: str) -> Optional[dict[str, Any]]:
        """
        법령 통합 검색 (정식명 → 공식약칭 → 비공식약칭)

        Args:
            query: 검색어 (예: "민소법", "민사소송법")

        Returns:
            {name, type, abbreviation} 또는 None
        """
        try:
            async with async_session_factory() as session:
                # 1단계: 정식명
                stmt = select(
                    LawDocument.law_name.label("name"),
                    LawDocument.law_type.label("type"),
                    LawDocument.abbreviation,
                ).where(LawDocument.law_name == query)
                result = await session.execute(stmt)
                row = result.first()
                if row:
                    return dict(row._mapping)

                # 2단계: 공식 약칭
                stmt = select(
                    LawDocument.law_name.label("name"),
                    LawDocument.law_type.label("type"),
                    LawDocument.abbreviation,
                ).where(LawDocument.abbreviation == query)
                result = await session.execute(stmt)
                row = result.first()
                if row:
                    return dict(row._mapping)

                # 3단계: 비공식 약칭 (alias)
                stmt = (
                    select(
                        LawDocument.law_name.label("name"),
                        LawDocument.law_type.label("type"),
                        LawDocument.abbreviation,
                    )
                    .join(
                        StatuteAlias,
                        StatuteAlias.law_doc_id == LawDocument.id,
                    )
                    .where(StatuteAlias.alias_name == query)
                )
                result = await session.execute(stmt)
                row = result.first()
                if row:
                    return dict(row._mapping)

                return None
        except Exception as e:
            logger.warning("search_statute 실패: %s", e)
            return None

    async def enrich_case_context(self, case_number: str) -> dict[str, Any]:
        """판례 컨텍스트 보강 (RAG용)"""
        return {
            "cited_statutes": await self.get_cited_statutes(case_number, limit=5),
            "similar_cases": await self.get_similar_cases(case_number, limit=3),
        }

    async def enrich_statute_context(self, statute_name: str) -> dict[str, Any]:
        """법령 컨텍스트 보강 (RAG용)"""
        return {
            "hierarchy": await self.get_statute_hierarchy(statute_name),
            "related": await self.get_related_statutes(statute_name, limit=3),
        }

    # ===================================================
    # case_precedent 라우터 전용 메서드
    # ===================================================

    async def search_statutes(
        self,
        query: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        법령 퍼지 검색 (pg_trgm similarity + ILIKE fallback)

        Returns:
            [{id, name, type, abbreviation, citation_count}, ...]
        """
        try:
            async with async_session_factory() as session:
                # pg_trgm similarity 검색
                sim_name = func.similarity(LawDocument.law_name, query)
                sim_abbr = func.similarity(
                    func.coalesce(LawDocument.abbreviation, literal("")), query
                )

                # 법령 직접 검색
                statute_stmt = select(
                    LawDocument.law_id.label("id"),
                    LawDocument.law_name.label("name"),
                    LawDocument.law_type.label("type"),
                    LawDocument.abbreviation,
                    LawDocument.citation_count,
                    func.greatest(sim_name, sim_abbr).label("score"),
                ).where(
                    or_(
                        sim_name > 0.1,
                        sim_abbr > 0.1,
                    )
                )

                # Alias 경유 검색
                sim_alias = func.similarity(StatuteAlias.alias_name, query)
                alias_stmt = select(
                    LawDocument.law_id.label("id"),
                    LawDocument.law_name.label("name"),
                    LawDocument.law_type.label("type"),
                    StatuteAlias.alias_name.label("abbreviation"),
                    LawDocument.citation_count,
                    sim_alias.label("score"),
                ).join(
                    StatuteAlias,
                    StatuteAlias.law_doc_id == LawDocument.id,
                ).where(sim_alias > 0.1)

                # UNION + ORDER BY score
                combined = union_all(statute_stmt, alias_stmt).subquery()
                final_stmt = (
                    select(
                        combined.c.id,
                        combined.c.name,
                        combined.c.type,
                        combined.c.abbreviation,
                        combined.c.citation_count,
                    )
                    .group_by(
                        combined.c.id,
                        combined.c.name,
                        combined.c.type,
                        combined.c.abbreviation,
                        combined.c.citation_count,
                    )
                    .order_by(desc(func.max(combined.c.score)))
                    .limit(limit)
                )
                result = await session.execute(final_stmt)
                results = [dict(row._mapping) for row in result]

                # Fallback: ILIKE
                if not results:
                    like_pattern = f"%{query}%"
                    fallback_stmt = (
                        select(
                            LawDocument.law_id.label("id"),
                            LawDocument.law_name.label("name"),
                            LawDocument.law_type.label("type"),
                            LawDocument.abbreviation,
                            LawDocument.citation_count,
                        )
                        .where(
                            or_(
                                LawDocument.law_name.ilike(like_pattern),
                                LawDocument.abbreviation.ilike(like_pattern),
                            )
                        )
                        .order_by(desc(LawDocument.citation_count))
                        .limit(limit)
                    )
                    result = await session.execute(fallback_stmt)
                    results = [dict(row._mapping) for row in result]

                    # Alias ILIKE fallback
                    if not results:
                        alias_fallback = (
                            select(
                                LawDocument.law_id.label("id"),
                                LawDocument.law_name.label("name"),
                                LawDocument.law_type.label("type"),
                                StatuteAlias.alias_name.label("abbreviation"),
                                LawDocument.citation_count,
                            )
                            .join(
                                StatuteAlias,
                                StatuteAlias.law_doc_id == LawDocument.id,
                            )
                            .where(StatuteAlias.alias_name.ilike(like_pattern))
                            .order_by(desc(LawDocument.citation_count))
                            .limit(limit)
                        )
                        result = await session.execute(alias_fallback)
                        results = [dict(row._mapping) for row in result]

                return results
        except Exception as e:
            logger.warning("search_statutes 실패: %s", e)
            return []

    async def get_statute_hierarchy_detail(
        self,
        statute_id: str,
    ) -> Optional[dict[str, Any]]:
        """
        법령 계층 상세 조회 (상위 + 하위 + 관련 법령)

        Args:
            statute_id: 법령 law_id (예: "002413")

        Returns:
            {
                "root": {id, name, type, abbreviation, citation_count},
                "upper": [...], "lower": [...], "related": [...]
            }
            or None if not found
        """
        try:
            async with async_session_factory() as session:
                # 루트 법령
                root_stmt = select(LawDocument).where(
                    LawDocument.law_id == statute_id
                )
                root_result = await session.execute(root_stmt)
                root = root_result.scalar_one_or_none()

                if not root:
                    return None

                def _node_dict(ld: LawDocument) -> dict[str, Any]:
                    return {
                        "id": ld.law_id,
                        "name": ld.law_name,
                        "type": ld.law_type,
                        "abbreviation": ld.abbreviation,
                        "citation_count": ld.citation_count or 0,
                    }

                # 상위 법령
                upper_stmt = (
                    select(LawDocument)
                    .join(
                        StatuteHierarchy,
                        StatuteHierarchy.parent_id == LawDocument.id,
                    )
                    .where(StatuteHierarchy.child_id == root.id)
                )
                upper_result = await session.execute(upper_stmt)
                upper = [_node_dict(row) for row in upper_result.scalars()]

                # 하위 법령
                lower_stmt = (
                    select(LawDocument)
                    .join(
                        StatuteHierarchy,
                        StatuteHierarchy.child_id == LawDocument.id,
                    )
                    .where(StatuteHierarchy.parent_id == root.id)
                    .order_by(desc(LawDocument.citation_count))
                )
                lower_result = await session.execute(lower_stmt)
                lower = [_node_dict(row) for row in lower_result.scalars()]

                # 관련 법령 (양방향)
                related_stmt1 = (
                    select(LawDocument)
                    .join(
                        StatuteRelation,
                        StatuteRelation.law_doc_id_2 == LawDocument.id,
                    )
                    .where(StatuteRelation.law_doc_id_1 == root.id)
                )
                related_stmt2 = (
                    select(LawDocument)
                    .join(
                        StatuteRelation,
                        StatuteRelation.law_doc_id_1 == LawDocument.id,
                    )
                    .where(StatuteRelation.law_doc_id_2 == root.id)
                )

                r1 = await session.execute(related_stmt1)
                r2 = await session.execute(related_stmt2)
                related_ids: set[int] = set()
                related: list[dict[str, Any]] = []
                for row in list(r1.scalars()) + list(r2.scalars()):
                    row_id: int = row.id  # type: ignore[assignment]
                    if row_id not in related_ids:
                        related_ids.add(row_id)
                        related.append(_node_dict(row))

                return {
                    "root": _node_dict(root),
                    "upper": upper,
                    "lower": lower,
                    "related": related,
                }
        except Exception as e:
            logger.warning("get_statute_hierarchy_detail 실패: %s", e)
            return None

    async def get_statute_children(
        self,
        statute_id: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        법령 하위 법령 목록 (지연 로딩용)

        Args:
            statute_id: 법령 law_id
            limit: 최대 반환 개수

        Returns:
            [{id, name, type, abbreviation, citation_count}, ...]
        """
        try:
            async with async_session_factory() as session:
                # 법령 PK 찾기
                parent_pk_stmt = select(LawDocument.id).where(
                    LawDocument.law_id == statute_id
                )
                parent_pk_result = await session.execute(parent_pk_stmt)
                parent_pk = parent_pk_result.scalar_one_or_none()

                if not parent_pk:
                    return []

                stmt = (
                    select(
                        LawDocument.law_id.label("id"),
                        LawDocument.law_name.label("name"),
                        LawDocument.law_type.label("type"),
                        LawDocument.abbreviation,
                        LawDocument.citation_count,
                    )
                    .join(
                        StatuteHierarchy,
                        StatuteHierarchy.child_id == LawDocument.id,
                    )
                    .where(StatuteHierarchy.parent_id == parent_pk)
                    .order_by(desc(LawDocument.citation_count))
                    .limit(limit)
                )
                result = await session.execute(stmt)
                return [dict(row._mapping) for row in result]
        except Exception as e:
            logger.warning("get_statute_children 실패: %s", e)
            return []

    async def get_statute_graph(
        self,
        center_id: Optional[str] = None,
        depth: int = 2,
        limit: int = 100,
    ) -> dict[str, Any]:
        """
        법령 그래프 데이터 (Force-directed 시각화용)

        center_id가 있으면 해당 법령 기준, 없으면 인용수 상위 법령 기준.
        Recursive CTE로 탐색 (depth/limit 하드캡 적용).

        Returns:
            {"nodes": [...], "links": [...]}
        """
        # 안전장치: depth/limit 하드캡
        depth = min(depth, MAX_GRAPH_DEPTH)
        limit = min(limit, MAX_GRAPH_NODES)

        try:
            async with async_session_factory() as session:
                # 쿼리 타임아웃 설정
                await session.execute(
                    text(f"SET LOCAL statement_timeout = '{QUERY_TIMEOUT_MS}'")
                )

                if center_id:
                    nodes, links = await self._graph_from_center(
                        session, center_id, depth, limit
                    )
                else:
                    nodes, links = await self._graph_from_top(session, limit)

                return {"nodes": nodes, "links": links}
        except Exception as e:
            logger.warning("get_statute_graph 실패: %s", e)
            return {"nodes": [], "links": []}

    async def _graph_from_center(
        self,
        session: AsyncSession,
        center_id: str,
        depth: int,
        limit: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """특정 법령 중심 그래프 (recursive CTE)"""
        cte_sql = text("""
            WITH RECURSIVE hier_connected AS (
                -- 시작 노드 (non-recursive term)
                SELECT ld.id, ld.law_id, 0 as depth
                FROM law_documents ld
                WHERE ld.law_id = :center_id

                UNION

                -- recursive term: HIERARCHY_OF 관계 (양방향)
                SELECT ld2.id, ld2.law_id, c.depth + 1
                FROM hier_connected c
                JOIN statute_hierarchy sh ON (sh.child_id = c.id OR sh.parent_id = c.id)
                JOIN law_documents ld2 ON ld2.id = CASE
                    WHEN sh.child_id = c.id THEN sh.parent_id
                    ELSE sh.child_id
                END
                WHERE c.depth < :depth
            ),
            rel_ids AS (
                -- RELATED_TO 관계 (hierarchy 노드에서 1-hop)
                SELECT DISTINCT ld3.id
                FROM hier_connected c
                JOIN statute_relations sr ON (sr.law_doc_id_1 = c.id OR sr.law_doc_id_2 = c.id)
                JOIN law_documents ld3 ON ld3.id = CASE
                    WHEN sr.law_doc_id_1 = c.id THEN sr.law_doc_id_2
                    ELSE sr.law_doc_id_1
                END
            )
            SELECT DISTINCT id FROM (
                SELECT id FROM hier_connected
                UNION ALL
                SELECT id FROM rel_ids
            ) combined
            LIMIT :limit
        """)
        id_result = await session.execute(
            cte_sql, {"center_id": center_id, "depth": depth, "limit": limit}
        )
        node_ids = [row[0] for row in id_result]

        if not node_ids:
            return [], []

        return await self._build_graph_data(session, node_ids)

    async def _graph_from_top(
        self,
        session: AsyncSession,
        limit: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """인용수 상위 법령 + 1-hop 이웃 그래프"""
        # 인용수 TOP 50
        top_stmt = (
            select(LawDocument.id)
            .where(LawDocument.citation_count > 0)
            .order_by(desc(LawDocument.citation_count))
            .limit(50)
        )
        top_result = await session.execute(top_stmt)
        top_ids = [row[0] for row in top_result]

        if not top_ids:
            return [], []

        # 1-hop 이웃 추가
        neighbor_sql = text("""
            SELECT DISTINCT neighbor_id FROM (
                SELECT CASE
                    WHEN sh.child_id = ANY(:ids) THEN sh.parent_id
                    ELSE sh.child_id
                END as neighbor_id
                FROM statute_hierarchy sh
                WHERE sh.child_id = ANY(:ids) OR sh.parent_id = ANY(:ids)

                UNION

                SELECT CASE
                    WHEN sr.law_doc_id_1 = ANY(:ids) THEN sr.law_doc_id_2
                    ELSE sr.law_doc_id_1
                END as neighbor_id
                FROM statute_relations sr
                WHERE sr.law_doc_id_1 = ANY(:ids) OR sr.law_doc_id_2 = ANY(:ids)
            ) sub
            LIMIT :limit
        """)
        neighbor_result = await session.execute(
            neighbor_sql, {"ids": top_ids, "limit": limit}
        )
        neighbor_ids = [row[0] for row in neighbor_result]

        all_ids = list(set(top_ids + neighbor_ids))[:limit]
        return await self._build_graph_data(session, all_ids)

    async def _build_graph_data(
        self,
        session: AsyncSession,
        node_ids: list[int],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """노드 ID 리스트로부터 그래프 데이터 구성"""
        if not node_ids:
            return [], []

        # 노드 데이터
        nodes_stmt = select(
            LawDocument.law_id.label("id"),
            LawDocument.law_name.label("name"),
            LawDocument.law_type.label("type"),
            LawDocument.abbreviation,
            LawDocument.citation_count,
        ).where(LawDocument.id.in_(node_ids))
        nodes_result = await session.execute(nodes_stmt)
        nodes = [dict(row._mapping) for row in nodes_result]

        # law_id 역매핑 (internal id → law_id)
        id_map_stmt = select(LawDocument.id, LawDocument.law_id).where(
            LawDocument.id.in_(node_ids)
        )
        id_map_result = await session.execute(id_map_stmt)
        id_to_law_id = {row[0]: row[1] for row in id_map_result}

        # HIERARCHY_OF 링크
        hier_stmt = select(
            StatuteHierarchy.child_id, StatuteHierarchy.parent_id
        ).where(
            and_(
                StatuteHierarchy.child_id.in_(node_ids),
                StatuteHierarchy.parent_id.in_(node_ids),
            )
        )
        hier_result = await session.execute(hier_stmt)
        links: list[dict[str, Any]] = []
        for row in hier_result:
            source_lid = id_to_law_id.get(row[0])
            target_lid = id_to_law_id.get(row[1])
            if source_lid and target_lid:
                links.append({
                    "source": source_lid,
                    "target": target_lid,
                    "relation": "HIERARCHY_OF",
                })

        # RELATED_TO 링크
        rel_stmt = select(
            StatuteRelation.law_doc_id_1, StatuteRelation.law_doc_id_2
        ).where(
            and_(
                StatuteRelation.law_doc_id_1.in_(node_ids),
                StatuteRelation.law_doc_id_2.in_(node_ids),
            )
        )
        rel_result = await session.execute(rel_stmt)
        for row in rel_result:
            source_lid = id_to_law_id.get(row[0])
            target_lid = id_to_law_id.get(row[1])
            if source_lid and target_lid:
                links.append({
                    "source": source_lid,
                    "target": target_lid,
                    "relation": "RELATED_TO",
                })

        return nodes, links


# 싱글톤 인스턴스
_pg_graph_service: Optional[PgGraphService] = None


def get_pg_graph_service() -> PgGraphService:
    """PgGraphService 싱글톤 인스턴스 반환"""
    global _pg_graph_service
    if _pg_graph_service is None:
        _pg_graph_service = PgGraphService()
    return _pg_graph_service
