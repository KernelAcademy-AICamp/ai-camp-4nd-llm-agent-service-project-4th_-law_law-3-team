"""
Neo4j 그래프 서비스

법령 계급, 판례 인용 관계를 활용한 컨텍스트 보강 서비스
RAG 파이프라인에서 검색 결과를 그래프 정보로 보강

그래프 스키마:
- Statute: id, name, type, abbreviation, citation_count
- Case: id, case_number, name, summary
- Alias: name, category (비공식 약칭)

관계:
- (Case)-[:CITES]->(Statute): 판례→법령 인용
- (Case)-[:CITES_CASE]->(Case): 판례→판례 인용
- (Statute)-[:HIERARCHY_OF]->(Statute): 시행령→법률
- (Statute)-[:RELATED_TO]->(Statute): 법령→법령 관련
- (Alias)-[:ALIAS_OF]->(Statute): 약칭→법령
"""

import logging
from contextlib import contextmanager
from functools import lru_cache
from typing import Generator, List, Optional

from neo4j import GraphDatabase, Driver

from app.core.config import settings

logger = logging.getLogger(__name__)


class GraphService:
    """Neo4j 그래프 서비스"""

    _instance: Optional["GraphService"] = None
    _driver: Optional[Driver] = None
    _initialized: bool = False

    def __new__(cls) -> "GraphService":
        """싱글톤 패턴"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """드라이버 초기화 (lazy, 중복 초기화 방지)"""
        if self._initialized:
            return
        self._initialized = True

        try:
            self._driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
            )
            logger.info("Neo4j 드라이버 연결 성공: %s", settings.NEO4J_URI)
        except Exception as e:
            logger.warning("Neo4j 연결 실패: %s", e)
            self._driver = None

    @property
    def is_connected(self) -> bool:
        """연결 상태 확인"""
        if self._driver is None:
            return False
        try:
            self._driver.verify_connectivity()
            return True
        except Exception:
            return False

    @contextmanager
    def session(self) -> Generator:
        """세션 컨텍스트 매니저"""
        if self._driver is None:
            raise ConnectionError("Neo4j 드라이버가 초기화되지 않았습니다")
        session = self._driver.session()
        try:
            yield session
        finally:
            session.close()

    def close(self) -> None:
        """드라이버 종료"""
        if self._driver:
            self._driver.close()
            self._driver = None
            GraphService._instance = None
            GraphService._initialized = False

    def get_cited_statutes(
        self,
        case_number: str,
        limit: int = 10,
    ) -> List[dict]:
        """
        판례가 인용한 법령 조회

        Args:
            case_number: 사건번호 (예: "2023다12345")
            limit: 최대 반환 개수

        Returns:
            인용 법령 목록 [{name, type, abbreviation, citation_count}, ...]
        """
        if not self.is_connected:
            return []

        query = """
        MATCH (c:Case {case_number: $case_number})-[:CITES]->(s:Statute)
        RETURN s.name as name, s.type as type,
               s.abbreviation as abbreviation,
               s.citation_count as citation_count
        ORDER BY s.citation_count DESC
        LIMIT $limit
        """
        try:
            with self.session() as session:
                result = session.run(query, case_number=case_number, limit=limit)
                return [dict(record) for record in result]
        except Exception as e:
            logger.warning("get_cited_statutes 실패: %s", e)
            return []

    def get_statute_hierarchy(self, statute_name: str) -> dict:
        """
        법령의 상위/하위 계급 조회

        Args:
            statute_name: 법령명 (예: "도로교통법 시행령")

        Returns:
            {
                "statute": {name, type, abbreviation},
                "upper": [{name, type}, ...],  # 상위 법령 (시행령→법률)
                "lower": [{name, type}, ...]   # 하위 법령 (법률→시행령)
            }
        """
        if not self.is_connected:
            return {"statute": None, "upper": [], "lower": []}

        query = """
        MATCH (s:Statute)
        WHERE s.name = $name OR s.abbreviation = $name
        OPTIONAL MATCH (s)-[:HIERARCHY_OF]->(upper:Statute)
        OPTIONAL MATCH (lower:Statute)-[:HIERARCHY_OF]->(s)
        RETURN s.name as name, s.type as type, s.abbreviation as abbreviation,
               collect(DISTINCT {name: upper.name, type: upper.type}) as upper_list,
               collect(DISTINCT {name: lower.name, type: lower.type}) as lower_list
        """
        try:
            with self.session() as session:
                result = session.run(query, name=statute_name)
                record = result.single()
                if record is None:
                    return {"statute": None, "upper": [], "lower": []}

                return {
                    "statute": {
                        "name": record["name"],
                        "type": record["type"],
                        "abbreviation": record["abbreviation"],
                    },
                    "upper": [
                        upper_statute for upper_statute in record["upper_list"]
                        if upper_statute.get("name") is not None
                    ],
                    "lower": [
                        lower_statute for lower_statute in record["lower_list"]
                        if lower_statute.get("name") is not None
                    ],
                }
        except Exception as e:
            logger.warning("get_statute_hierarchy 실패: %s", e)
            return {"statute": None, "upper": [], "lower": []}

    def get_similar_cases(
        self,
        case_number: str,
        limit: int = 5,
    ) -> List[dict]:
        """
        같은 법령을 인용한 유사 판례 조회

        Args:
            case_number: 사건번호
            limit: 최대 반환 개수

        Returns:
            유사 판례 목록 [{case_number, name, common_statutes}, ...]
        """
        if not self.is_connected:
            return []

        query = """
        MATCH (c1:Case {case_number: $case_number})-[:CITES]->(s:Statute)<-[:CITES]-(c2:Case)
        WHERE c1 <> c2
        WITH c2, count(DISTINCT s) as common_count, collect(DISTINCT s.name) as statutes
        RETURN c2.case_number as case_number, c2.name as name,
               common_count, statutes[0..3] as common_statutes
        ORDER BY common_count DESC
        LIMIT $limit
        """
        try:
            with self.session() as session:
                result = session.run(query, case_number=case_number, limit=limit)
                return [
                    {
                        "case_number": r["case_number"],
                        "name": r["name"],
                        "common_count": r["common_count"],
                        "common_statutes": r["common_statutes"],
                    }
                    for r in result
                ]
        except Exception as e:
            logger.warning("get_similar_cases 실패: %s", e)
            return []

    def get_related_statutes(
        self,
        statute_name: str,
        limit: int = 5,
    ) -> List[dict]:
        """
        관련 법령 조회 (RELATED_TO 관계)

        Args:
            statute_name: 법령명
            limit: 최대 반환 개수

        Returns:
            관련 법령 목록 [{name, type}, ...]
        """
        if not self.is_connected:
            return []

        query = """
        MATCH (s:Statute)-[:RELATED_TO]-(related:Statute)
        WHERE s.name = $name OR s.abbreviation = $name
        RETURN DISTINCT related.name as name, related.type as type
        LIMIT $limit
        """
        try:
            with self.session() as session:
                result = session.run(query, name=statute_name, limit=limit)
                return [dict(record) for record in result]
        except Exception as e:
            logger.warning("get_related_statutes 실패: %s", e)
            return []

    def search_statute(self, query: str) -> Optional[dict]:
        """
        법령 통합 검색 (정식명/공식약칭/비공식약칭)

        Args:
            query: 검색어 (예: "민소법", "민사소송법", "119법")

        Returns:
            {name, type, abbreviation} 또는 None
        """
        if not self.is_connected:
            return None

        cypher = """
        OPTIONAL MATCH (s1:Statute {name: $query})
        OPTIONAL MATCH (s2:Statute {abbreviation: $query})
        OPTIONAL MATCH (a:Alias {name: $query})-[:ALIAS_OF]->(s3:Statute)
        WITH coalesce(s1, s2, s3) as s
        WHERE s IS NOT NULL
        RETURN s.name as name, s.type as type, s.abbreviation as abbreviation
        LIMIT 1
        """
        try:
            with self.session() as session:
                result = session.run(cypher, query=query)
                record = result.single()
                if record:
                    return dict(record)
                return None
        except Exception as e:
            logger.warning("search_statute 실패: %s", e)
            return None

    def enrich_case_context(self, case_number: str) -> dict:
        """
        판례 컨텍스트 보강 (RAG용)

        검색된 판례에 대해 그래프 정보를 추가로 제공:
        - 인용 법령 목록
        - 유사 판례

        Args:
            case_number: 사건번호

        Returns:
            {
                "cited_statutes": [...],
                "similar_cases": [...]
            }
        """
        return {
            "cited_statutes": self.get_cited_statutes(case_number, limit=5),
            "similar_cases": self.get_similar_cases(case_number, limit=3),
        }

    def enrich_statute_context(self, statute_name: str) -> dict:
        """
        법령 컨텍스트 보강 (RAG용)

        검색된 법령에 대해 그래프 정보를 추가로 제공:
        - 계급 정보 (상위/하위 법령)
        - 관련 법령

        Args:
            statute_name: 법령명

        Returns:
            {
                "hierarchy": {...},
                "related": [...]
            }
        """
        return {
            "hierarchy": self.get_statute_hierarchy(statute_name),
            "related": self.get_related_statutes(statute_name, limit=3),
        }


@lru_cache(maxsize=1)
def get_graph_service() -> GraphService:
    """GraphService 싱글톤 인스턴스 반환"""
    return GraphService()
