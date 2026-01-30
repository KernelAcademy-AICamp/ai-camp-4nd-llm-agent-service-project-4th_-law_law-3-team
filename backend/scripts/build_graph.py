"""
Neo4j Graph DB 구축 스크립트

법령, 판례 데이터를 Neo4j 그래프로 구축합니다.

노드 (Nodes):
- Statute: 법령 정보 (id, name, type, promulgation_date, abbreviation, citation_count)
- Case: 판례 정보 (id, case_number, name, summary)
- Alias: 비공식 약칭 (name, category)

관계 (Relationships):
- HIERARCHY_OF: 법령 계급 (시행령 -> 법률)
- CITES: 판례 -> 법령 인용
- CITES_CASE: 판례 -> 판례 인용 (참조판례)
- RELATED_TO: 법령 -> 법령 관련
- ALIAS_OF: 비공식 약칭 -> 법령

데이터 파일:
- law_cleaned.json: 법령 데이터
- lsAbrv.json: 법령 약칭 데이터 (공식)
- informal_abbreviations.json: 비공식 약칭 데이터
- [cleaned]lsStmd-full.json: 법령 계급도/관련법령
- precedents_cleaned.json: 판례 데이터

사용법:
    cd backend
    uv run python scripts/build_graph.py
"""

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
from tqdm import tqdm

# .env 파일 로드
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# 프로젝트 루트 기준 데이터 파일 경로
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LAW_FILE = DATA_DIR / "law_cleaned.json"
HIERARCHY_FILE = DATA_DIR / "[cleaned]lsStmd-full.json"
CASE_FILE = DATA_DIR / "precedents_cleaned.json"
ABBREVIATION_FILE = DATA_DIR / "lsAbrv.json"

# 비공식 약칭 파일 (scripts 폴더)
INFORMAL_ABBR_FILE = Path(__file__).parent / "informal_abbreviations.json"

BATCH_SIZE = 1000


class GraphBuilder:
    """Neo4j 그래프 구축 클래스"""

    def __init__(self, uri: str, auth: tuple[str, str]) -> None:
        self.driver = GraphDatabase.driver(uri, auth=auth)
        self.verify_connection()

    def verify_connection(self) -> None:
        """Neo4j 연결 확인"""
        try:
            self.driver.verify_connectivity()
            print("Connected to Neo4j.")
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            raise SystemExit(1) from e

    def close(self) -> None:
        """드라이버 종료"""
        self.driver.close()

    def create_constraints(self) -> None:
        """인덱스 및 제약조건 생성"""
        # 기본 제약조건 및 인덱스
        queries = [
            "CREATE CONSTRAINT FOR (s:Statute) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT FOR (s:Statute) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT FOR (c:Case) REQUIRE c.id IS UNIQUE",
            "CREATE INDEX FOR (c:Case) ON (c.case_number)",
            "CREATE INDEX FOR (c:Case) ON (c.name)",
            "CREATE INDEX FOR (s:Statute) ON (s.abbreviation)",
            "CREATE CONSTRAINT FOR (a:Alias) REQUIRE a.name IS UNIQUE",
        ]

        # Full-text 인덱스 (텍스트 검색 최적화)
        fulltext_queries = [
            """CREATE FULLTEXT INDEX ft_statute_search FOR (s:Statute)
               ON EACH [s.name, s.abbreviation]""",
            """CREATE FULLTEXT INDEX ft_case_search FOR (c:Case)
               ON EACH [c.name, c.summary]""",
            """CREATE FULLTEXT INDEX ft_alias_search FOR (a:Alias)
               ON EACH [a.name]""",
        ]

        with self.driver.session() as session:
            for q in queries:
                try:
                    session.run(q)
                except Exception as e:
                    if "already exists" not in str(e).lower():
                        print(f"Constraint warning: {e}")

            for q in fulltext_queries:
                try:
                    session.run(q)
                except Exception as e:
                    if "already exists" not in str(e).lower():
                        print(f"Fulltext index warning: {e}")

        print("Constraints and indexes checked.")

    def load_statutes(self) -> None:
        """법령 데이터 로드 (law_cleaned.json)"""
        print(f"Loading Statutes from {LAW_FILE}...")

        if not LAW_FILE.exists():
            print(f"File not found: {LAW_FILE}")
            return

        with open(LAW_FILE, encoding="utf-8") as f:
            data = json.load(f)
            items = data.get("items", []) if isinstance(data, dict) else data

        query = """
        UNWIND $batch AS row
        MERGE (s:Statute {id: row.law_id})
        SET s.name = row.law_name,
            s.type = row.law_type,
            s.promulgation_date = row.promulgation_date
        """

        batch = []
        with self.driver.session() as session:
            for item in tqdm(items, desc="Statutes"):
                batch.append(item)
                if len(batch) >= BATCH_SIZE:
                    session.run(query, batch=batch)
                    batch = []
            if batch:
                session.run(query, batch=batch)

        print(f"Loaded {len(items)} statutes.")

    def load_hierarchy(self) -> None:
        """법령 계급도 로드 ([cleaned]lsStmd-full.json)"""
        print(f"Loading Hierarchy from {HIERARCHY_FILE}...")

        if not HIERARCHY_FILE.exists():
            print(f"Hierarchy file not found: {HIERARCHY_FILE}")
            return

        with open(HIERARCHY_FILE, encoding="utf-8") as f:
            data = json.load(f)

        real_relations: list[dict[str, str]] = []

        def parse_node(
            node_dict: dict, parent_id: str | None = None
        ) -> None:
            """재귀적으로 계급 관계 파싱"""
            current_id = None
            if "기본정보" in node_dict:
                current_id = node_dict["기본정보"].get("법령ID")
                if current_id and parent_id and current_id != parent_id:
                    real_relations.append({"lower": current_id, "upper": parent_id})

            next_parent = current_id if current_id else parent_id

            for key, val in node_dict.items():
                if key in ["기본정보", "관련법령", "제개정구분"]:
                    continue

                targets = []
                if isinstance(val, dict):
                    targets.append(val)
                elif isinstance(val, list):
                    targets.extend(val)

                for t in targets:
                    if isinstance(t, dict):
                        parse_node(t, next_parent)

        for item in tqdm(data, desc="Parsing Hierarchy"):
            parse_node(item)

        print(f"Found {len(real_relations)} hierarchy relationships.")

        query = """
        UNWIND $batch AS row
        MATCH (l:Statute {id: row.lower})
        MATCH (u:Statute {id: row.upper})
        MERGE (l)-[:HIERARCHY_OF]->(u)
        """

        batch = []
        with self.driver.session() as session:
            for rel in tqdm(real_relations, desc="Hierarchy Edges"):
                batch.append(rel)
                if len(batch) >= BATCH_SIZE:
                    session.run(query, batch=batch)
                    batch = []
            if batch:
                session.run(query, batch=batch)

    def load_cases(self) -> None:
        """판례 데이터 로드 (precedents_cleaned.json)"""
        print(f"Loading Cases from {CASE_FILE}...")

        if not CASE_FILE.exists():
            print(f"Case file not found: {CASE_FILE}")
            return

        with open(CASE_FILE, encoding="utf-8") as f:
            data = json.load(f)

        node_query = """
        UNWIND $batch AS row
        MERGE (c:Case {id: row.id})
        SET c.case_number = row.case_number,
            c.name = row.name,
            c.summary = row.summary
        """

        edge_query = """
        UNWIND $batch AS row
        MATCH (c:Case {id: row.case_id})
        MATCH (s:Statute {name: row.statute_name})
        MERGE (c)-[:CITES]->(s)
        """

        node_batch = []
        edge_batch = []

        regex_statute = re.compile(r"([가-힣]+법(?:시행령|시행규칙)?)")

        with self.driver.session() as session:
            for item in tqdm(data, desc="Cases"):
                c_id = item.get("판례정보일련번호")
                if not c_id:
                    continue

                node_batch.append({
                    "id": c_id,
                    "case_number": item.get("사건번호"),
                    "name": item.get("사건명"),
                    "summary": item.get("판결요지"),
                })

                refs = item.get("참조조문", "")
                if refs:
                    found_laws = set(regex_statute.findall(refs))
                    for law_name in found_laws:
                        if len(law_name) < 2:
                            continue
                        edge_batch.append({
                            "case_id": c_id,
                            "statute_name": law_name,
                        })

                if len(node_batch) >= BATCH_SIZE:
                    session.run(node_query, batch=node_batch)
                    node_batch = []

                if len(edge_batch) >= BATCH_SIZE:
                    session.run(edge_query, batch=edge_batch)
                    edge_batch = []

            if node_batch:
                session.run(node_query, batch=node_batch)
            if edge_batch:
                session.run(edge_query, batch=edge_batch)

        print(f"Loaded {len(data)} cases.")

    def load_case_citations(self) -> None:
        """판례 간 인용 관계 로드 (참조판례 필드)"""
        print(f"Loading Case Citations from {CASE_FILE}...")

        if not CASE_FILE.exists():
            print(f"Case file not found: {CASE_FILE}")
            return

        with open(CASE_FILE, encoding="utf-8") as f:
            data = json.load(f)

        # 사건번호 패턴: 연도(2-4자리) + 사건종류(한글) + 번호
        # 예: 80다268, 2023도1234, 99다12345
        regex_case_number = re.compile(
            r"(\d{2,4})"  # 연도 (2-4자리)
            r"([가-힣]{1,3})"  # 사건종류 (다, 도, 누, 카, 마 등)
            r"(\d+)"  # 번호
        )

        edge_batch: list[dict[str, str]] = []
        total_refs = 0

        for item in tqdm(data, desc="Parsing Case Citations"):
            c_id = item.get("판례정보일련번호")
            refs = item.get("참조판례", "")

            if not c_id or not refs:
                continue

            # 참조판례에서 사건번호 추출
            matches = regex_case_number.findall(refs)
            for match in matches:
                year, case_type, number = match
                # 사건번호 형식으로 조합 (예: 80다268)
                ref_case_number = f"{year}{case_type}{number}"
                edge_batch.append({
                    "from_id": c_id,
                    "to_case_number": ref_case_number,
                })
                total_refs += 1

        print(f"Found {total_refs} case citation references.")

        if not edge_batch:
            print("No case citations to load.")
            return

        # 사건번호로 매칭하여 CITES_CASE 관계 생성
        query = """
        UNWIND $batch AS row
        MATCH (from:Case {id: row.from_id})
        MATCH (to:Case {case_number: row.to_case_number})
        MERGE (from)-[:CITES_CASE]->(to)
        """

        batch = []
        created = 0
        with self.driver.session() as session:
            for rel in tqdm(edge_batch, desc="Case Citation Edges"):
                batch.append(rel)
                if len(batch) >= BATCH_SIZE:
                    result = session.run(query, batch=batch)
                    summary = result.consume()
                    created += summary.counters.relationships_created
                    batch = []
            if batch:
                result = session.run(query, batch=batch)
                summary = result.consume()
                created += summary.counters.relationships_created

        print(f"Created {created} CITES_CASE relationships.")

    def load_related_statutes(self) -> None:
        """법령 간 관련 관계 로드 (관련법령 필드)"""
        print(f"Loading Related Statutes from {HIERARCHY_FILE}...")

        if not HIERARCHY_FILE.exists():
            print(f"Hierarchy file not found: {HIERARCHY_FILE}")
            return

        with open(HIERARCHY_FILE, encoding="utf-8") as f:
            data = json.load(f)

        relations: list[dict[str, str]] = []

        for item in tqdm(data, desc="Parsing Related Statutes"):
            if "기본정보" not in item:
                continue

            source_id = item["기본정보"].get("법령ID")
            if not source_id:
                continue

            # 관련법령 추출
            related = item.get("관련법령", {})
            conlaw_list = related.get("conlaw", [])

            if not isinstance(conlaw_list, list):
                continue

            for conlaw in conlaw_list:
                target_id = conlaw.get("법령ID")
                if target_id and target_id != source_id:
                    relations.append({
                        "source_id": source_id,
                        "target_id": target_id,
                    })

        print(f"Found {len(relations)} related statute relationships.")

        if not relations:
            print("No related statutes to load.")
            return

        query = """
        UNWIND $batch AS row
        MATCH (s:Statute {id: row.source_id})
        MATCH (t:Statute {id: row.target_id})
        MERGE (s)-[:RELATED_TO]->(t)
        """

        batch = []
        with self.driver.session() as session:
            for rel in tqdm(relations, desc="Related Statute Edges"):
                batch.append(rel)
                if len(batch) >= BATCH_SIZE:
                    session.run(query, batch=batch)
                    batch = []
            if batch:
                session.run(query, batch=batch)

        print(f"Loaded {len(relations)} RELATED_TO relationships.")

    def load_abbreviations(self) -> None:
        """법령 약칭 로드 (lsAbrv.json)"""
        print(f"Loading Abbreviations from {ABBREVIATION_FILE}...")

        if not ABBREVIATION_FILE.exists():
            print(f"Abbreviation file not found: {ABBREVIATION_FILE}")
            return

        with open(ABBREVIATION_FILE, encoding="utf-8") as f:
            data = json.load(f)

        query = """
        UNWIND $batch AS row
        MATCH (s:Statute {id: row.statute_id})
        SET s.abbreviation = row.abbreviation
        """

        batch = []
        updated = 0
        with self.driver.session() as session:
            for item in tqdm(data, desc="Abbreviations"):
                statute_id = item.get("법령ID")
                abbreviation = item.get("법령약칭명")

                if not statute_id or not abbreviation:
                    continue

                batch.append({
                    "statute_id": statute_id,
                    "abbreviation": abbreviation,
                })

                if len(batch) >= BATCH_SIZE:
                    result = session.run(query, batch=batch)
                    summary = result.consume()
                    updated += summary.counters.properties_set
                    batch = []

            if batch:
                result = session.run(query, batch=batch)
                summary = result.consume()
                updated += summary.counters.properties_set

        print(f"Updated {updated} statutes with abbreviations.")

    def load_informal_abbreviations(self) -> None:
        """비공식 약칭 로드 (informal_abbreviations.json)

        Alias 노드를 생성하고 ALIAS_OF 관계로 Statute에 연결합니다.
        """
        print(f"Loading Informal Abbreviations from {INFORMAL_ABBR_FILE}...")

        if not INFORMAL_ABBR_FILE.exists():
            print(f"Informal abbreviation file not found: {INFORMAL_ABBR_FILE}")
            return

        with open(INFORMAL_ABBR_FILE, encoding="utf-8") as f:
            data = json.load(f)

        mappings = data.get("mappings", [])

        # Alias 노드 생성 및 ALIAS_OF 관계 연결
        query = """
        UNWIND $batch AS row
        MERGE (a:Alias {name: row.abbreviation})
        SET a.category = row.category
        WITH a, row
        MATCH (s:Statute {name: row.full_name})
        MERGE (a)-[:ALIAS_OF]->(s)
        """

        batch = []
        total = 0
        with self.driver.session() as session:
            for category_data in mappings:
                category = category_data.get("category", "")
                items = category_data.get("items", [])

                for item in items:
                    abbr = item.get("abbreviation")
                    full_name = item.get("full_name")

                    if not abbr or not full_name:
                        continue

                    batch.append({
                        "abbreviation": abbr,
                        "full_name": full_name,
                        "category": category,
                    })
                    total += 1

            if batch:
                result = session.run(query, batch=batch)
                summary = result.consume()
                nodes_created = summary.counters.nodes_created
                rels_created = summary.counters.relationships_created
                print(f"Created {nodes_created} Alias nodes, {rels_created} ALIAS_OF relationships.")

        print(f"Processed {total} informal abbreviations.")

    def compute_citation_counts(self) -> None:
        """법령별 인용 수 미리 계산 (성능 최적화)"""
        print("Computing citation counts...")

        query = """
        MATCH (s:Statute)
        OPTIONAL MATCH (c:Case)-[:CITES]->(s)
        WITH s, count(c) as cnt
        SET s.citation_count = cnt
        RETURN count(s) as updated
        """

        with self.driver.session() as session:
            result = session.run(query)
            updated = result.single()["updated"]

        print(f"Updated {updated} statutes with citation_count.")

    def run(self) -> None:
        """전체 그래프 구축 실행"""
        self.create_constraints()
        self.load_statutes()
        self.load_abbreviations()
        self.load_informal_abbreviations()
        self.load_hierarchy()
        self.load_related_statutes()
        self.load_cases()
        self.load_case_citations()
        self.compute_citation_counts()
        self.close()
        print("Graph construction complete.")


if __name__ == "__main__":
    builder = GraphBuilder(NEO4J_URI, (NEO4J_USER, NEO4J_PASSWORD))
    builder.run()
