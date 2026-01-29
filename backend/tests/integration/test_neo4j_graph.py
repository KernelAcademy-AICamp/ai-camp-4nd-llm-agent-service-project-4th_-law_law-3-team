#!/usr/bin/env python3
"""
Neo4j Graph DB 테스트 스크립트

Neo4j에 저장된 법령/판례 그래프 데이터가 정상적으로 구축되었는지 검증합니다.

사용법:
    cd backend
    NEO4J_PASSWORD=password1234 uv run pytest tests/integration/test_neo4j_graph.py -v

    # 또는 직접 실행
    NEO4J_PASSWORD=password1234 uv run python tests/integration/test_neo4j_graph.py
"""

import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# .env 파일 로드
env_path = PROJECT_ROOT / ".env"
load_dotenv(env_path)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# 예상 최소 개수 (데이터 무결성 검증용)
MIN_STATUTE_COUNT = 5000
MIN_CASE_COUNT = 60000
MIN_HIERARCHY_COUNT = 3000
MIN_CITES_COUNT = 70000
MIN_CITES_CASE_COUNT = 80000  # 판례 간 인용
MIN_RELATED_TO_COUNT = 50  # 법령 간 관련


def get_driver():
    """Neo4j 드라이버 생성"""
    from neo4j import GraphDatabase
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


class TestNeo4jConnection:
    """Neo4j 연결 테스트"""

    def test_connection(self):
        """Neo4j 연결 확인"""
        driver = get_driver()
        try:
            driver.verify_connectivity()
            print(f"[OK] Neo4j 연결 성공: {NEO4J_URI}")
        finally:
            driver.close()

    def test_database_exists(self):
        """데이터베이스 존재 확인"""
        driver = get_driver()
        try:
            with driver.session() as session:
                result = session.run("RETURN 1 as test")
                record = result.single()
                assert record["test"] == 1
                print("[OK] 데이터베이스 정상")
        finally:
            driver.close()


class TestNodeCounts:
    """노드 개수 검증 테스트"""

    def test_statute_count(self):
        """법령(Statute) 노드 개수 검증"""
        driver = get_driver()
        try:
            with driver.session() as session:
                result = session.run("MATCH (s:Statute) RETURN count(s) as count")
                count = result.single()["count"]
                print(f"[INFO] Statute 노드: {count:,}개")
                assert count >= MIN_STATUTE_COUNT, \
                    f"Statute 개수 부족: {count} < {MIN_STATUTE_COUNT}"
        finally:
            driver.close()

    def test_case_count(self):
        """판례(Case) 노드 개수 검증"""
        driver = get_driver()
        try:
            with driver.session() as session:
                result = session.run("MATCH (c:Case) RETURN count(c) as count")
                count = result.single()["count"]
                print(f"[INFO] Case 노드: {count:,}개")
                assert count >= MIN_CASE_COUNT, \
                    f"Case 개수 부족: {count} < {MIN_CASE_COUNT}"
        finally:
            driver.close()


class TestRelationshipCounts:
    """관계 개수 검증 테스트"""

    def test_hierarchy_count(self):
        """HIERARCHY_OF 관계 개수 검증"""
        driver = get_driver()
        try:
            with driver.session() as session:
                result = session.run(
                    "MATCH ()-[r:HIERARCHY_OF]->() RETURN count(r) as count"
                )
                count = result.single()["count"]
                print(f"[INFO] HIERARCHY_OF 관계: {count:,}개")
                assert count >= MIN_HIERARCHY_COUNT, \
                    f"HIERARCHY_OF 개수 부족: {count} < {MIN_HIERARCHY_COUNT}"
        finally:
            driver.close()

    def test_cites_count(self):
        """CITES 관계 개수 검증"""
        driver = get_driver()
        try:
            with driver.session() as session:
                result = session.run(
                    "MATCH ()-[r:CITES]->() RETURN count(r) as count"
                )
                count = result.single()["count"]
                print(f"[INFO] CITES 관계: {count:,}개")
                assert count >= MIN_CITES_COUNT, \
                    f"CITES 개수 부족: {count} < {MIN_CITES_COUNT}"
        finally:
            driver.close()

    def test_cites_case_count(self):
        """CITES_CASE 관계 개수 검증 (판례 간 인용)"""
        driver = get_driver()
        try:
            with driver.session() as session:
                result = session.run(
                    "MATCH ()-[r:CITES_CASE]->() RETURN count(r) as count"
                )
                count = result.single()["count"]
                print(f"[INFO] CITES_CASE 관계: {count:,}개")
                assert count >= MIN_CITES_CASE_COUNT, \
                    f"CITES_CASE 개수 부족: {count} < {MIN_CITES_CASE_COUNT}"
        finally:
            driver.close()

    def test_related_to_count(self):
        """RELATED_TO 관계 개수 검증 (법령 간 관련)"""
        driver = get_driver()
        try:
            with driver.session() as session:
                result = session.run(
                    "MATCH ()-[r:RELATED_TO]->() RETURN count(r) as count"
                )
                count = result.single()["count"]
                print(f"[INFO] RELATED_TO 관계: {count:,}개")
                assert count >= MIN_RELATED_TO_COUNT, \
                    f"RELATED_TO 개수 부족: {count} < {MIN_RELATED_TO_COUNT}"
        finally:
            driver.close()


class TestHierarchyIntegrity:
    """법령 계급 관계 정합성 테스트"""

    def test_enforcement_decree_has_parent(self):
        """시행령은 상위 법률을 가져야 함"""
        driver = get_driver()
        try:
            with driver.session() as session:
                # 시행령 중 상위법이 있는 것 조회
                result = session.run("""
                    MATCH (child:Statute)-[:HIERARCHY_OF]->(parent:Statute)
                    WHERE child.name CONTAINS '시행령'
                    RETURN child.name as child, parent.name as parent
                    LIMIT 10
                """)
                records = list(result)
                print(f"[INFO] 시행령 → 법률 관계: {len(records)}개 샘플")

                for r in records[:3]:
                    print(f"       {r['child']} → {r['parent']}")

                assert len(records) > 0, "시행령 → 법률 관계가 없습니다"
        finally:
            driver.close()

    def test_hierarchy_direction(self):
        """계급 관계 방향 검증 (하위 → 상위)"""
        driver = get_driver()
        try:
            with driver.session() as session:
                # 시행령이 법률을 가리키는지 확인
                result = session.run("""
                    MATCH (child:Statute)-[:HIERARCHY_OF]->(parent:Statute)
                    WHERE child.name CONTAINS '시행령'
                      AND NOT parent.name CONTAINS '시행령'
                      AND NOT parent.name CONTAINS '시행규칙'
                    RETURN count(*) as count
                """)
                count = result.single()["count"]
                print(f"[INFO] 시행령 → 법률 (정방향) 관계: {count:,}개")
                assert count > 0, "시행령 → 법률 관계가 없습니다"
        finally:
            driver.close()

    def test_no_self_loop(self):
        """자기 참조 루프 없음 확인"""
        driver = get_driver()
        try:
            with driver.session() as session:
                result = session.run("""
                    MATCH (s:Statute)-[:HIERARCHY_OF]->(s)
                    RETURN count(*) as count
                """)
                count = result.single()["count"]
                print(f"[INFO] 자기 참조 루프: {count}개")
                assert count == 0, f"자기 참조 루프 발견: {count}개"
        finally:
            driver.close()


class TestCitationIntegrity:
    """판례 인용 관계 정합성 테스트"""

    def test_case_cites_statute(self):
        """판례가 법령을 인용하는 관계 확인"""
        driver = get_driver()
        try:
            with driver.session() as session:
                result = session.run("""
                    MATCH (c:Case)-[:CITES]->(s:Statute)
                    RETURN c.name as case_name, s.name as statute_name
                    LIMIT 10
                """)
                records = list(result)
                print(f"[INFO] 판례 → 법령 인용: {len(records)}개 샘플")

                for r in records[:3]:
                    case_name = r['case_name'][:30] if r['case_name'] else 'N/A'
                    print(f"       {case_name}... → {r['statute_name']}")

                assert len(records) > 0, "판례 → 법령 인용 관계가 없습니다"
        finally:
            driver.close()

    def test_most_cited_statutes(self):
        """가장 많이 인용된 법령 확인"""
        driver = get_driver()
        try:
            with driver.session() as session:
                result = session.run("""
                    MATCH (c:Case)-[:CITES]->(s:Statute)
                    RETURN s.name as statute, count(c) as citations
                    ORDER BY citations DESC
                    LIMIT 5
                """)
                records = list(result)
                print("[INFO] 가장 많이 인용된 법령 TOP 5:")

                for r in records:
                    print(f"       {r['statute']}: {r['citations']:,}건")

                # 민법이 가장 많이 인용되어야 함
                assert records[0]['statute'] == '민법', \
                    f"예상: 민법이 1위, 실제: {records[0]['statute']}"
        finally:
            driver.close()


class TestCaseCitationIntegrity:
    """판례 간 인용 관계 정합성 테스트"""

    def test_case_cites_case(self):
        """판례가 다른 판례를 인용하는 관계 확인"""
        driver = get_driver()
        try:
            with driver.session() as session:
                result = session.run("""
                    MATCH (from:Case)-[:CITES_CASE]->(to:Case)
                    RETURN from.case_number as from_num, to.case_number as to_num
                    LIMIT 10
                """)
                records = list(result)
                print(f"[INFO] 판례 → 판례 인용: {len(records)}개 샘플")

                for r in records[:3]:
                    print(f"       {r['from_num']} → {r['to_num']}")

                assert len(records) > 0, "판례 → 판례 인용 관계가 없습니다"
        finally:
            driver.close()

    def test_most_cited_cases(self):
        """가장 많이 인용된 판례 확인"""
        driver = get_driver()
        try:
            with driver.session() as session:
                result = session.run("""
                    MATCH (from:Case)-[:CITES_CASE]->(to:Case)
                    RETURN to.case_number as case_num, to.name as case_name,
                           count(from) as citations
                    ORDER BY citations DESC
                    LIMIT 5
                """)
                records = list(result)
                print("[INFO] 가장 많이 인용된 판례 TOP 5:")

                for r in records:
                    name = r['case_name'][:30] if r['case_name'] else 'N/A'
                    print(f"       {r['case_num']}: {r['citations']:,}건 ({name}...)")

                assert len(records) > 0, "인용된 판례가 없습니다"
        finally:
            driver.close()

    def test_no_self_citation(self):
        """자기 인용 최소화 확인 (데이터 품질)"""
        driver = get_driver()
        try:
            with driver.session() as session:
                result = session.run("""
                    MATCH (c:Case)-[:CITES_CASE]->(c)
                    RETURN count(*) as count
                """)
                count = result.single()["count"]
                print(f"[INFO] 자기 인용: {count}개")
                # 소수의 자기 인용은 데이터 품질 이슈로 허용
                assert count < 100, f"자기 인용 과다: {count}개 (< 100 권장)"
        finally:
            driver.close()


class TestRelatedStatuteIntegrity:
    """법령 간 관련 관계 정합성 테스트"""

    def test_statute_related_to_statute(self):
        """법령 간 관련 관계 확인"""
        driver = get_driver()
        try:
            with driver.session() as session:
                result = session.run("""
                    MATCH (s:Statute)-[:RELATED_TO]->(t:Statute)
                    RETURN s.name as source, t.name as target
                    LIMIT 10
                """)
                records = list(result)
                print(f"[INFO] 법령 → 법령 관련: {len(records)}개 샘플")

                for r in records[:5]:
                    print(f"       {r['source']} → {r['target']}")

                assert len(records) > 0, "법령 → 법령 관련 관계가 없습니다"
        finally:
            driver.close()

    def test_most_related_statutes(self):
        """가장 많은 관련 법령을 가진 법령 확인"""
        driver = get_driver()
        try:
            with driver.session() as session:
                result = session.run("""
                    MATCH (s:Statute)-[:RELATED_TO]->(t:Statute)
                    RETURN s.name as statute, count(t) as related_count
                    ORDER BY related_count DESC
                    LIMIT 5
                """)
                records = list(result)
                print("[INFO] 가장 많은 관련 법령을 가진 법령:")

                for r in records:
                    print(f"       {r['statute']}: {r['related_count']}개")
        finally:
            driver.close()


class TestGraphTraversal:
    """그래프 탐색 테스트"""

    def test_case_to_statute_to_upper(self):
        """판례 → 법령 → 상위법령 경로 탐색"""
        driver = get_driver()
        try:
            with driver.session() as session:
                result = session.run("""
                    MATCH (c:Case)-[:CITES]->(s:Statute)-[:HIERARCHY_OF]->(upper:Statute)
                    RETURN c.name as case_name, s.name as statute, upper.name as upper_statute
                    LIMIT 5
                """)
                records = list(result)
                print(f"[INFO] Case → Statute → Upper 경로: {len(records)}개")

                for r in records[:3]:
                    case_name = r['case_name'][:25] if r['case_name'] else 'N/A'
                    print(f"       {case_name}... → {r['statute']} → {r['upper_statute']}")

                assert len(records) > 0, "Case → Statute → Upper 경로가 없습니다"
        finally:
            driver.close()

    def test_hierarchy_chain(self):
        """계급 체인 탐색 (시행규칙 → 시행령 → 법률)"""
        driver = get_driver()
        try:
            with driver.session() as session:
                result = session.run("""
                    MATCH path = (s:Statute)-[:HIERARCHY_OF*1..3]->(root:Statute)
                    WHERE s.name CONTAINS '시행규칙'
                      AND NOT (root)-[:HIERARCHY_OF]->()
                    RETURN s.name as start,
                           [n in nodes(path) | n.name] as chain,
                           length(path) as depth
                    LIMIT 5
                """)
                records = list(result)
                print(f"[INFO] 계급 체인 (깊이 1~3): {len(records)}개")

                for r in records[:3]:
                    print(f"       깊이 {r['depth']}: {' → '.join(r['chain'])}")
        finally:
            driver.close()


class TestStatuteTypes:
    """법령 타입별 분포 테스트"""

    def test_statute_type_distribution(self):
        """법령 타입별 분포 확인"""
        driver = get_driver()
        try:
            with driver.session() as session:
                result = session.run("""
                    MATCH (s:Statute)
                    RETURN s.type as type, count(s) as count
                    ORDER BY count DESC
                    LIMIT 10
                """)
                records = list(result)
                print("[INFO] 법령 타입별 분포:")

                for r in records:
                    type_name = r['type'] or '(없음)'
                    print(f"       {type_name}: {r['count']:,}개")

                # 대통령령과 법률이 많아야 함
                types = [r['type'] for r in records[:3]]
                assert '대통령령' in types or '법률' in types, \
                    "대통령령 또는 법률이 상위에 없습니다"
        finally:
            driver.close()


class TestRAGContextEnrichment:
    """RAG 컨텍스트 보강 시나리오 테스트"""

    def test_get_statute_context(self):
        """법령 검색 시 상위/하위/관련 법령 컨텍스트 조회"""
        driver = get_driver()
        try:
            with driver.session() as session:
                # 민법 기준 컨텍스트 조회
                result = session.run("""
                    MATCH (s:Statute {name: '민법'})
                    OPTIONAL MATCH (s)-[:HIERARCHY_OF]->(upper:Statute)
                    OPTIONAL MATCH (lower:Statute)-[:HIERARCHY_OF]->(s)
                    OPTIONAL MATCH (s)-[:RELATED_TO]->(related:Statute)
                    RETURN s.name as statute,
                           collect(DISTINCT upper.name) as upper_laws,
                           collect(DISTINCT lower.name) as lower_laws,
                           collect(DISTINCT related.name) as related_laws
                """)
                record = result.single()
                print(f"[INFO] 민법 컨텍스트:")
                print(f"       상위법: {record['upper_laws']}")
                print(f"       하위법: {record['lower_laws'][:5]}...")
                print(f"       관련법: {record['related_laws'][:5]}...")

                assert record['statute'] == '민법', "민법을 찾을 수 없습니다"
        finally:
            driver.close()

    def test_get_case_context(self):
        """판례 검색 시 인용 법령/인용 판례 컨텍스트 조회"""
        driver = get_driver()
        try:
            with driver.session() as session:
                # 손해배상 관련 판례 컨텍스트 조회
                result = session.run("""
                    MATCH (c:Case)
                    WHERE c.name CONTAINS '손해배상'
                    WITH c LIMIT 1
                    OPTIONAL MATCH (c)-[:CITES]->(statute:Statute)
                    OPTIONAL MATCH (c)-[:CITES_CASE]->(cited_case:Case)
                    RETURN c.case_number as case_num,
                           c.name as case_name,
                           collect(DISTINCT statute.name) as cited_statutes,
                           collect(DISTINCT cited_case.case_number) as cited_cases
                """)
                record = result.single()
                if record:
                    print(f"[INFO] 판례 컨텍스트:")
                    print(f"       사건번호: {record['case_num']}")
                    print(f"       인용 법령: {record['cited_statutes'][:5]}")
                    print(f"       인용 판례: {record['cited_cases'][:5]}")
        finally:
            driver.close()


class TestCaseRecommendation:
    """판례 추천 시나리오 테스트"""

    def test_find_similar_cases_by_statute(self):
        """같은 법령을 인용한 유사 판례 찾기"""
        driver = get_driver()
        try:
            with driver.session() as session:
                # 인용 수가 적은 법령 기준으로 유사 판례 찾기 (메모리 최적화)
                result = session.run("""
                    MATCH (c1:Case)-[:CITES]->(s:Statute {name: '상표법'})<-[:CITES]-(c2:Case)
                    WHERE c1 <> c2 AND id(c1) < id(c2)
                    WITH c1, c2
                    LIMIT 5
                    RETURN c1.case_number as case1, c2.case_number as case2
                """)
                records = list(result)
                print(f"[INFO] 상표법 인용 유사 판례 쌍: {len(records)}개")

                for r in records[:3]:
                    print(f"       {r['case1']} <-> {r['case2']}")

                assert len(records) > 0, "유사 판례를 찾을 수 없습니다"
        finally:
            driver.close()

    def test_find_citing_cases(self):
        """특정 판례를 인용한 후속 판례 찾기"""
        driver = get_driver()
        try:
            with driver.session() as session:
                # 가장 많이 인용된 판례의 후속 판례 찾기
                result = session.run("""
                    MATCH (citing:Case)-[:CITES_CASE]->(cited:Case)
                    WITH cited, count(citing) as citation_count
                    ORDER BY citation_count DESC
                    LIMIT 1
                    MATCH (citing:Case)-[:CITES_CASE]->(cited)
                    RETURN cited.case_number as original_case,
                           collect(citing.case_number)[..5] as citing_cases,
                           citation_count
                """)
                record = result.single()
                if record:
                    print(f"[INFO] 가장 많이 인용된 판례:")
                    print(f"       원본: {record['original_case']} ({record['citation_count']}회 인용)")
                    print(f"       인용한 판례: {record['citing_cases']}")
        finally:
            driver.close()


class TestLegalExplorerUI:
    """법령 탐색 UI 시나리오 테스트"""

    def test_get_full_hierarchy_tree(self):
        """법령의 전체 계급 트리 조회"""
        driver = get_driver()
        try:
            with driver.session() as session:
                result = session.run("""
                    MATCH (s:Statute {name: '도로교통법'})
                    OPTIONAL MATCH path = (child:Statute)-[:HIERARCHY_OF*]->(s)
                    RETURN s.name as root,
                           [n in nodes(path) | n.name] as hierarchy_path
                    LIMIT 10
                """)
                records = list(result)
                print(f"[INFO] 도로교통법 계급 트리:")

                for r in records[:5]:
                    if r['hierarchy_path']:
                        print(f"       {' → '.join(r['hierarchy_path'])}")
        finally:
            driver.close()

    def test_get_citation_network(self):
        """판례의 인용 네트워크 조회 (2-depth)"""
        driver = get_driver()
        try:
            with driver.session() as session:
                result = session.run("""
                    MATCH (c:Case)-[:CITES_CASE]->(cited:Case)
                    WITH c LIMIT 1
                    MATCH path = (c)-[:CITES_CASE*1..2]->(related:Case)
                    RETURN c.case_number as origin,
                           [n in nodes(path) | n.case_number] as citation_path
                    LIMIT 10
                """)
                records = list(result)
                print(f"[INFO] 판례 인용 네트워크 (2-depth):")

                for r in records[:5]:
                    print(f"       {' → '.join(r['citation_path'])}")
        finally:
            driver.close()


def run_all_tests():
    """모든 테스트 실행 (직접 실행용)"""
    print("=" * 60)
    print("Neo4j Graph DB 테스트")
    print("=" * 60)
    print(f"URI: {NEO4J_URI}")
    print(f"User: {NEO4J_USER}")
    print("=" * 60)

    test_classes = [
        TestNeo4jConnection,
        TestNodeCounts,
        TestRelationshipCounts,
        TestHierarchyIntegrity,
        TestCitationIntegrity,
        TestCaseCitationIntegrity,
        TestRelatedStatuteIntegrity,
        TestGraphTraversal,
        TestStatuteTypes,
        TestRAGContextEnrichment,
        TestCaseRecommendation,
        TestLegalExplorerUI,
    ]

    passed = 0
    failed = 0

    for test_class in test_classes:
        print(f"\n{'─' * 40}")
        print(f"테스트 클래스: {test_class.__name__}")
        print("─" * 40)

        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                try:
                    print(f"\n  ▶ {method_name}")
                    getattr(instance, method_name)()
                    print(f"    ✓ PASSED")
                    passed += 1
                except AssertionError as e:
                    print(f"    ✗ FAILED: {e}")
                    failed += 1
                except Exception as e:
                    print(f"    ✗ ERROR: {e}")
                    failed += 1

    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)
    print(f"  통과: {passed}")
    print(f"  실패: {failed}")
    print(f"  총계: {passed + failed}")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("[ERROR] neo4j 패키지가 설치되지 않았습니다.")
        print("        uv pip install neo4j")
        sys.exit(1)

    success = run_all_tests()
    sys.exit(0 if success else 1)
