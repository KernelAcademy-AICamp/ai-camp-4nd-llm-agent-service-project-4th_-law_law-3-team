"""
Neo4j Graph DB 검증 스크립트 (CLI)

그래프 통계 및 샘플 경로를 확인합니다.

사용법:
    cd backend
    uv run python scripts/verify_graph.py
"""

import os

from neo4j import GraphDatabase

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


def verify() -> None:
    """그래프 통계 및 샘플 경로 출력"""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        print("=== Graph Statistics ===")
        res = session.run("MATCH (n) RETURN labels(n) as label, count(n) as count")
        for record in res:
            labels = record["label"]
            label_str = labels[0] if labels else "Unknown"
            print(f"{label_str}: {record['count']}")

        print("\n=== Relationship Statistics ===")
        res = session.run(
            "MATCH ()-[r]->() RETURN type(r) as type, count(r) as count"
        )
        for record in res:
            print(f"{record['type']}: {record['count']}")

        print("\n=== Sample Path: Case -> Statute -> Upper Statute ===")
        query = """
        MATCH path = (c:Case)-[:CITES]->(s:Statute)-[:HIERARCHY_OF]->(upper:Statute)
        RETURN c.name, s.name, upper.name, length(path)
        LIMIT 5
        """
        res = list(session.run(query))
        if not res:
            print(
                "No paths found matching pattern "
                "(Case)-[:CITES]->(Statute)-[:HIERARCHY_OF]->(Upper)"
            )
            print("\n=== Sample Path: Case -> Statute ===")
            query2 = (
                "MATCH (c:Case)-[:CITES]->(s:Statute) "
                "RETURN c.name, s.name LIMIT 5"
            )
            res2 = session.run(query2)
            for record in res2:
                print(
                    f"Case: {record['c.name']} -> Cites -> Statute: {record['s.name']}"
                )
        else:
            for record in res:
                print(f"Case: {record['c.name']}")
                print(f"  -> Cites -> {record['s.name']}")
                print(f"  -> Hierarchy -> {record['upper.name']}")
                print("-" * 40)

    driver.close()


if __name__ == "__main__":
    verify()
