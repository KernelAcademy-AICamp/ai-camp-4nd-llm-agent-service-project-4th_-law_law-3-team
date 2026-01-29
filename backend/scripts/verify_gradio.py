"""
Neo4j Graph DB 검증 스크립트 (Gradio UI)

그래프 데이터를 시각적으로 탐색합니다.
- 법령 계급 탐색 (Hierarchy Explorer)
- 판례 인용 추적 (Citation Tracker)
- Cypher 쿼리 (Raw Query)

사용법:
    cd backend
    uv run python scripts/verify_gradio.py
    # → http://localhost:7860
"""

import json
import os

import gradio as gr
from neo4j import GraphDatabase

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


class GraphVerifier:
    """Neo4j 그래프 검증 클래스"""

    def __init__(self) -> None:
        self.driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )

    def close(self) -> None:
        """드라이버 종료"""
        self.driver.close()

    def get_hierarchy(self, statute_name: str) -> str:
        """
        법령 계급 조회

        Args:
            statute_name: 법령명

        Returns:
            계급 구조 JSON
        """
        query_upper = """
        MATCH (s:Statute {name: $name})
        OPTIONAL MATCH path = (s)-[:HIERARCHY_OF*]->(root)
        RETURN [n in nodes(path) | n.name] as path_names
        """

        query_lower = """
        MATCH (s:Statute {name: $name})
        OPTIONAL MATCH (child)-[:HIERARCHY_OF]->(s)
        RETURN child.name as child_name, child.type as child_type
        """

        result = {"law": statute_name, "parents_chain": [], "children": []}

        try:
            with self.driver.session() as session:
                res_upper = list(session.run(query_upper, name=statute_name))
                paths = []
                for record in res_upper:
                    if record["path_names"]:
                        paths.append(" -> ".join(record["path_names"]))
                result["parents_chain"] = paths

                res_lower = list(session.run(query_lower, name=statute_name))
                children = []
                for record in res_lower:
                    if record["child_name"]:
                        children.append(
                            f"{record['child_name']} ({record['child_type']})"
                        )
                result["children"] = children

        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)

        return json.dumps(result, indent=2, ensure_ascii=False)

    def search_citation(self, case_query: str) -> str:
        """
        판례 인용 법령 검색

        Args:
            case_query: 사건번호 또는 사건명

        Returns:
            인용 분석 결과 (Markdown)
        """
        query = """
        MATCH (c:Case)
        WHERE c.name CONTAINS $q OR c.case_number CONTAINS $q
        RETURN c.case_number, c.name, c.summary
        LIMIT 1
        """

        citation_query = """
        MATCH (c:Case {case_number: $c_num})-[:CITES]->(s:Statute)
        OPTIONAL MATCH (s)-[:HIERARCHY_OF*]->(root)
        WHERE NOT (root)-[:HIERARCHY_OF]->()
        RETURN s.name as cited_law, coalesce(root.name, s.name) as root_law
        """

        try:
            with self.driver.session() as session:
                res_case = list(session.run(query, q=case_query))
                if not res_case:
                    return "No case found."

                case_node = res_case[0]
                case_num = case_node["c.case_number"]
                summary = case_node["c.summary"] or "No summary"

                output = (
                    f"## [{case_num}] {case_node['c.name']}\n\n"
                    f"**판결요지**:\n{summary[:300]}...\n\n"
                    "### 인용 법령 분석\n"
                )

                res_cit = list(session.run(citation_query, c_num=case_num))
                if not res_cit:
                    output += "\n(인용된 법령이 그래프에 없습니다.)"
                else:
                    output += (
                        "| 인용 법령 (Cited) | 근거 상위법 (Root Hierarchy) |\n"
                        "|---|---|\n"
                    )
                    for row in res_cit:
                        cited = row["cited_law"]
                        root = row["root_law"]
                        output += f"| {cited} | **{root}** |\n"

                return output

        except Exception as e:
            return f"Error: {e!s}"

    def run_cypher(self, cypher: str) -> str:
        """
        Raw Cypher 쿼리 실행

        Args:
            cypher: Cypher 쿼리

        Returns:
            결과 JSON
        """
        try:
            with self.driver.session() as session:
                result = session.run(cypher)
                data = [dict(record) for record in result]
                return json.dumps(data, indent=2, ensure_ascii=False)
        except Exception as e:
            return str(e)


def create_demo() -> gr.Blocks:
    """Gradio UI 생성"""
    verifier = GraphVerifier()

    with gr.Blocks(title="Law Graph Verifier") as demo:
        gr.Markdown("# Law Graph Verification Tool")
        gr.Markdown(
            "Neo4j 그래프 데이터(법령 계급, 판례 인용)를 직접 탐색하고 검증합니다."
        )

        with gr.Tab("법령 계급 탐색 (Hierarchy Explorer)"):
            law_input = gr.Textbox(
                label="법령명 입력 (예: 도로교통법, 소득세법 시행령)",
                value="도로교통법 시행령",
            )
            law_btn = gr.Button("계급도 검색")
            law_output = gr.Code(label="계급 구조 (JSON)", language="json")
            law_btn.click(
                verifier.get_hierarchy, inputs=law_input, outputs=law_output
            )

        with gr.Tab("판례 인용 추적 (Citation Tracker)"):
            case_input = gr.Textbox(
                label="판례 검색 (사건번호 또는 사건명)", value="도로교통법"
            )
            case_btn = gr.Button("인용 분석")
            case_output = gr.Markdown(label="분석 결과")
            case_btn.click(
                verifier.search_citation, inputs=case_input, outputs=case_output
            )

        with gr.Tab("Cypher 쿼리 (Raw Query)"):
            cypher_input = gr.Textbox(
                label="Cypher Query",
                value="MATCH (n:Statute) RETURN n.name LIMIT 5",
                lines=3,
            )
            cypher_btn = gr.Button("실행")
            cypher_output = gr.Code(label="실행 결과 (JSON)", language="json")
            cypher_btn.click(
                verifier.run_cypher, inputs=cypher_input, outputs=cypher_output
            )

    return demo


if __name__ == "__main__":
    app = create_demo()
    app.launch(server_name="0.0.0.0", server_port=7860)
