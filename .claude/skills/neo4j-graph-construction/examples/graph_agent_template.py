"""
LangGraph 기반 법률 그래프 에이전트 템플릿

Neo4j GraphDB를 활용하여 법령 계급 및 판례 인용 관계를 탐색합니다.
"""

import os
from typing import Dict, List

from langchain_core.prompts import PromptTemplate
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class Statute(BaseModel):
    """법령 스키마"""

    name: str = Field(description="Name of the statute law")
    articles: List[str] = Field(description="List of article numbers")


class LegalGraphDB:
    """Neo4j 그래프 DB 연결 관리자"""

    def __init__(self, uri: str, auth: tuple[str, str]) -> None:
        self.graph = Neo4jGraph(
            url=uri,
            username=auth[0],
            password=auth[1],
            enhanced_schema=True,
        )

    def refresh_schema(self) -> None:
        """Neo4j 스키마 갱신"""
        self.graph.refresh_schema()
        print(self.graph.schema)

    def query(self, cypher: str, params: Dict | None = None) -> List[Dict]:
        """Raw Cypher 쿼리 실행"""
        if params is None:
            params = {}
        return self.graph.query(cypher, params)


class LegalGraphAgent:
    """
    LangGraph 호환 법률 그래프 에이전트

    법령 계급(Hierarchy) 및 판례 인용(Citation) 관계를 탐색합니다.
    """

    def __init__(self, llm_model: str = "gpt-4o") -> None:
        self.llm = ChatOpenAI(temperature=0, model=llm_model)

        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "password")

        self.db = LegalGraphDB(
            uri=neo4j_uri,
            auth=(neo4j_user, neo4j_password),
        )

        self.cypher_prompt = PromptTemplate(
            input_variables=["question", "schema"],
            template="""
            You are an expert Neo4j Cypher translator.
            Convert the user's question into a Cypher query based on the schema.

            Schema:
            {schema}

            Question: {question}
            Cypher Query:""",
        )

        self.chain = GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=self.db.graph,
            cypher_prompt=self.cypher_prompt,
            verbose=True,
            allow_dangerous_requests=True,
        )

    def find_statute_hierarchy(self, statute_name: str) -> List[Dict]:
        """
        특정 법령의 상하위 계급을 조회합니다.

        Args:
            statute_name: 법령명 (예: "도로교통법 시행령")

        Returns:
            계급 정보 리스트 (current, upper_laws, lower_laws)
        """
        query = """
        MATCH (s:Statute {name: $name})
        OPTIONAL MATCH (s)-[:HIERARCHY_OF]->(upper)
        OPTIONAL MATCH (s)<-[:HIERARCHY_OF]-(lower)
        RETURN
            s.name as current,
            collect(DISTINCT upper.name) as upper_laws,
            collect(DISTINCT lower.name) as lower_laws
        """
        return self.db.query(query, {"name": statute_name})

    def find_case_citations(self, case_number: str) -> List[Dict]:
        """
        특정 판례가 인용한 법령을 조회합니다.

        Args:
            case_number: 사건번호 (예: "2023다12345")

        Returns:
            인용 법령 리스트
        """
        query = """
        MATCH (c:Case {case_number: $case_number})-[:CITES]->(s:Statute)
        RETURN s.name as statute_name, s.type as statute_type
        """
        return self.db.query(query, {"case_number": case_number})

    def run(self, user_query: str) -> str:
        """
        자연어 질의를 처리합니다.

        Args:
            user_query: 사용자 질의

        Returns:
            처리 결과 문자열
        """
        try:
            response = self.chain.invoke(user_query)
            return response["result"]
        except Exception as e:
            return f"Error processing graph query: {e!s}"


if __name__ == "__main__":
    agent = LegalGraphAgent()

    print("Checking hierarchy for '도로교통법 시행령'...")
    hierarchy = agent.find_statute_hierarchy("도로교통법 시행령")
    print(hierarchy)
