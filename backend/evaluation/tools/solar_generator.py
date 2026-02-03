"""
Upstage Solar 자동 질문 생성기

판례/법령 데이터를 기반으로 Solar 모델을 사용하여
평가용 질문을 자동 생성

사용 흐름:
1. PostgreSQL에서 판례/법령 샘플링
2. Solar 모델로 질문 + Key Points 자동 생성
3. 원본 문서가 자동으로 Ground Truth
"""

import json
import os
import random
from typing import Optional

import httpx

from evaluation.schemas import (
    EvalQuery,
    GroundTruth,
    SourceDocument,
    QueryMetadata,
    DocumentType,
    QueryType,
    Category,
    Difficulty,
    Relevance,
)
from evaluation.config import eval_settings


PRECEDENT_PROMPT_TEMPLATE = """다음 판례의 판시사항을 보고, 일반인이 법률 상담 시 물어볼 만한 자연스러운 질문을 생성하세요.

[판시사항]
{summary}

[사건명]
{case_name}

[법원명]
{court_name}

요구사항:
1. 법률 전문용어 대신 일상적인 표현 사용
2. 구체적인 상황을 가정한 질문
3. 1개의 질문만 생성

출력 형식 (JSON):
{{"question": "...", "key_points": ["...", "..."], "category": "민사|형사|행정|헌법|노동|상사|조세|기타", "query_type": "단순조회|개념검색|비교검색|참조추적|시간검색|복합검색"}}
"""

LAW_PROMPT_TEMPLATE = """다음 법령 조문을 보고, 일반인이 이 법에 대해 물어볼 만한 질문을 생성하세요.

[법령명]
{law_name}

[조문 내용]
{content}

요구사항:
1. "~란 무엇인가요?" 또는 "~할 때 어떻게 해야 하나요?" 형태
2. 실생활에서 필요한 정보를 묻는 질문
3. 1개의 질문만 생성

출력 형식 (JSON):
{{"question": "...", "key_points": ["...", "..."], "citations": ["..."], "category": "민사|형사|행정|헌법|노동|상사|조세|기타", "query_type": "단순조회|개념검색|비교검색|참조추적|시간검색|복합검색"}}
"""


class SolarGenerator:
    """
    Upstage Solar 기반 질문 자동 생성기

    Usage:
        generator = SolarGenerator()

        # 판례 기반 질문 생성
        queries = await generator.generate_from_precedents(
            precedents=[...],
            count=10,
        )

        # 법령 기반 질문 생성
        queries = await generator.generate_from_laws(
            laws=[...],
            count=10,
        )
    """

    UPSTAGE_API_URL = "https://api.upstage.ai/v1/solar/chat/completions"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "solar-pro3-260126",
    ):
        """
        Args:
            api_key: Upstage API 키 (None이면 환경변수에서 로드)
            model: 사용할 모델 (solar-pro3-260126, solar-pro3)
        """
        self.api_key = api_key or os.getenv("UPSTAGE_API_KEY")
        if not self.api_key:
            self.api_key = eval_settings.upstage_api_key

        self.model = model
        self._query_counter = 0

    async def _call_solar(
        self,
        prompt: str,
        temperature: float = 0.7,
    ) -> Optional[dict]:
        """
        Solar API 호출

        Args:
            prompt: 프롬프트
            temperature: 생성 온도

        Returns:
            파싱된 응답 또는 None
        """
        if not self.api_key:
            raise ValueError("UPSTAGE_API_KEY가 설정되지 않았습니다")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": 1500,  # reasoning 모델은 사고 과정에 토큰을 많이 사용
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                self.UPSTAGE_API_URL,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            result = response.json()

        content = result["choices"][0]["message"]["content"]
        return self._parse_json_response(content)

    def _parse_json_response(self, content: str) -> Optional[dict]:
        """JSON 응답 파싱"""
        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = content[start:end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        return None

    def _get_next_id(self) -> str:
        """다음 쿼리 ID 반환"""
        self._query_counter += 1
        return f"Q-{self._query_counter:03d}"

    def _map_category(self, category_str: str) -> Category:
        """카테고리 매핑"""
        mapping = {
            "민사": Category.CIVIL,
            "형사": Category.CRIMINAL,
            "행정": Category.ADMINISTRATIVE,
            "헌법": Category.CONSTITUTIONAL,
            "노동": Category.LABOR,
            "상사": Category.COMMERCIAL,
            "조세": Category.TAX,
        }
        return mapping.get(category_str, Category.OTHER)

    def _map_query_type(self, type_str: str) -> QueryType:
        """쿼리 유형 매핑"""
        mapping = {
            "단순조회": QueryType.SIMPLE_LOOKUP,
            "개념검색": QueryType.CONCEPT_SEARCH,
            "비교검색": QueryType.COMPARISON,
            "참조추적": QueryType.REFERENCE_TRACE,
            "시간검색": QueryType.TEMPORAL,
            "복합검색": QueryType.COMPLEX,
        }
        return mapping.get(type_str, QueryType.CONCEPT_SEARCH)

    async def generate_from_precedent(
        self,
        precedent: dict,
    ) -> Optional[EvalQuery]:
        """
        판례에서 질문 생성

        Args:
            precedent: 판례 데이터 (PostgreSQL 조회 결과)

        Returns:
            생성된 EvalQuery 또는 None
        """
        summary = precedent.get("summary") or precedent.get("reasoning", "")
        case_name = precedent.get("case_name", "")
        court_name = precedent.get("court_name", "")

        if not summary:
            return None

        prompt = PRECEDENT_PROMPT_TEMPLATE.format(
            summary=summary[:1500],
            case_name=case_name,
            court_name=court_name,
        )

        result = await self._call_solar(prompt)
        if not result or "question" not in result:
            return None

        source_doc = SourceDocument(
            doc_id=str(precedent.get("serial_number", "")),
            doc_type=DocumentType.PRECEDENT,
            relevance=Relevance.HIGHLY_RELEVANT,
            reason="Solar 자동 생성의 원본 문서",
        )

        ground_truth = GroundTruth(
            source_documents=[source_doc],
            key_points=result.get("key_points", []),
            required_citations=[],
        )

        metadata = QueryMetadata(
            category=self._map_category(result.get("category", "기타")),
            query_type=self._map_query_type(result.get("query_type", "개념검색")),
            difficulty=Difficulty.MEDIUM,
        )

        return EvalQuery(
            id=self._get_next_id(),
            question=result["question"],
            metadata=metadata,
            ground_truth=ground_truth,
            source="solar",
        )

    async def generate_from_law(
        self,
        law: dict,
    ) -> Optional[EvalQuery]:
        """
        법령에서 질문 생성

        Args:
            law: 법령 데이터 (PostgreSQL 조회 결과)

        Returns:
            생성된 EvalQuery 또는 None
        """
        law_name = law.get("law_name", "")
        content = law.get("content", "")

        if not content:
            return None

        prompt = LAW_PROMPT_TEMPLATE.format(
            law_name=law_name,
            content=content[:2000],
        )

        result = await self._call_solar(prompt)
        if not result or "question" not in result:
            return None

        source_doc = SourceDocument(
            doc_id=str(law.get("law_id", "")),
            doc_type=DocumentType.LAW,
            relevance=Relevance.HIGHLY_RELEVANT,
            reason="Solar 자동 생성의 원본 문서",
        )

        ground_truth = GroundTruth(
            source_documents=[source_doc],
            key_points=result.get("key_points", []),
            required_citations=result.get("citations", []),
        )

        metadata = QueryMetadata(
            category=self._map_category(result.get("category", "기타")),
            query_type=self._map_query_type(result.get("query_type", "개념검색")),
            difficulty=Difficulty.MEDIUM,
        )

        return EvalQuery(
            id=self._get_next_id(),
            question=result["question"],
            metadata=metadata,
            ground_truth=ground_truth,
            source="solar",
        )

    async def generate_from_precedents(
        self,
        precedents: list[dict],
        count: Optional[int] = None,
    ) -> list[EvalQuery]:
        """
        여러 판례에서 질문 생성

        Args:
            precedents: 판례 목록
            count: 생성할 개수 (None이면 전체)

        Returns:
            생성된 EvalQuery 목록
        """
        if count and count < len(precedents):
            precedents = random.sample(precedents, count)

        queries = []
        for precedent in precedents:
            query = await self.generate_from_precedent(precedent)
            if query:
                queries.append(query)

        return queries

    async def generate_from_laws(
        self,
        laws: list[dict],
        count: Optional[int] = None,
    ) -> list[EvalQuery]:
        """
        여러 법령에서 질문 생성

        Args:
            laws: 법령 목록
            count: 생성할 개수 (None이면 전체)

        Returns:
            생성된 EvalQuery 목록
        """
        if count and count < len(laws):
            laws = random.sample(laws, count)

        queries = []
        for law in laws:
            query = await self.generate_from_law(law)
            if query:
                queries.append(query)

        return queries


async def main():
    """CLI 실행"""
    import argparse
    import asyncio
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from sqlalchemy import select
    from app.core.database import async_session_factory
    from app.models.precedent_document import PrecedentDocument
    from app.models.law_document import LawDocument
    from evaluation.tools.dataset_builder import DatasetBuilder

    parser = argparse.ArgumentParser(description="Solar 자동 질문 생성")
    parser.add_argument("--count", type=int, default=30, help="생성할 질문 수")
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/datasets/solar_generated.json",
        help="출력 파일 경로",
    )
    parser.add_argument(
        "--precedent-ratio",
        type=float,
        default=0.5,
        help="판례 비율 (0.0 ~ 1.0)",
    )

    args = parser.parse_args()

    generator = SolarGenerator()
    precedent_count = int(args.count * args.precedent_ratio)
    law_count = args.count - precedent_count

    async with async_session_factory() as session:
        precedent_result = await session.execute(
            select(PrecedentDocument)
            .where(PrecedentDocument.summary.isnot(None))
            .limit(precedent_count * 2)
        )
        precedents = [
            {
                "serial_number": p.serial_number,
                "case_name": p.case_name,
                "court_name": p.court_name,
                "summary": p.summary,
                "reasoning": p.reasoning,
            }
            for p in precedent_result.scalars()
        ]

        law_result = await session.execute(
            select(LawDocument)
            .where(LawDocument.content.isnot(None))
            .limit(law_count * 2)
        )
        laws = [
            {
                "law_id": l.law_id,
                "law_name": l.law_name,
                "content": l.content,
            }
            for l in law_result.scalars()
        ]

    print(f"판례 {len(precedents)}건, 법령 {len(laws)}건 로드됨")

    precedent_queries = await generator.generate_from_precedents(
        precedents, count=precedent_count
    )
    print(f"판례 기반 질문 {len(precedent_queries)}개 생성됨")

    law_queries = await generator.generate_from_laws(laws, count=law_count)
    print(f"법령 기반 질문 {len(law_queries)}개 생성됨")

    builder = DatasetBuilder(
        name="solar_generated",
        description="Upstage Solar 모델로 자동 생성된 평가 데이터셋",
    )

    for query in precedent_queries + law_queries:
        builder.add_query_from_eval_query(query)

    output_path = Path(args.output)
    builder.save(output_path)
    print(f"저장됨: {output_path}")
    print(f"총 {len(builder.dataset.queries)}개 질문")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
