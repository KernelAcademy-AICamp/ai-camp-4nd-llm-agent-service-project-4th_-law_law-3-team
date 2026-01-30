"""
데이터셋 빌더 (역추적 방식)

문서를 먼저 선택하고, 그 문서를 기반으로 질문을 작성하는 방식
선택한 문서가 자동으로 Ground Truth가 됨

사용 흐름:
1. 문서 검색/선택 → 2. 질문 작성 → 3. Ground Truth 자동 생성
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from evaluation.schemas import (
    EvalDataset,
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


class DatasetBuilder:
    """
    평가 데이터셋 빌더

    역추적 방식으로 데이터셋 생성:
    1. 문서(판례/법령)를 먼저 선택
    2. 해당 문서 기반으로 질문 작성
    3. 선택한 문서가 Ground Truth로 자동 설정

    Usage:
        builder = DatasetBuilder("eval_dataset_v1")

        # 문서 선택 및 질문 작성
        builder.add_query(
            question="임대차 보증금 반환 청구 요건은?",
            source_documents=[
                {"doc_id": "76396", "doc_type": "precedent"},
                {"doc_id": "010719", "doc_type": "law", "article": "제621조"},
            ],
            category="민사",
            query_type="개념검색",
            key_points=["계약 종료 요건", "동시이행"],
        )

        # 저장
        builder.save()
    """

    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        version: str = "1.0",
    ):
        """
        Args:
            name: 데이터셋 이름
            description: 데이터셋 설명
            version: 버전
        """
        self.dataset = EvalDataset(
            name=name,
            description=description,
            version=version,
        )

    def add_query(
        self,
        question: str,
        source_documents: list[dict],
        category: str,
        query_type: str,
        difficulty: str = "medium",
        subcategory: Optional[str] = None,
        key_points: Optional[list[str]] = None,
        required_citations: Optional[list[str]] = None,
        source: str = "manual",
    ) -> EvalQuery:
        """
        쿼리 추가

        Args:
            question: 질문 텍스트
            source_documents: 소스 문서 목록
                [{"doc_id": "...", "doc_type": "precedent|law", "article": "제X조"}]
            category: 카테고리 (민사, 형사, 행정 등)
            query_type: 쿼리 유형 (단순조회, 개념검색 등)
            difficulty: 난이도 (easy, medium, hard)
            subcategory: 하위 카테고리
            key_points: 핵심 포인트 목록
            required_citations: 필수 인용 목록
            source: 생성 방식 (manual, solar)

        Returns:
            생성된 EvalQuery
        """
        parsed_docs = []
        for doc in source_documents:
            parsed_docs.append(
                SourceDocument(
                    doc_id=doc["doc_id"],
                    doc_type=DocumentType(doc["doc_type"]),
                    relevance=Relevance(doc.get("relevance", "highly_relevant")),
                    reason=doc.get("reason"),
                    article=doc.get("article"),
                )
            )

        ground_truth = GroundTruth(
            source_documents=parsed_docs,
            key_points=key_points or [],
            required_citations=required_citations or [],
        )

        metadata = QueryMetadata(
            category=Category(category),
            subcategory=subcategory,
            query_type=QueryType(query_type),
            difficulty=Difficulty(difficulty),
        )

        query = EvalQuery(
            id=self.dataset.get_next_id(),
            question=question,
            metadata=metadata,
            ground_truth=ground_truth,
            source=source,
        )

        self.dataset.add_query(query)
        return query

    def add_query_from_eval_query(self, query: EvalQuery) -> None:
        """
        기존 EvalQuery 객체 추가

        Args:
            query: 추가할 EvalQuery
        """
        query.id = self.dataset.get_next_id()
        self.dataset.add_query(query)

    def get_query_by_id(self, query_id: str) -> Optional[EvalQuery]:
        """ID로 쿼리 조회"""
        for query in self.dataset.queries:
            if query.id == query_id:
                return query
        return None

    def update_query(self, query_id: str, **updates) -> Optional[EvalQuery]:
        """
        쿼리 업데이트

        Args:
            query_id: 쿼리 ID
            **updates: 업데이트할 필드

        Returns:
            업데이트된 쿼리
        """
        query = self.get_query_by_id(query_id)
        if not query:
            return None

        for key, value in updates.items():
            if hasattr(query, key):
                setattr(query, key, value)

        self.dataset.updated_at = datetime.now()
        return query

    def delete_query(self, query_id: str) -> bool:
        """쿼리 삭제"""
        original_len = len(self.dataset.queries)
        self.dataset.queries = [q for q in self.dataset.queries if q.id != query_id]
        return len(self.dataset.queries) < original_len

    def get_statistics(self) -> dict:
        """데이터셋 통계"""
        queries = self.dataset.queries
        if not queries:
            return {"total": 0}

        by_category: dict[str, int] = {}
        by_type: dict[str, int] = {}
        by_difficulty: dict[str, int] = {}
        by_source: dict[str, int] = {}

        for query in queries:
            cat = query.metadata.category.value
            by_category[cat] = by_category.get(cat, 0) + 1

            qtype = query.metadata.query_type.value
            by_type[qtype] = by_type.get(qtype, 0) + 1

            diff = query.metadata.difficulty.value
            by_difficulty[diff] = by_difficulty.get(diff, 0) + 1

            src = query.source
            by_source[src] = by_source.get(src, 0) + 1

        return {
            "total": len(queries),
            "by_category": by_category,
            "by_type": by_type,
            "by_difficulty": by_difficulty,
            "by_source": by_source,
        }

    def validate(self) -> list[str]:
        """
        데이터셋 검증

        Returns:
            오류 메시지 목록 (빈 리스트면 유효)
        """
        errors = []

        for query in self.dataset.queries:
            if not query.question.strip():
                errors.append(f"{query.id}: 질문이 비어있습니다")

            if not query.ground_truth.source_documents:
                errors.append(f"{query.id}: Ground Truth 문서가 없습니다")

        ids = [q.id for q in self.dataset.queries]
        if len(ids) != len(set(ids)):
            errors.append("중복된 쿼리 ID가 있습니다")

        return errors

    def save(
        self,
        path: Optional[Path] = None,
        format: str = "json",
    ) -> Path:
        """
        데이터셋 저장

        Args:
            path: 저장 경로 (None이면 기본 경로)
            format: 저장 형식 (json, yaml)

        Returns:
            저장된 파일 경로
        """
        if path is None:
            filename = f"{self.dataset.name}.{format}"
            path = eval_settings.datasets_dir / filename

        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    self.dataset.model_dump(mode="json"),
                    f,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
        elif format == "yaml":
            import yaml
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(
                    self.dataset.model_dump(mode="json"),
                    f,
                    allow_unicode=True,
                    default_flow_style=False,
                )

        return path

    @classmethod
    def load(cls, path: Path) -> "DatasetBuilder":
        """
        데이터셋 로드

        Args:
            path: 파일 경로

        Returns:
            DatasetBuilder 인스턴스
        """
        suffix = path.suffix.lower()

        if suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif suffix in (".yaml", ".yml"):
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"지원하지 않는 형식: {suffix}")

        dataset = EvalDataset.model_validate(data)

        builder = cls(
            name=dataset.name,
            description=dataset.description,
            version=dataset.version,
        )
        builder.dataset = dataset

        return builder

    def export_for_evaluation(self) -> list[dict]:
        """
        평가 실행용 데이터 내보내기

        Returns:
            평가 실행에 필요한 최소 데이터
        """
        return [
            {
                "id": q.id,
                "question": q.question,
                "ground_truth_ids": [
                    doc.doc_id for doc in q.ground_truth.source_documents
                ],
                "category": q.metadata.category.value,
                "query_type": q.metadata.query_type.value,
                "difficulty": q.metadata.difficulty.value,
            }
            for q in self.dataset.queries
        ]
