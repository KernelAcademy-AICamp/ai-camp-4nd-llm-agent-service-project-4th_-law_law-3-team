"""
평가 스키마 단위 테스트

Pydantic 모델 유효성 검증
"""

import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from pydantic import ValidationError

from evaluation.schemas import (
    DocumentType,
    QueryType,
    Difficulty,
    Relevance,
    Category,
    SourceDocument,
    GroundTruth,
    QueryMetadata,
    EvalQuery,
    EvalDataset,
    SearchResult,
    MetricsResult,
    ExperimentConfig,
)


class TestDocumentType:
    """DocumentType 열거형 테스트"""

    def test_valid_types(self):
        assert DocumentType.PRECEDENT.value == "precedent"
        assert DocumentType.LAW.value == "law"

    def test_from_string(self):
        assert DocumentType("precedent") == DocumentType.PRECEDENT
        assert DocumentType("law") == DocumentType.LAW


class TestSourceDocument:
    """SourceDocument 모델 테스트"""

    def test_create_precedent(self):
        doc = SourceDocument(
            doc_id="76396",
            doc_type=DocumentType.PRECEDENT,
        )
        assert doc.doc_id == "76396"
        assert doc.doc_type == DocumentType.PRECEDENT
        assert doc.relevance == Relevance.HIGHLY_RELEVANT  # 기본값

    def test_create_law_with_article(self):
        doc = SourceDocument(
            doc_id="010719",
            doc_type=DocumentType.LAW,
            article="제750조",
            reason="불법행위 조문",
        )
        assert doc.article == "제750조"
        assert doc.reason == "불법행위 조문"

    def test_invalid_doc_type(self):
        with pytest.raises(ValidationError):
            SourceDocument(
                doc_id="123",
                doc_type="invalid_type",
            )


class TestGroundTruth:
    """GroundTruth 모델 테스트"""

    def test_create_with_documents(self):
        gt = GroundTruth(
            source_documents=[
                SourceDocument(doc_id="76396", doc_type=DocumentType.PRECEDENT),
            ],
            key_points=["요건1", "요건2"],
            required_citations=["민법 제750조"],
        )
        assert len(gt.source_documents) == 1
        assert len(gt.key_points) == 2
        assert "민법 제750조" in gt.required_citations

    def test_empty_ground_truth(self):
        gt = GroundTruth()
        assert gt.source_documents == []
        assert gt.key_points == []


class TestQueryMetadata:
    """QueryMetadata 모델 테스트"""

    def test_create_metadata(self):
        meta = QueryMetadata(
            category=Category.CIVIL,
            query_type=QueryType.CONCEPT_SEARCH,
            difficulty=Difficulty.MEDIUM,
        )
        assert meta.category == Category.CIVIL
        assert meta.query_type == QueryType.CONCEPT_SEARCH
        assert meta.difficulty == Difficulty.MEDIUM

    def test_with_subcategory(self):
        meta = QueryMetadata(
            category=Category.CIVIL,
            subcategory="임대차",
            query_type=QueryType.CONCEPT_SEARCH,
        )
        assert meta.subcategory == "임대차"

    def test_all_categories(self):
        """모든 카테고리 유효성"""
        for cat in Category:
            meta = QueryMetadata(
                category=cat,
                query_type=QueryType.SIMPLE_LOOKUP,
            )
            assert meta.category == cat

    def test_all_query_types(self):
        """모든 쿼리 유형 유효성"""
        for qt in QueryType:
            meta = QueryMetadata(
                category=Category.CIVIL,
                query_type=qt,
            )
            assert meta.query_type == qt


class TestEvalQuery:
    """EvalQuery 모델 테스트"""

    def test_create_full_query(self):
        query = EvalQuery(
            id="Q-001",
            question="임대차 보증금 반환 청구 요건은?",
            metadata=QueryMetadata(
                category=Category.CIVIL,
                query_type=QueryType.CONCEPT_SEARCH,
                difficulty=Difficulty.MEDIUM,
            ),
            ground_truth=GroundTruth(
                source_documents=[
                    SourceDocument(doc_id="76396", doc_type=DocumentType.PRECEDENT),
                ],
                key_points=["계약 종료", "동시이행"],
            ),
            source="manual",
        )

        assert query.id == "Q-001"
        assert "임대차" in query.question
        assert query.source == "manual"
        assert query.metadata.category == Category.CIVIL

    def test_create_minimal_query(self):
        query = EvalQuery(
            id="Q-002",
            question="테스트 질문",
            metadata=QueryMetadata(
                category=Category.OTHER,
                query_type=QueryType.SIMPLE_LOOKUP,
            ),
            ground_truth=GroundTruth(),
        )
        assert query.id == "Q-002"

    def test_created_at_default(self):
        query = EvalQuery(
            id="Q-003",
            question="테스트",
            metadata=QueryMetadata(
                category=Category.CIVIL,
                query_type=QueryType.SIMPLE_LOOKUP,
            ),
            ground_truth=GroundTruth(),
        )
        assert isinstance(query.created_at, datetime)


class TestEvalDataset:
    """EvalDataset 모델 테스트"""

    def test_create_dataset(self):
        dataset = EvalDataset(
            name="test_dataset",
            description="테스트 데이터셋",
        )
        assert dataset.name == "test_dataset"
        assert dataset.queries == []
        assert dataset.version == "1.0"

    def test_add_query(self):
        dataset = EvalDataset(name="test")
        query = EvalQuery(
            id="Q-001",
            question="테스트",
            metadata=QueryMetadata(
                category=Category.CIVIL,
                query_type=QueryType.SIMPLE_LOOKUP,
            ),
            ground_truth=GroundTruth(),
        )

        dataset.add_query(query)

        assert len(dataset.queries) == 1
        assert dataset.queries[0].id == "Q-001"

    def test_get_next_id(self):
        dataset = EvalDataset(name="test")

        # 첫 번째 ID
        assert dataset.get_next_id() == "Q-001"

        # 쿼리 추가 후
        dataset.queries.append(
            EvalQuery(
                id="Q-001",
                question="테스트",
                metadata=QueryMetadata(
                    category=Category.CIVIL,
                    query_type=QueryType.SIMPLE_LOOKUP,
                ),
                ground_truth=GroundTruth(),
            )
        )
        assert dataset.get_next_id() == "Q-002"


class TestSearchResult:
    """SearchResult 모델 테스트"""

    def test_create_result(self):
        result = SearchResult(
            doc_id="76396",
            chunk_id="76396_0",
            doc_type=DocumentType.PRECEDENT,
            title="손해배상청구사건",
            content="판결 내용...",
            score=0.85,
            rank=1,
        )

        assert result.doc_id == "76396"
        assert result.score == 0.85
        assert result.rank == 1


class TestMetricsResult:
    """MetricsResult 모델 테스트"""

    def test_create_metrics(self):
        metrics = MetricsResult(
            recall_at_5=0.72,
            recall_at_10=0.84,
            mrr=0.68,
            hit_rate=0.92,
            ndcg_at_10=0.78,
            latency_p50_ms=150.0,
            latency_p95_ms=450.0,
        )

        assert metrics.recall_at_5 == 0.72
        assert metrics.latency_p50_ms == 150.0


class TestExperimentConfig:
    """ExperimentConfig 모델 테스트"""

    def test_create_config(self):
        config = ExperimentConfig(
            experiment_id="EXP-20260129-001",
            name="baseline",
            dataset_path="datasets/eval_v1.json",
        )

        assert config.experiment_id == "EXP-20260129-001"
        assert config.embedding_model == "nlpai-lab/KURE-v1"  # 기본값
        assert config.top_k == 10  # 기본값

    def test_custom_config(self):
        config = ExperimentConfig(
            experiment_id="EXP-001",
            name="custom",
            dataset_path="test.json",
            embedding_model="custom-model",
            top_k=20,
            filters={"data_type": "판례"},
        )

        assert config.embedding_model == "custom-model"
        assert config.top_k == 20
        assert config.filters == {"data_type": "판례"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
