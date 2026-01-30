"""
데이터셋 빌더 단위 테스트

DatasetBuilder 클래스의 기능 검증
"""

import sys
import json
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from evaluation.tools.dataset_builder import DatasetBuilder
from evaluation.schemas import (
    EvalQuery,
    GroundTruth,
    SourceDocument,
    QueryMetadata,
    DocumentType,
    QueryType,
    Category,
    Difficulty,
)


class TestDatasetBuilder:
    """DatasetBuilder 테스트"""

    def test_create_builder(self):
        """빌더 생성"""
        builder = DatasetBuilder(
            name="test_dataset",
            description="테스트용 데이터셋",
        )

        assert builder.dataset.name == "test_dataset"
        assert builder.dataset.description == "테스트용 데이터셋"
        assert len(builder.dataset.queries) == 0

    def test_add_query(self):
        """쿼리 추가"""
        builder = DatasetBuilder(name="test")

        query = builder.add_query(
            question="임대차 보증금 반환 청구 요건은?",
            source_documents=[
                {"doc_id": "76396", "doc_type": "precedent"},
                {"doc_id": "010719", "doc_type": "law", "article": "제621조"},
            ],
            category="민사",
            query_type="개념검색",
            difficulty="medium",
            key_points=["계약 종료", "동시이행"],
            required_citations=["민법 제621조"],
        )

        assert query.id == "Q-001"
        assert len(builder.dataset.queries) == 1
        assert len(query.ground_truth.source_documents) == 2
        assert query.metadata.category == Category.CIVIL
        assert query.metadata.query_type == QueryType.CONCEPT_SEARCH

    def test_add_multiple_queries(self):
        """여러 쿼리 추가"""
        builder = DatasetBuilder(name="test")

        builder.add_query(
            question="질문 1",
            source_documents=[{"doc_id": "1", "doc_type": "precedent"}],
            category="민사",
            query_type="단순조회",
        )
        builder.add_query(
            question="질문 2",
            source_documents=[{"doc_id": "2", "doc_type": "law"}],
            category="형사",
            query_type="개념검색",
        )

        assert len(builder.dataset.queries) == 2
        assert builder.dataset.queries[0].id == "Q-001"
        assert builder.dataset.queries[1].id == "Q-002"

    def test_get_query_by_id(self):
        """ID로 쿼리 조회"""
        builder = DatasetBuilder(name="test")
        builder.add_query(
            question="테스트 질문",
            source_documents=[{"doc_id": "1", "doc_type": "precedent"}],
            category="민사",
            query_type="단순조회",
        )

        query = builder.get_query_by_id("Q-001")
        assert query is not None
        assert query.question == "테스트 질문"

        # 없는 ID
        assert builder.get_query_by_id("Q-999") is None

    def test_delete_query(self):
        """쿼리 삭제"""
        builder = DatasetBuilder(name="test")
        builder.add_query(
            question="삭제될 질문",
            source_documents=[{"doc_id": "1", "doc_type": "precedent"}],
            category="민사",
            query_type="단순조회",
        )

        assert len(builder.dataset.queries) == 1

        result = builder.delete_query("Q-001")
        assert result is True
        assert len(builder.dataset.queries) == 0

        # 이미 삭제된 쿼리
        result = builder.delete_query("Q-001")
        assert result is False

    def test_get_statistics(self):
        """통계 조회"""
        builder = DatasetBuilder(name="test")

        # 빈 데이터셋
        stats = builder.get_statistics()
        assert stats["total"] == 0

        # 쿼리 추가
        builder.add_query(
            question="민사 질문",
            source_documents=[{"doc_id": "1", "doc_type": "precedent"}],
            category="민사",
            query_type="개념검색",
            difficulty="easy",
        )
        builder.add_query(
            question="형사 질문",
            source_documents=[{"doc_id": "2", "doc_type": "law"}],
            category="형사",
            query_type="단순조회",
            difficulty="hard",
        )

        stats = builder.get_statistics()

        assert stats["total"] == 2
        assert stats["by_category"]["민사"] == 1
        assert stats["by_category"]["형사"] == 1
        assert stats["by_type"]["개념검색"] == 1
        assert stats["by_type"]["단순조회"] == 1

    def test_validate_empty_question(self):
        """빈 질문 검증"""
        builder = DatasetBuilder(name="test")

        # 빈 질문 추가 시도
        builder.add_query(
            question="",
            source_documents=[{"doc_id": "1", "doc_type": "precedent"}],
            category="민사",
            query_type="단순조회",
        )

        errors = builder.validate()
        assert any("비어있습니다" in e for e in errors)

    def test_validate_no_ground_truth(self):
        """Ground Truth 없는 경우 검증"""
        builder = DatasetBuilder(name="test")
        builder.add_query(
            question="질문",
            source_documents=[],  # 빈 목록
            category="민사",
            query_type="단순조회",
        )

        errors = builder.validate()
        assert any("Ground Truth" in e for e in errors)

    def test_save_and_load_json(self):
        """JSON 저장 및 로드"""
        builder = DatasetBuilder(name="test_save")
        builder.add_query(
            question="저장 테스트 질문",
            source_documents=[{"doc_id": "123", "doc_type": "precedent"}],
            category="민사",
            query_type="개념검색",
            key_points=["키포인트1"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_dataset.json"

            # 저장
            result_path = builder.save(save_path, format="json")
            assert result_path.exists()

            # JSON 내용 확인
            with open(result_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert data["name"] == "test_save"
            assert len(data["queries"]) == 1

            # 로드
            loaded_builder = DatasetBuilder.load(result_path)
            assert loaded_builder.dataset.name == "test_save"
            assert len(loaded_builder.dataset.queries) == 1
            assert loaded_builder.dataset.queries[0].question == "저장 테스트 질문"

    def test_save_and_load_yaml(self):
        """YAML 저장 및 로드"""
        builder = DatasetBuilder(name="test_yaml")
        builder.add_query(
            question="YAML 테스트",
            source_documents=[{"doc_id": "456", "doc_type": "law"}],
            category="형사",
            query_type="단순조회",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test.yaml"

            # 저장
            result_path = builder.save(save_path, format="yaml")
            assert result_path.exists()

            # 로드
            loaded_builder = DatasetBuilder.load(result_path)
            assert loaded_builder.dataset.name == "test_yaml"

    def test_export_for_evaluation(self):
        """평가용 데이터 내보내기"""
        builder = DatasetBuilder(name="test")
        builder.add_query(
            question="질문 1",
            source_documents=[
                {"doc_id": "A", "doc_type": "precedent"},
                {"doc_id": "B", "doc_type": "law"},
            ],
            category="민사",
            query_type="개념검색",
            difficulty="medium",
        )

        exported = builder.export_for_evaluation()

        assert len(exported) == 1
        assert exported[0]["id"] == "Q-001"
        assert exported[0]["question"] == "질문 1"
        assert exported[0]["ground_truth_ids"] == ["A", "B"]
        assert exported[0]["category"] == "민사"

    def test_add_query_from_eval_query(self):
        """EvalQuery 객체 직접 추가"""
        builder = DatasetBuilder(name="test")

        query = EvalQuery(
            id="TEMP-001",  # 이 ID는 무시됨
            question="직접 추가된 질문",
            metadata=QueryMetadata(
                category=Category.ADMINISTRATIVE,
                query_type=QueryType.COMPARISON,
                difficulty=Difficulty.HARD,
            ),
            ground_truth=GroundTruth(
                source_documents=[
                    SourceDocument(doc_id="X", doc_type=DocumentType.LAW),
                ],
            ),
            source="solar",
        )

        builder.add_query_from_eval_query(query)

        assert len(builder.dataset.queries) == 1
        # ID가 새로 할당됨
        assert builder.dataset.queries[0].id == "Q-001"
        assert builder.dataset.queries[0].source == "solar"


class TestDatasetBuilderEdgeCases:
    """DatasetBuilder 엣지 케이스 테스트"""

    def test_unicode_content(self):
        """유니코드 내용 처리"""
        builder = DatasetBuilder(name="한글_테스트")
        builder.add_query(
            question="한글 질문: 민법 제750조란?",
            source_documents=[{"doc_id": "법령1", "doc_type": "law"}],
            category="민사",
            query_type="단순조회",
            key_points=["불법행위", "손해배상"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "unicode_test.json"
            builder.save(save_path)

            loaded = DatasetBuilder.load(save_path)
            assert "한글" in loaded.dataset.queries[0].question
            assert "불법행위" in loaded.dataset.queries[0].ground_truth.key_points

    def test_long_content(self):
        """긴 내용 처리"""
        builder = DatasetBuilder(name="test")
        long_question = "A" * 1000
        long_key_point = "B" * 500

        builder.add_query(
            question=long_question,
            source_documents=[{"doc_id": "1", "doc_type": "precedent"}],
            category="민사",
            query_type="복합검색",
            key_points=[long_key_point],
        )

        assert len(builder.dataset.queries[0].question) == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
