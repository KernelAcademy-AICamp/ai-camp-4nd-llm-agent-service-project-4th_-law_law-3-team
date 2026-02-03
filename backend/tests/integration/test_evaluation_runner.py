"""
평가 실행기 통합 테스트

LanceDB 연동 테스트 (실제 데이터 필요)
"""

import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from evaluation.tools.dataset_builder import DatasetBuilder
from evaluation.runners.experiment_tracker import ExperimentTracker
from evaluation.config import eval_settings, PerformanceTargets


class TestExperimentTracker:
    """ExperimentTracker 테스트"""

    def test_generate_next_id(self):
        """다음 실험 ID 생성"""
        exp_id = ExperimentTracker.generate_next_id()
        today = datetime.now().strftime("%Y%m%d")
        assert exp_id.startswith(f"EXP-{today}-")

    def test_create_tracker(self):
        """트래커 생성"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 임시 디렉토리 설정
            original_dir = eval_settings.experiments_dir
            eval_settings._evaluation_dir = Path(tmpdir)

            try:
                tracker = ExperimentTracker("EXP-TEST-001")
                assert tracker.experiment_id == "EXP-TEST-001"
                assert tracker.experiment_dir.exists()
            finally:
                eval_settings._evaluation_dir = original_dir.parent


class TestPerformanceTargets:
    """성능 목표 테스트"""

    def test_check_metrics_all_pass(self):
        """모든 목표 달성"""
        metrics = {
            "recall_at_5": 0.8,
            "recall_at_10": 0.9,
            "mrr": 0.8,
            "hit_rate": 0.95,
            "ndcg_at_10": 0.8,
            "latency_p50_ms": 100.0,
            "latency_p95_ms": 300.0,
        }

        results = PerformanceTargets.check_metrics(metrics)

        assert results["recall_at_5"] is True
        assert results["recall_at_10"] is True
        assert results["mrr"] is True
        assert results["hit_rate"] is True
        assert results["latency_p50"] is True

    def test_check_metrics_partial_fail(self):
        """일부 목표 미달"""
        metrics = {
            "recall_at_5": 0.5,  # 목표: 0.7
            "recall_at_10": 0.9,
            "mrr": 0.5,  # 목표: 0.7
            "hit_rate": 0.95,
            "ndcg_at_10": 0.8,
            "latency_p50_ms": 100.0,
            "latency_p95_ms": 600.0,  # 목표: < 500
        }

        results = PerformanceTargets.check_metrics(metrics)

        assert results["recall_at_5"] is False
        assert results["recall_at_10"] is True
        assert results["mrr"] is False
        assert results["latency_p95"] is False


class TestEvaluationWorkflow:
    """평가 워크플로우 통합 테스트"""

    @pytest.fixture
    def sample_dataset(self, tmp_path):
        """샘플 데이터셋 생성"""
        builder = DatasetBuilder(
            name="test_eval",
            description="테스트용 평가 데이터셋",
        )

        # 테스트 쿼리 추가 (실제 데이터 없어도 테스트 가능하도록)
        builder.add_query(
            question="손해배상 청구 요건은 무엇인가요?",
            source_documents=[
                {"doc_id": "test_doc_1", "doc_type": "precedent"},
            ],
            category="민사",
            query_type="개념검색",
            key_points=["불법행위", "손해", "인과관계"],
        )

        builder.add_query(
            question="민법 제750조의 내용은?",
            source_documents=[
                {"doc_id": "test_doc_2", "doc_type": "law"},
            ],
            category="민사",
            query_type="단순조회",
        )

        dataset_path = tmp_path / "test_dataset.json"
        builder.save(dataset_path)

        return dataset_path

    def test_dataset_creation_and_validation(self, sample_dataset):
        """데이터셋 생성 및 검증"""
        builder = DatasetBuilder.load(sample_dataset)

        assert builder.dataset.name == "test_eval"
        assert len(builder.dataset.queries) == 2

        # 통계 확인
        stats = builder.get_statistics()
        assert stats["total"] == 2
        assert stats["by_category"]["민사"] == 2

    def test_dataset_export(self, sample_dataset):
        """데이터셋 내보내기"""
        builder = DatasetBuilder.load(sample_dataset)
        exported = builder.export_for_evaluation()

        assert len(exported) == 2
        assert all("id" in e for e in exported)
        assert all("question" in e for e in exported)
        assert all("ground_truth_ids" in e for e in exported)


@pytest.mark.skipif(
    not Path("lancedb_data").exists(),
    reason="LanceDB 데이터 필요"
)
class TestEvaluationWithLanceDB:
    """LanceDB 연동 테스트 (실제 데이터 필요)"""

    def test_vector_search(self):
        """벡터 검색 테스트"""
        from app.tools.vectorstore.lancedb import LanceDBStore

        store = LanceDBStore()
        count = store.count()

        # 데이터가 있는지 확인
        assert count > 0, "LanceDB에 데이터가 없습니다"

    def test_embedding_model_load(self):
        """임베딩 모델 로드 테스트"""
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            "nlpai-lab/KURE-v1",
            trust_remote_code=True,
        )

        # 테스트 인코딩
        embedding = model.encode("테스트 문장")
        assert len(embedding) == 1024  # KURE-v1 차원


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
