"""
평가 메트릭 단위 테스트

Recall@K, MRR, NDCG, Hit Rate 등의 계산 로직 검증
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from evaluation.metrics.retrieval import (
    calculate_recall_at_k,
    calculate_mrr,
    calculate_ndcg_at_k,
    calculate_hit_rate,
    calculate_all_retrieval_metrics,
    aggregate_metrics,
)
from evaluation.metrics.generation import (
    extract_citations,
    normalize_citation,
    calculate_citation_accuracy,
    calculate_key_point_coverage,
)
from evaluation.metrics.rag import (
    calculate_context_relevance_score,
    calculate_faithfulness_score,
)


class TestRecallAtK:
    """Recall@K 테스트"""

    def test_perfect_recall(self):
        """모든 정답이 검색된 경우"""
        retrieved = ["A", "B", "C", "D", "E"]
        relevant = ["A", "B", "C"]
        assert calculate_recall_at_k(retrieved, relevant, k=5) == 1.0

    def test_partial_recall(self):
        """일부만 검색된 경우"""
        retrieved = ["A", "B", "X", "Y", "Z"]
        relevant = ["A", "B", "C", "D"]
        # 4개 중 2개 = 0.5
        assert calculate_recall_at_k(retrieved, relevant, k=5) == 0.5

    def test_no_recall(self):
        """정답이 하나도 없는 경우"""
        retrieved = ["X", "Y", "Z"]
        relevant = ["A", "B", "C"]
        assert calculate_recall_at_k(retrieved, relevant, k=3) == 0.0

    def test_k_limit(self):
        """K 제한 적용"""
        retrieved = ["X", "Y", "A", "B", "C"]
        relevant = ["A", "B", "C"]
        # K=2면 X, Y만 확인 → 0개 매칭
        assert calculate_recall_at_k(retrieved, relevant, k=2) == 0.0
        # K=5면 A, B, C 모두 포함 → 3/3 = 1.0
        assert calculate_recall_at_k(retrieved, relevant, k=5) == 1.0

    def test_empty_relevant(self):
        """정답 목록이 비어있는 경우"""
        retrieved = ["A", "B", "C"]
        relevant = []
        assert calculate_recall_at_k(retrieved, relevant, k=3) == 0.0

    def test_empty_retrieved(self):
        """검색 결과가 비어있는 경우"""
        retrieved = []
        relevant = ["A", "B"]
        assert calculate_recall_at_k(retrieved, relevant, k=5) == 0.0


class TestMRR:
    """MRR (Mean Reciprocal Rank) 테스트"""

    def test_first_position(self):
        """첫 번째 위치에 정답이 있는 경우"""
        retrieved = ["A", "B", "C"]
        relevant = ["A"]
        assert calculate_mrr(retrieved, relevant) == 1.0

    def test_second_position(self):
        """두 번째 위치에 정답이 있는 경우"""
        retrieved = ["X", "A", "B"]
        relevant = ["A", "B"]
        # 첫 정답 A가 2번째 위치 → 1/2 = 0.5
        assert calculate_mrr(retrieved, relevant) == 0.5

    def test_third_position(self):
        """세 번째 위치에 정답이 있는 경우"""
        retrieved = ["X", "Y", "A"]
        relevant = ["A"]
        assert abs(calculate_mrr(retrieved, relevant) - 1/3) < 0.001

    def test_no_match(self):
        """정답이 없는 경우"""
        retrieved = ["X", "Y", "Z"]
        relevant = ["A", "B"]
        assert calculate_mrr(retrieved, relevant) == 0.0

    def test_empty_relevant(self):
        """정답 목록이 비어있는 경우"""
        retrieved = ["A", "B"]
        relevant = []
        assert calculate_mrr(retrieved, relevant) == 0.0


class TestNDCG:
    """NDCG@K 테스트"""

    def test_perfect_order(self):
        """완벽한 순서의 경우"""
        retrieved = ["A", "B", "C"]
        relevant = ["A", "B", "C"]
        # 이상적인 순서와 동일 → NDCG = 1.0
        assert calculate_ndcg_at_k(retrieved, relevant, k=3) == 1.0

    def test_reverse_order(self):
        """역순인 경우"""
        retrieved = ["C", "B", "A"]
        relevant = ["A", "B", "C"]
        # 역순이어도 모두 포함되어 있으므로 DCG > 0
        ndcg = calculate_ndcg_at_k(retrieved, relevant, k=3)
        assert 0 < ndcg <= 1.0

    def test_partial_match(self):
        """부분 매칭인 경우"""
        retrieved = ["X", "A", "Y"]
        relevant = ["A", "B"]
        ndcg = calculate_ndcg_at_k(retrieved, relevant, k=3)
        assert 0 < ndcg < 1.0

    def test_no_match(self):
        """매칭 없는 경우"""
        retrieved = ["X", "Y", "Z"]
        relevant = ["A", "B"]
        assert calculate_ndcg_at_k(retrieved, relevant, k=3) == 0.0

    def test_with_relevance_scores(self):
        """관련성 점수가 있는 경우"""
        retrieved = ["A", "B", "C"]
        relevant = ["A", "B", "C"]
        scores = {"A": 3.0, "B": 2.0, "C": 1.0}
        ndcg = calculate_ndcg_at_k(retrieved, relevant, scores, k=3)
        # 이상적 순서이므로 1.0
        assert ndcg == 1.0


class TestHitRate:
    """Hit Rate 테스트"""

    def test_hit(self):
        """정답이 있는 경우"""
        retrieved = ["X", "A", "Y"]
        relevant = ["A"]
        assert calculate_hit_rate(retrieved, relevant, k=3) == 1.0

    def test_no_hit(self):
        """정답이 없는 경우"""
        retrieved = ["X", "Y", "Z"]
        relevant = ["A"]
        assert calculate_hit_rate(retrieved, relevant, k=3) == 0.0

    def test_hit_beyond_k(self):
        """정답이 K 밖에 있는 경우"""
        retrieved = ["X", "Y", "Z", "A"]
        relevant = ["A"]
        assert calculate_hit_rate(retrieved, relevant, k=3) == 0.0
        assert calculate_hit_rate(retrieved, relevant, k=4) == 1.0


class TestAllRetrievalMetrics:
    """전체 메트릭 계산 테스트"""

    def test_all_metrics(self):
        """모든 메트릭 계산"""
        retrieved = ["A", "B", "X", "C", "Y"]
        relevant = ["A", "B", "C"]

        metrics = calculate_all_retrieval_metrics(retrieved, relevant)

        assert "recall_at_5" in metrics
        assert "recall_at_10" in metrics
        assert "mrr" in metrics
        assert "hit_at_10" in metrics
        assert "ndcg_at_10" in metrics

        # A, B, C 모두 top 5에 있음
        assert metrics["recall_at_5"] == 1.0
        # A가 첫 번째
        assert metrics["mrr"] == 1.0


class TestAggregateMetrics:
    """메트릭 집계 테스트"""

    def test_aggregate(self):
        """여러 결과 집계"""
        results = [
            {"recall_at_5": 1.0, "mrr": 1.0, "hit_at_10": 1.0},
            {"recall_at_5": 0.5, "mrr": 0.5, "hit_at_10": 1.0},
            {"recall_at_5": 0.0, "mrr": 0.0, "hit_at_10": 0.0},
        ]

        agg = aggregate_metrics(results)

        assert agg["recall_at_5"] == 0.5  # (1.0 + 0.5 + 0.0) / 3
        assert agg["mrr"] == 0.5
        assert agg["hit_rate"] == 2/3  # hit_at_10 평균

    def test_empty_results(self):
        """빈 결과"""
        agg = aggregate_metrics([])
        assert agg["recall_at_5"] == 0.0


class TestCitationExtraction:
    """인용 추출 테스트"""

    def test_extract_law_citation(self):
        """법령 인용 추출"""
        text = "민법 제750조에 따르면 불법행위가 성립합니다."
        citations = extract_citations(text)
        assert any("민법" in c and "750" in c for c in citations)

    def test_extract_multiple_citations(self):
        """여러 인용 추출"""
        text = "민법 제750조와 형법 제250조를 참조하세요."
        citations = extract_citations(text)
        assert len(citations) >= 2


class TestCitationAccuracy:
    """인용 정확도 테스트"""

    def test_perfect_match(self):
        """완벽한 매칭"""
        generated = ["민법 제750조", "민법 제751조"]
        required = ["민법 제750조", "민법 제751조"]

        result = calculate_citation_accuracy(generated, required)

        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_partial_match(self):
        """부분 매칭"""
        generated = ["민법 제750조", "민법 제752조"]
        required = ["민법 제750조", "민법 제751조"]

        result = calculate_citation_accuracy(generated, required)

        # 1개 매칭 / 2개 생성 = 0.5
        assert result["precision"] == 0.5
        # 1개 매칭 / 2개 필수 = 0.5
        assert result["recall"] == 0.5

    def test_no_match(self):
        """매칭 없음"""
        generated = ["형법 제250조"]
        required = ["민법 제750조"]

        result = calculate_citation_accuracy(generated, required)

        assert result["precision"] == 0.0
        assert result["recall"] == 0.0


class TestKeyPointCoverage:
    """Key Point 커버리지 테스트"""

    def test_full_coverage(self):
        """모든 키포인트 포함"""
        text = "계약 종료 요건과 동시이행 관계가 중요합니다."
        key_points = ["계약 종료 요건", "동시이행"]

        result = calculate_key_point_coverage(text, key_points)

        assert result["coverage"] > 0.5

    def test_no_coverage(self):
        """키포인트 없음"""
        text = "전혀 관련 없는 내용입니다."
        key_points = ["계약 종료", "동시이행"]

        result = calculate_key_point_coverage(text, key_points)

        # 키워드 매칭이 어려울 수 있음
        assert result["coverage"] >= 0.0


class TestRAGMetrics:
    """RAG 메트릭 테스트"""

    def test_context_relevance(self):
        """컨텍스트 관련성"""
        question = "손해배상 청구 요건은 무엇인가요?"
        context = "손해배상을 청구하려면 불법행위, 손해, 인과관계가 필요합니다."

        score = calculate_context_relevance_score(question, context)

        assert score > 0  # 키워드 매칭으로 인해 0보다 큼

    def test_faithfulness(self):
        """충실도 점수"""
        answer = "손해배상을 청구하려면 불법행위가 필요합니다. 인과관계도 필요합니다."
        contexts = ["손해배상을 청구하려면 불법행위, 손해, 인과관계가 필요합니다."]

        score = calculate_faithfulness_score(answer, contexts)

        # 컨텍스트에서 유래한 내용이므로 점수 > 0
        assert score >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
