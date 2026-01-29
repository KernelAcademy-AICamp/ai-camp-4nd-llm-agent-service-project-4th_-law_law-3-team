"""
RAG 평가 메트릭 모듈

- retrieval: Recall@K, MRR, NDCG, Hit Rate
- generation: Citation Accuracy
- rag: Faithfulness, Relevance
"""

from evaluation.metrics.retrieval import (
    calculate_recall_at_k,
    calculate_mrr,
    calculate_ndcg_at_k,
    calculate_hit_rate,
    calculate_all_retrieval_metrics,
)

__all__ = [
    "calculate_recall_at_k",
    "calculate_mrr",
    "calculate_ndcg_at_k",
    "calculate_hit_rate",
    "calculate_all_retrieval_metrics",
]
