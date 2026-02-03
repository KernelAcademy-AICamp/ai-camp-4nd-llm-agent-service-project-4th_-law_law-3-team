"""
평가 실행기

데이터셋을 로드하고 각 쿼리에 대해 검색을 수행한 후
메트릭을 계산하여 결과를 저장

Usage:
    runner = EvaluationRunner(
        experiment_id="EXP-20260129-001",
        dataset_path="evaluation/datasets/eval_dataset_v1.json",
    )
    result = await runner.run()
    runner.save_result(result)
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from evaluation.schemas import (
    EvalDataset,
    ExperimentConfig,
    ExperimentResult,
    MetricsResult,
    QueryResult,
    RetrievalResult,
    SearchResult as SearchResultSchema,
    DocumentType,
)
from evaluation.config import eval_settings, PerformanceTargets
from evaluation.tools.dataset_builder import DatasetBuilder
from evaluation.metrics.retrieval import (
    calculate_all_retrieval_metrics,
    aggregate_metrics,
)
from evaluation.runners.experiment_tracker import ExperimentTracker


class EvaluationRunner:
    """
    RAG 평가 실행기

    주요 기능:
    1. 데이터셋 로드
    2. 각 쿼리에 대해 벡터 검색 수행
    3. Ground Truth와 비교하여 메트릭 계산
    4. 결과 저장 및 분석
    """

    def __init__(
        self,
        experiment_id: str,
        dataset_path: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[dict] = None,
    ):
        """
        Args:
            experiment_id: 실험 ID (EXP-YYYYMMDD-NNN 형식)
            dataset_path: 평가 데이터셋 경로
            name: 실험 이름
            description: 실험 설명
            top_k: 검색 결과 수
            filters: 검색 필터
        """
        self.config = ExperimentConfig(
            experiment_id=experiment_id,
            name=name or experiment_id,
            description=description,
            dataset_path=dataset_path,
            embedding_model=eval_settings.embedding_model,
            distance_metric=eval_settings.distance_metric,
            top_k=top_k,
            filters=filters,
        )

        self.dataset: Optional[EvalDataset] = None
        self.tracker = ExperimentTracker(experiment_id)
        self._embedding_model = None
        self._vector_store = None

    def _load_dataset(self) -> EvalDataset:
        """데이터셋 로드"""
        path = Path(self.config.dataset_path)
        if not path.is_absolute():
            path = Path(__file__).parent.parent / self.config.dataset_path

        builder = DatasetBuilder.load(path)
        return builder.dataset

    def _get_embedding_model(self):
        """임베딩 모델 로드 (lazy)"""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(
                self.config.embedding_model,
                trust_remote_code=True,
            )
        return self._embedding_model

    def _get_vector_store(self):
        """벡터 스토어 연결 (lazy)"""
        if self._vector_store is None:
            from app.tools.vectorstore.lancedb import LanceDBStore
            self._vector_store = LanceDBStore()
        return self._vector_store

    def _encode_query(self, query: str) -> list[float]:
        """쿼리 임베딩 생성"""
        model = self._get_embedding_model()
        embedding = model.encode(query, normalize_embeddings=True)
        return embedding.tolist()

    def _search(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: Optional[dict] = None,
    ) -> tuple[list[dict], float]:
        """
        벡터 검색 수행

        Returns:
            (검색 결과 목록, latency_ms)
        """
        store = self._get_vector_store()

        start_time = time.perf_counter()

        result = store.search(
            query_embedding=query_embedding,
            n_results=top_k,
            where=filters,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        search_results = []
        if result.ids and result.ids[0]:
            for i, doc_id in enumerate(result.ids[0]):
                meta = result.metadatas[0][i] if result.metadatas else {}
                content = result.documents[0][i] if result.documents else ""
                distance = result.distances[0][i] if result.distances else 0.0

                search_results.append({
                    "doc_id": meta.get("source_id", doc_id),
                    "chunk_id": doc_id,
                    "doc_type": meta.get("data_type", ""),
                    "title": meta.get("title", ""),
                    "content": content,
                    "score": 1.0 - distance,
                    "rank": i + 1,
                })

        return search_results, latency_ms

    async def run(self) -> ExperimentResult:
        """
        평가 실행

        Returns:
            ExperimentResult
        """
        self.dataset = self._load_dataset()
        started_at = datetime.now()

        self.tracker.log_start(self.config)

        all_query_results: list[QueryResult] = []
        all_metrics: list[dict[str, float]] = []
        all_latencies: list[float] = []
        failed_queries: list[dict] = []

        metrics_by_type: dict[str, list[dict]] = {}
        metrics_by_category: dict[str, list[dict]] = {}

        for query in self.dataset.queries:
            try:
                query_embedding = self._encode_query(query.question)

                results, latency_ms = self._search(
                    query_embedding,
                    self.config.top_k,
                    self.config.filters,
                )
                all_latencies.append(latency_ms)

                retrieved_ids = [r["doc_id"] for r in results]
                ground_truth_ids = [
                    doc.doc_id for doc in query.ground_truth.source_documents
                ]

                metrics = calculate_all_retrieval_metrics(
                    retrieved_ids, ground_truth_ids
                )
                all_metrics.append(metrics)

                search_results = [
                    SearchResultSchema(
                        doc_id=r["doc_id"],
                        chunk_id=r["chunk_id"],
                        doc_type=DocumentType(r["doc_type"])
                        if r["doc_type"] in ("precedent", "law", "판례", "법령")
                        else DocumentType.PRECEDENT,
                        title=r["title"],
                        content=r["content"],
                        score=r["score"],
                        rank=r["rank"],
                    )
                    for r in results
                ]

                retrieval_result = RetrievalResult(
                    query=query.question,
                    results=search_results,
                    latency_ms=latency_ms,
                )

                query_result = QueryResult(
                    query_id=query.id,
                    question=query.question,
                    retrieval=retrieval_result,
                    recall_at_5=metrics["recall_at_5"],
                    recall_at_10=metrics["recall_at_10"],
                    mrr=metrics["mrr"],
                    hit=metrics["hit_at_10"] > 0,
                    ndcg_at_10=metrics["ndcg_at_10"],
                )
                all_query_results.append(query_result)

                query_type = query.metadata.query_type.value
                if query_type not in metrics_by_type:
                    metrics_by_type[query_type] = []
                metrics_by_type[query_type].append(metrics)

                category = query.metadata.category.value
                if category not in metrics_by_category:
                    metrics_by_category[category] = []
                metrics_by_category[category].append(metrics)

                self.tracker.log_query_result(query.id, metrics)

            except Exception as e:
                failed_queries.append({
                    "query_id": query.id,
                    "question": query.question,
                    "error": str(e),
                })
                self.tracker.log_error(query.id, str(e))

        agg_metrics = aggregate_metrics(all_metrics)

        latency_array = np.array(all_latencies) if all_latencies else np.array([0])
        latency_p50 = float(np.percentile(latency_array, 50))
        latency_p95 = float(np.percentile(latency_array, 95))

        overall_metrics = MetricsResult(
            recall_at_5=agg_metrics["recall_at_5"],
            recall_at_10=agg_metrics["recall_at_10"],
            mrr=agg_metrics["mrr"],
            hit_rate=agg_metrics["hit_rate"],
            ndcg_at_10=agg_metrics["ndcg_at_10"],
            latency_p50_ms=latency_p50,
            latency_p95_ms=latency_p95,
        )

        type_metrics = {}
        for qtype, type_results in metrics_by_type.items():
            agg = aggregate_metrics(type_results)
            type_metrics[qtype] = MetricsResult(
                recall_at_5=agg["recall_at_5"],
                recall_at_10=agg["recall_at_10"],
                mrr=agg["mrr"],
                hit_rate=agg["hit_rate"],
                ndcg_at_10=agg["ndcg_at_10"],
                latency_p50_ms=0,
                latency_p95_ms=0,
            )

        cat_metrics = {}
        for cat, cat_results in metrics_by_category.items():
            agg = aggregate_metrics(cat_results)
            cat_metrics[cat] = MetricsResult(
                recall_at_5=agg["recall_at_5"],
                recall_at_10=agg["recall_at_10"],
                mrr=agg["mrr"],
                hit_rate=agg["hit_rate"],
                ndcg_at_10=agg["ndcg_at_10"],
                latency_p50_ms=0,
                latency_p95_ms=0,
            )

        result = ExperimentResult(
            config=self.config,
            metrics=overall_metrics,
            metrics_by_type=type_metrics,
            metrics_by_category=cat_metrics,
            query_results=all_query_results,
            failed_queries=failed_queries,
            started_at=started_at,
            completed_at=datetime.now(),
        )

        self.tracker.log_completion(result)

        return result

    def save_result(self, result: ExperimentResult) -> Path:
        """결과 저장"""
        return self.tracker.save_result(result)

    def check_targets(self, result: ExperimentResult) -> dict[str, bool]:
        """성능 목표 달성 여부 확인"""
        metrics_dict = {
            "recall_at_5": result.metrics.recall_at_5,
            "recall_at_10": result.metrics.recall_at_10,
            "mrr": result.metrics.mrr,
            "hit_rate": result.metrics.hit_rate,
            "ndcg_at_10": result.metrics.ndcg_at_10,
            "latency_p50_ms": result.metrics.latency_p50_ms,
            "latency_p95_ms": result.metrics.latency_p95_ms,
        }
        return PerformanceTargets.check_metrics(metrics_dict)


async def main():
    """CLI 실행"""
    import argparse
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    parser = argparse.ArgumentParser(description="RAG 평가 실행")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="평가 데이터셋 경로",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="실험 ID (기본: 자동 생성)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="검색 결과 수",
    )

    args = parser.parse_args()

    if args.experiment_id is None:
        from datetime import datetime
        today = datetime.now().strftime("%Y%m%d")
        args.experiment_id = f"EXP-{today}-001"

    runner = EvaluationRunner(
        experiment_id=args.experiment_id,
        dataset_path=args.dataset,
        top_k=args.top_k,
    )

    print(f"평가 시작: {args.experiment_id}")
    result = await runner.run()

    print("\n=== 평가 결과 ===")
    print(f"Recall@5:  {result.metrics.recall_at_5:.4f}")
    print(f"Recall@10: {result.metrics.recall_at_10:.4f}")
    print(f"MRR:       {result.metrics.mrr:.4f}")
    print(f"Hit Rate:  {result.metrics.hit_rate:.4f}")
    print(f"NDCG@10:   {result.metrics.ndcg_at_10:.4f}")
    print(f"Latency P50: {result.metrics.latency_p50_ms:.2f}ms")
    print(f"Latency P95: {result.metrics.latency_p95_ms:.2f}ms")

    targets = runner.check_targets(result)
    print("\n=== 목표 달성 여부 ===")
    for metric, achieved in targets.items():
        status = "✓" if achieved else "✗"
        print(f"  {status} {metric}")

    path = runner.save_result(result)
    print(f"\n결과 저장됨: {path}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
