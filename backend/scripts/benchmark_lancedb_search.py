"""
LanceDB 벡터 인덱스 타입별 검색 속도 벤치마크

Brute-force / IVF_PQ / IVF_FLAT / IVF_HNSW_SQ 를 비교합니다.
각 인덱스를 replace=True 로 교체하면서 순차 측정합니다.

Usage:
    cd backend
    uv run python scripts/benchmark_lancedb_search.py
"""

import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 프로젝트 루트 설정
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import lancedb  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.services.rag.embedding import create_query_embedding  # noqa: E402

# ─────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────

QUERIES = [
    "손해배상 청구 요건과 입증 책임",
    "임대차 계약 해지 사유",
    "교통사고 과실 비율 산정 기준",
    "명예훼손 성립 요건",
    "상속 포기 절차와 기한",
]

N_RESULTS = 10
WARMUP_RUNS = 2
MEASURE_RUNS = 10


# ─────────────────────────────────────────────────────────────
# 인덱스 타입 정의
# ─────────────────────────────────────────────────────────────

INDEX_CONFIGS: List[Dict[str, Any]] = [
    {
        "label": "IVF_PQ",
        "params": {
            "metric": "cosine",
            "index_type": "IVF_PQ",
            "num_sub_vectors": 32,
            "replace": True,
        },
    },
    {
        "label": "IVF_FLAT",
        "params": {
            "metric": "cosine",
            "index_type": "IVF_FLAT",
            "replace": True,
        },
    },
    {
        "label": "IVF_HNSW_SQ",
        "params": {
            "metric": "cosine",
            "index_type": "IVF_HNSW_SQ",
            "replace": True,
        },
    },
]


def embed_queries(queries: List[str]) -> Tuple[List[List[float]], float]:
    """쿼리를 임베딩 벡터로 변환하고 총 소요 시간을 반환한다."""
    embeddings: List[List[float]] = []
    start = time.perf_counter()
    for q in queries:
        embeddings.append(create_query_embedding(q))
    elapsed = time.perf_counter() - start
    return embeddings, elapsed


def run_search(
    table: lancedb.table.Table,
    embedding: List[float],
    n_results: int,
) -> Tuple[float, List[str]]:
    """단일 벡터 검색을 실행하고 (소요시간, 상위 ID 리스트)를 반환한다."""
    start = time.perf_counter()
    df = (
        table.search(embedding)
        .metric("cosine")
        .limit(n_results)
        .to_pandas()
    )
    elapsed = time.perf_counter() - start
    ids = df["id"].tolist() if not df.empty else []
    return elapsed, ids


def benchmark_searches(
    table: lancedb.table.Table,
    embeddings: List[List[float]],
    queries: List[str],
    n_results: int,
    label: str,
) -> Dict[str, Any]:
    """Warm-up 후 반복 측정하여 통계를 반환한다."""
    all_times: List[float] = []
    top_ids_per_query: Dict[str, List[str]] = {}

    for emb, query in zip(embeddings, queries):
        # Warm-up
        for _ in range(WARMUP_RUNS):
            run_search(table, emb, n_results)

        # 측정
        for _ in range(MEASURE_RUNS):
            elapsed, ids = run_search(table, emb, n_results)
            all_times.append(elapsed)

        # 마지막 실행의 상위 ID 저장
        top_ids_per_query[query] = ids

    return {
        "label": label,
        "n_results": n_results,
        "mean_ms": statistics.mean(all_times) * 1000,
        "median_ms": statistics.median(all_times) * 1000,
        "p95_ms": sorted(all_times)[int(len(all_times) * 0.95)] * 1000,
        "min_ms": min(all_times) * 1000,
        "max_ms": max(all_times) * 1000,
        "top_ids": top_ids_per_query,
    }


def create_index(
    table: lancedb.table.Table,
    config: Dict[str, Any],
    row_count: int,
) -> float:
    """인덱스를 생성하고 소요 시간(초)을 반환한다."""
    params = dict(config["params"])
    # num_partitions 자동 설정 (sqrt(N), 최소 16)
    if "num_partitions" not in params:
        params["num_partitions"] = max(16, int(row_count**0.5))
    params["vector_column_name"] = "vector"

    start = time.perf_counter()
    table.create_index(**params)
    elapsed = time.perf_counter() - start
    return elapsed


def compute_recall(
    baseline_ids: Dict[str, List[str]],
    target_ids: Dict[str, List[str]],
) -> Tuple[float, Dict[str, str]]:
    """baseline 대비 recall (상위 결과 일치율)을 계산한다.

    Returns:
        (평균 recall, 쿼리별 상세 dict)
    """
    recalls: List[float] = []
    details: Dict[str, str] = {}
    for query, b_ids in baseline_ids.items():
        t_ids = target_ids.get(query, [])
        b_set = set(b_ids)
        t_set = set(t_ids)
        overlap = len(b_set & t_set)
        total = len(b_set) if b_set else 1
        recall = overlap / total
        recalls.append(recall)
        details[query] = f"{overlap}/{total} ({recall:.0%})"
    avg_recall = statistics.mean(recalls) if recalls else 0.0
    return avg_recall, details


def print_result_row(
    stats: Dict[str, Any],
    build_time: Optional[float],
    recall: Optional[float],
) -> None:
    """결과 한 줄을 출력한다."""
    build_str = f"{build_time:6.1f}s" if build_time is not None else "    - "
    recall_str = f"{recall:5.0%}" if recall is not None else "  100%"
    print(
        f"  {stats['label']:<16}"
        f"  {stats['mean_ms']:8.2f}"
        f"  {stats['median_ms']:8.2f}"
        f"  {stats['p95_ms']:8.2f}"
        f"  {stats['min_ms']:8.2f}"
        f"  {stats['max_ms']:8.2f}"
        f"  {build_str}"
        f"  {recall_str}"
    )


def main() -> None:
    """벤치마크를 실행한다."""
    db_path = str(Path(settings.LANCEDB_URI))
    table_name = settings.LANCEDB_TABLE_NAME

    print("=" * 65)
    print("LanceDB Vector Index Benchmark (Multi-Type)")
    print("=" * 65)

    # DB 연결
    db = lancedb.connect(db_path)
    if table_name not in db.table_names():
        print(f"[ERROR] 테이블 '{table_name}'이 존재하지 않습니다.")
        sys.exit(1)

    table = db.open_table(table_name)
    row_count = table.count_rows()
    print(f"\n  DB 경로     : {db_path}")
    print(f"  테이블      : {table_name}")
    print(f"  레코드 수   : {row_count:,}")
    print(f"  쿼리 수     : {len(QUERIES)}")
    print(f"  n_results   : {N_RESULTS}")
    print(f"  측정 반복   : {MEASURE_RUNS} (warm-up {WARMUP_RUNS})")

    # 기존 인덱스 확인
    indices = table.list_indices()
    print(f"  기존 인덱스 : {indices if indices else '없음'}")

    # ─────────────────────────────────────────────────────────
    # Step 1: 임베딩 생성
    # ─────────────────────────────────────────────────────────
    print("\n" + "-" * 65)
    print("Step 1: 쿼리 임베딩 생성")
    print("-" * 65)

    embeddings, embed_time = embed_queries(QUERIES)
    print(f"  임베딩 생성 시간: {embed_time:.2f}s ({len(QUERIES)}개 쿼리)")
    print(f"  벡터 차원: {len(embeddings[0])}")

    # ─────────────────────────────────────────────────────────
    # Step 2: 현재 상태에서 측정 (인덱스 있으면 인덱스 적용 상태)
    # ─────────────────────────────────────────────────────────
    # 이전 벤치마크의 brute-force 참조값 (인덱스 없는 상태에서 측정)
    print("\n" + "-" * 65)
    print("Step 2: 이전 Brute-force 참조값 (인덱스 없는 상태에서 측정)")
    print("-" * 65)
    print("  n=10  Mean=90.86ms  Median=89.06ms  (이전 세션 측정값)")

    # ─────────────────────────────────────────────────────────
    # Step 3: 인덱스 타입별 벤치마크
    # ─────────────────────────────────────────────────────────
    all_results: List[Dict[str, Any]] = []
    all_build_times: List[Optional[float]] = []
    all_recalls: List[Optional[float]] = []

    # 첫 번째 인덱스 결과를 baseline으로 사용하기 위해 순회
    baseline_ids: Optional[Dict[str, List[str]]] = None

    for i, config in enumerate(INDEX_CONFIGS):
        label = config["label"]
        step_num = i + 3
        print(f"\n{'─' * 65}")
        print(f"Step {step_num}: {label} 인덱스")
        print("─" * 65)

        # 인덱스 생성
        print("  생성 중...")
        build_time = create_index(table, config, row_count)
        print(f"  생성 시간: {build_time:.1f}s")
        print(f"  인덱스: {table.list_indices()}")

        # 벤치마크
        stats = benchmark_searches(
            table, embeddings, QUERIES, N_RESULTS, label,
        )
        print(f"  Mean: {stats['mean_ms']:.2f}ms  Median: {stats['median_ms']:.2f}ms")

        # baseline 설정 (IVF_FLAT을 recall baseline으로 사용)
        if label == "IVF_FLAT":
            baseline_ids = stats["top_ids"]

        all_results.append(stats)
        all_build_times.append(build_time)

    # Recall 계산 (IVF_FLAT을 baseline으로)
    if baseline_ids is not None:
        for stats in all_results:
            if stats["label"] == "IVF_FLAT":
                all_recalls.append(1.0)
            else:
                avg_recall, _ = compute_recall(baseline_ids, stats["top_ids"])
                all_recalls.append(avg_recall)
    else:
        all_recalls = [None] * len(all_results)

    # ─────────────────────────────────────────────────────────
    # 최종 비교 테이블
    # ─────────────────────────────────────────────────────────
    print("\n" + "=" * 95)
    print(f"비교 결과 (n_results={N_RESULTS}, {row_count:,} 청크)")
    print("=" * 95)
    print(
        f"  {'Index Type':<16}"
        f"  {'Mean':>8}"
        f"  {'Median':>8}"
        f"  {'P95':>8}"
        f"  {'Min':>8}"
        f"  {'Max':>8}"
        f"  {'Build':>7}"
        f"  {'Recall':>6}"
    )
    print(
        f"  {'':─<16}"
        f"  {'':─>8}"
        f"  {'':─>8}"
        f"  {'':─>8}"
        f"  {'':─>8}"
        f"  {'':─>8}"
        f"  {'':─>7}"
        f"  {'':─>6}"
    )

    # Brute-force 참조 행
    print(
        f"  {'Brute-force':<16}"
        f"  {'90.86':>8}"
        f"  {'89.06':>8}"
        f"  {'114.20':>8}"
        f"  {'78.45':>8}"
        f"  {'125.03':>8}"
        f"  {'    - ':>7}"
        f"  {'  100%':>6}"
    )

    for stats, build_time, recall in zip(
        all_results, all_build_times, all_recalls,
    ):
        print_result_row(stats, build_time, recall)

    print("─" * 95)
    print("  * 단위: ms (밀리초), Recall = IVF_FLAT 대비 상위 결과 일치율")
    print("  * Brute-force 값은 이전 벤치마크 참조값 (인덱스 없는 상태)")

    # ─────────────────────────────────────────────────────────
    # 쿼리별 Recall 상세
    # ─────────────────────────────────────────────────────────
    if baseline_ids is not None:
        print(f"\n{'=' * 95}")
        print("쿼리별 Recall 상세 (IVF_FLAT 대비)")
        print("=" * 95)
        for stats in all_results:
            if stats["label"] == "IVF_FLAT":
                continue
            _, details = compute_recall(baseline_ids, stats["top_ids"])
            print(f"\n  [{stats['label']}]")
            for query, detail in details.items():
                print(f"    {query[:30]:<32} {detail}")

    print(f"\n{'=' * 95}")
    print("Done (마지막 인덱스 IVF_HNSW_SQ 유지)")
    print("=" * 95)


if __name__ == "__main__":
    main()
