"""
RAG 파이프라인 단계별 타이밍 벤치마크

각 단계의 소요 시간과 결과 건수를 측정합니다.
사용법: cd backend && uv run python scripts/benchmark_rag_pipeline.py
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ─────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────

QUERY = "교통사고 손해배상 판례"
N_RESULTS = 15
RERANK_TOP_K = 4
DOC_TYPE = "precedent"

SEPARATOR = "=" * 70


def format_ms(seconds: float) -> str:
    """초 → ms 문자열."""
    return f"{seconds * 1000:.1f}ms"


def print_header(title: str) -> None:
    """섹션 헤더 출력."""
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def print_step(step_num: str, name: str, elapsed: float, detail: str = "") -> None:
    """단계 결과 출력."""
    ms = format_ms(elapsed)
    extra = f" | {detail}" if detail else ""
    print(f"  [{step_num}] {name}: {ms}{extra}")


def main() -> None:
    """벤치마크 메인 실행."""
    print_header("RAG 파이프라인 벤치마크")
    print(f"  쿼리: {QUERY}")
    print(f"  검색 수: {N_RESULTS} | 리랭킹 top_k: {RERANK_TOP_K}")
    print(f"  doc_type: {DOC_TYPE}")

    total_start = time.monotonic()
    timings: list[tuple[str, str, float, str]] = []

    # ─────────────────────────────────────────────────
    # [0] 모델 로딩 (임베딩 + 리랭커)
    # ─────────────────────────────────────────────────
    print_header("Step 0: 모델 로딩")

    t0 = time.monotonic()
    from app.services.rag.embedding import create_query_embedding  # noqa: E402
    _ = create_query_embedding("테스트")  # warm-up
    embedding_load_time = time.monotonic() - t0
    print_step("0a", "임베딩 모델 로드 + warm-up", embedding_load_time)
    timings.append(("0a", "임베딩 모델 로드", embedding_load_time, ""))

    t0 = time.monotonic()
    from app.services.rag.rerank import _load_reranker_model  # noqa: E402
    reranker = _load_reranker_model()
    reranker_load_time = time.monotonic() - t0
    reranker_status = "성공" if reranker else "실패/미설치"
    print_step("0b", "리랭커 모델 로드", reranker_load_time, reranker_status)
    timings.append(("0b", "리랭커 모델 로드", reranker_load_time, reranker_status))

    # ─────────────────────────────────────────────────
    # [1] 쿼리 임베딩 생성
    # ─────────────────────────────────────────────────
    print_header("Step 1: 쿼리 임베딩 생성")

    t0 = time.monotonic()
    query_embedding = create_query_embedding(QUERY)
    embed_time = time.monotonic() - t0
    dim = len(query_embedding) if query_embedding is not None else 0
    print_step("1", "쿼리 임베딩", embed_time, f"차원: {dim}")
    timings.append(("1", "쿼리 임베딩", embed_time, f"dim={dim}"))

    # ─────────────────────────────────────────────────
    # [2] 벡터 검색 (LanceDB IVF)
    # ─────────────────────────────────────────────────
    print_header("Step 2: 벡터 검색 (LanceDB)")

    from app.services.rag.retrieval import _search_vector_ids  # noqa: E402

    vector_fetch = N_RESULTS * 3

    t0 = time.monotonic()
    vector_results = _search_vector_ids(QUERY, vector_fetch, DOC_TYPE)
    vector_time = time.monotonic() - t0
    print_step("2", "벡터 검색", vector_time, f"{len(vector_results)}건 (fetch={vector_fetch})")
    timings.append(("2", "벡터 검색 (LanceDB)", vector_time, f"{len(vector_results)}건"))

    if vector_results:
        top3 = vector_results[:3]
        for i, doc in enumerate(top3):
            meta = doc.get("metadata", {})
            doc_id = meta.get("doc_id", "")
            case_name = meta.get("case_name", "")[:30]
            sim = doc.get("similarity", 0)
            print(f"    top-{i+1}: [{doc_id}] {case_name}  (sim={sim:.3f})")

    # ─────────────────────────────────────────────────
    # [3] 키워드 검색 (PostgreSQL FTS)
    # ─────────────────────────────────────────────────
    print_header("Step 3: 키워드 검색 (PostgreSQL FTS)")

    from app.services.rag.keyword_search import is_fts_available, search_by_keyword  # noqa: E402, I001

    fts_available = is_fts_available()
    keyword_results: list[dict] = []
    keyword_time = 0.0

    if fts_available:
        t0 = time.monotonic()
        keyword_results = search_by_keyword(QUERY, n_results=vector_fetch, doc_type=DOC_TYPE)
        keyword_time = time.monotonic() - t0
        print_step("3", "키워드 검색 (FTS)", keyword_time, f"{len(keyword_results)}건")
    else:
        print("  [3] FTS 비활성화 → 스킵")
    timings.append(("3", "키워드 검색 (FTS)", keyword_time, f"{len(keyword_results)}건"))

    # ─────────────────────────────────────────────────
    # [4] RRF 병합
    # ─────────────────────────────────────────────────
    print_header("Step 4: RRF 병합")

    from app.services.rag.fusion import reciprocal_rank_fusion  # noqa: E402
    from app.services.rag.retrieval import (  # noqa: E402
        _best_doc_per_source,
        _unique_source_ids,
    )

    t0 = time.monotonic()
    vector_source_ids = _unique_source_ids(vector_results)
    keyword_source_ids = _unique_source_ids(keyword_results)
    fused_ids = reciprocal_rank_fusion(vector_source_ids, keyword_source_ids)

    vector_best = _best_doc_per_source(vector_results)
    keyword_best = _best_doc_per_source(keyword_results)

    merged: list[dict] = []
    for sid in fused_ids:
        if sid in vector_best:
            merged.append(vector_best[sid])
        elif sid in keyword_best:
            merged.append(keyword_best[sid])
        if len(merged) >= N_RESULTS:
            break

    rrf_time = time.monotonic() - t0
    overlap = len(set(vector_source_ids) & set(keyword_source_ids))
    print_step(
        "4", "RRF 병합", rrf_time,
        f"벡터 {len(vector_source_ids)} + FTS {len(keyword_source_ids)} "
        f"→ {len(merged)}건 (중복 {overlap}건)"
    )
    timings.append(("4", "RRF 병합", rrf_time, f"{len(merged)}건"))

    # ─────────────────────────────────────────────────
    # [5] 요약문 조회 (PostgreSQL ai_summary, 리랭킹용)
    # ─────────────────────────────────────────────────
    print_header("Step 5: 요약문 조회 (PostgreSQL)")

    from app.services.rag.retrieval import (  # noqa: E402
        _extract_id_data_type_map,
        _populate_content,
        fetch_ai_summaries,
    )

    id_type_map_summary = _extract_id_data_type_map(merged)

    t0 = time.monotonic()
    summaries = fetch_ai_summaries(id_type_map_summary)
    _populate_content(merged, summaries)
    summary_time = time.monotonic() - t0
    filled_count = sum(1 for d in merged if d.get("content"))
    print_step("5", "요약문 조회", summary_time, f"{len(summaries)}/{len(id_type_map_summary)}건 매칭")
    timings.append(("5", "요약문 조회 (PostgreSQL)", summary_time, f"{filled_count}건 content 주입"))

    # ─────────────────────────────────────────────────
    # [6] Cross-encoder 리랭킹
    # ─────────────────────────────────────────────────
    print_header("Step 6: Cross-encoder 리랭킹")

    from app.services.rag.rerank import rerank_documents  # noqa: E402

    t0 = time.monotonic()
    reranked = rerank_documents(QUERY, merged, top_k=RERANK_TOP_K)
    rerank_time = time.monotonic() - t0
    print_step(
        "6", "리랭킹", rerank_time,
        f"{len(merged)}건 → {len(reranked)}건 (top_k={RERANK_TOP_K})"
    )
    timings.append(("6", "Cross-encoder 리랭킹", rerank_time, f"{len(reranked)}건"))

    if reranked:
        for i, doc in enumerate(reranked[:3]):
            meta = doc.get("metadata", {})
            doc_id = meta.get("doc_id", "")
            case_name = meta.get("case_name", "")[:30]
            score = doc.get("rerank_score", 0)
            print(f"    top-{i+1}: [{doc_id}] {case_name}  (score={score:.4f})")

    # ─────────────────────────────────────────────────
    # [7] 원문 조회 (PostgreSQL) - top-k만
    # ─────────────────────────────────────────────────
    print_header("Step 7: 원문 조회 (PostgreSQL)")

    from app.services.rag.retrieval import (  # noqa: E402
        _extract_id_data_type_map,
        fetch_document_contents,
    )

    t0 = time.monotonic()
    id_type_map = _extract_id_data_type_map(reranked)
    contents = fetch_document_contents(id_type_map)
    _populate_content(reranked, contents)
    content_time = time.monotonic() - t0
    content_lengths = [len(d.get("content", "")) for d in reranked]
    avg_len = sum(content_lengths) / len(content_lengths) if content_lengths else 0
    print_step(
        "7", "원문 조회 (PostgreSQL)", content_time,
        f"{len(contents)}/{len(reranked)}건 | 평균 {avg_len:.0f}자"
    )
    timings.append(("7", "원문 조회 (PostgreSQL)", content_time, f"{len(contents)}건"))

    # ─────────────────────────────────────────────────
    # [8] 판례 상세 조회 (PrecedentService)
    # ─────────────────────────────────────────────────
    print_header("Step 8: 판례 상세 조회 (PrecedentService)")

    from app.services.service_function import get_precedent_service  # noqa: E402

    detail_source_ids = [
        d.get("metadata", {}).get("doc_id", "")
        for d in reranked
        if d.get("metadata", {}).get("doc_id")
    ]

    t0 = time.monotonic()
    precedent_service = get_precedent_service()
    details = precedent_service.get_details(detail_source_ids)
    detail_time = time.monotonic() - t0
    print_step("8", "판례 상세 조회", detail_time, f"{len(details)}/{len(detail_source_ids)}건")
    timings.append(("8", "판례 상세 조회", detail_time, f"{len(details)}건"))

    # ─────────────────────────────────────────────────
    # [9] 그래프 보강 (Neo4j)
    # ─────────────────────────────────────────────────
    print_header("Step 9: 그래프 보강 (Neo4j)")

    from app.tools.graph import get_graph_service  # noqa: E402

    t0 = time.monotonic()
    graph_contexts: dict = {}
    try:
        graph_service = get_graph_service()
        if graph_service.is_connected:
            for doc in reranked:
                case_number = doc.get("metadata", {}).get("case_number", "")
                if case_number:
                    ctx = graph_service.enrich_case_context(case_number)
                    if ctx.get("cited_statutes") or ctx.get("similar_cases"):
                        graph_contexts[case_number] = ctx
            graph_status = f"{len(graph_contexts)}건 보강"
        else:
            graph_status = "Neo4j 미연결"
    except Exception as e:
        graph_status = f"실패: {e}"
    graph_time = time.monotonic() - t0
    print_step("9", "그래프 보강 (Neo4j)", graph_time, graph_status)
    timings.append(("9", "그래프 보강 (Neo4j)", graph_time, graph_status))

    # ─────────────────────────────────────────────────
    # [*] LLM 응답 생성 (측정만, 실제 호출 선택)
    # ─────────────────────────────────────────────────
    print_header("Step *: LLM 응답 생성 (스킵 - API 키 필요)")
    print("  LLM 호출은 벤치마크에서 제외합니다.")
    print("  일반적으로 Solar Pro: ~2-5초, GPT-4o: ~3-8초")

    # ─────────────────────────────────────────────────
    # 요약
    # ─────────────────────────────────────────────────
    total_time = time.monotonic() - total_start

    print_header("요약 (모델 로딩 제외)")

    # 모델 로딩 제외 시간 계산
    pipeline_time = sum(
        t[2] for t in timings if not t[0].startswith("0")
    )

    print(f"\n  {'단계':<5} {'이름':<30} {'시간':>10} {'비율':>8}  비고")
    print(f"  {'-' * 5} {'-' * 30} {'-' * 10} {'-' * 8}  {'-' * 20}")

    for step_num, name, elapsed, detail in timings:
        ms_str = format_ms(elapsed)
        if pipeline_time > 0 and not step_num.startswith("0"):
            pct = elapsed / pipeline_time * 100
            pct_str = f"{pct:.1f}%"
        else:
            pct_str = "-"
        print(f"  {step_num:<5} {name:<30} {ms_str:>10} {pct_str:>8}  {detail}")

    print(f"\n  {'총 파이프라인 (모델 로딩 제외)':<40} {format_ms(pipeline_time):>10}")
    print(f"  {'총 시간 (모델 로딩 포함)':<40} {format_ms(total_time):>10}")

    # ─────────────────────────────────────────────────
    # 파이프라인 플로우 다이어그램
    # ─────────────────────────────────────────────────
    print_header("파이프라인 플로우")
    print("""
  쿼리 입력
    │
    ├─[1] 쿼리 임베딩 생성 (KURE-v1, 1024차원)
    │
    ├─[2] 벡터 검색 (LanceDB IVF_FLAT, nprobes=40)
    │     └─ source_id 단위 deduplicate, 유사도 내림차순
    │
    ├─[3] 키워드 검색 (PostgreSQL tsvector, 개념AND→OR fallback)
    │     └─ MeCab 토크나이징 → tsquery 생성
    │
    ├─[4] RRF 병합 (벡터 + FTS 결과 합산)
    │     └─ reciprocal_rank_fusion → source_id별 최고 유사도 선택
    │
    ├─[5] 요약문 조회 (PostgreSQL ai_summary → 리랭킹 입력)
    │
    ├─[6] Cross-encoder 리랭킹 (bge-reranker-v2-m3-ko)
    │     └─ 적응형 truncation (head 3000 + tail 1000자)
    │     └─ sigmoid 점수 → top-k 선택
    │
    ├─[7] 원문 조회 (PostgreSQL, top-k만 배치 조회)
    │     └─ 테이블 레지스트리 기반 (ruling, reasoning 등)
    │
    ├─[8] 판례 상세 조회 (PrecedentService)
    │     └─ ruling, reasoning, full_reason, reference_provisions 등
    │
    ├─[9] 그래프 보강 (Neo4j)
    │     └─ cited_statutes, similar_cases
    │
    └─[*] LLM 응답 생성 (Solar Pro / GPT-4o)
""")


if __name__ == "__main__":
    main()
