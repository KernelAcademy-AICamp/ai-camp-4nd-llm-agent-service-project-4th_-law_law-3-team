"""
RAG 파이프라인 단계별 타이밍 벤치마크 v2

warm-up 완료 후 steady-state 측정 (3회 반복 평균)
그래프 보강 제거 반영

사용법: cd backend && uv run python scripts/benchmark_rag_pipeline_v2.py
"""

import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

QUERY = "교통사고 손해배상 판례"
N_RESULTS = 15
RERANK_TOP_K = 4
DOC_TYPE = "precedent"
N_TRIALS = 3

SEPARATOR = "=" * 60


def fmt(seconds: float) -> str:
    return f"{seconds * 1000:.0f}ms"


def main() -> None:
    print(f"\n{SEPARATOR}")
    print("  RAG 파이프라인 벤치마크 v2 (warm-up 후 steady-state)")
    print(f"  쿼리: {QUERY}")
    print(f"  N_RESULTS={N_RESULTS}, RERANK_TOP_K={RERANK_TOP_K}, 반복={N_TRIALS}회")
    print(SEPARATOR)

    # ── 모델 warm-up ──────────────────────────────────
    print("\n  모델 warm-up...")
    from app.services.rag.embedding import create_query_embedding
    from app.services.rag.rerank import rerank_documents

    create_query_embedding("warm-up 테스트")

    import torch
    from sentence_transformers import CrossEncoder
    _model = CrossEncoder(
        "dragonkue/bge-reranker-v2-m3-ko",
        activation_fn=torch.nn.Sigmoid(),
    )
    _model.predict([("warm-up", "warm-up 문서")])
    del _model

    # rerank_documents 내부 모델도 warm-up
    dummy_docs = [{"content": "warm-up 문서", "metadata": {"doc_id": "x"}, "similarity": 0.5}]
    rerank_documents("warm-up", dummy_docs, top_k=1)
    print("  warm-up 완료\n")

    # ── 반복 측정 ──────────────────────────────────
    from app.services.rag.retrieval import (
        _search_vector_ids,
        _best_doc_per_source,
        _extract_id_data_type_map,
        _populate_content,
        _unique_source_ids,
        fetch_document_contents,
        fetch_lancedb_summaries,
    )
    from app.services.rag.keyword_search import is_fts_available, search_by_keyword
    from app.services.rag.fusion import reciprocal_rank_fusion
    from app.services.service_function import get_precedent_service

    fts_ok = is_fts_available()
    precedent_service = get_precedent_service()

    step_names = [
        "쿼리 임베딩",
        "벡터 검색 (LanceDB)",
        "키워드 검색 (FTS)",
        "RRF 병합",
        "요약문 조회 (LanceDB)",
        "Cross-encoder 리랭킹",
        "원문 조회 (PostgreSQL)",
        "판례 상세 조회",
    ]
    step_times: list[list[float]] = [[] for _ in step_names]
    step_details: list[str] = [""] * len(step_names)

    for trial in range(N_TRIALS):
        print(f"  Trial {trial + 1}/{N_TRIALS}...", end="", flush=True)

        vector_fetch = N_RESULTS * 3

        # [0] 쿼리 임베딩
        t0 = time.monotonic()
        _emb = create_query_embedding(QUERY)
        step_times[0].append(time.monotonic() - t0)

        # [1] 벡터 검색
        t0 = time.monotonic()
        vector_results = _search_vector_ids(QUERY, vector_fetch, DOC_TYPE)
        step_times[1].append(time.monotonic() - t0)

        # [2] 키워드 검색
        keyword_results: list[dict[str, Any]] = []
        t0 = time.monotonic()
        if fts_ok:
            keyword_results = search_by_keyword(QUERY, n_results=vector_fetch, doc_type=DOC_TYPE)
        step_times[2].append(time.monotonic() - t0)

        # [3] RRF 병합
        t0 = time.monotonic()
        v_ids = _unique_source_ids(vector_results)
        k_ids = _unique_source_ids(keyword_results)
        fused_ids = reciprocal_rank_fusion(v_ids, k_ids)
        v_best = _best_doc_per_source(vector_results)
        k_best = _best_doc_per_source(keyword_results)
        merged: list[dict[str, Any]] = []
        for sid in fused_ids:
            if sid in v_best:
                merged.append(v_best[sid])
            elif sid in k_best:
                merged.append(k_best[sid])
            if len(merged) >= N_RESULTS:
                break
        step_times[3].append(time.monotonic() - t0)

        # [4] 요약문 조회
        t0 = time.monotonic()
        src_ids = [d.get("metadata", {}).get("doc_id", "") for d in merged]
        summaries = fetch_lancedb_summaries(src_ids)
        _populate_content(merged, summaries)
        step_times[4].append(time.monotonic() - t0)

        # [5] 리랭킹
        t0 = time.monotonic()
        reranked = rerank_documents(QUERY, merged, top_k=RERANK_TOP_K)
        step_times[5].append(time.monotonic() - t0)

        # [6] 원문 조회
        t0 = time.monotonic()
        id_type_map = _extract_id_data_type_map(reranked)
        contents = fetch_document_contents(id_type_map)
        _populate_content(reranked, contents)
        step_times[6].append(time.monotonic() - t0)

        # [7] 판례 상세 조회
        detail_ids = [
            d.get("metadata", {}).get("doc_id", "")
            for d in reranked if d.get("metadata", {}).get("doc_id")
        ]
        t0 = time.monotonic()
        details = precedent_service.get_details(detail_ids)
        step_times[7].append(time.monotonic() - t0)

        # 마지막 trial의 상세 정보 저장
        overlap = len(set(v_ids) & set(k_ids))
        step_details = [
            f"dim={len(_emb)}",
            f"{len(vector_results)}건",
            f"{len(keyword_results)}건",
            f"벡터 {len(v_ids)} + FTS {len(k_ids)} → {len(merged)}건 (중복 {overlap})",
            f"{len(summaries)}/{len(src_ids)}건",
            f"{len(merged)}건 → {len(reranked)}건",
            f"{len(contents)}/{len(reranked)}건",
            f"{len(details)}/{len(detail_ids)}건",
        ]

        trial_total = sum(step_times[i][-1] for i in range(len(step_names)))
        print(f" {fmt(trial_total)}")

    # ── 결과 출력 ──────────────────────────────────
    print(f"\n{SEPARATOR}")
    print(f"  결과 ({N_TRIALS}회 평균)")
    print(SEPARATOR)

    import numpy as np

    avgs = [np.mean(t) for t in step_times]
    total = sum(avgs)

    print(f"\n  {'#':<4} {'단계':<28} {'평균':>8} {'비율':>7}  비고")
    print(f"  {'-'*4} {'-'*28} {'-'*8} {'-'*7}  {'-'*25}")

    for i, (name, avg, detail) in enumerate(zip(step_names, avgs, step_details)):
        pct = avg / total * 100 if total > 0 else 0
        print(f"  [{i+1}]  {name:<26} {fmt(avg):>8} {pct:>6.1f}%  {detail}")

    print(f"\n  {'총 파이프라인 (LLM 제외)':<36} {fmt(total):>8}")
    print(f"  {'+ LLM 응답 생성 (추정)':<36} {'~2-5초':>8}")
    print(f"  {'= 예상 총 응답 시간':<36} {fmt(total + 3.5):>8}")

    # 이전 대비 비교
    print(f"\n  이전 벤치마크 (그래프 포함, warm-up 미제거): 24,400ms")
    print(f"  보정 벤치마크 (그래프 포함, warm-up 제거):   ~9,300ms")
    print(f"  현재 벤치마크 (그래프 제거, warm-up 제거):   {fmt(total)}")


if __name__ == "__main__":
    main()
