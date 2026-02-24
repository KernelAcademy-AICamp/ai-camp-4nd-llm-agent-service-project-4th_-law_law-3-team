"""
자치법규 전용 LanceDB 벡터 라이터

1문서 → 다중 벡터 (전체요약 1 + 조문요약 N) 확장 로직.
기존 vector_writer.py의 15개 최적화를 모두 재사용합니다.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, cast

_backend_root = Path(__file__).parent.parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))

from app.tools.vectorstore.local_ordinance_schema import (  # noqa: I001
    LOCAL_ORDINANCE_SCHEMA,
    TABLE_NAME as LOCAL_ORDINANCE_TABLE,
    VECTOR_DIM,
    create_article_summary_chunk,
    create_overall_summary_chunk,
)
from scripts.embedding_common.cache import EmbeddingCache
from scripts.embedding_common.config import (
    DEFAULT_CONFIG,
    HardwareProfile,
    get_optimal_config,
)
from scripts.embedding_common.device import print_device_info
from scripts.embedding_common.memory import check_memory_pressure, print_memory_status
from scripts.embedding_common.model import (
    clear_memory,
    create_embeddings,
    get_embedding_model,
    set_seed,
)
from scripts.embedding_common.store import EmbeddingStore
from scripts.embedding_common.temperature import TemperatureMonitor
from scripts.ingest.config import IngestConfig
from scripts.ingest.vector_writer import _load_json_streaming

logger = logging.getLogger(__name__)

MAX_TEXT_LENGTH = 4000


# ---------------------------------------------------------------------------
# 1문서 → 다중 텍스트 확장
# ---------------------------------------------------------------------------


def _expand_item_to_texts(
    item: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """1개 자치법규 → (텍스트, 부분 메타데이터) 리스트로 확장

    Returns:
        [(텍스트, {"summary_type", "article_number", "chunk_index", "total_chunks"}), ...]
    """
    results: list[tuple[str, dict[str, Any]]] = []
    articles = item.get("조", [])
    if not isinstance(articles, list):
        articles = []
    total_chunks = 1 + len(articles)

    # 1. 전체요약 (Basic)
    overall = str(item.get("전체요약", "") or "").strip()
    if overall:
        results.append((overall, {
            "summary_type": "Basic",
            "article_number": None,
            "chunk_index": 0,
            "total_chunks": total_chunks,
        }))

    # 2. 조문요약 (Specific, N개)
    for idx, article in enumerate(articles):
        if not isinstance(article, dict):
            continue
        art_summary = str(article.get("조문요약", "") or "").strip()
        if art_summary:
            results.append((art_summary, {
                "summary_type": "Specific",
                "article_number": str(article.get("조문번호", "")),
                "chunk_index": idx + 1,
                "total_chunks": total_chunks,
            }))

    return results


# ---------------------------------------------------------------------------
# 배치 임베딩 + 저장
# ---------------------------------------------------------------------------


def _embed_and_store_batch(
    batch_texts: list[str],
    batch_meta: list[dict[str, Any]],
    model: Any,
    cache: EmbeddingCache | None,
    store: EmbeddingStore,
    stats: dict[str, int],
    batch_size: int,
    dim_verified: bool,
) -> bool:
    """배치 임베딩 생성 + LanceDB 저장"""
    texts = [t[:MAX_TEXT_LENGTH] for t in batch_texts]

    # 캐시 조회 → 미스분만 임베딩
    if cache:
        cached_vectors: list[list[float] | None] = []
        uncached_texts: list[str] = []
        uncached_indices: list[int] = []

        for i, text in enumerate(texts):
            cached = cache.get(text)
            if cached is not None:
                cached_vectors.append(cached)
            else:
                cached_vectors.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)

        if uncached_texts:
            try:
                new_vectors = create_embeddings(
                    uncached_texts, model=model, batch_size=batch_size
                )
                for idx, vec in zip(uncached_indices, new_vectors):
                    cached_vectors[idx] = vec
                    cache.set(texts[idx], vec)
            except Exception as e:
                logger.error("임베딩 생성 실패: %s", e)
                stats["errors"] += len(batch_texts)
                return dim_verified

        vectors = cast(list[list[float]], cached_vectors)
    else:
        try:
            vectors = create_embeddings(
                texts, model=model, batch_size=batch_size
            )
        except Exception as e:
            logger.error("임베딩 생성 실패: %s", e)
            stats["errors"] += len(batch_texts)
            return dim_verified

    # 차원 검증 (첫 배치만)
    if not dim_verified and vectors:
        dim = len(vectors[0])
        if dim != VECTOR_DIM:
            raise ValueError(
                f"임베딩 차원 불일치: {dim} != {VECTOR_DIM}. 모델 확인 필요"
            )
        logger.info("임베딩 차원 검증 통과: %d", dim)
        dim_verified = True

    # LanceDB record 생성
    records: list[dict[str, Any]] = []
    for meta, vector, text in zip(batch_meta, vectors, texts):
        source_id = meta["source_id"]
        title = meta["title"]
        source_name = meta["source_name"]
        summary_type = meta["summary_type"]
        article_number = meta.get("article_number")
        chunk_index = meta["chunk_index"]
        total_chunks = meta["total_chunks"]

        if summary_type == "Basic":
            record = create_overall_summary_chunk(
                source_id=source_id,
                title=title,
                content=text,
                vector=vector,
                source_name=source_name,
                total_chunks=total_chunks,
            )
        else:
            record = create_article_summary_chunk(
                source_id=source_id,
                title=title,
                content=text,
                vector=vector,
                source_name=source_name,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                article_number=article_number or "",
            )
        records.append(record)

    if records:
        store.add_batch(records)
        stats["embedded"] += len(records)

    return dim_verified


# ---------------------------------------------------------------------------
# 매니페스트 저장
# ---------------------------------------------------------------------------


def _save_run_manifest(
    store: EmbeddingStore,
    config: IngestConfig,
    stats: dict[str, int],
    device_info: Any,
    elapsed: float,
) -> None:
    """임베딩 실행 메타데이터 저장"""
    manifest = {
        "data_type": config.data_type_label,
        "table_name": LOCAL_ORDINANCE_TABLE,
        "model": str(DEFAULT_CONFIG["EMBEDDING_MODEL"]),
        "vector_dim": VECTOR_DIM,
        "device": str(device_info),
        "stats": stats,
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": datetime.now().isoformat(),
    }
    manifest_path = Path(store.db_path) / f"manifest_{config.name}.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    logger.info("매니페스트 저장: %s", manifest_path)


# ---------------------------------------------------------------------------
# 메인 인제스트 함수
# ---------------------------------------------------------------------------


def run_local_ordinance_vector_ingest(
    config: IngestConfig,
    source_path: Path | None = None,
    reset: bool = False,
    batch_size: int | None = None,
    device: str | None = None,
    profile: str | None = None,
    use_cache: bool = True,
    backend: str | None = None,
) -> dict[str, int]:
    """
    자치법규 JSON → 다중 벡터 임베딩 → local_ordinance_chunks 테이블 저장

    1문서 = 1(전체요약) + N(조문요약) 벡터를 생성합니다.

    Args:
        config: 인제스트 설정
        source_path: JSON 소스 경로
        reset: 기존 데이터 삭제 후 재실행
        batch_size: 임베딩 배치 크기
        device: 임베딩 디바이스
        profile: 하드웨어 프로필
        use_cache: 임베딩 캐시 사용 여부

    Returns:
        통계 dict
    """
    from tqdm import tqdm

    set_seed(42)

    device_info, hw_config = print_device_info()

    if profile:
        hw_config = get_optimal_config(device_info, HardwareProfile(profile))
        logger.info("프로필 지정: %s → batch_size=%d", profile, hw_config.batch_size)

    if batch_size is None:
        batch_size = hw_config.batch_size
        logger.info("배치 크기 자동 설정: %d", batch_size)

    model = get_embedding_model(device=device, backend=backend)
    logger.info("임베딩 모델 로드 완료 (backend=%s)", backend or "pytorch")

    cache = EmbeddingCache() if use_cache else None

    thermal_monitor: TemperatureMonitor | None = None
    if hw_config.temp_monitoring:
        thermal_monitor = TemperatureMonitor(threshold=hw_config.temp_threshold)

    source = source_path or config.source_path

    stats: dict[str, int] = {
        "total_docs": 0,
        "total_vectors": 0,
        "embedded": 0,
        "skipped_no_summary": 0,
        "skipped_existing": 0,
        "errors": 0,
    }

    # 별도 테이블 사용
    store = EmbeddingStore(
        table_name=LOCAL_ORDINANCE_TABLE,
        schema=LOCAL_ORDINANCE_SCHEMA,
    )

    if reset:
        logger.info("기존 local_ordinance_chunks 테이블 리셋")
        store.reset()
        existing_ids: set[str] = set()
    else:
        existing_ids = store.get_existing_source_ids("자치법규")
        logger.info("기존 자치법규 문서: %d건 (자동 스킵)", len(existing_ids))

    start_time = time.time()
    dim_verified = False
    current_batch_size = batch_size
    batch_count = 0

    pbar = tqdm(
        desc=f"임베딩 ({config.data_type_label})",
        unit="vecs",
        mininterval=1.0,
    )

    # 배치 수집 버퍼 (텍스트 + 메타데이터)
    batch_texts: list[str] = []
    batch_meta: list[dict[str, Any]] = []

    for item in _load_json_streaming(source):
        stats["total_docs"] += 1
        source_id = str(item.get("자치법규ID", ""))

        if not source_id:
            stats["errors"] += 1
            continue

        if source_id in existing_ids:
            stats["skipped_existing"] += 1
            continue

        title = str(item.get("자치법규명", "") or "")
        source_name = str(item.get("지자체기관명", "") or "")

        # 1문서 → 다중 텍스트 확장
        expanded = _expand_item_to_texts(item)
        if not expanded:
            stats["skipped_no_summary"] += 1
            continue

        stats["total_vectors"] += len(expanded)

        for text, partial_meta in expanded:
            partial_meta["source_id"] = source_id
            partial_meta["title"] = title
            partial_meta["source_name"] = source_name

            batch_texts.append(text)
            batch_meta.append(partial_meta)

            if len(batch_texts) < current_batch_size:
                continue

            # --- 배치 처리 ---
            dim_verified = _embed_and_store_batch(
                batch_texts, batch_meta, model, cache, store, stats,
                batch_size, dim_verified,
            )
            pbar.update(len(batch_texts))
            batch_texts = []
            batch_meta = []
            batch_count += 1

            # 메모리 관리
            if batch_count % hw_config.gc_interval == 0:
                print_memory_status()
                if check_memory_pressure():
                    logger.warning("메모리 압력 감지! 강제 GC 실행")
                clear_memory()
                if cache:
                    cache_size = len(cache._memory_cache)
                    if cache_size > 0:
                        cache.clear_memory_cache()
                        logger.info("캐시 메모리 정리: %d 엔트리 해제", cache_size)

            # GPU 온도 모니터링
            if thermal_monitor:
                current_batch_size, should_stop = thermal_monitor.check_and_adjust(
                    current_batch_size
                )
                if should_stop:
                    logger.warning("GPU 온도 위험! 중단 (재실행 시 자동 재개)")
                    pbar.close()
                    return stats

    # 잔여 배치 처리
    if batch_texts:
        dim_verified = _embed_and_store_batch(
            batch_texts, batch_meta, model, cache, store, stats,
            batch_size, dim_verified,
        )
        pbar.update(len(batch_texts))

    pbar.close()

    elapsed = time.time() - start_time

    logger.info(
        "스캔 완료: 문서 %d건 → 벡터 %d건 임베딩 %d건 "
        "(요약 없음: %d, 기존: %d, 에러: %d)",
        stats["total_docs"],
        stats["total_vectors"],
        stats["embedded"],
        stats["skipped_no_summary"],
        stats["skipped_existing"],
        stats["errors"],
    )

    # compact
    if store.table is not None and stats["embedded"] > 0:
        logger.info("LanceDB compact 실행 중...")
        try:
            store.table.compact_files()
        except Exception as e:
            logger.warning("compact 실패 (무시): %s", e)

    # 매니페스트
    if stats["embedded"] > 0:
        _save_run_manifest(store, config, stats, device_info, elapsed)

    # 캐시 통계
    if cache:
        logger.info("캐시 통계: %s", cache.get_stats())

    logger.info(
        "벡터 저장 완료: %d건, %.1f초 (%.1f vecs/s)",
        stats["embedded"],
        elapsed,
        stats["embedded"] / elapsed if elapsed > 0 else 0,
    )

    return stats


def build_local_ordinance_ann_index(
    num_partitions: int = 512,
) -> None:
    """local_ordinance_chunks 전용 ANN 인덱스 생성"""
    store = EmbeddingStore(
        table_name=LOCAL_ORDINANCE_TABLE,
        schema=LOCAL_ORDINANCE_SCHEMA,
    )
    table = store.table

    if table is None:
        logger.warning("local_ordinance_chunks 테이블이 없습니다.")
        return

    total = store.count()
    logger.info(
        "ANN 인덱스 생성 시작 (%s, 총 %d건, partitions=%d)",
        LOCAL_ORDINANCE_TABLE, total, num_partitions,
    )

    start_time = time.time()
    table.create_index(
        metric="cosine",
        num_partitions=num_partitions,
    )

    elapsed = time.time() - start_time
    logger.info("ANN 인덱스 생성 완료: %.1f초", elapsed)
