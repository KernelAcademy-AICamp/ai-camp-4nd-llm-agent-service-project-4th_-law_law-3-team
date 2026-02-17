"""
LanceDB 벡터 저장 + IVF_FLAT ANN 인덱스

요약문 기반 1문서=1벡터 임베딩을 LanceDB에 저장합니다.
기존 embedding_common 모듈을 재사용하며, 15개 최적화 기능을 통합합니다.

최적화 기능:
 #1  LanceDB 기반 자동 재개 (existing_ids로 이미 임베딩된 문서 스킵)
 #2  하드웨어 자동 설정
 #3  메모리 압력 감지
 #4  GPU 온도 모니터링
 #5  임베딩 캐시
 #6  ijson 스트리밍 + 배치 단위 처리
 #7  compact()
 #8  차원 검증
 #9  스키마 검증
 #10 품질 검증
 #11 재현성 시드 고정
 #12 tqdm 진행바
 #13 디바이스 정보 출력
 #14 메모리 상태 로그
 #15 실행 매니페스트
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, cast

# 백엔드 app 모듈 import를 위한 경로 추가
_backend_root = Path(__file__).parent.parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))

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
from scripts.embedding_common.schema import VECTOR_DIM, validate_chunk
from scripts.embedding_common.store import EmbeddingStore
from scripts.embedding_common.temperature import TemperatureMonitor
from scripts.ingest.config import IngestConfig

logger = logging.getLogger(__name__)

# 임베딩 텍스트 최대 길이 (모델 토큰 제한 방지)
MAX_TEXT_LENGTH = 4000

# ANN 인덱스 파라미터
ANN_NUM_PARTITIONS = 256  # sqrt(92055) ≈ 303, 보수적으로 256


# ---------------------------------------------------------------------------
# #6 ijson 스트리밍 로드
# ---------------------------------------------------------------------------


def _extract_group_from_filename(filename: str) -> str:
    """파일명에서 그룹명(위원회/부처/기관명) 추출

    패턴: prefix_그룹명_v숫자.json
    예: dec_comm_공정거래위원회_v1.json → 공정거래위원회
        intp_min_고용노동부_v1.json → 고용노동부
        sadm_case_조세심판원_v1.json → 조세심판원
    """
    import re

    m = re.search(r"(?:dec_comm|intp_min|sadm_case)_(.+?)_v\d+\.json", filename)
    return m.group(1) if m else Path(filename).stem


def _load_json_streaming(source_path: Path) -> Iterator[dict[str, Any]]:
    """JSON 스트리밍 로드 (ijson 필수).

    단일 파일이면 그대로 스트리밍 로드.
    디렉토리이면 내부 .json 파일을 순차 스트리밍하고
    각 item에 __source_group__ 키를 추가.
    """
    import ijson

    if not source_path.exists():
        raise FileNotFoundError(f"소스를 찾을 수 없습니다: {source_path}")

    if source_path.is_dir():
        yield from _load_json_directory_streaming(source_path)
        return

    logger.info("JSON 스트리밍 로드: %s", source_path)
    with open(source_path, "rb") as f:
        for item in ijson.items(f, "item"):
            yield item


def _load_json_directory_streaming(dir_path: Path) -> Iterator[dict[str, Any]]:
    """디렉토리 내 모든 .json 파일을 순차 스트리밍 로드

    각 item에 __source_group__ 키를 추가하여
    어느 파일(위원회/부처)에서 왔는지 식별 가능하게 함.
    """
    import ijson

    json_files = sorted(dir_path.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"디렉토리에 .json 파일이 없습니다: {dir_path}")

    logger.info("디렉토리 스트리밍 로드: %s (%d개 파일)", dir_path, len(json_files))

    for json_file in json_files:
        group_name = _extract_group_from_filename(json_file.name)
        logger.info("  스트리밍: %s (%s)", json_file.name, group_name)

        with open(json_file, "rb") as f:
            for item in ijson.items(f, "item"):
                item["__source_group__"] = group_name
                yield item


# ---------------------------------------------------------------------------
# #10 품질 검증
# ---------------------------------------------------------------------------

def _run_quality_check(model: Any) -> bool:
    """임베딩 모델 품질 간이 검증 (유사/비유사 쌍 테스트)"""
    import numpy as np

    similar_pairs = [
        ("손해배상 청구권", "손해배상 청구"),
        ("민법 제750조 불법행위", "민법상 불법행위 책임"),
    ]
    dissimilar_pairs = [
        ("민법 제750조 불법행위", "형법 제250조 살인죄"),
        ("손해배상 청구권", "회사 설립 절차"),
    ]
    all_texts = [t for pair in similar_pairs + dissimilar_pairs for t in pair]

    try:
        vectors = create_embeddings(all_texts, model=model)
    except Exception as e:
        logger.warning("품질 검증 실패 (임베딩 오류): %s", e)
        return True  # 검증 실패해도 진행

    sim_scores: list[float] = []
    dissim_scores: list[float] = []
    idx = 0
    for _ in similar_pairs:
        sim_scores.append(float(np.dot(vectors[idx], vectors[idx + 1])))
        idx += 2
    for _ in dissimilar_pairs:
        dissim_scores.append(float(np.dot(vectors[idx], vectors[idx + 1])))
        idx += 2

    avg_sim = float(np.mean(sim_scores))
    avg_dissim = float(np.mean(dissim_scores))
    separation = avg_sim - avg_dissim

    if separation > 0.2:
        quality = "GOOD"
    elif separation > 0.1:
        quality = "FAIR"
    else:
        quality = "POOR"

    logger.info(
        "품질 검증: 유사=%.3f, 비유사=%.3f, 분리도=%.3f (%s)",
        avg_sim,
        avg_dissim,
        separation,
        quality,
    )

    if quality == "POOR":
        logger.warning("임베딩 품질이 낮습니다 (POOR). 모델 확인을 권장합니다.")
        return False

    return True


# ---------------------------------------------------------------------------
# #15 실행 매니페스트
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
# 배치 임베딩 헬퍼
# ---------------------------------------------------------------------------

def _embed_and_store_batch(
    batch_items: list[dict[str, Any]],
    config: IngestConfig,
    model: Any,
    cache: EmbeddingCache | None,
    store: EmbeddingStore,
    stats: dict[str, int],
    batch_size: int,
    dim_verified: bool,
) -> bool:
    """
    배치 임베딩 생성 + LanceDB 저장

    Returns:
        dim_verified 여부 (첫 배치에서 True로 변경)
    """
    # 요약문 추출 + 길이 제한
    texts = [
        str(item.get(config.summary_field, "")).strip()[:MAX_TEXT_LENGTH]
        for item in batch_items
    ]

    # #5 캐시 조회 → 미스분만 임베딩
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
                stats["errors"] += len(batch_items)
                return dim_verified

        # create_embeddings는 성공 시 N→N, 실패 시 예외 (early return)
        # 따라서 이 시점에서 None은 남지 않음
        vectors = cast(list[list[float]], cached_vectors)
    else:
        # 캐시 미사용: 전체 배치 임베딩
        try:
            vectors = create_embeddings(
                texts, model=model, batch_size=batch_size
            )
        except Exception as e:
            logger.error("임베딩 생성 실패: %s", e)
            stats["errors"] += len(batch_items)
            return dim_verified

    # #8 차원 검증 (첫 배치만)
    if not dim_verified and vectors:
        dim = len(vectors[0])
        if dim != VECTOR_DIM:
            raise ValueError(
                f"임베딩 차원 불일치: {dim} != {VECTOR_DIM}. 모델 확인 필요"
            )
        logger.info("임베딩 차원 검증 통과: %d", dim)
        dim_verified = True

    # #9 스키마 검증 + LanceDB record 생성
    records: list[dict[str, Any]] = []
    for item, vector in zip(batch_items, vectors):
        try:
            record = config.vector_metadata_fn(item, vector)
            if not validate_chunk(record):
                logger.error(
                    "스키마 검증 실패 (id=%s)",
                    item.get(config.id_field, "?"),
                )
                stats["errors"] += 1
                continue
            records.append(record)
        except Exception as e:
            logger.error(
                "레코드 생성 실패 (id=%s): %s",
                item.get(config.id_field, "?"),
                e,
            )
            stats["errors"] += 1
            continue

    # LanceDB 저장
    if records:
        store.add_batch(records)
        stats["embedded"] += len(records)

    return dim_verified


# ---------------------------------------------------------------------------
# 메인 인제스트 함수
# ---------------------------------------------------------------------------

def run_vector_ingest(
    config: IngestConfig,
    source_path: Path | None = None,
    reset: bool = False,
    batch_size: int | None = None,
    device: str | None = None,
    profile: str | None = None,
    use_cache: bool = True,
) -> dict[str, int]:
    """
    JSON -> 요약문 임베딩 -> LanceDB 저장 (스트리밍 배치 처리)

    중단 후 재실행 시 LanceDB의 existing_ids로 이미 임베딩된 문서를
    자동 스킵하므로 별도 체크포인트 파일이 필요 없습니다.

    Args:
        config: 인제스트 설정
        source_path: JSON 소스 경로 (None이면 config.source_path)
        reset: 기존 데이터 삭제 후 재실행
        batch_size: 임베딩 배치 크기 (None=하드웨어 자동 설정)
        device: 임베딩 디바이스 (None=자동)
        profile: 하드웨어 프로필 (desktop/laptop/mac/cpu, None=자동)
        use_cache: 임베딩 캐시 사용 여부

    Returns:
        통계 dict: total, embedded, skipped_no_summary, skipped_existing, errors
    """
    from tqdm import tqdm

    # #11 재현성 시드 고정
    set_seed(42)

    # #13 디바이스 정보 출력 + #2 하드웨어 자동 설정
    device_info, hw_config = print_device_info()

    if profile:
        hw_config = get_optimal_config(device_info, HardwareProfile(profile))
        logger.info("프로필 지정: %s → batch_size=%d", profile, hw_config.batch_size)

    if batch_size is None:
        batch_size = hw_config.batch_size
        logger.info("배치 크기 자동 설정: %d", batch_size)

    # 모델 로드
    model = get_embedding_model(device=device)
    logger.info("임베딩 모델 로드 완료")

    # #10 품질 검증
    _run_quality_check(model)

    # #5 임베딩 캐시
    cache = EmbeddingCache() if use_cache else None

    # #4 GPU 온도 모니터링
    thermal_monitor: TemperatureMonitor | None = None
    if hw_config.temp_monitoring:
        thermal_monitor = TemperatureMonitor(threshold=hw_config.temp_threshold)
        logger.info(
            "GPU 온도 모니터링 활성화 (임계: %d°C)", hw_config.temp_threshold
        )

    source = source_path or config.source_path

    stats: dict[str, int] = {
        "total": 0,
        "embedded": 0,
        "skipped_no_summary": 0,
        "skipped_existing": 0,
        "errors": 0,
    }

    # LanceDB 저장소
    store = EmbeddingStore()

    if reset:
        logger.info("기존 %s 데이터 삭제 중...", config.data_type_label)
        store.reset()
        existing_ids: set[str] = set()
    else:
        # #1 LanceDB 기반 자동 재개: 이미 임베딩된 문서는 자동 스킵
        existing_ids = store.get_existing_source_ids(config.data_type_label)
        logger.info(
            "기존 %s 문서: %d건 (자동 스킵)", config.data_type_label, len(existing_ids)
        )

    # #6 스트리밍 + 배치 단위 처리 (전체 JSON을 메모리에 올리지 않음)
    start_time = time.time()
    dim_verified = False
    current_batch_size = batch_size
    batch_count = 0

    # #12 tqdm 진행바 (스트리밍이므로 전체 건수 미지정)
    pbar = tqdm(
        desc=f"임베딩 ({config.data_type_label})",
        unit="docs",
        mininterval=1.0,
    )

    batch_items: list[dict[str, Any]] = []

    for item in _load_json_streaming(source):
        stats["total"] += 1
        doc_id = str(item.get(config.id_field, ""))
        summary = item.get(config.summary_field)

        if not doc_id:
            stats["errors"] += 1
            continue

        if not summary or not str(summary).strip():
            stats["skipped_no_summary"] += 1
            continue

        if doc_id in existing_ids:
            stats["skipped_existing"] += 1
            continue

        batch_items.append(item)

        if len(batch_items) < current_batch_size:
            continue

        # --- 배치 처리 ---
        dim_verified = _embed_and_store_batch(
            batch_items, config, model, cache, store, stats,
            batch_size, dim_verified,
        )
        pbar.update(len(batch_items))
        batch_items = []
        batch_count += 1

        # #14 메모리 상태 로그 + #3 메모리 압력 감지
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

        # #4 GPU 온도 모니터링
        if thermal_monitor:
            current_batch_size, should_stop = thermal_monitor.check_and_adjust(
                current_batch_size
            )
            if should_stop:
                logger.warning(
                    "GPU 온도 위험! 중단 (재실행 시 자동 재개)"
                )
                pbar.close()
                return stats

    # 잔여 배치 처리
    if batch_items:
        dim_verified = _embed_and_store_batch(
            batch_items, config, model, cache, store, stats,
            batch_size, dim_verified,
        )
        pbar.update(len(batch_items))

    pbar.close()

    elapsed = time.time() - start_time

    # 필터링 통계 로그
    logger.info(
        "스캔 완료: 전체 %d건 → 임베딩 %d건 "
        "(요약 없음: %d, 기존: %d, 에러: %d)",
        stats["total"],
        stats["embedded"],
        stats["skipped_no_summary"],
        stats["skipped_existing"],
        stats["errors"],
    )

    # #7 compact
    if store.table is not None and stats["embedded"] > 0:
        logger.info("LanceDB compact 실행 중...")
        try:
            store.table.compact_files()
        except Exception as e:
            logger.warning("compact 실패 (무시): %s", e)

    # #15 매니페스트
    if stats["embedded"] > 0:
        _save_run_manifest(store, config, stats, device_info, elapsed)

    # #5 캐시 통계
    if cache:
        logger.info("캐시 통계: %s", cache.get_stats())

    logger.info(
        "벡터 저장 완료: %d건, %.1f초 (%.1f docs/s)",
        stats["embedded"],
        elapsed,
        stats["embedded"] / elapsed if elapsed > 0 else 0,
    )

    return stats


def build_ann_index(
    num_partitions: int = ANN_NUM_PARTITIONS,
) -> None:
    """
    LanceDB IVF_FLAT ANN 인덱스 생성

    Args:
        num_partitions: IVF 파티션 수 (기본: 256)
    """
    store = EmbeddingStore()
    table = store.table

    if table is None:
        logger.warning("테이블이 없습니다. 인덱스를 건너뜁니다.")
        return

    total = store.count()
    logger.info("ANN 인덱스 생성 시작 (총 %d건, partitions=%d)", total, num_partitions)

    start_time = time.time()
    table.create_index(
        metric="cosine",
        num_partitions=num_partitions,
    )

    elapsed = time.time() - start_time
    logger.info("ANN 인덱스 생성 완료: %.1f초", elapsed)
