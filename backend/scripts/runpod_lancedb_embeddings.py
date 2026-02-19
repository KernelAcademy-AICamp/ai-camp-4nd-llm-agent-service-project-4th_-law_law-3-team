#!/usr/bin/env python3
"""
LanceDB 임베딩 생성 스크립트 (RunPod 전용 - thin wrapper)

ingest 파이프라인(scripts/ingest/)을 내부 호출하여 임베딩을 생성합니다.
기존 Public API를 100% 유지하므로 노트북에서 동일하게 사용 가능합니다.

=============================================================================
RunPod 사용법
=============================================================================

1. RunPod에서 GPU Pod 생성
   - GPU: RTX 3090 (24GB) 권장
   - Template: RunPod Pytorch 2.1

2. Jupyter Lab 접속 후 새 노트북 생성

3. 셀 1: 패키지 설치
   !pip install lancedb sentence-transformers pyarrow ijson psutil tqdm gdown pyyaml sqlalchemy pydantic-settings -q

4. 셀 2: 스크립트 import
   import sys; sys.path.insert(0, '/workspace/backend')
   from scripts.runpod_lancedb_embeddings import *

5. 셀 3: 디바이스 확인
   print_device_info()

6. 셀 4: 임베딩 생성
   run_law_embedding('law_v3.json', reset=True)
   run_precedent_embedding('precedents_v2.json', reset=True)

7. 셀 5: 결과 다운로드
   !zip -r lancedb_data.zip ./lancedb_data

=============================================================================
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

# 백엔드 모듈 import를 위한 경로 설정
_BACKEND_ROOT = Path(__file__).parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from tqdm import tqdm  # noqa: E402

from scripts.embedding_common.cache import EmbeddingCache  # noqa: F401, E402
from scripts.embedding_common.device import (  # noqa: E402
    get_device,  # noqa: F401
    print_device_info,  # noqa: F401
)
from scripts.embedding_common.memory import print_memory_status  # noqa: F401, E402
from scripts.embedding_common.model import (  # noqa: E402
    clear_memory,
    clear_model_cache,  # noqa: F401
    create_embeddings,  # noqa: F401
    get_embedding_model,  # noqa: F401
    set_seed,  # noqa: F401
)
from scripts.embedding_common.quality import EmbeddingQualityChecker  # noqa: F401, E402
from scripts.embedding_common.store import EmbeddingStore  # noqa: E402

# ijson 가용 여부 (split 함수에서 사용)
try:
    import ijson

    IJSON_AVAILABLE = True
except ImportError:
    IJSON_AVAILABLE = False


# ============================================================================
# ingest 파이프라인 연동
# ============================================================================


def _get_ingest_config(data_type: str) -> Any:
    """ingest 파이프라인의 IngestConfig 조회

    Args:
        data_type: "law" 또는 "precedent"
    """
    # types/ 자동 등록 트리거
    import scripts.ingest.types  # noqa: F401
    from scripts.ingest.config import get_config

    return get_config(data_type)


# ============================================================================
# Public API (하위 호환)
# ============================================================================


def run_law_embedding(
    source_path: str,
    reset: bool = False,
    batch_size: Optional[int] = None,
    auto_config: bool = True,
) -> dict[str, Any]:
    """법령 임베딩 실행

    Args:
        source_path: JSON 파일 경로
        reset: 기존 데이터 삭제 후 시작
        batch_size: 배치 크기 (None=자동)
        auto_config: 하위 호환용 (무시됨)
    """
    from scripts.ingest.vector_writer import run_vector_ingest

    config = _get_ingest_config("law")
    stats = run_vector_ingest(
        config,
        source_path=Path(source_path),
        reset=reset,
        batch_size=batch_size,
    )
    show_stats()
    clear_memory()
    return stats


def run_precedent_embedding(
    source_path: str,
    reset: bool = False,
    batch_size: Optional[int] = None,
    auto_config: bool = True,
) -> dict[str, Any]:
    """판례 임베딩 실행

    Args:
        source_path: JSON 파일 경로
        reset: 기존 데이터 삭제 후 시작
        batch_size: 배치 크기 (None=자동)
        auto_config: 하위 호환용 (무시됨)
    """
    from scripts.ingest.vector_writer import run_vector_ingest

    config = _get_ingest_config("precedent")
    stats = run_vector_ingest(
        config,
        source_path=Path(source_path),
        reset=reset,
        batch_size=batch_size,
    )
    show_stats()
    clear_memory()
    return stats


def run_precedent_embedding_part(
    source_path: str,
    reset: bool = False,
    batch_size: int = 64,
) -> dict[str, Any]:
    """분할된 판례 파일 하나를 처리"""
    from scripts.ingest.vector_writer import run_vector_ingest

    config = _get_ingest_config("precedent")
    return run_vector_ingest(
        config,
        source_path=Path(source_path),
        reset=reset,
        batch_size=batch_size,
    )


def run_law_embedding_part(
    source_path: str,
    reset: bool = False,
    batch_size: int = 64,
) -> dict[str, Any]:
    """분할된 법령 파일 하나를 처리"""
    from scripts.ingest.vector_writer import run_vector_ingest

    config = _get_ingest_config("law")
    return run_vector_ingest(
        config,
        source_path=Path(source_path),
        reset=reset,
        batch_size=batch_size,
    )


def run_all_precedent_parts(
    pattern: str = "precedents_part_*.json",
    batch_size: int = 64,
) -> dict[str, int]:
    """모든 분할된 판례 파일 처리"""
    import glob as glob_mod

    files = sorted(glob_mod.glob(pattern))
    if not files:
        print(f"[ERROR] No files matching: {pattern}")
        return {}

    print(f"[INFO] Found {len(files)} files to process")

    total_stats: dict[str, int] = {
        "total": 0,
        "embedded": 0,
        "skipped_no_summary": 0,
        "skipped_existing": 0,
        "errors": 0,
    }

    for i, filepath in enumerate(files):
        is_reset = i == 0
        print(f"\n[{i + 1}/{len(files)}] Processing: {filepath}")
        stats = run_precedent_embedding_part(
            filepath, reset=is_reset, batch_size=batch_size
        )
        for key in total_stats:
            total_stats[key] += stats.get(key, 0)
        clear_memory()
        print(f"\n[Progress] {i + 1}/{len(files)} files done")
        print("-" * 60)

    print("\n" + "=" * 60)
    print("ALL DONE!")
    print("=" * 60)
    print(f"Total: {total_stats['total']:,}")
    print(f"Embedded: {total_stats['embedded']:,}")
    print(f"Errors: {total_stats['errors']:,}")
    show_stats()
    return total_stats


def run_all_law_parts(
    pattern: str = "laws_part_*.json",
    batch_size: int = 64,
) -> dict[str, int]:
    """모든 분할된 법령 파일 처리"""
    import glob as glob_mod

    files = sorted(glob_mod.glob(pattern))
    if not files:
        print(f"[ERROR] No files matching: {pattern}")
        return {}

    print(f"[INFO] Found {len(files)} files to process")

    total_stats: dict[str, int] = {
        "total": 0,
        "embedded": 0,
        "skipped_no_summary": 0,
        "skipped_existing": 0,
        "errors": 0,
    }

    for i, filepath in enumerate(files):
        is_reset = i == 0
        print(f"\n[{i + 1}/{len(files)}] Processing: {filepath}")
        stats = run_law_embedding_part(
            filepath, reset=is_reset, batch_size=batch_size
        )
        for key in total_stats:
            total_stats[key] += stats.get(key, 0)
        clear_memory()
        print(f"\n[Progress] {i + 1}/{len(files)} files done")
        print("-" * 60)

    print("\n" + "=" * 60)
    print("ALL DONE!")
    print("=" * 60)
    print(f"Total: {total_stats['total']:,}")
    print(f"Embedded: {total_stats['embedded']:,}")
    print(f"Errors: {total_stats['errors']:,}")
    show_stats()
    return total_stats


# ============================================================================
# 통계 출력
# ============================================================================


def show_stats(store: Optional[EmbeddingStore] = None) -> None:
    """LanceDB 통계 출력"""
    if store is None:
        store = EmbeddingStore()

    print("\n" + "=" * 60)
    print("LanceDB Statistics")
    print("=" * 60)

    total = store.count()
    print(f"Total chunks: {total:,}")

    if total > 0:
        law_count = store.count_by_type("법령")
        precedent_count = store.count_by_type("판례")
        print("\nBy data_type:")
        print(f"  - 법령: {law_count:,}")
        print(f"  - 판례: {precedent_count:,}")


# ============================================================================
# 데이터 분할 유틸리티
# ============================================================================


def detect_json_format(file_path: str) -> str:
    """JSON 파일 형식 감지 (array 또는 object)"""
    with open(file_path, "rb") as f:
        first_char = f.read(1).decode("utf-8").strip()
        while first_char in ("\ufeff", " ", "\n", "\r", "\t", ""):
            first_char = f.read(1).decode("utf-8")
    return "array" if first_char == "[" else "object"


def split_precedents(
    source_path: str,
    chunk_size: int = 5000,
    output_dir: str = ".",
) -> list[str]:
    """판례 JSON을 작은 파일들로 분할 (스트리밍 방식)

    Args:
        source_path: 원본 JSON 파일 경로
        chunk_size: 파일당 항목 수 (기본: 5000)
        output_dir: 출력 디렉토리

    Returns:
        생성된 파일명 리스트
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not IJSON_AVAILABLE:
        print("[WARN] ijson not available. Using full load (higher memory usage).")
        return _split_full_load(
            source_path, chunk_size, output_path, prefix="precedents"
        )

    return _split_streaming(
        source_path, chunk_size, output_path, prefix="precedents"
    )


def split_laws(
    source_path: str,
    chunk_size: int = 2000,
    output_dir: str = ".",
) -> list[str]:
    """법령 JSON을 작은 파일들로 분할 (스트리밍 방식)

    Args:
        source_path: 원본 JSON 파일 경로
        chunk_size: 파일당 항목 수 (기본: 2000)
        output_dir: 출력 디렉토리

    Returns:
        생성된 파일명 리스트
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not IJSON_AVAILABLE:
        print("[WARN] ijson not available. Using full load (higher memory usage).")
        return _split_full_load(
            source_path, chunk_size, output_path, prefix="laws"
        )

    return _split_streaming(source_path, chunk_size, output_path, prefix="laws")


def _split_streaming(
    source_path: str,
    chunk_size: int,
    output_path: Path,
    prefix: str,
) -> list[str]:
    """ijson 스트리밍 분할 (공통)"""
    print(f"[INFO] Splitting {source_path} using streaming (memory-safe)...")

    json_format = detect_json_format(source_path)
    print(f"[INFO] JSON format: {json_format}")

    part_files: list[str] = []
    current_chunk: list[Any] = []
    part_num = 1
    total_count = 0

    with open(source_path, "rb") as f:
        if json_format == "array":
            parser = ijson.items(f, "item")
        else:
            parser = ijson.items(f, "items.item")

        for item in tqdm(parser, desc="Splitting"):
            current_chunk.append(item)
            total_count += 1

            if len(current_chunk) >= chunk_size:
                filename = f"{prefix}_part_{part_num:03d}.json"
                filepath = output_path / filename

                with open(filepath, "w", encoding="utf-8") as out:
                    json.dump(current_chunk, out, ensure_ascii=False)

                part_files.append(str(filepath))
                print(f"  - {filename}: {len(current_chunk):,} items")

                current_chunk = []
                part_num += 1
                clear_memory()

    if current_chunk:
        filename = f"{prefix}_part_{part_num:03d}.json"
        filepath = output_path / filename

        with open(filepath, "w", encoding="utf-8") as out:
            json.dump(current_chunk, out, ensure_ascii=False)

        part_files.append(str(filepath))
        print(f"  - {filename}: {len(current_chunk):,} items")

    clear_memory()
    print(
        f"\n[INFO] Split complete! {len(part_files)} files, "
        f"{total_count:,} items total."
    )
    return part_files


def _split_full_load(
    source_path: str,
    chunk_size: int,
    output_path: Path,
    prefix: str,
) -> list[str]:
    """ijson 없을 때 폴백: 전체 로드 방식"""
    print(f"[INFO] Loading {source_path} (full load)...")

    with open(source_path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        items = data.get("items", data.get("precedents", []))
    else:
        items = data

    total = len(items)
    num_parts = (total + chunk_size - 1) // chunk_size

    print(f"[INFO] Total items: {total:,}")
    print(f"[INFO] Splitting into {num_parts} parts ({chunk_size} items each)")

    part_files: list[str] = []
    for i in range(num_parts):
        start = i * chunk_size
        end = min(start + chunk_size, total)
        part_data = items[start:end]

        filename = f"{prefix}_part_{i + 1:03d}.json"
        filepath = output_path / filename

        with open(filepath, "w", encoding="utf-8") as out:
            json.dump(part_data, out, ensure_ascii=False)

        part_files.append(str(filepath))
        print(f"  - {filename}: {len(part_data):,} items")

    del data, items
    clear_memory()

    print(f"\n[INFO] Split complete! {num_parts} files created.")
    return part_files


# ============================================================================
# Jupyter/Notebook 환경 감지
# ============================================================================


def is_notebook() -> bool:
    """Jupyter/Colab/RunPod Notebook 환경인지 확인"""
    try:
        from IPython import get_ipython

        if get_ipython() is not None:
            return True
    except ImportError:
        pass
    return False


# ============================================================================
# CLI
# ============================================================================


def main() -> None:
    """CLI 메인 함수"""
    import argparse
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="LanceDB 임베딩 생성 (RunPod)")
    parser.add_argument(
        "--type", choices=["law", "precedent", "all"], default="all"
    )
    parser.add_argument("--law-source", type=str, default="law_v3.json")
    parser.add_argument(
        "--precedent-source", type=str, default="precedents_v2.json"
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--stats", action="store_true")

    args = parser.parse_args()

    print("=" * 60)
    print("LanceDB Embedding Creator (RunPod Edition)")
    print("=" * 60)

    if args.stats:
        show_stats()
        return

    if args.type in ("law", "all"):
        run_law_embedding(args.law_source, args.reset, args.batch_size)

    if args.type in ("precedent", "all"):
        run_precedent_embedding(
            args.precedent_source, args.reset, args.batch_size
        )


if __name__ == "__main__":
    if not is_notebook():
        main()
    else:
        print("=" * 60)
        print("LanceDB Embedding Creator (RunPod Edition)")
        print("=" * 60)

        device_info, optimal = print_device_info()

        print("\n[기본 함수]")
        print("  - print_device_info()           # 디바이스 정보")
        print("  - print_memory_status()         # 메모리 사용량 확인")
        print("  - show_stats()                  # 통계 확인")
        print("  - clear_model_cache()           # 모델 메모리 정리")
        print("  - clear_memory()                # GPU/CPU 메모리 정리")
        print("  - set_seed(42)                  # 랜덤 시드 고정")
        print("")
        print("[품질 검증]")
        print("  - checker = EmbeddingQualityChecker()")
        print("  - checker.quick_test()          # 법률 도메인 기본 테스트")
        print("")
        print("[임베딩 캐싱]")
        print("  - cache = EmbeddingCache('./cache')")
        print("  - cache.get_stats()             # 캐시 통계")
        print("")
        print("[분할 처리 (권장)]")
        print("  - split_precedents(path, chunk_size=5000)")
        print("  - split_laws(path, chunk_size=2000)")
        print("  - run_all_precedent_parts(pattern)  # 모든 판례 분할 파일")
        print("  - run_all_law_parts(pattern)        # 모든 법령 분할 파일")
        print("")
        print("[일반 처리 (소규모 데이터)]")
        print("  - run_law_embedding(path)       # 법령 임베딩")
        print("  - run_precedent_embedding(path) # 판례 임베딩")
        print("=" * 60)
