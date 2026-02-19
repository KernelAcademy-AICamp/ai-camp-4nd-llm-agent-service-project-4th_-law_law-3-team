#!/usr/bin/env python3
"""
LanceDB 임베딩 생성 스크립트 (Google Colab 전용 - thin wrapper)

runpod_lancedb_embeddings.py의 Public API를 re-export하고
Google Drive 마운트 유틸을 추가합니다.

사용법 (Colab):
    1. 셀 1: 패키지 설치
       !pip install lancedb sentence-transformers pyarrow ijson psutil tqdm pyyaml sqlalchemy pydantic-settings -q

    2. 셀 2: import
       import sys; sys.path.insert(0, '/content/backend')
       from scripts.colab_lancedb_embeddings import *
       setup_google_drive()

    3. 셀 3: 임베딩 생성
       run_law_embedding('law_v3.json', reset=True)
       run_precedent_embedding('precedents_v2.json', reset=True)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 백엔드 모듈 import를 위한 경로 설정
_BACKEND_ROOT = Path(__file__).parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

# ============================================================================
# runpod wrapper의 전체 Public API re-export
# ============================================================================

from scripts.runpod_lancedb_embeddings import (  # noqa: F401, E402, I001
    EmbeddingCache,
    EmbeddingQualityChecker,
    EmbeddingStore,
    clear_memory,
    clear_model_cache,
    create_embeddings,
    detect_json_format,
    get_device,
    get_embedding_model,
    is_notebook,
    print_device_info,
    print_memory_status,
    run_all_law_parts,
    run_all_precedent_parts,
    run_law_embedding,
    run_law_embedding_part,
    run_precedent_embedding,
    run_precedent_embedding_part,
    set_seed,
    show_stats,
    split_laws,
    split_precedents,
)


# ============================================================================
# Colab 특화 유틸리티
# ============================================================================


def setup_google_drive(
    mount_path: str = "/content/drive",
    lancedb_dir: str = "MyDrive/lancedb_data",
) -> str:
    """Google Drive 마운트 + LANCEDB_URI 설정

    Args:
        mount_path: Drive 마운트 경로
        lancedb_dir: Drive 내 LanceDB 저장 디렉토리

    Returns:
        설정된 LANCEDB_URI 경로
    """
    try:
        from google.colab import drive  # type: ignore[import-untyped]

        drive.mount(mount_path)
    except ImportError:
        print("[WARN] google.colab 미설치 (Colab 환경이 아닐 수 있음)")
        print("[INFO] 로컬 lancedb_data/ 사용")
        return os.getenv("LANCEDB_URI", "./lancedb_data")

    lancedb_path = f"{mount_path}/{lancedb_dir}"
    Path(lancedb_path).mkdir(parents=True, exist_ok=True)
    os.environ["LANCEDB_URI"] = lancedb_path

    print(f"[INFO] LANCEDB_URI = {lancedb_path}")
    return lancedb_path


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

    parser = argparse.ArgumentParser(
        description="LanceDB 임베딩 생성 (Colab)"
    )
    parser.add_argument(
        "--type", choices=["law", "precedent", "all"], default="all"
    )
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--stats", action="store_true")

    args = parser.parse_args()

    print("=" * 60)
    print("LanceDB Embedding Creator (Colab Edition)")
    print("=" * 60)

    if args.stats:
        show_stats()
        return

    if args.type in ("law", "all"):
        source = args.source or "law_v3.json"
        run_law_embedding(source, args.reset, args.batch_size)

    if args.type in ("precedent", "all"):
        source = args.source or "precedents_v2.json"
        run_precedent_embedding(source, args.reset, args.batch_size)


if __name__ == "__main__":
    if not is_notebook():
        main()
    else:
        print("=" * 60)
        print("LanceDB Embedding Creator (Colab Edition)")
        print("=" * 60)
        print("\n1. setup_google_drive()  # Drive 마운트 + LANCEDB_URI 설정")
        print("2. run_law_embedding(path, reset=True)")
        print("3. run_precedent_embedding(path, reset=True)")
        print("4. show_stats()")
        print("=" * 60)
