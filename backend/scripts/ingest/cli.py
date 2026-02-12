"""
인제스트 파이프라인 통합 CLI

Usage:
    cd backend

    # 전체 파이프라인 (DB+FTS → Vector+ANN)
    uv run python -m scripts.ingest.cli --type precedent --step all

    # 단계별 실행
    uv run python -m scripts.ingest.cli --type precedent --step db       # PostgreSQL + FTS
    uv run python -m scripts.ingest.cli --type precedent --step vector   # LanceDB + ANN
    uv run python -m scripts.ingest.cli --type precedent --step fts      # FTS만 재빌드
    uv run python -m scripts.ingest.cli --type precedent --step index    # ANN 인덱스만 재빌드

    # 옵션
    --reset         # 기존 데이터 삭제 후 재실행
    --source PATH   # 커스텀 JSON 경로
    --batch-size N  # 배치 크기 (기본: 1000)
    --stats         # 통계만 출력
    --verify        # 검증만 실행
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# 백엔드 app 모듈 import를 위한 경로 추가
_backend_root = Path(__file__).parent.parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))

# 타입 등록을 위해 types 패키지 import (자동 등록)
import scripts.ingest.types  # noqa: F401
from scripts.ingest.config import get_config, list_configs
from scripts.ingest.db_writer import run_db_ingest, verify_db
from scripts.ingest.fts_builder import run_fts_rebuild
from scripts.ingest.vector_writer import build_ann_index, run_vector_ingest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

VALID_STEPS = ("all", "db", "vector", "fts", "index")
BATCH_SIZE_DB = 1000


def _print_stats(config_name: str) -> None:
    """LanceDB + PostgreSQL 통계 출력"""
    config = get_config(config_name)

    print(f"\n{'=' * 60}")
    print(f"  인제스트 통계: {config.data_type_label} ({config.name})")
    print(f"{'=' * 60}")

    # PostgreSQL
    try:
        result = verify_db(config)
        print("\n  [PostgreSQL]")
        print(f"    ORM 테이블: {result['orm_count']:,}건")
        print(f"    FTS 인덱스: {result['fts_count']:,}건")
        print(
            f"    tsvector 보유: {result['fts_with_tsvector']:,}/{result['fts_count']:,}"
            f" ({result['fts_with_tsvector'] / result['fts_count'] * 100:.1f}%)"
            if result["fts_count"]
            else "    tsvector 보유: 0/0"
        )
    except Exception as e:
        print(f"\n  [PostgreSQL] 조회 실패: {e}")

    # LanceDB
    try:
        from scripts.embedding_common.store import EmbeddingStore

        store = EmbeddingStore()
        total = store.count()
        by_type = store.count_by_type(config.data_type_label)
        print("\n  [LanceDB]")
        print(f"    전체 레코드: {total:,}건")
        print(f"    {config.data_type_label}: {by_type:,}건")
    except Exception as e:
        print(f"\n  [LanceDB] 조회 실패: {e}")

    print(f"\n{'=' * 60}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="인제스트 파이프라인 CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 판례 전체 파이프라인
  uv run python -m scripts.ingest.cli --type precedent --step all --reset

  # DB+FTS만
  uv run python -m scripts.ingest.cli --type precedent --step db

  # 벡터만
  uv run python -m scripts.ingest.cli --type precedent --step vector

  # FTS 재빌드 (토크나이저 변경 후)
  uv run python -m scripts.ingest.cli --type precedent --step fts --reset

  # ANN 인덱스만 재빌드
  uv run python -m scripts.ingest.cli --type precedent --step index
        """,
    )

    available_types = list_configs()
    parser.add_argument(
        "--type",
        choices=available_types,
        required=True,
        help=f"인제스트 대상 타입 ({', '.join(available_types)})",
    )
    parser.add_argument(
        "--step",
        choices=VALID_STEPS,
        default="all",
        help="실행 단계 (default: all)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="커스텀 JSON 소스 경로",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="배치 크기 (DB: 기본 1000, 벡터: 미지정 시 하드웨어 자동)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="기존 데이터 삭제 후 재실행",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="통계만 출력",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="검증만 실행",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="임베딩 디바이스 (cuda, mps, cpu, 기본: 자동)",
    )
    parser.add_argument(
        "--profile",
        choices=["desktop", "laptop", "mac", "cpu"],
        default=None,
        help="하드웨어 프로필 (기본: 자동 감지)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="임베딩 캐시 비활성화",
    )

    args = parser.parse_args()

    config = get_config(args.type)
    source_path = Path(args.source) if args.source else None

    # 통계 모드
    if args.stats:
        _print_stats(args.type)
        return

    # 검증 모드
    if args.verify:
        verify_db(config)
        return

    # 실행
    print(f"\n{'=' * 60}")
    print(f"  인제스트 파이프라인: {config.data_type_label}")
    print(f"{'=' * 60}")
    print(f"  타입: {config.name}")
    print(f"  단계: {args.step}")
    print(f"  소스: {source_path or config.source_path}")
    print(f"  배치: {args.batch_size or '자동'}")
    print(f"  리셋: {args.reset}")
    print(f"  프로필: {args.profile or '자동'}")
    print(f"  캐시: {'비활성' if args.no_cache else '활성'}")
    print(f"{'=' * 60}\n")

    overall_start = time.time()
    results: dict[str, dict[str, int]] = {}

    # Step: DB + FTS
    if args.step in ("all", "db"):
        logger.info("=== DB + FTS 적재 시작 ===")
        db_batch = args.batch_size or BATCH_SIZE_DB
        results["db"] = run_db_ingest(
            config=config,
            source_path=source_path,
            reset=args.reset,
            batch_size=db_batch,
        )

    # Step: Vector (LanceDB)
    if args.step in ("all", "vector"):
        logger.info("=== 벡터 임베딩 시작 ===")
        # 명시 시 64 상한 적용, 미지정 시 None(하드웨어 자동)
        vector_batch = min(args.batch_size, 64) if args.batch_size else None
        results["vector"] = run_vector_ingest(
            config=config,
            source_path=source_path,
            reset=args.reset,
            batch_size=vector_batch,
            device=args.device,
            profile=args.profile,
            use_cache=not args.no_cache,
        )

    # Step: FTS 재빌드
    if args.step == "fts":
        logger.info("=== FTS 재빌드 시작 ===")
        results["fts"] = run_fts_rebuild(
            config=config,
            reset=args.reset,
        )

    # Step: ANN 인덱스
    if args.step in ("all", "index"):
        logger.info("=== ANN 인덱스 빌드 ===")
        build_ann_index()

    # 결과 출력
    overall_elapsed = time.time() - overall_start

    print(f"\n{'=' * 60}")
    print(f"  인제스트 결과: {config.data_type_label}")
    print(f"{'=' * 60}")

    for step_name, step_stats in results.items():
        print(f"\n  [{step_name}]")
        for key, value in step_stats.items():
            print(f"    {key}: {value:,}")

    print(f"\n  총 소요 시간: {overall_elapsed:.1f}초")
    print(f"{'=' * 60}")

    # 최종 검증
    _print_stats(args.type)


if __name__ == "__main__":
    main()
