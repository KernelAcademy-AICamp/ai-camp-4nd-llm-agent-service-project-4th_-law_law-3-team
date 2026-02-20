"""
인제스트 파이프라인 통합 CLI

Usage:
    cd backend

    # 전체 타입 × 전체 파이프라인
    uv run python -m scripts.ingest.cli --type all --step all --reset

    # 특정 타입 전체 파이프라인 (DB+FTS → Vector+ANN)
    uv run python -m scripts.ingest.cli --type precedent --step all

    # 단계별 실행
    uv run python -m scripts.ingest.cli --type precedent --step db       # PostgreSQL + FTS
    uv run python -m scripts.ingest.cli --type precedent --step vector   # LanceDB + ANN
    uv run python -m scripts.ingest.cli --type precedent --step fts      # FTS만 재빌드
    uv run python -m scripts.ingest.cli --type precedent --step index    # ANN 인덱스만 재빌드

    # 옵션
    --reset         # 기존 데이터 삭제 후 재실행
    --source PATH   # 커스텀 JSON 경로 (단일 타입만)
    --batch-size N  # 배치 크기 (기본: 1000)
    --stats         # 통계만 출력
    --verify        # 검증만 실행
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# 백엔드 app 모듈 import를 위한 경로 추가
_backend_root = Path(__file__).parent.parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))

# 타입 등록을 위해 types 패키지 import (자동 등록)
import scripts.ingest.types  # noqa: F401
from scripts.common.logging_config import setup_logging
from scripts.ingest.config import IngestConfig, get_config, list_configs
from scripts.ingest.db_writer import run_db_ingest, verify_db
from scripts.ingest.fts_builder import run_fts_rebuild
from scripts.ingest.vector_writer import build_ann_index, run_vector_ingest

logger = setup_logging(__name__)

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

        if config.name == "local_ordinance":
            from app.tools.vectorstore.local_ordinance_schema import (  # noqa: I001
                LOCAL_ORDINANCE_SCHEMA,
                TABLE_NAME as LO_TABLE,
            )

            lo_store = EmbeddingStore(
                table_name=LO_TABLE, schema=LOCAL_ORDINANCE_SCHEMA
            )
            lo_total = lo_store.count()
            print("\n  [LanceDB - local_ordinance_chunks]")
            print(f"    전체 레코드: {lo_total:,}건")
        else:
            store = EmbeddingStore()
            total = store.count()
            by_type = store.count_by_type(config.data_type_label)
            print("\n  [LanceDB]")
            print(f"    전체 레코드: {total:,}건")
            print(f"    {config.data_type_label}: {by_type:,}건")
    except Exception as e:
        print(f"\n  [LanceDB] 조회 실패: {e}")

    print(f"\n{'=' * 60}")


def _run_single_type(
    config: IngestConfig,
    step: str,
    source_path: Path | None,
    batch_size: int | None,
    reset: bool,
    device: str | None,
    profile: str | None,
    no_cache: bool,
) -> dict[str, dict[str, int]]:
    """단일 타입 인제스트 실행. 결과 dict 반환."""
    results: dict[str, dict[str, int]] = {}

    # Step: DB + FTS
    if step in ("all", "db"):
        logger.info("=== DB + FTS 적재 시작 ===")
        db_batch = batch_size or BATCH_SIZE_DB
        results["db"] = run_db_ingest(
            config=config,
            source_path=source_path,
            reset=reset,
            batch_size=db_batch,
        )

    # Step: Vector (LanceDB)
    if step in ("all", "vector"):
        logger.info("=== 벡터 임베딩 시작 ===")
        # 명시 시 64 상한 적용, 미지정 시 None(하드웨어 자동)
        vector_batch = min(batch_size, 64) if batch_size else None

        if config.name == "local_ordinance":
            # 자치법규: 전용 라이터 (1문서 → 다중 벡터)
            from scripts.ingest.local_ordinance_vector_writer import (
                run_local_ordinance_vector_ingest,
            )

            results["vector"] = run_local_ordinance_vector_ingest(
                config=config,
                source_path=source_path,
                reset=reset,
                batch_size=vector_batch,
                device=device,
                profile=profile,
                use_cache=not no_cache,
            )
        else:
            results["vector"] = run_vector_ingest(
                config=config,
                source_path=source_path,
                reset=reset,
                batch_size=vector_batch,
                device=device,
                profile=profile,
                use_cache=not no_cache,
            )

    # Step: FTS 재빌드
    if step == "fts":
        logger.info("=== FTS 재빌드 시작 ===")
        results["fts"] = run_fts_rebuild(
            config=config,
            reset=reset,
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="인제스트 파이프라인 CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 전체 타입 전체 파이프라인
  uv run python -m scripts.ingest.cli --type all --step all --reset

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
    type_choices = ["all"] + available_types
    parser.add_argument(
        "--type",
        choices=type_choices,
        required=True,
        help=f"인제스트 대상 타입 (all: 전체, {', '.join(available_types)})",
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
        help="커스텀 JSON 소스 경로 (단일 타입만 사용 가능)",
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

    # --type all + --source 조합 차단
    if args.type == "all" and args.source:
        parser.error("--source는 단일 타입에서만 사용 가능합니다 (--type all과 함께 사용 불가)")

    # 대상 타입 목록 결정
    target_types = available_types if args.type == "all" else [args.type]
    source_path = Path(args.source) if args.source else None

    # 통계 모드
    if args.stats:
        for type_name in target_types:
            _print_stats(type_name)
        return

    # 검증 모드
    if args.verify:
        for type_name in target_types:
            config = get_config(type_name)
            verify_db(config)
        return

    # ===== 실행 =====
    overall_start = time.time()
    succeeded: list[str] = []
    failed: list[tuple[str, str]] = []

    print(f"\n{'=' * 60}")
    print("  인제스트 파이프라인")
    print(f"{'=' * 60}")
    print(f"  대상: {args.type} ({len(target_types)}개 타입)")
    print(f"  단계: {args.step}")
    print(f"  리셋: {args.reset}")
    print(f"  프로필: {args.profile or '자동'}")
    print(f"  캐시: {'비활성' if args.no_cache else '활성'}")
    print(f"{'=' * 60}\n")

    for i, type_name in enumerate(target_types, 1):
        config = get_config(type_name)

        print(f"\n{'─' * 60}")
        print(f"  [{i}/{len(target_types)}] {config.data_type_label} ({type_name})")
        print(f"  소스: {source_path or config.source_path}")
        print(f"{'─' * 60}")

        type_start = time.time()
        try:
            results = _run_single_type(
                config=config,
                step=args.step,
                source_path=source_path,
                batch_size=args.batch_size,
                reset=args.reset,
                device=args.device,
                profile=args.profile,
                no_cache=args.no_cache,
            )

            # 타입별 결과 출력
            for step_name, step_stats in results.items():
                for key, value in step_stats.items():
                    logger.info("  %s.%s: %s", step_name, key, f"{value:,}")

            type_elapsed = time.time() - type_start
            logger.info("  %s 완료 (%.1f초)", type_name, type_elapsed)
            succeeded.append(type_name)

        except Exception as e:
            type_elapsed = time.time() - type_start
            logger.error("  %s 실패 (%.1f초): %s", type_name, type_elapsed, e)
            failed.append((type_name, str(e)))

    # ANN 인덱스는 전체 LanceDB 대상이므로 마지막에 1회만 실행
    if args.step in ("all", "index") and succeeded:
        logger.info("=== ANN 인덱스 빌드 ===")
        try:
            build_ann_index()
        except Exception as e:
            logger.error("ANN 인덱스 빌드 실패: %s", e)

        # 자치법규 별도 테이블 ANN 인덱스
        if "local_ordinance" in succeeded:
            try:
                from scripts.ingest.local_ordinance_vector_writer import (
                    build_local_ordinance_ann_index,
                )

                build_local_ordinance_ann_index()
            except Exception as e:
                logger.error("자치법규 ANN 인덱스 빌드 실패: %s", e)

    # ===== 최종 요약 =====
    overall_elapsed = time.time() - overall_start

    print(f"\n{'=' * 60}")
    print("  인제스트 완료 요약")
    print(f"{'=' * 60}")
    print(f"  성공: {len(succeeded)}/{len(target_types)}개 타입")

    if failed:
        print(f"  실패: {len(failed)}개 타입")
        for name, err in failed:
            print(f"    - {name}: {err}")

    print(f"  총 소요 시간: {overall_elapsed:.1f}초")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
