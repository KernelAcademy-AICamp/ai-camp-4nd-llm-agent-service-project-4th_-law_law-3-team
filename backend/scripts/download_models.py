#!/usr/bin/env python3
"""
모델 사전 다운로드 스크립트

서버 실행 전에 필요한 임베딩/리랭커 모델을 미리 다운로드합니다.

Usage:
    uv run python scripts/download_models.py           # 전체 모델 다운로드
    uv run python scripts/download_models.py --check    # 캐시 상태만 확인
    uv run python scripts/download_models.py --model nlpai-lab/KURE-v1  # 특정 모델만
    uv run python scripts/download_models.py --embedding-only  # 임베딩만
    uv run python scripts/download_models.py --reranker-only   # 리랭커만
"""

import argparse
import sys
from pathlib import Path

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 기본 모델 목록
DEFAULT_EMBEDDING_MODEL = "nlpai-lab/KURE-v1"
DEFAULT_RERANKER_MODEL = "dragonkue/bge-reranker-v2-m3-ko"


def get_cache_dir() -> Path:
    """모델 캐시 디렉토리 반환"""
    return PROJECT_ROOT / "data" / "models"


def check_model_cached(model_name: str, cache_dir: Path) -> bool:
    """모델이 완전히 캐시되어 있는지 확인"""
    sanitized = model_name.replace("/", "--")
    model_path = cache_dir / f"models--{sanitized}"

    if not model_path.exists():
        return False

    # .incomplete 파일 확인
    blobs_dir = model_path / "blobs"
    if blobs_dir.exists():
        for file in blobs_dir.iterdir():
            if file.name.endswith(".incomplete"):
                return False

    # snapshots 확인
    snapshots_dir = model_path / "snapshots"
    if not snapshots_dir.exists():
        return False

    return len(list(snapshots_dir.iterdir())) > 0


def get_cache_size(model_name: str, cache_dir: Path) -> str:
    """캐시된 모델 크기 반환"""
    sanitized = model_name.replace("/", "--")
    model_path = cache_dir / f"models--{sanitized}"

    if not model_path.exists():
        return "0 B"

    total_size = 0
    for file in model_path.rglob("*"):
        if file.is_file():
            total_size += file.stat().st_size

    for unit in ["B", "KB", "MB", "GB"]:
        if total_size < 1024:
            return f"{total_size:.1f} {unit}"
        total_size /= 1024

    return f"{total_size:.1f} TB"


def _clean_incomplete_blobs(model_name: str, cache_dir: Path) -> None:
    """불완전한 다운로드 파일 정리"""
    sanitized = model_name.replace("/", "--")
    model_path = cache_dir / f"models--{sanitized}"
    if not model_path.exists():
        return
    blobs_dir = model_path / "blobs"
    if not blobs_dir.exists():
        return
    for file in blobs_dir.iterdir():
        if file.name.endswith(".incomplete"):
            print(f"   불완전한 파일 삭제: {file.name}")
            file.unlink()


def download_embedding_model(
    model_name: str, cache_dir: Path, force: bool = False
) -> bool:
    """임베딩 모델(SentenceTransformer) 다운로드"""
    from sentence_transformers import SentenceTransformer

    cache_dir.mkdir(parents=True, exist_ok=True)

    if not force and check_model_cached(model_name, cache_dir):
        print(f"  ✓ 이미 캐시됨: {model_name} ({get_cache_size(model_name, cache_dir)})")
        return True

    print(f"  ⏳ 다운로드 중: {model_name}")
    print(f"     캐시 경로: {cache_dir}")
    print()

    try:
        _clean_incomplete_blobs(model_name, cache_dir)

        model = SentenceTransformer(
            model_name,
            cache_folder=str(cache_dir),
            trust_remote_code=True,
        )

        test_embedding = model.encode("테스트 문장", show_progress_bar=False)
        print(f"  ✓ 완료: {model_name}")
        print(f"    임베딩 차원: {len(test_embedding)}")
        print(f"    캐시 크기: {get_cache_size(model_name, cache_dir)}")
        return True

    except Exception as e:
        print(f"  ✗ 실패: {e}")
        return False


def download_reranker_model(
    model_name: str, cache_dir: Path, force: bool = False
) -> bool:
    """리랭커 모델(CrossEncoder) 다운로드"""
    import torch
    from sentence_transformers import CrossEncoder

    cache_dir.mkdir(parents=True, exist_ok=True)

    if not force and check_model_cached(model_name, cache_dir):
        print(f"  ✓ 이미 캐시됨: {model_name} ({get_cache_size(model_name, cache_dir)})")
        return True

    print(f"  ⏳ 다운로드 중: {model_name}")
    print(f"     캐시 경로: {cache_dir}")
    print()

    try:
        _clean_incomplete_blobs(model_name, cache_dir)

        model = CrossEncoder(
            model_name,
            cache_folder=str(cache_dir),
            activation_fn=torch.nn.Sigmoid(),
        )

        test_scores = model.predict([("테스트 쿼리", "테스트 문서")])
        print(f"  ✓ 완료: {model_name}")
        print(f"    테스트 점수: {float(test_scores[0]):.4f}")
        print(f"    캐시 크기: {get_cache_size(model_name, cache_dir)}")
        return True

    except Exception as e:
        print(f"  ✗ 실패: {e}")
        return False


def check_all_models(cache_dir: Path) -> int:
    """전체 모델 캐시 상태 확인"""
    models = [
        ("임베딩", DEFAULT_EMBEDDING_MODEL),
        ("리랭커", DEFAULT_RERANKER_MODEL),
    ]

    all_cached = True
    for label, model_name in models:
        is_cached = check_model_cached(model_name, cache_dir)
        status = "✓ 완료" if is_cached else "✗ 없음 또는 불완전"
        size = f" ({get_cache_size(model_name, cache_dir)})" if is_cached else ""
        print(f"  [{label}] {model_name}: {status}{size}")
        if not is_cached:
            all_cached = False

    return 0 if all_cached else 1


def main() -> int:
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="임베딩/리랭커 모델 사전 다운로드",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
    uv run python scripts/download_models.py              # 전체 다운로드
    uv run python scripts/download_models.py --check      # 캐시 상태 확인
    uv run python scripts/download_models.py --embedding-only  # 임베딩만
    uv run python scripts/download_models.py --reranker-only   # 리랭커만
    uv run python scripts/download_models.py --model nlpai-lab/KURE-v1  # 특정 모델
    uv run python scripts/download_models.py --force      # 재다운로드
        """,
    )
    parser.add_argument(
        "--model",
        default=None,
        help="특정 모델만 다운로드 (SentenceTransformer로 로드)",
    )
    parser.add_argument(
        "--embedding-only",
        action="store_true",
        help="임베딩 모델만 다운로드",
    )
    parser.add_argument(
        "--reranker-only",
        action="store_true",
        help="리랭커 모델만 다운로드",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="캐시 상태만 확인 (다운로드 안함)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="이미 캐시되어 있어도 재다운로드",
    )

    args = parser.parse_args()
    cache_dir = get_cache_dir()

    print("=" * 50)
    print("모델 다운로드 스크립트")
    print(f"캐시 경로: {cache_dir}")
    print("=" * 50)
    print()

    if args.check:
        return check_all_models(cache_dir)

    # --model: 특정 모델 (기존 하위 호환)
    if args.model:
        print(f"[임베딩] {args.model}")
        success = download_embedding_model(args.model, cache_dir, force=args.force)
        return 0 if success else 1

    results: list[bool] = []

    # 임베딩 모델
    if not args.reranker_only:
        print(f"[1/2] 임베딩: {DEFAULT_EMBEDDING_MODEL}")
        results.append(
            download_embedding_model(DEFAULT_EMBEDDING_MODEL, cache_dir, force=args.force)
        )
        print()

    # 리랭커 모델
    if not args.embedding_only:
        step = "1/1" if args.reranker_only else "2/2"
        print(f"[{step}] 리랭커: {DEFAULT_RERANKER_MODEL}")
        results.append(
            download_reranker_model(DEFAULT_RERANKER_MODEL, cache_dir, force=args.force)
        )
        print()

    # 결과 요약
    print("=" * 50)
    if all(results):
        print("✓ 전체 모델 준비 완료")
        return 0
    else:
        failed = results.count(False)
        print(f"✗ {failed}개 모델 다운로드 실패")
        return 1


if __name__ == "__main__":
    sys.exit(main())
