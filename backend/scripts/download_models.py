#!/usr/bin/env python3
"""
임베딩 모델 사전 다운로드 스크립트

서버 실행 전에 필요한 모델을 미리 다운로드합니다.

Usage:
    uv run python scripts/download_models.py
    uv run python scripts/download_models.py --model nlpai-lab/KURE-v1
    uv run python scripts/download_models.py --check  # 캐시 상태만 확인
"""

import argparse
import sys
from pathlib import Path

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


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

    # 읽기 좋은 형태로 변환
    for unit in ["B", "KB", "MB", "GB"]:
        if total_size < 1024:
            return f"{total_size:.1f} {unit}"
        total_size /= 1024

    return f"{total_size:.1f} TB"


def download_model(model_name: str, cache_dir: Path, force: bool = False) -> bool:
    """
    모델 다운로드

    Args:
        model_name: HuggingFace 모델명
        cache_dir: 캐시 디렉토리
        force: True면 이미 캐시되어 있어도 재다운로드

    Returns:
        True if 성공
    """
    from sentence_transformers import SentenceTransformer

    cache_dir.mkdir(parents=True, exist_ok=True)

    # 이미 캐시되어 있으면 스킵 (force가 아닐 때)
    if not force and check_model_cached(model_name, cache_dir):
        print(f"✓ 모델이 이미 캐시되어 있습니다: {model_name}")
        print(f"  캐시 크기: {get_cache_size(model_name, cache_dir)}")
        return True

    print(f"⏳ 모델 다운로드 중: {model_name}")
    print(f"   캐시 경로: {cache_dir}")
    print("   (수 GB의 모델이므로 네트워크 상태에 따라 시간이 걸릴 수 있습니다)")
    print()

    try:
        # 불완전한 다운로드 파일 정리
        sanitized = model_name.replace("/", "--")
        model_path = cache_dir / f"models--{sanitized}"
        if model_path.exists():
            blobs_dir = model_path / "blobs"
            if blobs_dir.exists():
                for file in blobs_dir.iterdir():
                    if file.name.endswith(".incomplete"):
                        print(f"   불완전한 파일 삭제: {file.name}")
                        file.unlink()

        # 모델 다운로드 (SentenceTransformer 초기화 시 자동 다운로드)
        model = SentenceTransformer(
            model_name,
            cache_folder=str(cache_dir),
            trust_remote_code=True,
        )

        # 간단한 테스트
        test_embedding = model.encode("테스트 문장", show_progress_bar=False)
        print()
        print(f"✓ 모델 다운로드 완료: {model_name}")
        print(f"  임베딩 차원: {len(test_embedding)}")
        print(f"  캐시 크기: {get_cache_size(model_name, cache_dir)}")

        return True

    except Exception as e:
        print(f"✗ 모델 다운로드 실패: {e}")
        return False


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="임베딩 모델 사전 다운로드",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
    uv run python scripts/download_models.py
    uv run python scripts/download_models.py --model nlpai-lab/KURE-v1
    uv run python scripts/download_models.py --check
    uv run python scripts/download_models.py --force  # 재다운로드
        """,
    )
    parser.add_argument(
        "--model",
        default="nlpai-lab/KURE-v1",
        help="다운로드할 모델명 (기본값: nlpai-lab/KURE-v1)",
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
    print("임베딩 모델 다운로드 스크립트")
    print("=" * 50)
    print()

    if args.check:
        # 캐시 상태 확인만
        is_cached = check_model_cached(args.model, cache_dir)
        print(f"모델: {args.model}")
        print(f"캐시 경로: {cache_dir}")
        print(f"캐시 상태: {'✓ 완료' if is_cached else '✗ 없음 또는 불완전'}")
        if is_cached:
            print(f"캐시 크기: {get_cache_size(args.model, cache_dir)}")
        return 0 if is_cached else 1

    # 모델 다운로드
    success = download_model(args.model, cache_dir, force=args.force)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
