"""
RAG 평가 시스템 CLI 진입점

사용법:
    # Gradio UI 실행
    uv run python -m evaluation

    # Solar 자동 생성
    uv run python -m evaluation.tools.solar_generator --count 30

    # 평가 실행
    uv run python -m evaluation.runners.evaluation_runner --dataset eval_dataset_v1.json

    # 데이터셋 검증
    uv run python -m evaluation.tools.validate_dataset eval_dataset_v1.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="RAG 평가 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # Gradio UI 실행 (기본)
  uv run python -m evaluation

  # 특정 포트로 실행
  uv run python -m evaluation --port 7861

  # 공유 링크 생성
  uv run python -m evaluation --share

서브커맨드:
  solar      Upstage Solar 자동 질문 생성
  evaluate   평가 실행
  validate   데이터셋 검증
  report     리포트 생성
        """,
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Gradio 호스트 (기본: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Gradio 포트 (기본: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="공유 링크 생성",
    )

    args = parser.parse_args()

    print("=" * 50)
    print("RAG 평가 시스템 - Gradio UI")
    print("=" * 50)
    print()
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"URL: http://localhost:{args.port}")
    print()
    print("시작 중...")
    print()

    from evaluation.ui.gradio_app import launch_app
    launch_app(host=args.host, port=args.port, share=args.share)


if __name__ == "__main__":
    main()
