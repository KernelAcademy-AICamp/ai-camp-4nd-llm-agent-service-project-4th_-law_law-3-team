"""
그래프 데이터를 PostgreSQL로 로드하는 스크립트 (레거시 래퍼)

실제 로직은 scripts/ingest/graph_writer.py에 있습니다.

사용법 (권장):
    cd backend
    uv run python -m scripts.ingest.cli --step graph

레거시:
    uv run python scripts/load_graph_data.py
    uv run python scripts/load_graph_data.py --verify
    uv run python scripts/load_graph_data.py --reset
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

env_path = PROJECT_ROOT / ".env"
load_dotenv(env_path)

from scripts.common.logging_config import setup_logging  # noqa: E402
from scripts.ingest.graph_writer import run_graph_ingest, verify_graph  # noqa: E402

logger = setup_logging(__name__, level=logging.INFO)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load graph data to PostgreSQL (레거시 래퍼)",
        epilog="권장: uv run python -m scripts.ingest.cli --step graph",
    )
    parser.add_argument("--verify", action="store_true", help="검증만 수행")
    parser.add_argument("--reset", action="store_true", help="기존 데이터 삭제 후 재로드")
    args = parser.parse_args()

    if args.verify:
        results = verify_graph()
        for key, count in results.items():
            print(f"  {key}: {count:,}")
        return

    stats = run_graph_ingest(reset=args.reset)

    print("\n=== Graph data loading complete ===")
    for key, count in stats.items():
        print(f"  {key}: {count:,}")


if __name__ == "__main__":
    main()
