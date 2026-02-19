"""행정규칙 조문요약 플랫 추출 스크립트

중첩된 조문내용[].조문요약을 플랫 JSON으로 추출하여
기존 audit_summary_quality.py에 전달 가능하게 변환합니다.

Usage:
    cd backend
    uv run python scripts/extract_article_summaries.py
    uv run python scripts/extract_article_summaries.py --input ../data/admin_rule_v3.json
    uv run python scripts/extract_article_summaries.py --output ../data/temp_article_summaries.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_INPUT = _PROJECT_ROOT / "data" / "admin_rule_v3.json"
_DEFAULT_OUTPUT = _PROJECT_ROOT / "data" / "temp_article_summaries.json"


def extract_article_summaries(
    data: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """중첩된 조문내용[].조문요약을 플랫 리스트로 추출

    Args:
        data: 행정규칙 레코드 리스트

    Returns:
        플랫 추출된 조문요약 리스트
    """
    flat_items: list[dict[str, Any]] = []
    skipped_no_articles = 0
    skipped_no_summary = 0

    for item in data:
        rule_id = str(item.get("행정규칙ID", ""))
        rule_name = str(item.get("행정규칙명", ""))
        articles = item.get("조문내용", [])

        if not isinstance(articles, list) or not articles:
            skipped_no_articles += 1
            continue

        for article in articles:
            if not isinstance(article, dict):
                continue

            article_summary = article.get("조문요약")
            article_number = str(article.get("조문번호", ""))

            if article_summary is None or (
                isinstance(article_summary, str) and not article_summary.strip()
            ):
                skipped_no_summary += 1
                continue

            flat_items.append({
                "id": f"{rule_id}_{article_number}",
                "조문요약": str(article_summary),
                "행정규칙명": rule_name,
                "행정규칙ID": rule_id,
                "조문번호": article_number,
            })

    logger.info(
        "추출 완료: %d건 (조문내용 없음: %d, 조문요약 없음: %d)",
        len(flat_items),
        skipped_no_articles,
        skipped_no_summary,
    )
    return flat_items


def main() -> None:
    parser = argparse.ArgumentParser(
        description="행정규칙 조문요약 플랫 추출",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(_DEFAULT_INPUT),
        help=f"입력 JSON 파일 경로 (기본: {_DEFAULT_INPUT.name})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(_DEFAULT_OUTPUT),
        help=f"출력 JSON 파일 경로 (기본: {_DEFAULT_OUTPUT.name})",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    if not input_path.exists():
        logger.error("입력 파일이 존재하지 않습니다: %s", input_path)
        sys.exit(1)

    logger.info("입력: %s", input_path)

    with open(input_path, encoding="utf-8") as f:
        data: list[dict[str, Any]] = json.load(f)

    logger.info("원본 레코드: %d건", len(data))

    flat_items = extract_article_summaries(data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(flat_items, f, ensure_ascii=False, indent=2)

    logger.info("저장: %s (%d건)", output_path, len(flat_items))


if __name__ == "__main__":
    main()
