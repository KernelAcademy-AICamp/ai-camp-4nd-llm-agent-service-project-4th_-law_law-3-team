"""자치법규 조문요약 평탄화 + 감사 스크립트

local_rules_v1.json의 중첩된 조[] 배열에서 조문요약을 추출하여
audit_summary_quality.py의 감사 함수를 직접 호출합니다.

중간 파일을 디스크에 쓰지 않고, 메모리에서 직접 감사합니다.
(2.3M 조문 × id+summary ≈ ~1.5GB RAM)

Usage:
    cd backend

    # 감사 실행 (기본)
    uv run python scripts/flatten_local_rules_summaries.py

    # JSON 보고서 저장
    uv run python scripts/flatten_local_rules_summaries.py \
        --output eda_output/audit_local_rules_article.json

    # 통계만 (감사 없이 평탄화 통계만)
    uv run python scripts/flatten_local_rules_summaries.py --stats

    # 샘플 수 조정
    uv run python scripts/flatten_local_rules_summaries.py --samples 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import ijson

_BACKEND_ROOT = Path(__file__).parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from scripts.audit_summary_quality import (  # noqa: E402
    format_terminal_report,
    run_audit,
    to_json_report,
)
from scripts.common.logging_config import setup_logging  # noqa: E402

logger = setup_logging(__name__)

_DEFAULT_INPUT = _BACKEND_ROOT.parent / "data" / "local_rules_v1.json"


def _stream_flat_items(input_path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """조문요약을 스트리밍으로 추출하여 최소 dict 리스트 반환.

    디스크에 쓰지 않고 메모리에서 직접 구성합니다.
    조내용은 포함하지 않아 메모리 사용을 최소화합니다.

    Args:
        input_path: local_rules JSON 파일 경로

    Returns:
        (flat_items, stats) 튜플
        flat_items: [{"id": "...", "조문요약": "..."}, ...]
        stats: 통계 dict
    """
    total_records = 0
    total_articles = 0
    has_summary = 0
    empty_summary = 0
    null_summary = 0

    flat_items: list[dict[str, Any]] = []

    logger.info("입력: %s (스트리밍 평탄화)", input_path)

    with open(input_path, "rb") as f:
        for item in ijson.items(f, "item"):
            total_records += 1
            serial = str(item.get("자치법규일련번호", ""))
            articles = item.get("조")

            if not isinstance(articles, list):
                continue

            for article in articles:
                if not isinstance(article, dict):
                    continue

                total_articles += 1
                article_num = str(article.get("조문번호", ""))
                summary = article.get("조문요약")

                if summary is None:
                    null_summary += 1
                    summary_str = ""
                elif isinstance(summary, str) and not summary.strip():
                    empty_summary += 1
                    summary_str = ""
                else:
                    has_summary += 1
                    summary_str = str(summary)

                flat_id = f"{serial}_{article_num}" if article_num else f"{serial}_{total_articles}"

                flat_items.append({
                    "id": flat_id,
                    "조문요약": summary_str,
                })

            if total_records % 20000 == 0:
                logger.info(
                    "  진행: %d 레코드, %d 조문 처리됨",
                    total_records, total_articles,
                )

    stats = {
        "total_records": total_records,
        "total_articles": total_articles,
        "has_summary": has_summary,
        "empty_summary": empty_summary,
        "null_summary": null_summary,
    }

    logger.info("평탄화 완료: %d 레코드 → %d 조문", total_records, total_articles)
    logger.info(
        "  유효: %d | 빈값: %d | null: %d",
        has_summary, empty_summary, null_summary,
    )

    return flat_items, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="자치법규 조문요약 평탄화 + 감사",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(_DEFAULT_INPUT),
        help=f"입력 JSON 파일 (기본: {_DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="감사 JSON 보고서 저장 경로 (예: eda_output/audit_local_rules_article.json)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="이상치 샘플 수 (기본: 5)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="평탄화 통계만 출력 (감사 생략)",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        logger.error("입력 파일이 존재하지 않습니다: %s", input_path)
        sys.exit(1)

    # 1. 스트리밍 평탄화
    flat_items, flatten_stats = _stream_flat_items(input_path)

    if args.stats:
        print("\n=== 평탄화 통계 ===")
        for key, val in flatten_stats.items():
            print(f"  {key}: {val:,}")
        return

    # 2. 감사 실행 (audit_summary_quality.py 함수 직접 호출)
    logger.info("감사 시작: %d 조문", len(flat_items))
    audit_report = run_audit(
        items=flat_items,
        summary_field="조문요약",
        id_field="id",
        compare_field=None,
        max_samples=args.samples,
        source_info=f"{input_path} (조문요약 평탄화)",
    )

    # 3. 터미널 출력
    print(format_terminal_report(audit_report))

    # 4. JSON 저장
    if args.output:
        output_path = _BACKEND_ROOT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(to_json_report(audit_report), f, ensure_ascii=False, indent=2)
        logger.info("JSON 보고서 저장: %s", output_path)


if __name__ == "__main__":
    main()
