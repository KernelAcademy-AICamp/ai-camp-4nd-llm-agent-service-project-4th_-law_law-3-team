"""행정규칙 전체요약 + 조문요약 통합 클리닝 스크립트

기존 SummaryCleaner를 재사용하여 두 필드를 모두 클리닝합니다.

클리닝 대상:
  - 전체요약: 문서당 1개 (17,332/17,335건)
  - 조문요약: 조문내용[].조문요약 (중첩, 조문당 1개)

Usage:
    cd backend
    uv run python scripts/clean_admin_rule_summaries.py --dry-run
    uv run python scripts/clean_admin_rule_summaries.py
    uv run python scripts/clean_admin_rule_summaries.py --output ../data/admin_rule_v3_cleaned.json
    uv run python scripts/clean_admin_rule_summaries.py --report eda_output/clean_admin_rule_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# backend/ 루트를 sys.path에 추가
_BACKEND_ROOT = Path(__file__).parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from scripts.clean_summaries import SummaryCleaner  # noqa: E402

# 행정규칙 전용: "핵심 키워드 및 (검색) 상황" 섹션 제거
# LLM이 요약 끝에 생성한 키워드 섹션 (전체의 98.7%가 본문 70% 이후에 위치)
# 변형: "핵심 키워드 및 검색 상황", "핵심 키워드 및 상황", "핵심 키워드·검색 상황" 등
_RE_KEYWORD_SECTION = re.compile(
    r"[\s.。,]*핵심\s?키워드\s*(?:및|·)\s*(?:검색\s?)?상황\s*[:：]?\s*[-–]?\s*.*$",
    re.DOTALL,
)

_PROJECT_ROOT = _BACKEND_ROOT.parent
_DEFAULT_INPUT = _PROJECT_ROOT / "data" / "admin_rule_v3.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class FieldStats:
    """단일 필드 클리닝 통계"""

    total: int = 0
    cleaned: int = 0
    markdown_removed: int = 0
    incomplete_fixed: int = 0
    artifact_removed: int = 0
    keyword_section_removed: int = 0
    null_count: int = 0


def clean_admin_rule_summaries(
    data: list[dict[str, Any]],
    dry_run: bool = False,
) -> tuple[list[dict[str, Any]], FieldStats, FieldStats]:
    """전체요약 + 조문요약 통합 클리닝

    Args:
        data: 행정규칙 레코드 리스트
        dry_run: True면 파일 수정 없이 통계만 수집

    Returns:
        (클리닝된 데이터, 전체요약 통계, 조문요약 통계)
    """
    cleaner = SummaryCleaner()
    summary_stats = FieldStats()
    article_stats = FieldStats()

    for item in data:
        # 1) 전체요약 클리닝
        summary = item.get("전체요약")
        summary_stats.total += 1

        if summary is None or (isinstance(summary, str) and not summary.strip()):
            summary_stats.null_count += 1
        elif isinstance(summary, str):
            cleaned, md_removed, inc_fixed, art_removed, changed = cleaner.clean_text(
                summary
            )
            # 행정규칙 전용 후처리: "핵심 키워드 및 검색 상황" 섹션 제거
            after_kw = _RE_KEYWORD_SECTION.sub("", cleaned).strip()
            kw_removed = after_kw != cleaned
            if kw_removed:
                cleaned = after_kw
                changed = True

            if changed:
                summary_stats.cleaned += 1
                if md_removed:
                    summary_stats.markdown_removed += 1
                if inc_fixed:
                    summary_stats.incomplete_fixed += 1
                if art_removed:
                    summary_stats.artifact_removed += 1
                if kw_removed:
                    summary_stats.keyword_section_removed += 1
                if not dry_run:
                    item["전체요약"] = cleaned

        # 2) 조문요약 클리닝 (중첩)
        articles = item.get("조문내용", [])
        if not isinstance(articles, list):
            continue

        for article in articles:
            if not isinstance(article, dict):
                continue

            art_summary = article.get("조문요약")
            article_stats.total += 1

            if art_summary is None or (
                isinstance(art_summary, str) and not art_summary.strip()
            ):
                article_stats.null_count += 1
            elif isinstance(art_summary, str):
                cleaned, md_removed, inc_fixed, art_removed, changed = (
                    cleaner.clean_text(art_summary)
                )
                if changed:
                    article_stats.cleaned += 1
                    if md_removed:
                        article_stats.markdown_removed += 1
                    if inc_fixed:
                        article_stats.incomplete_fixed += 1
                    if art_removed:
                        article_stats.artifact_removed += 1
                    if not dry_run:
                        article["조문요약"] = cleaned

    return data, summary_stats, article_stats


def _format_stats(label: str, stats: FieldStats) -> str:
    """통계를 터미널 출력용 문자열로 변환"""
    lines: list[str] = []
    lines.append(f"  [{label}]")
    pct = (stats.cleaned / stats.total * 100) if stats.total else 0
    lines.append(f"    대상: {stats.total:,}건 | 수정: {stats.cleaned:,}건 ({pct:.2f}%)")

    details: list[str] = []
    if stats.markdown_removed:
        details.append(f"마크다운 제거: {stats.markdown_removed:,}건")
    if stats.incomplete_fixed:
        details.append(f"불완전 보정: {stats.incomplete_fixed:,}건")
    if stats.artifact_removed:
        details.append(f"아티팩트 제거: {stats.artifact_removed:,}건")
    if stats.keyword_section_removed:
        details.append(f"키워드 섹션 제거: {stats.keyword_section_removed:,}건")
    if stats.null_count:
        details.append(f"null/empty: {stats.null_count:,}건")
    if details:
        lines.append(f"    {' | '.join(details)}")

    return "\n".join(lines)


def _to_json_report(
    summary_stats: FieldStats, article_stats: FieldStats
) -> dict[str, Any]:
    """JSON 직렬화 가능한 보고서"""
    def _stat_dict(label: str, stats: FieldStats) -> dict[str, Any]:
        return {
            "field": label,
            "total": stats.total,
            "cleaned": stats.cleaned,
            "markdown_removed": stats.markdown_removed,
            "incomplete_fixed": stats.incomplete_fixed,
            "artifact_removed": stats.artifact_removed,
            "keyword_section_removed": stats.keyword_section_removed,
            "null_count": stats.null_count,
        }

    total_cleaned = summary_stats.cleaned + article_stats.cleaned
    total_records = summary_stats.total + article_stats.total
    return {
        "fields": [
            _stat_dict("전체요약", summary_stats),
            _stat_dict("조문요약", article_stats),
        ],
        "summary": {
            "total_records": total_records,
            "total_cleaned": total_cleaned,
            "total_markdown_removed": (
                summary_stats.markdown_removed + article_stats.markdown_removed
            ),
            "total_incomplete_fixed": (
                summary_stats.incomplete_fixed + article_stats.incomplete_fixed
            ),
            "total_artifact_removed": (
                summary_stats.artifact_removed + article_stats.artifact_removed
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="행정규칙 전체요약 + 조문요약 통합 클리닝",
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
        default=None,
        help="출력 JSON 파일 경로 (미지정 시 입력 파일 덮어쓰기)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="파일 수정 없이 통계만 출력",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="JSON 클리닝 리포트 저장 경로",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        logger.error("입력 파일이 존재하지 않습니다: %s", input_path)
        sys.exit(1)

    if args.dry_run:
        logger.info("드라이런 모드 (파일 수정 없음)")

    logger.info("입력: %s", input_path)

    with open(input_path, encoding="utf-8") as f:
        data: list[dict[str, Any]] = json.load(f)

    logger.info("레코드: %d건", len(data))

    data, summary_stats, article_stats = clean_admin_rule_summaries(
        data, dry_run=args.dry_run
    )

    # 터미널 출력
    print()
    print("=" * 55)
    print("  행정규칙 요약 클리닝 보고서")
    print("=" * 55)
    print()
    print(_format_stats("전체요약", summary_stats))
    print()
    print(_format_stats("조문요약", article_stats))
    print()
    total_cleaned = summary_stats.cleaned + article_stats.cleaned
    total_records = summary_stats.total + article_stats.total
    pct = (total_cleaned / total_records * 100) if total_records else 0
    print("─" * 55)
    print(f"  종합: {total_records:,}건 중 {total_cleaned:,}건 수정 ({pct:.2f}%)")
    print("─" * 55)
    print()

    # 파일 저장
    if not args.dry_run:
        output_path = Path(args.output).resolve() if args.output else input_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("저장: %s", output_path)

    # JSON 리포트 저장
    if args.report:
        report_path = _BACKEND_ROOT / args.report
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(
                _to_json_report(summary_stats, article_stats),
                f,
                ensure_ascii=False,
                indent=2,
            )
        logger.info("리포트 저장: %s", report_path)


if __name__ == "__main__":
    main()
