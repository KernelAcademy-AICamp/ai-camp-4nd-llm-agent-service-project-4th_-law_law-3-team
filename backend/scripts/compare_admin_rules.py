"""행정규칙 원본 vs v3 비교 및 전체요약 null 레코드 클리닝 스크립트

원본 파일([DONE]administrative_rules-1.json, 537MB)과
v3 파일(admin_rule_v3.json, 485MB)을 비교하여 차이점을 보고합니다.

비교 필드:
  - 행정규칙명, 행정규칙종류, 소관부처명 (문자열 직접 비교)
  - 전체요약, 행정규칙요약, 부칙내용 (MD5 해시)
  - 조문내용 (list → JSON 직렬화 후 MD5 해시)

클리닝 대상:
  - 전체요약이 null이고 조문내용이 빈 리스트인 레코드 삭제

Usage:
    cd backend

    # 비교만
    uv run python scripts/compare_admin_rules.py compare

    # 클리닝만 (전체요약 null 삭제)
    uv run python scripts/compare_admin_rules.py clean --dry-run
    uv run python scripts/compare_admin_rules.py clean

    # 비교 + 클리닝 동시
    uv run python scripts/compare_admin_rules.py both --report eda_output/admin_rule_diff.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ijson  # type: ignore[import-untyped]

_BACKEND_ROOT = Path(__file__).parent.parent
_PROJECT_ROOT = _BACKEND_ROOT.parent
_DEFAULT_ORIGINAL = _PROJECT_ROOT / "data" / "[DONE]administrative_rules-1.json"
_DEFAULT_V3 = _PROJECT_ROOT / "data" / "admin_rule_v3.json"

from scripts.common.logging_config import setup_logging

logger = setup_logging(__name__)

# ---------------------------------------------------------------------------
# 비교용 데이터 구조
# ---------------------------------------------------------------------------

# 비교 대상 필드: (필드명, 비교방식)
_STR_FIELDS = ("행정규칙명", "행정규칙종류", "소관부처명")
_HASH_FIELDS = ("전체요약", "행정규칙요약", "부칙내용")
_LIST_HASH_FIELDS = ("조문내용",)


def _md5(value: Any) -> str:
    """값의 MD5 해시를 반환. None이면 빈 문자열."""
    if value is None:
        return ""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(text.encode("utf-8")).hexdigest()


@dataclass
class RecordDigest:
    """레코드의 비교용 다이제스트"""

    record_id: str
    str_values: dict[str, str | None] = field(default_factory=dict)
    hash_values: dict[str, str] = field(default_factory=dict)


def _build_digest(item: dict[str, Any]) -> RecordDigest:
    """JSON 레코드 → RecordDigest"""
    record_id = str(item.get("행정규칙ID", ""))
    digest = RecordDigest(record_id=record_id)

    for f in _STR_FIELDS:
        digest.str_values[f] = item.get(f)

    for f in _HASH_FIELDS:
        digest.hash_values[f] = _md5(item.get(f))

    for f in _LIST_HASH_FIELDS:
        digest.hash_values[f] = _md5(item.get(f))

    return digest


# ---------------------------------------------------------------------------
# Phase 1: 원본 인덱싱 (ijson 스트리밍)
# ---------------------------------------------------------------------------


def _index_original(path: Path) -> dict[str, RecordDigest]:
    """원본 파일을 스트리밍으로 읽어 {행정규칙ID: RecordDigest} 딕셔너리 구축"""
    logger.info("원본 인덱싱: %s", path.name)
    index: dict[str, RecordDigest] = {}
    count = 0

    with open(path, "rb") as f:
        for item in ijson.items(f, "item"):
            digest = _build_digest(item)
            index[digest.record_id] = digest
            count += 1
            if count % 5000 == 0:
                logger.info("  원본 인덱싱 진행: %d건", count)

    logger.info("  원본 인덱싱 완료: %d건", count)
    return index


# ---------------------------------------------------------------------------
# Phase 2: v3 스트리밍 비교
# ---------------------------------------------------------------------------


@dataclass
class DiffReport:
    """비교 결과"""

    original_count: int = 0
    v3_count: int = 0
    common_count: int = 0
    only_in_original: list[str] = field(default_factory=list)
    only_in_v3: list[str] = field(default_factory=list)
    field_diffs: dict[str, list[str]] = field(default_factory=dict)


def _compare_v3(
    v3_path: Path,
    original_index: dict[str, RecordDigest],
) -> DiffReport:
    """v3를 스트리밍으로 읽으며 원본 인덱스와 비교"""
    logger.info("v3 비교 시작: %s", v3_path.name)

    report = DiffReport(original_count=len(original_index))
    remaining_ids = set(original_index.keys())

    with open(v3_path, "rb") as f:
        for item in ijson.items(f, "item"):
            report.v3_count += 1
            record_id = str(item.get("행정규칙ID", ""))
            v3_digest = _build_digest(item)

            if record_id not in original_index:
                report.only_in_v3.append(record_id)
                continue

            remaining_ids.discard(record_id)
            report.common_count += 1
            orig_digest = original_index[record_id]

            # 필드별 비교
            for f in _STR_FIELDS:
                if v3_digest.str_values.get(f) != orig_digest.str_values.get(f):
                    report.field_diffs.setdefault(f, []).append(record_id)

            for f in (*_HASH_FIELDS, *_LIST_HASH_FIELDS):
                if v3_digest.hash_values.get(f) != orig_digest.hash_values.get(f):
                    report.field_diffs.setdefault(f, []).append(record_id)

            if report.v3_count % 5000 == 0:
                logger.info("  v3 비교 진행: %d건", report.v3_count)

    report.only_in_original = sorted(remaining_ids)
    logger.info("  v3 비교 완료: %d건", report.v3_count)
    return report


def _print_diff_report(report: DiffReport) -> None:
    """비교 결과를 터미널에 출력"""
    print()
    print("=" * 60)
    print("  행정규칙 원본 vs v3 비교 보고서")
    print("=" * 60)
    print()
    print(f"  원본 레코드: {report.original_count:,}건")
    print(f"  v3 레코드:   {report.v3_count:,}건")
    print(f"  공통 ID:     {report.common_count:,}건")
    print()

    if report.only_in_original:
        print(f"  원본에만 존재 ({len(report.only_in_original)}건):")
        for rid in report.only_in_original[:10]:
            print(f"    - {rid}")
        if len(report.only_in_original) > 10:
            print(f"    ... 외 {len(report.only_in_original) - 10}건")
    else:
        print("  원본에만 존재: 없음")
    print()

    if report.only_in_v3:
        print(f"  v3에만 존재 ({len(report.only_in_v3)}건):")
        for rid in report.only_in_v3[:10]:
            print(f"    - {rid}")
        if len(report.only_in_v3) > 10:
            print(f"    ... 외 {len(report.only_in_v3) - 10}건")
    else:
        print("  v3에만 존재: 없음")
    print()

    if report.field_diffs:
        print("  필드별 차이:")
        for field_name, ids in sorted(report.field_diffs.items()):
            print(f"    {field_name}: {len(ids):,}건")
            for rid in ids[:5]:
                print(f"      - ID {rid}")
            if len(ids) > 5:
                print(f"      ... 외 {len(ids) - 5}건")
    else:
        print("  필드별 차이: 없음 (완전 일치)")
    print()
    print("─" * 60)


# ---------------------------------------------------------------------------
# Phase 3: 전체요약 null 레코드 클리닝
# ---------------------------------------------------------------------------


@dataclass
class CleanReport:
    """클리닝 결과"""

    total_before: int = 0
    total_after: int = 0
    removed_ids: list[str] = field(default_factory=list)
    removed_names: list[str] = field(default_factory=list)


def _clean_null_summaries(
    v3_path: Path,
    output_path: Path | None,
    dry_run: bool = False,
) -> CleanReport:
    """전체요약이 null이고 조문내용이 빈 리스트인 레코드를 삭제"""
    logger.info("v3 로드: %s", v3_path.name)
    with open(v3_path, encoding="utf-8") as f:
        data: list[dict[str, Any]] = json.load(f)

    report = CleanReport(total_before=len(data))

    # 삭제 대상 필터링
    cleaned: list[dict[str, Any]] = []
    for item in data:
        summary = item.get("전체요약")
        articles = item.get("조문내용", [])
        is_empty = summary is None and (not isinstance(articles, list) or len(articles) == 0)

        if is_empty:
            record_id = str(item.get("행정규칙ID", ""))
            record_name = item.get("행정규칙명", "(이름 없음)")
            report.removed_ids.append(record_id)
            report.removed_names.append(record_name)
        else:
            cleaned.append(item)

    report.total_after = len(cleaned)

    # 터미널 출력
    print()
    print("=" * 60)
    print("  전체요약 null 레코드 클리닝 보고서")
    print("=" * 60)
    print()
    print(f"  변경 전: {report.total_before:,}건")
    print(f"  삭제:    {len(report.removed_ids):,}건")
    print(f"  변경 후: {report.total_after:,}건")
    print()

    if report.removed_ids:
        print("  삭제된 레코드:")
        for rid, rname in zip(report.removed_ids, report.removed_names, strict=True):
            print(f"    - ID {rid}: {rname}")
    print()
    print("─" * 60)

    # 파일 저장
    if not dry_run and report.removed_ids:
        save_path = output_path or v3_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, ensure_ascii=False, indent=2)
        logger.info("저장: %s (%d건)", save_path, report.total_after)
    elif dry_run:
        logger.info("드라이런 모드 — 파일 수정 없음")

    return report


# ---------------------------------------------------------------------------
# JSON 보고서 생성
# ---------------------------------------------------------------------------


def _to_json_report(
    diff: DiffReport | None,
    clean: CleanReport | None,
) -> dict[str, Any]:
    """JSON 직렬화 가능한 통합 보고서"""
    result: dict[str, Any] = {}

    if diff is not None:
        result["comparison"] = {
            "original_count": diff.original_count,
            "v3_count": diff.v3_count,
            "common_count": diff.common_count,
            "only_in_original": diff.only_in_original,
            "only_in_v3": diff.only_in_v3,
            "field_diffs": {
                k: {"count": len(v), "ids": v}
                for k, v in sorted(diff.field_diffs.items())
            },
        }

    if clean is not None:
        result["cleaning"] = {
            "total_before": clean.total_before,
            "total_after": clean.total_after,
            "removed_count": len(clean.removed_ids),
            "removed_records": [
                {"id": rid, "name": rname}
                for rid, rname in zip(clean.removed_ids, clean.removed_names, strict=True)
            ],
        }

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="행정규칙 원본 vs v3 비교 및 전체요약 null 레코드 클리닝",
    )
    parser.add_argument(
        "command",
        choices=["compare", "clean", "both"],
        help="실행 명령 (compare: 비교만, clean: 클리닝만, both: 비교+클리닝)",
    )
    parser.add_argument(
        "--original",
        type=str,
        default=str(_DEFAULT_ORIGINAL),
        help=f"원본 JSON 파일 경로 (기본: {_DEFAULT_ORIGINAL.name})",
    )
    parser.add_argument(
        "--v3",
        type=str,
        default=str(_DEFAULT_V3),
        help=f"v3 JSON 파일 경로 (기본: {_DEFAULT_V3.name})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="클리닝 결과 저장 경로 (미지정 시 v3 파일 덮어쓰기)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="파일 수정 없이 통계만 출력 (clean 명령용)",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="JSON 보고서 저장 경로 (예: eda_output/admin_rule_diff.json)",
    )
    args = parser.parse_args()

    original_path = Path(args.original).resolve()
    v3_path = Path(args.v3).resolve()

    diff_report: DiffReport | None = None
    clean_report: CleanReport | None = None

    # 비교
    if args.command in ("compare", "both"):
        if not original_path.exists():
            logger.error("원본 파일이 존재하지 않습니다: %s", original_path)
            sys.exit(1)
        if not v3_path.exists():
            logger.error("v3 파일이 존재하지 않습니다: %s", v3_path)
            sys.exit(1)

        original_index = _index_original(original_path)
        diff_report = _compare_v3(v3_path, original_index)
        _print_diff_report(diff_report)

    # 클리닝
    if args.command in ("clean", "both"):
        if not v3_path.exists():
            logger.error("v3 파일이 존재하지 않습니다: %s", v3_path)
            sys.exit(1)

        output_path = Path(args.output).resolve() if args.output else None
        clean_report = _clean_null_summaries(v3_path, output_path, dry_run=args.dry_run)

    # JSON 보고서 저장
    if args.report and (diff_report is not None or clean_report is not None):
        report_path = _BACKEND_ROOT / args.report
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(
                _to_json_report(diff_report, clean_report),
                f,
                ensure_ascii=False,
                indent=2,
            )
        logger.info("JSON 보고서 저장: %s", report_path)


if __name__ == "__main__":
    main()
