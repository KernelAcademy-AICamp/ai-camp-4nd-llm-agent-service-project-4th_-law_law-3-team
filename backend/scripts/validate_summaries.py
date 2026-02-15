"""
LLM 요약 필드 검증 스크립트

19개 데이터 타입의 JSON 소스에서 LLM이 생성한 요약 필드를 검증합니다.

검증 항목:
  1. 길이 제약: 법령(전체 ≤600자, 조문 ≤200자), 나머지 ≤300자
  2. LLM 오류 패턴: 프롬프트 누출, 불완전 문장, HTML 잔여, 반복 등
  3. 통계 보고: 타입별 길이 분포, null/초과/이상 패턴 건수

Usage:
    cd backend
    uv run python scripts/validate_summaries.py                 # 전체 검증
    uv run python scripts/validate_summaries.py --type law      # 특정 타입
    uv run python scripts/validate_summaries.py --samples 10    # 이상치 샘플 수
    uv run python scripts/validate_summaries.py --output eda_output/summary_validation.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# backend/ 루트를 sys.path에 추가
_BACKEND_ROOT = Path(__file__).parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

# 인제스트 config 레지스트리 로드 (types/ 자동 등록 트리거)
import scripts.ingest.types  # noqa: F401, E402
from scripts.ingest.config import (  # noqa: E402
    DATA_DIR,
    IngestConfig,
    get_config,
    list_configs,
)

# 인제스트 config의 기본 소스 베이스 디렉토리
_DEFAULT_INGEST_SOURCE_DIR = DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 구조
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Issue:
    """검증 이상 항목"""

    severity: str  # HIGH, MEDIUM, LOW
    pattern: str  # 패턴 이름
    detail: str  # 상세 설명
    sample_text: str = ""  # 요약 텍스트 일부 (최대 80자)
    item_id: str = ""  # 원본 식별자


@dataclass
class FieldReport:
    """단일 필드(요약 종류)에 대한 검증 결과"""

    field_name: str
    max_length: int
    total: int = 0
    null_count: int = 0
    over_count: int = 0
    short_count: int = 0  # < 10자
    lengths: list[int] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)

    def percentiles(self) -> dict[str, int]:
        """길이 분포 백분위수 계산"""
        if not self.lengths:
            return {}
        arr = np.array(self.lengths)
        return {
            "min": int(np.min(arr)),
            "P25": int(np.percentile(arr, 25)),
            "P50": int(np.percentile(arr, 50)),
            "P75": int(np.percentile(arr, 75)),
            "P90": int(np.percentile(arr, 90)),
            "P99": int(np.percentile(arr, 99)),
            "max": int(np.max(arr)),
        }


@dataclass
class TypeReport:
    """타입별 검증 결과"""

    type_name: str
    label: str
    fields: list[FieldReport] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# LLM 오류 패턴 검사
# ─────────────────────────────────────────────────────────────────────────────

# 프롬프트 누출 패턴
_PROMPT_LEAK_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("한국어 프롬프트", re.compile(r"요약해|요약하시오|요약을 작성|다음 텍스트|아래 내용")),
    ("영어 프롬프트", re.compile(r"summarize|as an ai|please provide|in summary", re.IGNORECASE)),
    ("지시문 잔여", re.compile(r"^\s*[-*]\s*(요약|summary)", re.IGNORECASE | re.MULTILINE)),
]

# 불완전 문장 패턴
_INCOMPLETE_ENDINGS = re.compile(r"(\.\.\.|\.\.|\(|「|『)\s*$")

# HTML/마크다운 잔여
_HTML_MARKDOWN = re.compile(r"<\s*/?\s*(p|br|div|span|b|i|em|strong|h[1-6])\s*/?\s*>|(\*\*|##|###|\n\n\n)")

# 반복 패턴: 동일 10자 이상 구절이 2회 이상
_REPEAT_MIN_LEN = 10


def _check_patterns(text: str) -> list[Issue]:
    """LLM 오류 패턴 검사"""
    issues: list[Issue] = []

    # 프롬프트 누출
    for name, pattern in _PROMPT_LEAK_PATTERNS:
        if pattern.search(text):
            issues.append(Issue(
                severity="HIGH",
                pattern="프롬프트 누출",
                detail=f"{name}: {pattern.pattern}",
                sample_text=text[:80],
            ))

    # 불완전 문장
    if _INCOMPLETE_ENDINGS.search(text):
        issues.append(Issue(
            severity="MEDIUM",
            pattern="불완전 문장",
            detail=f"끝부분: ...{text[-30:]}",
            sample_text=text[:80],
        ))

    # 언어 혼용: 연속 영문 3단어 이상 (법률 고유명사 제외)
    english_runs = re.findall(r"[A-Za-z]{2,}(?:\s+[A-Za-z]{2,}){2,}", text)
    for run in english_runs:
        # 법률/고유명사로 빈번한 패턴 제외
        if re.match(
            r"(?:UN|EU|WTO|OECD|UNESCO|WHO|ILO|ICC|ICJ|NATO|ASEAN|"
            r"GDP|GNP|IMF|FTA|MOU|ODA|NGO|"
            r"No\s+\d|Art\s+\d|Section\s+\d)",
            run,
            re.IGNORECASE,
        ):
            continue
        issues.append(Issue(
            severity="LOW",
            pattern="언어 혼용",
            detail=f"영문 구절: {run[:50]}",
            sample_text=text[:80],
        ))

    # HTML/마크다운 잔여
    if _HTML_MARKDOWN.search(text):
        issues.append(Issue(
            severity="MEDIUM",
            pattern="HTML/마크다운 잔여",
            detail=f"패턴 발견: {_HTML_MARKDOWN.search(text).group()!r}",  # type: ignore[union-attr]
            sample_text=text[:80],
        ))

    # 반복 패턴
    if len(text) >= _REPEAT_MIN_LEN * 2:
        for i in range(len(text) - _REPEAT_MIN_LEN):
            substr = text[i : i + _REPEAT_MIN_LEN]
            if text.count(substr) >= 2 and i == text.index(substr):
                issues.append(Issue(
                    severity="MEDIUM",
                    pattern="반복 패턴",
                    detail=f"반복 구절: {substr!r}",
                    sample_text=text[:80],
                ))
                break  # 타입별 1건만 보고

    return issues


# ─────────────────────────────────────────────────────────────────────────────
# JSON 로딩
# ─────────────────────────────────────────────────────────────────────────────


def _load_json_all(source_path: Path) -> list[dict[str, Any]]:
    """JSON 파일 또는 디렉토리 전체 로드"""
    if not source_path.exists():
        raise FileNotFoundError(f"소스를 찾을 수 없습니다: {source_path}")

    if source_path.is_dir():
        return _load_json_directory(source_path)

    with open(source_path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("items", [])


def _load_json_directory(dir_path: Path) -> list[dict[str, Any]]:
    """디렉토리 내 모든 .json 파일 합산 로드"""
    json_files = sorted(dir_path.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"디렉토리에 .json 파일이 없습니다: {dir_path}")

    all_items: list[dict[str, Any]] = []
    for json_file in json_files:
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.warning("JSON 파싱 실패, 건너뜀: %s (%s)", json_file.name, e)
            continue

        items = data if isinstance(data, list) else data.get("items", [])
        all_items.extend(items)
    return all_items


def _stream_json(source_path: Path) -> Any:
    """ijson 스트리밍 (대용량 파일)"""
    import ijson

    with open(source_path, "rb") as f:
        yield from ijson.items(f, "item")


def _iter_items(source_path: Path, use_streaming: bool = False) -> Any:
    """소스 경로에서 아이템 순회 (파일/디렉토리/스트리밍)"""
    if source_path.is_dir():
        yield from _load_json_all(source_path)
    elif use_streaming:
        yield from _stream_json(source_path)
    else:
        yield from _load_json_all(source_path)


# ─────────────────────────────────────────────────────────────────────────────
# 스트리밍 크기 기준 (200MB 이상이면 스트리밍)
# ─────────────────────────────────────────────────────────────────────────────

_STREAMING_THRESHOLD = 200 * 1024 * 1024  # 200MB


def _should_stream(source_path: Path) -> bool:
    """파일 크기 기반 스트리밍 여부 결정"""
    if source_path.is_dir():
        return False  # 디렉토리는 개별 파일 로드
    return source_path.stat().st_size > _STREAMING_THRESHOLD


# ─────────────────────────────────────────────────────────────────────────────
# 검증기
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_source_path(cfg: IngestConfig, data_dir: Path | None) -> Path:
    """소스 경로 해석 (--data-dir 재매핑 지원)

    인제스트 config의 source_path가 data/ 하위를 가리킵니다.
    --data-dir로 다른 디렉토리를 지정하면 상대 경로를 재매핑합니다.

    추가로 파일명이 _v1 → _v2 등으로 바뀐 경우도 glob으로 탐색합니다.
    """
    # data_dir 미지정이면 config 기본 경로 사용
    if data_dir is None:
        return cfg.source_path

    # config 경로에서 DATA_DIR 이후 상대 경로 추출
    try:
        relative = cfg.source_path.relative_to(_DEFAULT_INGEST_SOURCE_DIR)
    except ValueError:
        # DATA_DIR 밖의 경로면 그대로 반환
        return cfg.source_path

    candidate = data_dir / relative
    if candidate.exists():
        return candidate

    # 파일명 버전 차이 대응: law_v1.json → law_v2.json 등
    if not candidate.is_dir():
        stem = candidate.stem  # 예: "law_v1"
        parent = candidate.parent
        if parent.exists():
            # _v숫자 패턴 제거 후 glob
            base_stem = re.sub(r"_v\d+$", "", stem)  # "law"
            matches = sorted(parent.glob(f"{base_stem}_v*.json"))
            if matches:
                # 가장 높은 버전 사용
                return matches[-1]

    return candidate


class SummaryValidator:
    """LLM 요약 필드 검증기"""

    def __init__(self, max_samples: int = 5, data_dir: Path | None = None) -> None:
        self.max_samples = max_samples
        self.data_dir = data_dir

    def validate_type(self, type_name: str) -> TypeReport:
        """단일 타입의 모든 레코드 검증"""
        cfg = get_config(type_name)
        source = _resolve_source_path(cfg, self.data_dir)
        logger.info("[%s] %s 검증 시작: %s", type_name, cfg.data_type_label, source)

        report = TypeReport(type_name=type_name, label=cfg.data_type_label)

        if type_name == "law":
            self._validate_law(cfg, source, report)
        else:
            self._validate_generic(cfg, source, report)

        return report

    def _validate_law(
        self, cfg: IngestConfig, source: Path, report: TypeReport
    ) -> None:
        """법령 타입: 전체 요약(≤600) + 조문별 요약(≤200) 검증"""
        overall_field = FieldReport(field_name="법령 요약", max_length=600)
        article_field = FieldReport(field_name="조문요약", max_length=200)

        use_stream = _should_stream(source)
        count = 0

        for item in _iter_items(source, use_streaming=use_stream):
            count += 1
            item_id = str(item.get(cfg.id_field, f"item_{count}"))

            # 전체 법령 요약
            summary = item.get("법령 요약")
            self._check_field(summary, item_id, overall_field)

            # 조문별 요약
            articles = item.get("조문")
            if isinstance(articles, list):
                for idx, article in enumerate(articles):
                    if isinstance(article, dict):
                        art_summary = article.get("조문요약")
                        art_id = f"{item_id}/조문[{idx}]"
                        self._check_field(art_summary, art_id, article_field)

            if count % 1000 == 0:
                logger.info("  [law] %d건 처리...", count)

        report.fields = [overall_field, article_field]
        logger.info("[law] 검증 완료: 전체 %d건, 조문요약 %d건", overall_field.total, article_field.total)

    def _validate_generic(
        self, cfg: IngestConfig, source: Path, report: TypeReport
    ) -> None:
        """일반 타입: 요약 필드 ≤300자 검증"""
        fld = FieldReport(field_name=cfg.summary_field, max_length=300)

        use_stream = _should_stream(source)
        count = 0

        for item in _iter_items(source, use_streaming=use_stream):
            count += 1
            item_id = str(item.get(cfg.id_field, f"item_{count}"))

            summary = item.get(cfg.summary_field)
            self._check_field(summary, item_id, fld)

            if count % 10000 == 0:
                logger.info("  [%s] %d건 처리...", cfg.name, count)

        report.fields = [fld]
        logger.info("[%s] 검증 완료: %d건", cfg.name, fld.total)

    def _check_field(
        self, value: Any, item_id: str, fld: FieldReport
    ) -> None:
        """단일 요약 값 검증"""
        fld.total += 1

        # null/empty
        if value is None or (isinstance(value, str) and not value.strip()):
            fld.null_count += 1
            fld.issues.append(Issue(
                severity="HIGH",
                pattern="null/empty",
                detail="값이 없거나 빈 문자열",
                item_id=item_id,
            ))
            return

        text = str(value).strip()
        length = len(text)
        fld.lengths.append(length)

        # 극단적으로 짧은 요약
        if length < 10:
            fld.short_count += 1
            fld.issues.append(Issue(
                severity="MEDIUM",
                pattern="극단적 단문",
                detail=f"길이 {length}자",
                sample_text=text,
                item_id=item_id,
            ))

        # 길이 초과
        if length > fld.max_length:
            fld.over_count += 1

        # LLM 오류 패턴
        pattern_issues = _check_patterns(text)
        for issue in pattern_issues:
            issue.item_id = item_id
        fld.issues.extend(pattern_issues)


# ─────────────────────────────────────────────────────────────────────────────
# 보고서 생성
# ─────────────────────────────────────────────────────────────────────────────


def _format_terminal_report(results: list[TypeReport], max_samples: int) -> str:
    """터미널 출력용 보고서"""
    lines: list[str] = []
    total_records = 0
    total_over = 0
    total_null = 0
    total_pattern_issues = 0

    lines.append("")
    lines.append("=" * 55)
    lines.append("  LLM 요약 필드 검증 보고서")
    lines.append("=" * 55)
    lines.append("")

    for tr in results:
        lines.append(f"[{tr.type_name}] {tr.label} " + "─" * max(1, 45 - len(tr.type_name) - len(tr.label)))

        for fld in tr.fields:
            total_records += fld.total
            total_over += fld.over_count
            total_null += fld.null_count

            lines.append(f"  {fld.field_name} (≤{fld.max_length}자):")
            lines.append(
                f"    건수: {fld.total:,} | null: {fld.null_count:,} | "
                f"초과(>{fld.max_length}자): {fld.over_count:,} | 단문(<10자): {fld.short_count:,}"
            )

            pct = fld.percentiles()
            if pct:
                pct_str = " ".join(f"{k}={v}" for k, v in pct.items())
                lines.append(f"    길이: {pct_str}")

            # 패턴별 이슈 집계
            pattern_counts: dict[str, int] = {}
            for issue in fld.issues:
                pattern_counts[issue.pattern] = pattern_counts.get(issue.pattern, 0) + 1
            non_null_patterns = {
                k: v for k, v in pattern_counts.items() if k not in ("null/empty",)
            }
            total_pattern_issues += sum(non_null_patterns.values())

            if non_null_patterns:
                lines.append("  패턴 이상:")
                for pname, pcount in sorted(non_null_patterns.items(), key=lambda x: -x[1]):
                    lines.append(f"    {pname}: {pcount:,}건")

            # 이상치 샘플
            sample_issues = [i for i in fld.issues if i.sample_text][:max_samples]
            if sample_issues:
                lines.append(f"  이상치 샘플 (최대 {max_samples}건):")
                for si in sample_issues:
                    lines.append(
                        f"    [{si.severity}] {si.pattern} | id={si.item_id} | {si.sample_text!r}"
                    )

        lines.append("")

    lines.append("=" * 55)
    lines.append("  종합 요약")
    lines.append("=" * 55)
    pct_over = (total_over / total_records * 100) if total_records else 0
    lines.append(f"  총 검증 건수: {total_records:,}")
    lines.append(f"  길이 초과: {total_over:,}건 ({pct_over:.1f}%)")
    lines.append(f"  LLM 오류 패턴: {total_pattern_issues:,}건")
    lines.append(f"  null/empty: {total_null:,}건")
    lines.append("")

    return "\n".join(lines)


def _to_json_report(results: list[TypeReport]) -> dict[str, Any]:
    """JSON 직렬화 가능한 보고서"""
    report: dict[str, Any] = {"types": {}}
    total_records = 0
    total_over = 0
    total_null = 0

    for tr in results:
        type_data: dict[str, Any] = {"label": tr.label, "fields": {}}

        for fld in tr.fields:
            total_records += fld.total
            total_over += fld.over_count
            total_null += fld.null_count

            pattern_counts: dict[str, int] = {}
            for issue in fld.issues:
                pattern_counts[issue.pattern] = pattern_counts.get(issue.pattern, 0) + 1

            type_data["fields"][fld.field_name] = {
                "max_length": fld.max_length,
                "total": fld.total,
                "null_count": fld.null_count,
                "over_count": fld.over_count,
                "short_count": fld.short_count,
                "percentiles": fld.percentiles(),
                "pattern_counts": pattern_counts,
            }

        report["types"][tr.type_name] = type_data

    report["summary"] = {
        "total_records": total_records,
        "total_over": total_over,
        "total_null": total_null,
    }
    return report


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM 요약 필드 검증",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--type",
        dest="type_name",
        default=None,
        help="검증할 타입 (미지정 시 전체). 예: law, precedent, dec_fair_trade",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="이상치 샘플 수 (기본: 5)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=(
            "데이터 기본 디렉토리 (config의 data/ 경로를 대체). "
            "예: ../data"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="JSON 보고서 저장 경로 (예: eda_output/summary_validation.json)",
    )
    args = parser.parse_args()

    data_dir: Path | None = None
    if args.data_dir:
        data_dir = Path(args.data_dir).resolve()
        if not data_dir.exists():
            logger.error("--data-dir 경로가 존재하지 않습니다: %s", data_dir)
            sys.exit(1)
        logger.info("데이터 디렉토리 재매핑: %s", data_dir)

    # 검증 대상 결정
    if args.type_name:
        type_names = [args.type_name]
    else:
        type_names = list_configs()

    # 소스 존재 여부 사전 확인
    missing: list[str] = []
    for tn in type_names:
        cfg = get_config(tn)
        resolved = _resolve_source_path(cfg, data_dir)
        if not resolved.exists():
            missing.append(f"  {tn}: {resolved}")

    if missing:
        logger.warning("소스 파일 없는 타입 (%d개):\n%s", len(missing), "\n".join(missing))
        type_names = [
            tn for tn in type_names
            if _resolve_source_path(get_config(tn), data_dir).exists()
        ]

    if not type_names:
        logger.error("검증할 타입이 없습니다.")
        sys.exit(1)

    logger.info("검증 대상: %d개 타입 — %s", len(type_names), ", ".join(type_names))

    # 검증 실행
    validator = SummaryValidator(max_samples=args.samples, data_dir=data_dir)
    results: list[TypeReport] = []

    for tn in type_names:
        try:
            report = validator.validate_type(tn)
            results.append(report)
        except Exception as e:
            logger.error("[%s] 검증 실패: %s", tn, e)

    # 터미널 출력
    print(_format_terminal_report(results, max_samples=args.samples))

    # JSON 저장
    if args.output:
        output_path = _BACKEND_ROOT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(_to_json_report(results), f, ensure_ascii=False, indent=2)
        logger.info("JSON 보고서 저장: %s", output_path)


if __name__ == "__main__":
    main()
