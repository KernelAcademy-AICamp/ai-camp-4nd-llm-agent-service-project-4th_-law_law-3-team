"""
LLM 요약 필드 탐색적 품질 감사 스크립트

JSON 데이터의 LLM 요약 필드를 탐색적으로 감사합니다.
7개 검사: 기본 통계, 마크다운, LLM 아티팩트, 포맷 일관성, 이상치, 중복, 교차 비교.

기존 도구와의 관계:
  - validate_summaries.py — 사전 정의 규칙 기반 검증 (길이, 프롬프트 누출)
  - clean_summaries.py — 규칙 기반 클리닝 (마크다운 제거, 불완전 문장 보정)
  - 이 스크립트 — 탐색적(EDA) 품질 감사 (마크다운/LLM 아티팩트/포맷/이상치/중복)

Usage:
    cd backend

    # Mode 1: IngestConfig 등록 타입 (자동 필드 해석)
    uv run python scripts/audit_summary_quality.py --type special_admin_appeal
    uv run python scripts/audit_summary_quality.py --type special_admin_appeal --data-dir ../data

    # Mode 2: 임의 JSON 파일 (필드 직접 지정)
    uv run python scripts/audit_summary_quality.py \\
        --file ../data/special_admin_appeal/sadm_case_조세심판원_v2.json \\
        --summary-field 심판례요약 \\
        --id-field 특별행정심판재결례일련번호

    # 공통 옵션
    uv run python scripts/audit_summary_quality.py --type precedent \\
        --compare-field 판결요지 \\
        --output eda_output/audit_precedent.json \\
        --samples 10
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# backend/ 루트를 sys.path에 추가
_BACKEND_ROOT = Path(__file__).parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

# 인제스트 config 레지스트리 로드 (types/ 자동 등록 트리거)
import scripts.ingest.types  # noqa: F401, E402
from scripts.common.json_loader import (  # noqa: E402
    load_items as _common_load_items,
)
from scripts.common.json_loader import (
    resolve_source_path,
)
from scripts.common.logging_config import setup_logging  # noqa: E402
from scripts.ingest.config import (  # noqa: E402
    DATA_DIR,
    IngestConfig,
    get_config,
    list_configs,
)

_DEFAULT_INGEST_SOURCE_DIR = DATA_DIR

logger = setup_logging(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 구조
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PatternResult:
    """개별 패턴 탐지 결과"""

    count: int = 0
    samples: list[dict[str, str]] = field(default_factory=list)


@dataclass
class BasicStats:
    """기본 통계"""

    total: int = 0
    null_count: int = 0
    empty_count: int = 0
    present_count: int = 0
    length_min: int = 0
    length_max: int = 0
    length_avg: float = 0.0
    length_median: float = 0.0
    length_p25: float = 0.0
    length_p75: float = 0.0
    length_p90: float = 0.0
    length_p99: float = 0.0


@dataclass
class MarkdownReport:
    """마크다운 패턴 종합"""

    total_affected: int = 0
    patterns: dict[str, PatternResult] = field(default_factory=dict)


@dataclass
class LlmArtifactReport:
    """LLM 아티팩트 종합"""

    total_affected: int = 0
    patterns: dict[str, PatternResult] = field(default_factory=dict)


@dataclass
class FormatReport:
    """포맷 일관성"""

    patterns: dict[str, int] = field(default_factory=dict)


@dataclass
class OutlierReport:
    """이상치"""

    extremely_short: PatternResult = field(default_factory=PatternResult)
    extremely_long: PatternResult = field(default_factory=PatternResult)
    infinite_loop: PatternResult = field(default_factory=PatternResult)


@dataclass
class DuplicateReport:
    """중복"""

    duplicate_groups: int = 0
    duplicate_records: int = 0
    samples: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CrossFieldReport:
    """교차 필드 비교"""

    compare_field: str = ""
    both_present: int = 0
    identical: int = 0
    summary_contains_compare: int = 0
    compare_contains_summary: int = 0
    independent: int = 0
    identical_pct: float = 0.0
    samples_identical: list[dict[str, str]] = field(default_factory=list)


@dataclass
class AuditReport:
    """전체 감사 보고서"""

    metadata: dict[str, Any] = field(default_factory=dict)
    basic_stats: BasicStats = field(default_factory=BasicStats)
    markdown_patterns: MarkdownReport = field(default_factory=MarkdownReport)
    llm_artifacts: LlmArtifactReport = field(default_factory=LlmArtifactReport)
    format_patterns: FormatReport = field(default_factory=FormatReport)
    outliers: OutlierReport = field(default_factory=OutlierReport)
    duplicates: DuplicateReport = field(default_factory=DuplicateReport)
    cross_field_comparison: CrossFieldReport | None = None
    severity_summary: dict[str, list[str]] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# 마크다운 패턴 정의
# ─────────────────────────────────────────────────────────────────────────────

_MARKDOWN_PATTERNS: dict[str, re.Pattern[str]] = {
    "bold_asterisk": re.compile(r"\*\*[^*]+\*\*"),
    "bold_underscore": re.compile(r"__[^_]+__"),
    "italic": re.compile(r"(?<!\*)\*(?!\*)[^*]+\*(?!\*)"),
    "header": re.compile(r"^#{1,6}\s", re.MULTILINE),
    "unordered_list_dash": re.compile(r"^- ", re.MULTILINE),
    "unordered_list_asterisk": re.compile(r"^\* ", re.MULTILINE),
    "ordered_list": re.compile(r"^\d+\.\s", re.MULTILINE),
    "code_block": re.compile(r"```"),
    "inline_code": re.compile(r"`[^`]+`"),
    "link": re.compile(r"\[([^\]]+)\]\([^)]+\)"),
    "horizontal_rule_dash": re.compile(r"^---+$", re.MULTILINE),
    "horizontal_rule_asterisk": re.compile(r"^\*\*\*+$", re.MULTILINE),
    "blockquote": re.compile(r"^>\s", re.MULTILINE),
    "table": re.compile(r"\|.*\|.*\|"),
}

# ─────────────────────────────────────────────────────────────────────────────
# LLM 아티팩트 패턴 정의
# ─────────────────────────────────────────────────────────────────────────────

_LLM_ARTIFACT_PATTERNS: dict[str, re.Pattern[str]] = {
    "char_count": re.compile(r"\(\d+자\)"),
    "keyword_list": re.compile(r"핵심\s?키워드[:\s]"),
    "bracket_header": re.compile(r"\[(쟁점|판단|결론|요약|배경|사실관계|법리)\]"),
    "instruction_leak": re.compile(
        r"요약해|요약하시오|요약을 작성|다음 텍스트|아래 내용"
    ),
    "english_instruction": re.compile(
        r"summarize|as an ai|please provide|in summary", re.IGNORECASE
    ),
    "self_reference": re.compile(r"저는\s|제가\s|AI로서"),
    "html_tag": re.compile(r"<[a-zA-Z][^>]*>"),
}

# 자기참조 오탐 패턴 (법률 텍스트에서 흔한 어미)
_SELF_REF_FALSE_POSITIVE = re.compile(
    r"(?:면제|배제|전제|폐제|부제|공제|규제|통제|억제|제재|절제|자제|조제"
    r"|저는점|제가격|제가치)(?:가|는)"
)


# ─────────────────────────────────────────────────────────────────────────────
# 이상치 기준
# ─────────────────────────────────────────────────────────────────────────────

_SHORT_THRESHOLD = 80  # 극단적 단문 기준 (자)
_LONG_THRESHOLD = 1000  # 극단적 장문 기준 (자)
_LOOP_MIN_LEN = 15  # 반복 판정 최소 구절 길이 (자)
_LOOP_MIN_REPEAT = 5  # 반복 판정 최소 반복 횟수
_LOOP_MIN_RATIO = 0.2  # 반복 구절이 전체 텍스트에서 차지하는 최소 비율


# ─────────────────────────────────────────────────────────────────────────────
# 검사 함수
# ─────────────────────────────────────────────────────────────────────────────


def analyze_basic_stats(texts: list[tuple[str, str]]) -> BasicStats:
    """기본 통계 분석

    Args:
        texts: (item_id, summary_text_or_empty) 튜플 리스트.
               null/None은 빈 문자열로 전달.

    Returns:
        BasicStats 결과
    """
    stats = BasicStats(total=len(texts))
    lengths: list[int] = []

    for _, text in texts:
        if not text:
            stats.null_count += 1
            continue
        stripped = text.strip()
        if not stripped:
            stats.empty_count += 1
            continue
        lengths.append(len(stripped))

    stats.present_count = len(lengths)

    if lengths:
        arr = np.array(lengths)
        stats.length_min = int(np.min(arr))
        stats.length_max = int(np.max(arr))
        stats.length_avg = round(float(np.mean(arr)), 1)
        stats.length_median = float(np.median(arr))
        stats.length_p25 = float(np.percentile(arr, 25))
        stats.length_p75 = float(np.percentile(arr, 75))
        stats.length_p90 = float(np.percentile(arr, 90))
        stats.length_p99 = float(np.percentile(arr, 99))

    return stats


def detect_markdown_patterns(
    texts: list[tuple[str, str]],
    max_samples: int = 5,
) -> MarkdownReport:
    """마크다운 패턴 탐지

    Args:
        texts: (item_id, summary_text) 튜플 리스트
        max_samples: 패턴별 샘플 수

    Returns:
        MarkdownReport 결과
    """
    report = MarkdownReport()
    affected_ids: set[str] = set()

    for name, pattern in _MARKDOWN_PATTERNS.items():
        result = PatternResult()
        for item_id, text in texts:
            if not text or not text.strip():
                continue
            if pattern.search(text):
                result.count += 1
                affected_ids.add(item_id)
                if len(result.samples) < max_samples:
                    match = pattern.search(text)
                    result.samples.append({
                        "id": item_id,
                        "match": match.group()[:80] if match else "",
                        "context": text[:120],
                    })
        report.patterns[name] = result

    report.total_affected = len(affected_ids)
    return report


def detect_llm_artifacts(
    texts: list[tuple[str, str]],
    max_samples: int = 5,
) -> LlmArtifactReport:
    """LLM 아티팩트 탐지

    Args:
        texts: (item_id, summary_text) 튜플 리스트
        max_samples: 패턴별 샘플 수

    Returns:
        LlmArtifactReport 결과
    """
    report = LlmArtifactReport()
    affected_ids: set[str] = set()

    for name, pattern in _LLM_ARTIFACT_PATTERNS.items():
        result = PatternResult()
        for item_id, text in texts:
            if not text or not text.strip():
                continue

            match = pattern.search(text)
            if not match:
                continue

            # 자기참조 오탐 필터링
            if name == "self_reference":
                matched_str = match.group()
                # "제가" → "면제가", "배제가" 등 오탐 확인
                start = max(0, match.start() - 2)
                context_window = text[start : match.end() + 2]
                if _SELF_REF_FALSE_POSITIVE.search(context_window):
                    continue

            result.count += 1
            affected_ids.add(item_id)
            if len(result.samples) < max_samples:
                result.samples.append({
                    "id": item_id,
                    "match": matched_str if name == "self_reference" else match.group()[:80],
                    "context": text[:120],
                })
        report.patterns[name] = result

    report.total_affected = len(affected_ids)
    return report


def analyze_format_patterns(texts: list[tuple[str, str]]) -> FormatReport:
    """포맷 일관성 분석

    텍스트 구조를 키워드 기반으로 분류합니다.

    Args:
        texts: (item_id, summary_text) 튜플 리스트

    Returns:
        FormatReport 결과
    """
    counter: Counter[str] = Counter()
    char_count_re = re.compile(r"\(\d+자\)")

    for _, text in texts:
        if not text or not text.strip():
            counter["empty"] += 1
            continue

        has_issue = "쟁점" in text
        has_judgment = "판단" in text
        has_keyword = bool(re.search(r"핵심\s?키워드", text))
        has_char_count = bool(char_count_re.search(text))

        if has_issue and has_judgment and has_keyword and has_char_count:
            label = "쟁점+판단+키워드+(N자)"
        elif has_issue and has_judgment and has_keyword:
            label = "쟁점+판단+키워드"
        elif has_issue and has_judgment and has_char_count:
            label = "쟁점+판단+(N자)"
        elif has_issue and has_judgment:
            label = "쟁점+판단"
        elif has_issue:
            label = "쟁점만"
        elif has_judgment:
            label = "판단만"
        else:
            label = "자유형식"

        counter[label] += 1

    return FormatReport(patterns=dict(counter.most_common()))


def detect_outliers(
    texts: list[tuple[str, str]],
    max_samples: int = 5,
) -> OutlierReport:
    """이상치 탐지

    극단적 단문(<80자), 극단적 장문(>1000자), 무한 반복(동일 10자+ 구절 3회+).

    Args:
        texts: (item_id, summary_text) 튜플 리스트
        max_samples: 카테고리별 샘플 수

    Returns:
        OutlierReport 결과
    """
    report = OutlierReport()

    for item_id, text in texts:
        if not text or not text.strip():
            continue
        stripped = text.strip()
        length = len(stripped)

        # 극단적 단문
        if length < _SHORT_THRESHOLD:
            report.extremely_short.count += 1
            if len(report.extremely_short.samples) < max_samples:
                report.extremely_short.samples.append({
                    "id": item_id,
                    "length": str(length),
                    "text": stripped[:200],
                })

        # 극단적 장문
        if length > _LONG_THRESHOLD:
            report.extremely_long.count += 1
            if len(report.extremely_long.samples) < max_samples:
                report.extremely_long.samples.append({
                    "id": item_id,
                    "length": str(length),
                    "text": stripped[:200],
                })

        # 무한 반복 감지: 동일 구절이 5회+ 반복 & 전체의 30%+ 차지
        if length >= _LOOP_MIN_LEN * _LOOP_MIN_REPEAT:
            found_loop = False
            for substr_len in range(_LOOP_MIN_LEN, min(50, length // _LOOP_MIN_REPEAT) + 1):
                for start in range(0, length - substr_len, substr_len):
                    substr = stripped[start : start + substr_len]
                    repeat_count = stripped.count(substr)
                    if repeat_count >= _LOOP_MIN_REPEAT:
                        # 반복 구절이 전체 텍스트에서 차지하는 비율 확인
                        repeated_chars = repeat_count * substr_len
                        ratio = repeated_chars / length
                        if ratio >= _LOOP_MIN_RATIO:
                            found_loop = True
                            report.infinite_loop.count += 1
                            if len(report.infinite_loop.samples) < max_samples:
                                report.infinite_loop.samples.append({
                                    "id": item_id,
                                    "length": str(length),
                                    "repeating_phrase": substr[:60],
                                    "repeat_count": str(repeat_count),
                                    "ratio": f"{ratio:.1%}",
                                })
                            break
                if found_loop:
                    break

    return report


def detect_duplicates(
    texts: list[tuple[str, str]],
    max_samples: int = 5,
) -> DuplicateReport:
    """중복 탐지

    완전 동일한 요약 텍스트를 찾습니다.

    Args:
        texts: (item_id, summary_text) 튜플 리스트
        max_samples: 샘플 그룹 수

    Returns:
        DuplicateReport 결과
    """
    # 텍스트 → ID 목록 매핑
    text_to_ids: defaultdict[str, list[str]] = defaultdict(list)
    for item_id, text in texts:
        if not text or not text.strip():
            continue
        text_to_ids[text.strip()].append(item_id)

    # 2건 이상 동일 텍스트 = 중복 그룹
    dup_groups = {t: ids for t, ids in text_to_ids.items() if len(ids) >= 2}

    report = DuplicateReport(
        duplicate_groups=len(dup_groups),
        duplicate_records=sum(len(ids) for ids in dup_groups.values()),
    )

    for text, ids in list(dup_groups.items())[:max_samples]:
        report.samples.append({
            "ids": ids[:5],
            "count": len(ids),
            "text_preview": text[:120],
        })

    return report


def compare_cross_fields(
    items: list[dict[str, Any]],
    summary_field: str,
    compare_field: str,
    id_field: str,
    max_samples: int = 5,
) -> CrossFieldReport:
    """교차 필드 비교

    요약 필드와 비교 필드 간 관계를 분석합니다.

    Args:
        items: 원본 레코드 리스트
        summary_field: 요약 필드명
        compare_field: 비교 필드명
        id_field: ID 필드명
        max_samples: 동일 샘플 수

    Returns:
        CrossFieldReport 결과
    """
    report = CrossFieldReport(compare_field=compare_field)

    for item in items:
        summary = item.get(summary_field)
        compare = item.get(compare_field)
        item_id = str(item.get(id_field, ""))

        if not summary or not compare:
            continue
        s_stripped = str(summary).strip()
        c_stripped = str(compare).strip()
        if not s_stripped or not c_stripped:
            continue

        report.both_present += 1

        if s_stripped == c_stripped:
            report.identical += 1
            if len(report.samples_identical) < max_samples:
                report.samples_identical.append({
                    "id": item_id,
                    "text_preview": s_stripped[:120],
                })
        elif c_stripped in s_stripped:
            report.summary_contains_compare += 1
        elif s_stripped in c_stripped:
            report.compare_contains_summary += 1
        else:
            report.independent += 1

    if report.both_present > 0:
        report.identical_pct = round(
            report.identical / report.both_present * 100, 2
        )

    return report


# ─────────────────────────────────────────────────────────────────────────────
# JSON 로딩 (scripts.common.json_loader 위임)
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_source_path(cfg: IngestConfig, data_dir: Path | None) -> Path:
    """소스 경로 해석 (--data-dir 재매핑 지원)"""
    return resolve_source_path(cfg.source_path, data_dir, _DEFAULT_INGEST_SOURCE_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# 오케스트레이터
# ─────────────────────────────────────────────────────────────────────────────


def run_audit(
    items: list[dict[str, Any]],
    summary_field: str,
    id_field: str,
    compare_field: str | None = None,
    max_samples: int = 5,
    source_info: str = "",
) -> AuditReport:
    """전체 감사 실행

    Args:
        items: 원본 JSON 레코드 리스트
        summary_field: 요약 필드명
        id_field: ID 필드명
        compare_field: 교차 비교 필드명 (선택)
        max_samples: 이상치 샘플 수
        source_info: 보고서 메타데이터용 소스 정보

    Returns:
        AuditReport 결과
    """
    report = AuditReport()
    report.metadata = {
        "source": source_info,
        "summary_field": summary_field,
        "id_field": id_field,
        "compare_field": compare_field,
        "total_records": len(items),
    }

    # 텍스트 추출
    texts: list[tuple[str, str]] = []
    for item in items:
        item_id = str(item.get(id_field, ""))
        summary = item.get(summary_field)
        text = str(summary).strip() if summary is not None else ""
        texts.append((item_id, text))

    logger.info("감사 시작: %d건, 필드=%s", len(texts), summary_field)

    # 1. 기본 통계
    logger.info("  [1/7] 기본 통계 분석...")
    report.basic_stats = analyze_basic_stats(texts)

    # 2. 마크다운 패턴
    logger.info("  [2/7] 마크다운 패턴 탐지...")
    report.markdown_patterns = detect_markdown_patterns(texts, max_samples)

    # 3. LLM 아티팩트
    logger.info("  [3/7] LLM 아티팩트 탐지...")
    report.llm_artifacts = detect_llm_artifacts(texts, max_samples)

    # 4. 포맷 일관성
    logger.info("  [4/7] 포맷 일관성 분석...")
    report.format_patterns = analyze_format_patterns(texts)

    # 5. 이상치
    logger.info("  [5/7] 이상치 탐지...")
    report.outliers = detect_outliers(texts, max_samples)

    # 6. 중복
    logger.info("  [6/7] 중복 탐지...")
    report.duplicates = detect_duplicates(texts, max_samples)

    # 7. 교차 비교 (선택)
    if compare_field:
        logger.info("  [7/7] 교차 필드 비교 (%s)...", compare_field)
        report.cross_field_comparison = compare_cross_fields(
            items, summary_field, compare_field, id_field, max_samples
        )
    else:
        logger.info("  [7/7] 교차 비교 생략 (--compare-field 미지정)")

    # 심각도 요약
    report.severity_summary = _compute_severity(report)

    logger.info("감사 완료")
    return report


def run_audit_for_type(
    type_name: str,
    data_dir: Path | None = None,
    compare_field: str | None = None,
    max_samples: int = 5,
) -> AuditReport:
    """IngestConfig 연동 감사 래퍼

    Args:
        type_name: 등록된 인제스트 타입명
        data_dir: 데이터 디렉토리 재매핑 (선택)
        compare_field: 교차 비교 필드명 (선택)
        max_samples: 이상치 샘플 수

    Returns:
        AuditReport 결과
    """
    cfg = get_config(type_name)
    source = _resolve_source_path(cfg, data_dir)
    logger.info("[%s] %s — 소스: %s", type_name, cfg.data_type_label, source)

    items = _common_load_items(source)
    return run_audit(
        items=items,
        summary_field=cfg.summary_field,
        id_field=cfg.id_field,
        compare_field=compare_field,
        max_samples=max_samples,
        source_info=str(source),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 심각도 판정
# ─────────────────────────────────────────────────────────────────────────────


def _compute_severity(report: AuditReport) -> dict[str, list[str]]:
    """검사 결과를 심각도별로 분류"""
    high: list[str] = []
    medium: list[str] = []
    low: list[str] = []

    total = report.basic_stats.total
    if total == 0:
        return {"HIGH": high, "MEDIUM": medium, "LOW": low}

    # null/empty 비율
    null_empty = report.basic_stats.null_count + report.basic_stats.empty_count
    if null_empty > 0:
        pct = null_empty / total * 100
        if pct > 5:
            high.append(f"null/empty: {null_empty:,}건 ({pct:.1f}%)")
        elif pct > 1:
            medium.append(f"null/empty: {null_empty:,}건 ({pct:.1f}%)")
        else:
            low.append(f"null/empty: {null_empty:,}건 ({pct:.1f}%)")

    # LLM 아티팩트
    for name, result in report.llm_artifacts.patterns.items():
        if result.count == 0:
            continue
        pct = result.count / total * 100
        label = f"LLM 아티팩트({name}): {result.count:,}건 ({pct:.1f}%)"
        if pct > 10:
            high.append(label)
        elif pct > 1:
            medium.append(label)
        else:
            low.append(label)

    # 마크다운
    if report.markdown_patterns.total_affected > 0:
        pct = report.markdown_patterns.total_affected / total * 100
        label = f"마크다운 오염: {report.markdown_patterns.total_affected:,}건 ({pct:.1f}%)"
        if pct > 5:
            medium.append(label)
        else:
            low.append(label)

    # 이상치
    if report.outliers.extremely_short.count > 0:
        low.append(
            f"극단적 단문(<{_SHORT_THRESHOLD}자): "
            f"{report.outliers.extremely_short.count:,}건"
        )
    if report.outliers.extremely_long.count > 0:
        low.append(
            f"극단적 장문(>{_LONG_THRESHOLD}자): "
            f"{report.outliers.extremely_long.count:,}건"
        )
    if report.outliers.infinite_loop.count > 0:
        high.append(
            f"무한 반복: {report.outliers.infinite_loop.count:,}건"
        )

    # 중복
    if report.duplicates.duplicate_groups > 0:
        low.append(
            f"중복: {report.duplicates.duplicate_groups}개 그룹, "
            f"{report.duplicates.duplicate_records:,}건"
        )

    return {"HIGH": high, "MEDIUM": medium, "LOW": low}


# ─────────────────────────────────────────────────────────────────────────────
# 보고서 생성
# ─────────────────────────────────────────────────────────────────────────────


def format_terminal_report(report: AuditReport) -> str:
    """ASCII 터미널 출력용 보고서 포맷"""
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 60)
    lines.append("  LLM 요약 품질 탐색적 감사 보고서")
    lines.append("=" * 60)

    # 메타
    meta = report.metadata
    lines.append(f"  소스: {meta.get('source', '?')}")
    lines.append(f"  필드: {meta.get('summary_field', '?')}")
    lines.append(f"  총 레코드: {meta.get('total_records', 0):,}")
    lines.append("")

    # 1. 기본 통계
    bs = report.basic_stats
    lines.append("─" * 60)
    lines.append("  [1] 기본 통계")
    lines.append("─" * 60)
    lines.append(
        f"  총: {bs.total:,} | 유효: {bs.present_count:,} | "
        f"null: {bs.null_count:,} | empty: {bs.empty_count:,}"
    )
    if bs.present_count > 0:
        lines.append(
            f"  길이: min={bs.length_min} P25={bs.length_p25:.0f} "
            f"median={bs.length_median:.0f} P75={bs.length_p75:.0f} "
            f"P90={bs.length_p90:.0f} P99={bs.length_p99:.0f} "
            f"max={bs.length_max} avg={bs.length_avg:.1f}"
        )
    lines.append("")

    # 2. 마크다운
    md = report.markdown_patterns
    lines.append("─" * 60)
    lines.append(f"  [2] 마크다운 패턴 (영향: {md.total_affected:,}건)")
    lines.append("─" * 60)
    for name, result in md.patterns.items():
        if result.count > 0:
            lines.append(f"    {name}: {result.count:,}건")
    if md.total_affected == 0:
        lines.append("    (감지 없음)")
    lines.append("")

    # 3. LLM 아티팩트
    llm = report.llm_artifacts
    lines.append("─" * 60)
    lines.append(f"  [3] LLM 아티팩트 (영향: {llm.total_affected:,}건)")
    lines.append("─" * 60)
    for name, result in llm.patterns.items():
        if result.count > 0:
            pct = result.count / max(bs.total, 1) * 100
            lines.append(f"    {name}: {result.count:,}건 ({pct:.2f}%)")
    if llm.total_affected == 0:
        lines.append("    (감지 없음)")
    lines.append("")

    # 4. 포맷 일관성
    fmt = report.format_patterns
    lines.append("─" * 60)
    lines.append("  [4] 포맷 일관성")
    lines.append("─" * 60)
    for label, count in fmt.patterns.items():
        pct = count / max(bs.total, 1) * 100
        lines.append(f"    {label}: {count:,}건 ({pct:.1f}%)")
    lines.append("")

    # 5. 이상치
    ol = report.outliers
    lines.append("─" * 60)
    lines.append("  [5] 이상치")
    lines.append("─" * 60)
    lines.append(f"    극단적 단문(<{_SHORT_THRESHOLD}자): {ol.extremely_short.count:,}건")
    lines.append(f"    극단적 장문(>{_LONG_THRESHOLD}자): {ol.extremely_long.count:,}건")
    lines.append(f"    무한 반복: {ol.infinite_loop.count:,}건")
    lines.append("")

    # 6. 중복
    dup = report.duplicates
    lines.append("─" * 60)
    lines.append("  [6] 중복")
    lines.append("─" * 60)
    lines.append(
        f"    중복 그룹: {dup.duplicate_groups}개 | "
        f"중복 레코드: {dup.duplicate_records:,}건"
    )
    lines.append("")

    # 7. 교차 비교
    if report.cross_field_comparison:
        cf = report.cross_field_comparison
        lines.append("─" * 60)
        lines.append(f"  [7] 교차 비교 (vs {cf.compare_field})")
        lines.append("─" * 60)
        lines.append(f"    양쪽 모두 존재: {cf.both_present:,}건")
        lines.append(f"    완전 동일: {cf.identical:,}건 ({cf.identical_pct:.1f}%)")
        lines.append(f"    요약⊃비교: {cf.summary_contains_compare:,}건")
        lines.append(f"    비교⊃요약: {cf.compare_contains_summary:,}건")
        lines.append(f"    독립적: {cf.independent:,}건")
        lines.append("")

    # 심각도 요약
    sev = report.severity_summary
    lines.append("=" * 60)
    lines.append("  심각도 요약")
    lines.append("=" * 60)
    for level in ("HIGH", "MEDIUM", "LOW"):
        items = sev.get(level, [])
        if items:
            lines.append(f"  [{level}]")
            for item in items:
                lines.append(f"    - {item}")
    if not any(sev.values()):
        lines.append("  (이슈 없음)")
    lines.append("")

    return "\n".join(lines)


def to_json_report(report: AuditReport) -> dict[str, Any]:
    """JSON 직렬화 가능한 보고서 생성"""
    data = asdict(report)
    # numpy 타입을 Python 기본 타입으로 변환
    result: dict[str, Any] = json.loads(
        json.dumps(data, default=_json_default, ensure_ascii=False)
    )
    return result


def _json_default(obj: Any) -> Any:
    """JSON 직렬화 헬퍼 (numpy 등 변환)"""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM 요약 필드 탐색적 품질 감사",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "예시:\n"
            "  # IngestConfig 등록 타입\n"
            "  uv run python scripts/audit_summary_quality.py "
            "--type special_admin_appeal --data-dir ../data\n\n"
            "  # 임의 JSON 파일\n"
            "  uv run python scripts/audit_summary_quality.py "
            "--file ../data/file.json --summary-field 심판례요약 "
            "--id-field ID\n"
        ),
    )
    # 소스 지정 (type 또는 file)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--type",
        dest="type_name",
        default=None,
        help=(
            f"인제스트 타입명 (자동 필드 해석). "
            f"등록 타입: {', '.join(list_configs())}"
        ),
    )
    source_group.add_argument(
        "--file",
        dest="file_path",
        default=None,
        help="임의 JSON 파일 경로 (--summary-field, --id-field 필수)",
    )

    # 필드 지정 (--file 모드용)
    parser.add_argument(
        "--summary-field",
        default=None,
        help="요약 필드명 (--file 모드 필수)",
    )
    parser.add_argument(
        "--id-field",
        default=None,
        help="ID 필드명 (--file 모드 필수, 미지정 시 인덱스 사용)",
    )

    # 공통 옵션
    parser.add_argument(
        "--data-dir",
        default=None,
        help="데이터 기본 디렉토리 (--type 전용, config 경로 재매핑)",
    )
    parser.add_argument(
        "--compare-field",
        default=None,
        help="교차 비교 대상 필드명 (예: 재결요지, 판결요지)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="JSON 보고서 저장 경로 (예: eda_output/audit_report.json)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="이상치 샘플 수 (기본: 5)",
    )
    args = parser.parse_args()

    # --type 모드
    if args.type_name:
        data_dir: Path | None = None
        if args.data_dir:
            data_dir = Path(args.data_dir).resolve()
            if not data_dir.exists():
                logger.error("--data-dir 경로가 존재하지 않습니다: %s", data_dir)
                sys.exit(1)

        audit_report = run_audit_for_type(
            type_name=args.type_name,
            data_dir=data_dir,
            compare_field=args.compare_field,
            max_samples=args.samples,
        )

    # --file 모드
    else:
        file_path = Path(args.file_path).resolve()
        if not file_path.exists():
            logger.error("파일이 존재하지 않습니다: %s", file_path)
            sys.exit(1)

        if not args.summary_field:
            logger.error("--file 모드에서 --summary-field는 필수입니다")
            sys.exit(1)

        id_field = args.id_field or "_index"
        items = _common_load_items(file_path)

        # ID 필드가 없으면 인덱스로 대체
        if id_field == "_index":
            for i, item in enumerate(items):
                item["_index"] = str(i)

        audit_report = run_audit(
            items=items,
            summary_field=args.summary_field,
            id_field=id_field,
            compare_field=args.compare_field,
            max_samples=args.samples,
            source_info=str(file_path),
        )

    # 터미널 출력
    print(format_terminal_report(audit_report))

    # JSON 저장
    if args.output:
        output_path = _BACKEND_ROOT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                to_json_report(audit_report), f, ensure_ascii=False, indent=2
            )
        logger.info("JSON 보고서 저장: %s", output_path)


if __name__ == "__main__":
    main()
