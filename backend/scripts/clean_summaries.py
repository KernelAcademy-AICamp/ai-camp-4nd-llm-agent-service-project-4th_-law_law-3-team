"""
LLM 요약 필드 후처리(클리닝) 스크립트

19개 데이터 타입의 JSON 소스에서 LLM이 생성한 요약 필드를 규칙 기반으로 클리닝합니다.

클리닝 규칙 (우선순위 순):
  Phase 1: 마크다운/HTML 잔여 제거 (**볼드**, ##헤딩, <p> 등)
  Phase 2: 불완전 문장 보정 (트레일링 "...", 열린 괄호)
  Phase 3: null/empty → 제목 필드 fallback
  Phase 4: LLM 아티팩트 제거 (핵심키워드 리스트, (NNN자), [쟁점]/[판단] 브라켓, 자기참조, 무한 반복)

수정하지 않는 항목:
  - 길이 초과 (내용 정상이면 허용)
  - 언어 혼용 (대부분 법률 고유명사)
  - 극단적 단문 (실제 짧은 내용일 수 있음)

Usage:
    cd backend
    uv run python scripts/clean_summaries.py --data-dir ../data                    # 전체 클리닝
    uv run python scripts/clean_summaries.py --data-dir ../data --type law         # 특정 타입
    uv run python scripts/clean_summaries.py --data-dir ../data --dry-run          # 드라이런
    uv run python scripts/clean_summaries.py --data-dir ../data --output-dir ../data
    uv run python scripts/clean_summaries.py --data-dir ../data --report eda_output/clean_report.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# backend/ 루트를 sys.path에 추가
_BACKEND_ROOT = Path(__file__).parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

# 인제스트 config 레지스트리 로드 (types/ 자동 등록 트리거)
import scripts.ingest.types  # noqa: F401, E402
from scripts.common.json_loader import resolve_source_path  # noqa: E402
from scripts.common.logging_config import setup_logging  # noqa: E402
from scripts.ingest.config import (  # noqa: E402
    DATA_DIR,
    IngestConfig,
    get_config,
    list_configs,
)

# 인제스트 config의 기본 소스 베이스 디렉토리
_DEFAULT_INGEST_SOURCE_DIR = DATA_DIR

logger = setup_logging(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 구조
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class FieldCleanReport:
    """단일 필드(요약 종류)에 대한 클리닝 결과"""

    field_name: str
    total: int = 0
    cleaned: int = 0
    markdown_removed: int = 0
    incomplete_fixed: int = 0
    null_replaced: int = 0
    artifact_removed: int = 0


@dataclass
class TypeCleanReport:
    """타입별 클리닝 결과"""

    type_name: str
    label: str
    fields: list[FieldCleanReport] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# 클리닝 정규식
# ─────────────────────────────────────────────────────────────────────────────

# Phase 1: 마크다운/HTML 제거
_RE_BOLD_ASTERISK = re.compile(r"\*\*(.+?)\*\*")
_RE_BOLD_UNDERSCORE = re.compile(r"__(.+?)__")
_RE_ITALIC_ASTERISK = re.compile(r"\*(.+?)\*")
_RE_ITALIC_UNDERSCORE = re.compile(r"(?<!\w)_(.+?)_(?!\w)")
_RE_HEADING = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_RE_HTML_TAG = re.compile(
    r"<\s*/?\s*(p|br|div|span|b|i|em|strong|h[1-6]|ul|ol|li|a|table|tr|td|th)\s*/?\s*>",
    re.IGNORECASE,
)
_RE_MULTI_NEWLINE = re.compile(r"\n{3,}")
_RE_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_RE_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_RE_MARKDOWN_LIST = re.compile(r"^[\s]*[-*+]\s+", re.MULTILINE)
_RE_BACKTICK = re.compile(r"`([^`]+)`")
_RE_BLOCKQUOTE = re.compile(r"^>\s+", re.MULTILINE)
_RE_HORIZONTAL_RULE = re.compile(r"^[-*_]{3,}\s*$", re.MULTILINE)

# Phase 4: LLM 아티팩트 제거
# (NNN자) 글자수 표기
_RE_CHAR_COUNT = re.compile(r"\s*\(\s*\d{2,5}\s*자\s*\)\s*")
# 핵심키워드: 리스트 (줄 끝에 위치하는 경우)
_RE_KEYWORD_LIST = re.compile(
    r"[\s.。]*(?:핵심\s*키워드|키워드|핵심\s*쟁점\s*키워드|관련\s*키워드|주요\s*키워드)\s*[:：]\s*.+$",
    re.MULTILINE,
)
# [쟁점], [판단], [결론] 등 브라켓 헤더
_RE_BRACKET_HEADER = re.compile(r"\[(?:쟁점|판단|결론|판결|요지|사실관계|주문|이유)\]\s*[:：]?\s*")
# 자기참조 ("이 요약은", "본 요약문은", "위 판례는" 등)
_RE_SELF_REFERENCE = re.compile(
    r"(?:이|본|위|해당)\s*(?:요약|요약문|판례요약|결정요약|해석요약|심판례요약)(?:은|는|에서|의)\s*",
)
# 무한 반복 탐지: 15자 이상 구문이 5회 이상 반복
_INFINITE_LOOP_MIN_PHRASE = 15
_INFINITE_LOOP_MIN_REPEATS = 5
_INFINITE_LOOP_MAX_LENGTH = 1000  # 반복 제거 후 최대 길이

# Phase 2: 불완전 문장 보정
_RE_TRAILING_DOTS = re.compile(r"\.{2,}\s*$")
_RE_TRAILING_OPEN_PAREN = re.compile(r"\([^)]*$")
_RE_TRAILING_OPEN_BRACKET = re.compile(r"[「『\[][^」』\]]*$")
# 마지막 완전한 문장 경계 (한국어 종결어미 + 마침표)
_RE_LAST_COMPLETE_SENTENCE = re.compile(
    r".*[.다함됨임음있없었였됨짐짐][\s.]*",
    re.DOTALL,
)


# ─────────────────────────────────────────────────────────────────────────────
# 클리너
# ─────────────────────────────────────────────────────────────────────────────


class SummaryCleaner:
    """LLM 요약 필드 후처리기"""

    def clean_text(self, text: str) -> tuple[str, bool, bool, bool, bool]:
        """단일 요약 텍스트 클리닝 (Phase 1→4→2 순차 적용)

        Returns:
            (클리닝된 텍스트, 마크다운_제거됨, 불완전_보정됨, 아티팩트_제거됨, 변경됨)
        """
        original = text
        markdown_removed = False
        incomplete_fixed = False
        artifact_removed = False

        # Phase 1: 마크다운/HTML 제거
        cleaned = self._remove_markdown_html(text)
        if cleaned != text:
            markdown_removed = True
            text = cleaned

        # Phase 4: LLM 아티팩트 제거 (Phase 2 전에 실행)
        cleaned = self._remove_llm_artifacts(text)
        if cleaned != text:
            artifact_removed = True
            text = cleaned

        # Phase 2: 불완전 문장 보정
        cleaned = self._fix_incomplete(text)
        if cleaned != text:
            incomplete_fixed = True
            text = cleaned

        changed = text != original
        return text, markdown_removed, incomplete_fixed, artifact_removed, changed

    def _remove_markdown_html(self, text: str) -> str:
        """Phase 1: 마크다운/HTML 잔여 제거"""
        # 마크다운 링크 → 텍스트만
        text = _RE_MARKDOWN_LINK.sub(r"\1", text)
        # 볼드 **text** → text
        text = _RE_BOLD_ASTERISK.sub(r"\1", text)
        # 볼드 __text__ → text
        text = _RE_BOLD_UNDERSCORE.sub(r"\1", text)
        # 이탤릭 *text* → text
        text = _RE_ITALIC_ASTERISK.sub(r"\1", text)
        # 이탤릭 _text_ → text
        text = _RE_ITALIC_UNDERSCORE.sub(r"\1", text)
        # 헤딩 ## 제목 → 제목
        text = _RE_HEADING.sub("", text)
        # HTML 태그 제거
        text = _RE_HTML_TAG.sub(" ", text)
        # 인라인 코드 `code` → code
        text = _RE_BACKTICK.sub(r"\1", text)
        # 블록 인용 > text → text
        text = _RE_BLOCKQUOTE.sub("", text)
        # 리스트 마커 - text → text
        text = _RE_MARKDOWN_LIST.sub("", text)
        # 수평선 --- → 제거
        text = _RE_HORIZONTAL_RULE.sub("", text)
        # 연속 줄바꿈 정리
        text = _RE_MULTI_NEWLINE.sub("\n\n", text)
        # 연속 공백 정리
        text = _RE_MULTI_SPACE.sub(" ", text)
        # 앞뒤 공백 정리
        text = text.strip()
        return text

    def _fix_incomplete(self, text: str) -> str:
        """Phase 2: 불완전 문장 보정"""
        # 트레일링 "..." 또는 ".." → "."
        if _RE_TRAILING_DOTS.search(text):
            text = _RE_TRAILING_DOTS.sub(".", text)

        # 열린 괄호로 끝나는 경우 → 해당 부분 제거
        if _RE_TRAILING_OPEN_PAREN.search(text):
            text = _RE_TRAILING_OPEN_PAREN.sub("", text).rstrip()
            if text and not text.endswith((".", "다", "함", "됨", "임", "음")):
                text = text.rstrip(",;: ") + "."

        # 열린 인용부호로 끝나는 경우 → 해당 부분 제거
        if _RE_TRAILING_OPEN_BRACKET.search(text):
            text = _RE_TRAILING_OPEN_BRACKET.sub("", text).rstrip()
            if text and not text.endswith((".", "다", "함", "됨", "임", "음")):
                text = text.rstrip(",;: ") + "."

        return text.strip()

    def _remove_llm_artifacts(self, text: str) -> str:
        """Phase 4: LLM 아티팩트 제거"""
        # (NNN자) 글자수 표기 제거
        text = _RE_CHAR_COUNT.sub(" ", text)
        # 핵심키워드: 리스트 제거 (줄 끝 패턴)
        text = _RE_KEYWORD_LIST.sub("", text)
        # [쟁점], [판단] 등 브라켓 헤더 → 제거 (내용은 유지)
        text = _RE_BRACKET_HEADER.sub("", text)
        # 자기참조 표현 제거
        text = _RE_SELF_REFERENCE.sub("", text)
        # 무한 반복 탐지 및 잘라내기
        text = self._truncate_infinite_loop(text)
        # 연속 공백/줄바꿈 정리
        text = _RE_MULTI_SPACE.sub(" ", text)
        text = _RE_MULTI_NEWLINE.sub("\n\n", text)
        return text.strip()

    def _truncate_infinite_loop(self, text: str) -> str:
        """무한 반복 구간을 탐지하여 첫 등장까지만 남기고 잘라냄"""
        if len(text) <= _INFINITE_LOOP_MAX_LENGTH:
            return text
        # 15자 이상 구문이 5회 이상 반복되는지 검사
        for phrase_len in range(30, _INFINITE_LOOP_MIN_PHRASE - 1, -1):
            # 텍스트 중반부터 반복 구문 후보 추출
            mid = len(text) // 2
            candidate = text[mid : mid + phrase_len]
            count = text.count(candidate)
            if count >= _INFINITE_LOOP_MIN_REPEATS:
                # 첫 2회 등장까지 유지
                first = text.find(candidate)
                second = text.find(candidate, first + phrase_len)
                if second != -1:
                    # 두 번째 등장 이후 적절한 문장 종결 위치에서 잘라냄
                    cut_pos = second + phrase_len
                    # 문장 종결 위치 탐색 (다, 함, 됨, 임, 음, .)
                    for end_marker in ("다.", "함.", "됨.", "임.", "음.", ". "):
                        end_idx = text.find(end_marker, cut_pos)
                        if end_idx != -1 and end_idx < cut_pos + 200:
                            return text[: end_idx + len(end_marker)].strip()
                    # 마커 못 찾으면 cut_pos에서 자르기
                    return text[:cut_pos].rstrip(",;: ") + "."
        return text

    def _make_fallback_summary(self, title: str) -> str:
        """Phase 3: null/empty에 대한 fallback 요약 생성"""
        if title:
            return f"{title}에 관한 내용"
        return ""

    # ─────────────────────────────────────────────────────────────────────
    # 타입별 클리닝
    # ─────────────────────────────────────────────────────────────────────

    def clean_law_type(
        self,
        items: list[dict[str, Any]],
        cfg: IngestConfig,
        dry_run: bool = False,
    ) -> tuple[list[dict[str, Any]], TypeCleanReport]:
        """법령 타입 클리닝 (전체 요약 + 조문별 요약)"""
        report = TypeCleanReport(type_name=cfg.name, label=cfg.data_type_label)
        overall_rpt = FieldCleanReport(field_name="법령 요약")
        article_rpt = FieldCleanReport(field_name="조문요약")

        for item in items:
            # 전체 법령 요약
            summary = item.get("법령 요약")
            overall_rpt.total += 1

            if summary is None or (isinstance(summary, str) and not summary.strip()):
                overall_rpt.null_replaced += 1
                overall_rpt.cleaned += 1
                if not dry_run:
                    title = item.get(cfg.title_field, "")
                    item["법령 요약"] = self._make_fallback_summary(str(title))
            elif isinstance(summary, str):
                cleaned, md, inc, art, changed = self.clean_text(summary)
                if changed:
                    overall_rpt.cleaned += 1
                    if md:
                        overall_rpt.markdown_removed += 1
                    if inc:
                        overall_rpt.incomplete_fixed += 1
                    if art:
                        overall_rpt.artifact_removed += 1
                    if not dry_run:
                        item["법령 요약"] = cleaned

            # 조문별 요약
            articles = item.get("조문")
            if isinstance(articles, list):
                for article in articles:
                    if not isinstance(article, dict):
                        continue
                    art_summary = article.get("조문요약")
                    article_rpt.total += 1

                    if art_summary is None or (
                        isinstance(art_summary, str) and not art_summary.strip()
                    ):
                        article_rpt.null_replaced += 1
                        article_rpt.cleaned += 1
                        if not dry_run:
                            art_title = article.get("조문제목", "")
                            article["조문요약"] = self._make_fallback_summary(
                                str(art_title)
                            )
                    elif isinstance(art_summary, str):
                        cleaned, md, inc, artf, changed = self.clean_text(art_summary)
                        if changed:
                            article_rpt.cleaned += 1
                            if md:
                                article_rpt.markdown_removed += 1
                            if inc:
                                article_rpt.incomplete_fixed += 1
                            if artf:
                                article_rpt.artifact_removed += 1
                            if not dry_run:
                                article["조문요약"] = cleaned

        report.fields = [overall_rpt, article_rpt]
        return items, report

    def clean_generic_type(
        self,
        items: list[dict[str, Any]],
        cfg: IngestConfig,
        dry_run: bool = False,
    ) -> tuple[list[dict[str, Any]], TypeCleanReport]:
        """일반 타입 클리닝 (단일 요약 필드)"""
        report = TypeCleanReport(type_name=cfg.name, label=cfg.data_type_label)
        fld_rpt = FieldCleanReport(field_name=cfg.summary_field)

        for item in items:
            summary = item.get(cfg.summary_field)
            fld_rpt.total += 1

            if summary is None or (isinstance(summary, str) and not summary.strip()):
                fld_rpt.null_replaced += 1
                fld_rpt.cleaned += 1
                if not dry_run:
                    title = item.get(cfg.title_field, "")
                    item[cfg.summary_field] = self._make_fallback_summary(str(title))
            elif isinstance(summary, str):
                cleaned, md, inc, art, changed = self.clean_text(summary)
                if changed:
                    fld_rpt.cleaned += 1
                    if md:
                        fld_rpt.markdown_removed += 1
                    if inc:
                        fld_rpt.incomplete_fixed += 1
                    if art:
                        fld_rpt.artifact_removed += 1
                    if not dry_run:
                        item[cfg.summary_field] = cleaned

        report.fields = [fld_rpt]
        return items, report


# ─────────────────────────────────────────────────────────────────────────────
# JSON I/O
# ─────────────────────────────────────────────────────────────────────────────


def _load_json(source_path: Path) -> tuple[Any, bool]:
    """JSON 파일 로드. 반환: (데이터, is_list)"""
    with open(source_path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data, True
    # {"items": [...]} 형식
    if isinstance(data, dict) and "items" in data:
        return data, False
    return data, False


def _load_json_directory(dir_path: Path) -> list[tuple[Path, Any, bool]]:
    """디렉토리 내 모든 .json 파일을 개별 로드"""
    json_files = sorted(dir_path.glob("*.json"))
    results: list[tuple[Path, Any, bool]] = []
    for json_file in json_files:
        try:
            data, is_list = _load_json(json_file)
            results.append((json_file, data, is_list))
        except json.JSONDecodeError as e:
            logger.warning("JSON 파싱 실패, 건너뜀: %s (%s)", json_file.name, e)
    return results


def _save_json(data: Any, output_path: Path, is_list: bool) -> None:
    """JSON 저장 (원본 형식 유지)"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        if is_list:
            json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            # dict 형식이면 원본 구조 유지
            json.dump(data, f, ensure_ascii=False, indent=2)


def _get_items(data: Any, is_list: bool) -> list[dict[str, Any]]:
    """데이터에서 아이템 리스트 추출"""
    if is_list and isinstance(data, list):
        return data
    if isinstance(data, dict) and "items" in data:
        return data["items"]
    return []


# ─────────────────────────────────────────────────────────────────────────────
# 소스 경로 해석 (scripts.common.json_loader 위임)
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_source_path(cfg: IngestConfig, data_dir: Path | None) -> Path:
    """소스 경로 해석 (--data-dir 재매핑 지원)"""
    return resolve_source_path(cfg.source_path, data_dir, _DEFAULT_INGEST_SOURCE_DIR)


def _resolve_output_path(
    source_path: Path, data_dir: Path, output_dir: Path
) -> Path:
    """출력 경로 결정 (소스 경로의 상대 위치를 output_dir에 매핑)"""
    try:
        relative = source_path.relative_to(data_dir)
    except ValueError:
        # data_dir 밖의 경로면 파일명만 사용
        relative = Path(source_path.name)
    return output_dir / relative


# ─────────────────────────────────────────────────────────────────────────────
# 메인 처리
# ─────────────────────────────────────────────────────────────────────────────


def process_type(
    type_name: str,
    data_dir: Path,
    output_dir: Path,
    dry_run: bool = False,
) -> TypeCleanReport | None:
    """단일 타입의 JSON 파일 클리닝"""
    cfg = get_config(type_name)
    source = _resolve_source_path(cfg, data_dir)

    if not source.exists():
        logger.warning("[%s] 소스 없음: %s", type_name, source)
        return None

    logger.info("[%s] %s 클리닝 시작: %s", type_name, cfg.data_type_label, source)

    cleaner = SummaryCleaner()

    if source.is_dir():
        return _process_directory(
            type_name, cfg, source, data_dir, output_dir, cleaner, dry_run
        )
    return _process_single_file(
        type_name, cfg, source, data_dir, output_dir, cleaner, dry_run
    )


def _process_single_file(
    type_name: str,
    cfg: IngestConfig,
    source: Path,
    data_dir: Path,
    output_dir: Path,
    cleaner: SummaryCleaner,
    dry_run: bool,
) -> TypeCleanReport:
    """단일 JSON 파일 클리닝"""
    data, is_list = _load_json(source)
    items = _get_items(data, is_list)

    if type_name == "law":
        items, report = cleaner.clean_law_type(items, cfg, dry_run=dry_run)
    else:
        items, report = cleaner.clean_generic_type(items, cfg, dry_run=dry_run)

    if not dry_run:
        output_path = _resolve_output_path(source, data_dir, output_dir)
        _save_json(data, output_path, is_list)
        logger.info("[%s] 저장: %s", type_name, output_path)

    return report


def _process_directory(
    type_name: str,
    cfg: IngestConfig,
    source: Path,
    data_dir: Path,
    output_dir: Path,
    cleaner: SummaryCleaner,
    dry_run: bool,
) -> TypeCleanReport:
    """디렉토리(여러 JSON 파일) 클리닝"""
    merged_report = TypeCleanReport(type_name=type_name, label=cfg.data_type_label)
    merged_fields: dict[str, FieldCleanReport] = {}

    file_entries = _load_json_directory(source)
    for json_file, data, is_list in file_entries:
        items = _get_items(data, is_list)
        items, report = cleaner.clean_generic_type(items, cfg, dry_run=dry_run)

        # 필드별 통계 합산
        for fld in report.fields:
            if fld.field_name not in merged_fields:
                merged_fields[fld.field_name] = FieldCleanReport(
                    field_name=fld.field_name
                )
            merged = merged_fields[fld.field_name]
            merged.total += fld.total
            merged.cleaned += fld.cleaned
            merged.markdown_removed += fld.markdown_removed
            merged.incomplete_fixed += fld.incomplete_fixed
            merged.null_replaced += fld.null_replaced

        if not dry_run:
            output_path = _resolve_output_path(json_file, data_dir, output_dir)
            _save_json(data, output_path, is_list)
            logger.info("[%s] 저장: %s", type_name, output_path)

    merged_report.fields = list(merged_fields.values())
    return merged_report


# ─────────────────────────────────────────────────────────────────────────────
# 보고서 생성
# ─────────────────────────────────────────────────────────────────────────────


def _format_terminal_report(results: list[TypeCleanReport]) -> str:
    """터미널 출력용 보고서"""
    lines: list[str] = []
    total_records = 0
    total_cleaned = 0
    total_md = 0
    total_inc = 0
    total_null = 0
    total_art = 0

    lines.append("")
    lines.append("=" * 55)
    lines.append("  LLM 요약 클리닝 보고서")
    lines.append("=" * 55)
    lines.append("")

    for tr in results:
        header_pad = max(1, 45 - len(tr.type_name) - len(tr.label))
        lines.append(f"[{tr.type_name}] {tr.label} " + "─" * header_pad)

        for fld in tr.fields:
            total_records += fld.total
            total_cleaned += fld.cleaned
            total_md += fld.markdown_removed
            total_inc += fld.incomplete_fixed
            total_null += fld.null_replaced
            total_art += fld.artifact_removed

            pct = (fld.cleaned / fld.total * 100) if fld.total else 0
            lines.append(f"  {fld.field_name}:")
            lines.append(
                f"    대상: {fld.total:,}건 | 수정: {fld.cleaned:,}건 ({pct:.3f}%)"
            )
            details: list[str] = []
            if fld.markdown_removed:
                details.append(f"마크다운 제거: {fld.markdown_removed:,}건")
            if fld.incomplete_fixed:
                details.append(f"불완전 보정: {fld.incomplete_fixed:,}건")
            if fld.artifact_removed:
                details.append(f"아티팩트 제거: {fld.artifact_removed:,}건")
            if fld.null_replaced:
                details.append(f"null 대체: {fld.null_replaced:,}건")
            if details:
                lines.append(f"    {' | '.join(details)}")

        lines.append("")

    lines.append("=" * 55)
    lines.append("  종합")
    lines.append("=" * 55)
    pct_total = (total_cleaned / total_records * 100) if total_records else 0
    lines.append(f"  총 대상: {total_records:,}건")
    lines.append(f"  수정됨: {total_cleaned:,}건 ({pct_total:.1f}%)")
    lines.append(f"  마크다운/HTML 제거: {total_md:,}건")
    lines.append(f"  불완전 문장 보정: {total_inc:,}건")
    lines.append(f"  LLM 아티팩트 제거: {total_art:,}건")
    lines.append(f"  null 대체: {total_null:,}건")
    lines.append("")

    return "\n".join(lines)


def _to_json_report(results: list[TypeCleanReport]) -> dict[str, Any]:
    """JSON 직렬화 가능한 보고서"""
    report: dict[str, Any] = {"types": {}}
    total_records = 0
    total_cleaned = 0
    total_md = 0
    total_inc = 0
    total_null = 0
    total_art = 0

    for tr in results:
        type_data: dict[str, Any] = {"label": tr.label, "fields": {}}
        for fld in tr.fields:
            total_records += fld.total
            total_cleaned += fld.cleaned
            total_md += fld.markdown_removed
            total_inc += fld.incomplete_fixed
            total_null += fld.null_replaced
            total_art += fld.artifact_removed

            type_data["fields"][fld.field_name] = {
                "total": fld.total,
                "cleaned": fld.cleaned,
                "markdown_removed": fld.markdown_removed,
                "incomplete_fixed": fld.incomplete_fixed,
                "artifact_removed": fld.artifact_removed,
                "null_replaced": fld.null_replaced,
            }
        report["types"][tr.type_name] = type_data

    report["summary"] = {
        "total_records": total_records,
        "total_cleaned": total_cleaned,
        "markdown_removed": total_md,
        "incomplete_fixed": total_inc,
        "artifact_removed": total_art,
        "null_replaced": total_null,
    }
    return report


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM 요약 필드 후처리(클리닝)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--type",
        dest="type_name",
        default=None,
        help="클리닝할 타입 (미지정 시 전체). 예: law, precedent, dec_fair_trade",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="원본 데이터 디렉토리. 예: ../data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "클리닝 결과 저장 디렉토리 (기본: <data-dir>). "
            "예: ../data"
        ),
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
        help="JSON 보고서 저장 경로. 예: eda_output/clean_report.json",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        logger.error("--data-dir 경로가 존재하지 않습니다: %s", data_dir)
        sys.exit(1)

    output_dir: Path
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = data_dir

    if args.dry_run:
        logger.info("드라이런 모드 (파일 수정 없음)")
    else:
        logger.info("출력 디렉토리: %s", output_dir)

    # 클리닝 대상 결정
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
        logger.warning(
            "소스 파일 없는 타입 (%d개):\n%s", len(missing), "\n".join(missing)
        )
        type_names = [
            tn
            for tn in type_names
            if _resolve_source_path(get_config(tn), data_dir).exists()
        ]

    if not type_names:
        logger.error("클리닝할 타입이 없습니다.")
        sys.exit(1)

    logger.info("클리닝 대상: %d개 타입 — %s", len(type_names), ", ".join(type_names))

    # 클리닝 실행
    results: list[TypeCleanReport] = []
    for tn in type_names:
        try:
            report = process_type(
                tn, data_dir=data_dir, output_dir=output_dir, dry_run=args.dry_run
            )
            if report:
                results.append(report)
        except Exception as e:
            logger.error("[%s] 클리닝 실패: %s", tn, e)

    # 터미널 출력
    print(_format_terminal_report(results))

    # JSON 보고서 저장
    if args.report:
        report_path = _BACKEND_ROOT / args.report
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(_to_json_report(results), f, ensure_ascii=False, indent=2)
        logger.info("JSON 보고서 저장: %s", report_path)


if __name__ == "__main__":
    main()
