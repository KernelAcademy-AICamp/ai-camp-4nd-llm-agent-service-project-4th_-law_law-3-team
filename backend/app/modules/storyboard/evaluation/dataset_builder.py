"""변호사시험 파일에서 핵심 사건 섹션을 추출하는 유틸리티

각 문서 유형별로 사건 사실관계가 포함된 핵심 섹션만 발췌한다.
- CIVIL: "상 담 내 용" 이후 ~ 다음 페이지 구분자("- N -") 또는 "상담인의 희망사항" 전까지
- CRIMINAL: "공소사실" 또는 "범 죄 사 실" 이후 ~ "증거목록" 또는 "증 거 목 록" 전까지
- PUBLIC: "법률상담일지" 내 "상 담 내 용" 이후 ~ 다음 섹션 전까지
"""

from __future__ import annotations

import re
from pathlib import Path


def _find_section(
    lines: list[str],
    start_patterns: list[str],
    end_patterns: list[str],
    *,
    include_start: bool = True,
    max_lines: int = 500,
) -> str:
    """패턴 기반으로 시작-끝 사이 텍스트를 추출한다.

    Args:
        lines: 전체 텍스트의 라인 목록
        start_patterns: 시작 지점을 찾는 정규식 패턴 목록 (OR)
        end_patterns: 종료 지점을 찾는 정규식 패턴 목록 (OR)
        include_start: 시작 라인을 포함할지 여부
        max_lines: 최대 추출 라인 수

    Returns:
        추출된 텍스트 (빈 문자열이면 매칭 실패)
    """
    start_idx: int | None = None
    end_idx: int | None = None

    for i, line in enumerate(lines):
        if start_idx is None:
            for pattern in start_patterns:
                if re.search(pattern, line):
                    start_idx = i if include_start else i + 1
                    break
        elif end_idx is None:
            for pattern in end_patterns:
                if re.search(pattern, line):
                    end_idx = i
                    break
            if start_idx is not None and i - start_idx >= max_lines:
                end_idx = i
                break

    if start_idx is None:
        return ""

    if end_idx is None:
        end_idx = min(start_idx + max_lines, len(lines))

    section_lines = lines[start_idx:end_idx]
    return "\n".join(section_lines).strip()


def _extract_civil_section(lines: list[str]) -> str:
    """민사 기록에서 상담 내용 섹션 추출"""
    # 1차: <상 담 내 용> ~ <상담인의 희망사항> 또는 페이지 구분
    text = _find_section(
        lines,
        start_patterns=[r"상\s*담\s*내\s*용"],
        end_patterns=[r"상담인의\s*희망사항", r"희\s*망\s*사\s*항"],
        include_start=True,
    )
    if text:
        return text

    # 2차: 의뢰인 상담일지 전체
    return _find_section(
        lines,
        start_patterns=[r"의뢰인\s*상담일지", r"상\s*담\s*일\s*지"],
        end_patterns=[r"^#\s*서\s*울|^#\s*판\s*결|^#\s*소\s*장"],
        include_start=True,
    )


def _extract_criminal_section(lines: list[str]) -> str:
    """형사 기록에서 공소사실/범죄사실 섹션 추출"""
    # 1차: 공소사실 ~ 증거목록
    text = _find_section(
        lines,
        start_patterns=[r"공\s*소\s*사\s*실", r"범\s*죄\s*사\s*실"],
        end_patterns=[r"증\s*거\s*목\s*록", r"증\s*거\s*의\s*요\s*지", r"증거관계"],
        include_start=True,
    )
    if text:
        return text

    # 2차: 기소장 또는 변론요지서 전체에서 사실관계 추출
    return _find_section(
        lines,
        start_patterns=[r"기\s*소\s*장", r"변론요지서"],
        end_patterns=[r"^#\s*법\s*령", r"^#\s*판\s*결"],
        include_start=True,
    )


def _extract_public_section(lines: list[str]) -> str:
    """공법 기록에서 법률상담일지 상담 내용 추출"""
    # 1차: 상 담 내 용 ~ 행정처분서 또는 다음 섹션
    text = _find_section(
        lines,
        start_patterns=[r"상\s*담\s*내\s*용"],
        end_patterns=[r"행\s*정\s*처\s*분\s*서", r"처분통지", r"내부회의록"],
        include_start=True,
    )
    if text:
        return text

    # 2차: 법률상담일지 전체
    return _find_section(
        lines,
        start_patterns=[r"법률상담일지"],
        end_patterns=[r"^#\s*소\s*장", r"^#\s*행정소송"],
        include_start=True,
    )


_EXTRACTORS: dict[str, type[None] | None] = {}  # placeholder for typing

_DOC_TYPE_EXTRACTORS = {
    "civil": _extract_civil_section,
    "criminal": _extract_criminal_section,
    "public": _extract_public_section,
}


def extract_case_section(filepath: str | Path, doc_type: str) -> str:
    """변호사시험 파일에서 사건 사실관계 섹션만 발췌한다.

    Args:
        filepath: 마크다운 파일 경로
        doc_type: 문서 유형 ("civil", "criminal", "public")

    Returns:
        추출된 핵심 섹션 텍스트. 추출 실패 시 빈 문자열.
    """
    path = Path(filepath)
    if not path.exists():
        return ""

    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")

    extractor = _DOC_TYPE_EXTRACTORS.get(doc_type)
    if extractor is None:
        return ""

    return extractor(lines)


def list_bar_exam_files(
    base_dir: str | Path,
    doc_type: str | None = None,
) -> list[Path]:
    """변호사시험 마크다운 파일 목록을 반환한다.

    Args:
        base_dir: 변호사시험 파일이 있는 디렉토리
        doc_type: 필터링할 문서 유형 (None이면 전체)

    Returns:
        파일 경로 목록 (번호순 정렬)
    """
    base = Path(base_dir)
    if not base.exists():
        return []

    type_prefix_map = {
        "civil": "[CIVIL]",
        "criminal": "[CRIMINAL]",
        "public": "[PUBLIC]",
    }

    if doc_type:
        prefix = type_prefix_map.get(doc_type, "")
        if not prefix:
            return []
        files = sorted(base.glob(f"{prefix}*.md"))
    else:
        files = sorted(base.glob("[*.md"))

    return files
