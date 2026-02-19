"""인용 추출 유틸리티.

법령 인용, 법령명, 사건번호 추출 정규식을 통합합니다.

Usage:
    from scripts.common.citation import extract_citations, extract_case_numbers
"""

from __future__ import annotations

import re

# 「법령명」 제N조 패턴 (가장 정확)
_CITATION_BRACKET_RE = re.compile(
    r"「([^」]+)」\s*제(\d+)\s*조(?:의(\d+))?"
)

# 법령명 제N조 패턴 (꺾쇠 없이)
_CITATION_PLAIN_RE = re.compile(
    r"((?:[가-힣]+법|[가-힣]+령|[가-힣]+규칙|[가-힣]+조례)(?:\s*시행[령규칙])?)"
    r"\s+제(\d+)\s*조(?:의(\d+))?"
)

# 「법령명」만 추출 (조문 번호 없이)
_LAW_NAME_BRACKET_RE = re.compile(r"「([^」]+)」")

# 꺾쇠 없는 법령명 추출
_STATUTE_NAME_PLAIN_RE = re.compile(
    r"((?:[가-힣]+법|[가-힣]+령|[가-힣]+규칙|[가-힣]+조례)(?:\s*시행[령규칙])?)"
)

# 사건번호 패턴: 연도(2-4자리) + 사건종류(한글 1-3자) + 번호
_CASE_NUMBER_RE = re.compile(
    r"(\d{2,4})"  # 연도
    r"([가-힣]{1,3})"  # 사건종류
    r"(\d+)"  # 번호
)


def extract_citations(text: str) -> list[str]:
    """텍스트에서 법령 인용을 추출.

    「법령명」 제N조 또는 법령명 제N조 패턴을 모두 추출합니다.

    Args:
        text: 분석할 텍스트

    Returns:
        인용 문자열 리스트 (발견 순서, 중복 제거)
    """
    if not text or not isinstance(text, str):
        return []

    seen: set[str] = set()
    citations: list[str] = []

    # 패턴 1: 「법령명」 제N조
    for match in _CITATION_BRACKET_RE.finditer(text):
        law_name = match.group(1).strip()
        article = match.group(2)
        suffix = f"의{match.group(3)}" if match.group(3) else ""
        citation = f"{law_name} 제{article}조{suffix}"
        if citation not in seen:
            seen.add(citation)
            citations.append(citation)

    # 패턴 2: 법령명 제N조 (꺾쇠 없이)
    for match in _CITATION_PLAIN_RE.finditer(text):
        law_name = match.group(1).strip()
        article = match.group(2)
        suffix = f"의{match.group(3)}" if match.group(3) else ""
        citation = f"{law_name} 제{article}조{suffix}"
        if citation not in seen:
            seen.add(citation)
            citations.append(citation)

    return citations


def extract_law_names(text: str) -> list[str]:
    """텍스트에서 「법령명」 패턴을 추출.

    Args:
        text: 분석할 텍스트

    Returns:
        법령명 리스트 (발견 순서, 중복 제거)
    """
    if not text or not isinstance(text, str):
        return []

    seen: set[str] = set()
    names: list[str] = []
    for match in _LAW_NAME_BRACKET_RE.finditer(text):
        name = match.group(1).strip()
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def extract_statute_names_plain(text: str) -> list[str]:
    """꺾쇠 없는 법령명 추출.

    Args:
        text: 분석할 텍스트

    Returns:
        법령명 리스트 (발견 순서, 중복 제거)
    """
    if not text or not isinstance(text, str):
        return []

    seen: set[str] = set()
    names: list[str] = []
    for match in _STATUTE_NAME_PLAIN_RE.finditer(text):
        name = match.group(1).strip()
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def extract_case_numbers(text: str) -> list[str]:
    """텍스트에서 사건번호 패턴을 추출.

    예: '2022다12345', '80도268'

    Args:
        text: 분석할 텍스트

    Returns:
        사건번호 리스트 (발견 순서, 중복 제거)
    """
    if not text or not isinstance(text, str):
        return []

    seen: set[str] = set()
    case_numbers: list[str] = []
    for match in _CASE_NUMBER_RE.finditer(text):
        year, case_type, number = match.groups()
        case_number = f"{year}{case_type}{number}"
        if case_number not in seen:
            seen.add(case_number)
            case_numbers.append(case_number)
    return case_numbers
