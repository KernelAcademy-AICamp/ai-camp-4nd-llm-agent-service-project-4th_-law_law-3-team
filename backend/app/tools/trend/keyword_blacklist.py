"""키워드 Sanitize + Blacklist 필터

Red Team 피드백 반영: Search Query Injection 방지 + 혐오/정치 편향 키워드 필터링
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# 허용 문자: 한글, 영문, 숫자, 공백
_ALLOWED_PATTERN = re.compile(r"^[가-힣a-zA-Z0-9\s]+$")

# Injection 패턴 (SQL, SSRF 등)
_INJECTION_PATTERNS = [
    re.compile(r"[;'\"\-\-]"),
    re.compile(r"https?://", re.IGNORECASE),
    re.compile(r"<[^>]+>"),
    re.compile(r"\{.*\}"),
]

_MAX_KEYWORD_LENGTH = 50


def _load_blacklist() -> dict[str, list[str]]:
    """blacklist.json 로드"""
    blacklist_path = Path(__file__).parent / "blacklist.json"
    if not blacklist_path.exists():
        logger.warning("blacklist.json 파일이 없습니다: %s", blacklist_path)
        return {"exact": [], "contains": [], "regex": []}
    with open(blacklist_path, encoding="utf-8") as f:
        data: dict[str, list[str]] = json.load(f)
        return data


_blacklist: dict[str, list[str]] | None = None


def _get_blacklist() -> dict[str, list[str]]:
    """싱글턴 블랙리스트 로드"""
    global _blacklist  # noqa: PLW0603
    if _blacklist is None:
        _blacklist = _load_blacklist()
    return _blacklist


def sanitize_keyword(keyword: str) -> str | None:
    """키워드 Sanitize: 허용 문자만 통과, injection 패턴 차단

    Returns:
        정제된 키워드 문자열. 무효하면 None.
    """
    cleaned = keyword.strip()

    if not cleaned or len(cleaned) > _MAX_KEYWORD_LENGTH:
        return None

    # Injection 패턴 검사
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(cleaned):
            logger.warning("Injection 패턴 감지, 키워드 차단: %s", cleaned[:30])
            return None

    # 허용 문자 검사
    if not _ALLOWED_PATTERN.match(cleaned):
        # 허용 문자만 추출 후 재검증
        cleaned = re.sub(r"[^가-힣a-zA-Z0-9\s]", "", cleaned).strip()
        if not cleaned:
            return None

    return cleaned


def is_blacklisted(keyword: str) -> bool:
    """블랙리스트 매칭 여부 확인"""
    bl = _get_blacklist()
    lower = keyword.lower()

    # Exact match (frozenset으로 O(1) 조회)
    exact_set = frozenset(e.lower() for e in bl.get("exact", []))
    if lower in exact_set:
        return True

    # Contains match
    for term in bl.get("contains", []):
        if term.lower() in lower:
            return True

    # Regex match
    for pattern_str in bl.get("regex", []):
        if re.search(pattern_str, keyword, re.IGNORECASE):
            return True

    return False


def filter_keywords(keywords: list[str]) -> list[str]:
    """키워드 리스트에 Sanitize + Blacklist 필터 일괄 적용

    Returns:
        필터링 통과한 키워드 리스트
    """
    result: list[str] = []
    blocked_count = 0

    for kw in keywords:
        sanitized = sanitize_keyword(kw)
        if sanitized is None:
            blocked_count += 1
            continue
        if is_blacklisted(sanitized):
            blocked_count += 1
            continue
        result.append(sanitized)

    if blocked_count > 0:
        logger.info("키워드 필터링: %d개 차단 (총 %d개 중)", blocked_count, len(keywords))

    return result
