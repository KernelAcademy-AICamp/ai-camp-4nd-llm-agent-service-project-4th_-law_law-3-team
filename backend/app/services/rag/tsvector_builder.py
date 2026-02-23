"""
tsvector 생성 유틸리티

MeCab 토크나이저를 활용하여 한국어 텍스트를 position 포함 tsvector 문자열로 변환.
PostgreSQL의 to_tsvector()는 한국어를 지원하지 않으므로,
사전 토크나이징 후 'simple' 설정으로 tsvector에 삽입한다.
"""

import logging
import re

logger = logging.getLogger(__name__)

# tsvector 토큰에 허용되지 않는 문자 (PostgreSQL 파싱 오류 방지)
_INVALID_TOKEN_RE = re.compile(r"['\\\x00]")


# PostgreSQL tsvector 최대 1,048,575 bytes — 안전 마진 확보
_MAX_TSVECTOR_BYTES = 1_000_000


def build_tsvector_string(tokens: list[str]) -> str:
    """
    MeCab 토큰 리스트를 position 포함 tsvector 문자열로 변환.

    Position 정보를 포함하여 phraseto_tsquery 구 검색도 가능.
    PostgreSQL의 tsvector 1MB 제한을 초과하지 않도록 토큰을 절삭한다.

    Args:
        tokens: MeCab morphs() 결과 리스트

    Returns:
        tsvector 문자열 (예: "'손해':1 '배상':2 '청구':3")
        빈 토큰 리스트일 경우 빈 문자열 반환.

    Example:
        >>> build_tsvector_string(["손해", "배상", "청구"])
        "'손해':1 '배상':2 '청구':3"
    """
    if not tokens:
        return ""

    parts: list[str] = []
    total_bytes = 0
    pos = 1
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        # tsvector에 안전하지 않은 문자 제거
        token = _INVALID_TOKEN_RE.sub("", token)
        if not token:
            continue
        part = f"'{token}':{pos}"
        # 구분자 공백(1byte) 포함하여 누적 크기 확인
        part_bytes = len(part.encode("utf-8")) + (1 if parts else 0)
        if total_bytes + part_bytes > _MAX_TSVECTOR_BYTES:
            logger.debug(
                "tsvector 크기 제한 도달: %d tokens, %d bytes",
                len(parts),
                total_bytes,
            )
            break
        parts.append(part)
        total_bytes += part_bytes
        pos += 1

    return " ".join(parts)


def tokens_to_tsquery(tokens: list[str], operator: str = "|") -> str:
    """
    MeCab 토큰 리스트를 tsquery 문자열로 변환.

    Args:
        tokens: MeCab morphs() 결과 리스트
        operator: 토큰 간 연산자 ("|" = OR, "&" = AND)

    Returns:
        tsquery 문자열 (예: "'손해' | '배상' | '청구'")
        빈 토큰 리스트일 경우 빈 문자열 반환.
    """
    if not tokens:
        return ""

    clean_tokens: list[str] = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        token = _INVALID_TOKEN_RE.sub("", token)
        if token:
            clean_tokens.append(f"'{token}'")

    if not clean_tokens:
        return ""

    sep = f" {operator} "
    return sep.join(clean_tokens)
