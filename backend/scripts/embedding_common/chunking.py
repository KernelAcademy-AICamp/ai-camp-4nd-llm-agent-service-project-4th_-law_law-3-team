"""
텍스트 청킹 유틸리티 (법령/판례)

법령과 판례의 텍스트를 적절한 크기로 분할합니다.
"""

import re
from dataclasses import dataclass
from typing import Optional

from scripts.embedding_common.config import DEFAULT_CONFIG


@dataclass
class PrecedentChunkConfig:
    """판례 청킹 설정 (글자 수 기반)"""

    max_chunk_size: int = int(DEFAULT_CONFIG["PRECEDENT_CHUNK_SIZE"])
    overlap: int = int(DEFAULT_CONFIG["PRECEDENT_CHUNK_OVERLAP"])
    min_chunk_size: int = int(DEFAULT_CONFIG["PRECEDENT_MIN_CHUNK_SIZE"])


@dataclass
class LawChunkConfig:
    """법령 청킹 설정 (토큰 기반)"""

    max_tokens: int = int(DEFAULT_CONFIG["LAW_MAX_TOKENS"])
    min_tokens: int = int(DEFAULT_CONFIG["LAW_MIN_TOKENS"])


def chunk_text_by_chars(
    text: str,
    config: Optional[PrecedentChunkConfig] = None,
) -> list[str]:
    """
    글자 수 기반 텍스트 청킹 (판례용)

    Args:
        text: 원본 텍스트
        config: 청킹 설정

    Returns:
        청크 목록
    """
    config = config or PrecedentChunkConfig()

    if len(text) <= config.max_chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + config.max_chunk_size

        if end >= len(text):
            chunk = text[start:]
        else:
            # 문장 경계에서 분할 시도
            split_pos = _find_sentence_boundary(
                text, start, end, config.min_chunk_size
            )
            chunk = text[start:split_pos]
            end = split_pos

        chunk = chunk.strip()
        if len(chunk) >= config.min_chunk_size:
            chunks.append(chunk)

        start = end - config.overlap
        if start <= 0 and chunks:
            break

    return chunks if chunks else [text]


def chunk_text_by_tokens(
    text: str,
    config: Optional[LawChunkConfig] = None,
) -> list[str]:
    """
    토큰(공백 분리) 기반 텍스트 청킹 (법령용)

    Args:
        text: 원본 텍스트
        config: 청킹 설정

    Returns:
        청크 목록
    """
    config = config or LawChunkConfig()
    tokens = text.split()

    if len(tokens) <= config.max_tokens:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < len(tokens):
        end = min(start + config.max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk = " ".join(chunk_tokens)

        if len(chunk_tokens) >= config.min_tokens:
            chunks.append(chunk)

        start = end

    return chunks if chunks else [text]


def _find_sentence_boundary(
    text: str, start: int, end: int, min_size: int
) -> int:
    """문장 경계 찾기 (마침표, 느낌표, 물음표, 줄바꿈)"""
    search_start = max(start + min_size, end - 200)
    search_text = text[search_start:end]

    # 역순으로 문장 경계 탐색
    for pattern in [r"\.\s", r"\n", r";\s", r",\s"]:
        matches = list(re.finditer(pattern, search_text))
        if matches:
            last_match = matches[-1]
            return search_start + last_match.end()

    return end
