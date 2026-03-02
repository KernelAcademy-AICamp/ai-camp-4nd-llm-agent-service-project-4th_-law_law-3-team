"""SimHash 기반 Fuzzy 중복 탐지 (v0.3.0)

동일 사건을 다른 매체가 약간 다른 표현으로 보도하는 '재탕 기사' 탐지.
SimHash 해밍 거리 기반으로 95%+ 유사도 기사를 필터링.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter

# SimHash 비트 수
SIMHASH_BITS = 64


def compute_simhash(text: str, *, ngram_size: int = 3) -> int:
    """텍스트의 SimHash 값을 계산

    Args:
        text: 입력 텍스트 (한국어 포함)
        ngram_size: 문자 n-gram 크기 (기본 3)

    Returns:
        64비트 SimHash 정수값
    """
    text = re.sub(r"\s+", " ", text.strip().lower())

    tokens = [text[i:i + ngram_size] for i in range(len(text) - ngram_size + 1)]
    token_counts = Counter(tokens)

    vector = [0] * SIMHASH_BITS

    for token, weight in token_counts.items():
        token_hash = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)  # noqa: S324
        for i in range(SIMHASH_BITS):
            bitmask = 1 << i
            if token_hash & bitmask:
                vector[i] += weight
            else:
                vector[i] -= weight

    fingerprint = 0
    for i in range(SIMHASH_BITS):
        if vector[i] > 0:
            fingerprint |= (1 << i)

    return fingerprint


def hamming_distance(hash_a: int, hash_b: int) -> int:
    """두 SimHash 값 간의 해밍 거리 계산

    Returns:
        0~64 범위의 해밍 거리 (작을수록 유사)
    """
    xor = hash_a ^ hash_b
    return bin(xor).count("1")


def is_near_duplicate(
    hash_a: int,
    hash_b: int,
    threshold: int = 3,
) -> bool:
    """두 해시가 유사 문서인지 판정

    threshold=3 → 해밍 거리 3 이하 → 약 95%+ 유사도
    """
    return hamming_distance(hash_a, hash_b) <= threshold
