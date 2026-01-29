"""
Generation 평가 메트릭

RAG 생성 응답의 품질을 평가하기 위한 지표들:
- Citation Accuracy: 인용 정확도
- Response Completeness: 응답 완전성
"""

import re
from typing import Optional


def extract_citations(text: str) -> list[str]:
    """
    텍스트에서 인용 추출

    법령, 판례 인용을 패턴 매칭으로 추출

    Args:
        text: 검사할 텍스트

    Returns:
        추출된 인용 목록

    Example:
        >>> extract_citations("민법 제750조에 따르면...")
        ["민법 제750조"]
    """
    patterns = [
        r"([\w가-힣]+법)\s*제?(\d+)조(?:의?\d*)?",
        r"(\d{2,4}[다나마가]\d+)",
        r"(대법원\s*\d{4}[가-힣]*\.\s*\d{1,2}\.\s*\d{1,2}\.?\s*선고)",
    ]

    citations = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if isinstance(match, tuple):
                citations.append("".join(match))
            else:
                citations.append(match)

    return list(set(citations))


def normalize_citation(citation: str) -> str:
    """
    인용 정규화

    다양한 형식의 인용을 표준 형식으로 변환

    Args:
        citation: 원본 인용

    Returns:
        정규화된 인용
    """
    normalized = citation.strip()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"제(\d+)", r"\1", normalized)
    normalized = normalized.replace("조의", "조 제")
    return normalized


def calculate_citation_accuracy(
    generated_citations: list[str],
    required_citations: list[str],
) -> dict[str, float]:
    """
    Citation Accuracy 계산

    생성된 응답의 인용이 필수 인용을 얼마나 포함하는지 측정

    Args:
        generated_citations: 생성된 응답에서 추출한 인용
        required_citations: 필수 인용 목록

    Returns:
        precision, recall, f1 스코어

    Example:
        >>> calculate_citation_accuracy(
        ...     ["민법 제750조", "민법 제751조"],
        ...     ["민법 제750조", "민법 제756조"]
        ... )
        {"precision": 0.5, "recall": 0.5, "f1": 0.5}
    """
    if not required_citations:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    if not generated_citations:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    gen_normalized = {normalize_citation(c) for c in generated_citations}
    req_normalized = {normalize_citation(c) for c in required_citations}

    true_positives = len(gen_normalized & req_normalized)

    precision = true_positives / len(gen_normalized) if gen_normalized else 0.0
    recall = true_positives / len(req_normalized) if req_normalized else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def calculate_key_point_coverage(
    generated_text: str,
    key_points: list[str],
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Key Point 커버리지 계산

    생성된 응답이 핵심 포인트를 얼마나 포함하는지 측정
    (간단한 키워드 매칭 기반, 정확한 측정은 LLM 평가 필요)

    Args:
        generated_text: 생성된 응답 텍스트
        key_points: 핵심 포인트 목록
        threshold: 매칭 threshold

    Returns:
        coverage 스코어
    """
    if not key_points:
        return {"coverage": 1.0, "matched": 0, "total": 0}

    matched = 0
    text_lower = generated_text.lower()

    for point in key_points:
        keywords = point.lower().split()
        keyword_matches = sum(1 for kw in keywords if kw in text_lower)
        if keywords and (keyword_matches / len(keywords)) >= threshold:
            matched += 1

    return {
        "coverage": matched / len(key_points),
        "matched": matched,
        "total": len(key_points),
    }


def calculate_response_metrics(
    generated_text: str,
    required_citations: Optional[list[str]] = None,
    key_points: Optional[list[str]] = None,
) -> dict[str, float]:
    """
    생성 응답 전체 메트릭 계산

    Args:
        generated_text: 생성된 응답 텍스트
        required_citations: 필수 인용 목록
        key_points: 핵심 포인트 목록

    Returns:
        모든 생성 메트릭
    """
    result: dict[str, float] = {}

    if required_citations:
        extracted = extract_citations(generated_text)
        citation_metrics = calculate_citation_accuracy(extracted, required_citations)
        result["citation_precision"] = citation_metrics["precision"]
        result["citation_recall"] = citation_metrics["recall"]
        result["citation_f1"] = citation_metrics["f1"]

    if key_points:
        coverage = calculate_key_point_coverage(generated_text, key_points)
        result["key_point_coverage"] = coverage["coverage"]

    return result
