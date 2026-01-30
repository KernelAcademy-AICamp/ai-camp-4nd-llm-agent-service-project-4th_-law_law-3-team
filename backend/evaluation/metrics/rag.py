"""
RAG 평가 메트릭

RAG 시스템 전체의 품질을 평가하기 위한 지표들:
- Faithfulness: 검색된 컨텍스트에 충실한 응답인지
- Relevance: 질문에 관련된 응답인지
- Context Precision: 검색된 컨텍스트의 정밀도
"""

from typing import Optional


def calculate_context_precision(
    contexts: list[str],
    relevant_contexts: list[str],
) -> float:
    """
    Context Precision 계산

    검색된 컨텍스트 중 관련된 것의 비율

    Args:
        contexts: 검색된 컨텍스트 목록
        relevant_contexts: 관련된 컨텍스트 목록

    Returns:
        Precision 값 (0.0 ~ 1.0)
    """
    if not contexts:
        return 0.0

    relevant_set = set(relevant_contexts)
    relevant_count = sum(1 for ctx in contexts if ctx in relevant_set)

    return relevant_count / len(contexts)


def calculate_context_relevance_score(
    question: str,
    context: str,
) -> float:
    """
    Context Relevance Score 계산 (간단한 키워드 기반)

    질문의 키워드가 컨텍스트에 얼마나 포함되는지 측정
    더 정확한 측정은 LLM 기반 평가 필요

    Args:
        question: 질문 텍스트
        context: 컨텍스트 텍스트

    Returns:
        Relevance 점수 (0.0 ~ 1.0)
    """
    question_words = set(question.lower().split())
    context_lower = context.lower()

    stopwords = {"은", "는", "이", "가", "을", "를", "의", "에", "에서", "로", "으로"}
    keywords = question_words - stopwords

    if not keywords:
        return 0.0

    matched = sum(1 for kw in keywords if kw in context_lower)
    return matched / len(keywords)


def calculate_answer_relevance_score(
    question: str,
    answer: str,
) -> float:
    """
    Answer Relevance Score 계산 (간단한 키워드 기반)

    답변이 질문에 얼마나 관련되는지 측정
    더 정확한 측정은 LLM 기반 평가 필요

    Args:
        question: 질문 텍스트
        answer: 답변 텍스트

    Returns:
        Relevance 점수 (0.0 ~ 1.0)
    """
    return calculate_context_relevance_score(question, answer)


def calculate_faithfulness_score(
    answer: str,
    contexts: list[str],
) -> float:
    """
    Faithfulness Score 계산 (간단한 overlap 기반)

    답변의 내용이 컨텍스트에서 유래했는지 측정
    더 정확한 측정은 LLM 기반 평가 필요 (hallucination 검출)

    Args:
        answer: 답변 텍스트
        contexts: 참조 컨텍스트 목록

    Returns:
        Faithfulness 점수 (0.0 ~ 1.0)
    """
    if not contexts or not answer:
        return 0.0

    answer_sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]
    if not answer_sentences:
        return 0.0

    combined_context = " ".join(contexts).lower()

    supported_count = 0
    for sentence in answer_sentences:
        words = sentence.lower().split()
        if len(words) < 3:
            continue

        key_words = words[:5]
        if sum(1 for w in key_words if w in combined_context) >= 2:
            supported_count += 1

    return supported_count / len(answer_sentences) if answer_sentences else 0.0


def calculate_rag_metrics(
    question: str,
    answer: str,
    retrieved_contexts: list[str],
    relevant_context_ids: Optional[list[str]] = None,
    retrieved_context_ids: Optional[list[str]] = None,
) -> dict[str, float]:
    """
    RAG 전체 메트릭 계산

    Args:
        question: 질문 텍스트
        answer: 생성된 답변 텍스트
        retrieved_contexts: 검색된 컨텍스트 내용 목록
        relevant_context_ids: 관련 컨텍스트 ID 목록 (optional)
        retrieved_context_ids: 검색된 컨텍스트 ID 목록 (optional)

    Returns:
        모든 RAG 메트릭
    """
    result: dict[str, float] = {}

    if retrieved_contexts:
        avg_context_relevance = sum(
            calculate_context_relevance_score(question, ctx)
            for ctx in retrieved_contexts
        ) / len(retrieved_contexts)
        result["avg_context_relevance"] = avg_context_relevance

    result["answer_relevance"] = calculate_answer_relevance_score(question, answer)

    if retrieved_contexts:
        result["faithfulness"] = calculate_faithfulness_score(answer, retrieved_contexts)

    if relevant_context_ids and retrieved_context_ids:
        result["context_precision"] = calculate_context_precision(
            retrieved_context_ids, relevant_context_ids
        )

    return result
