"""
쿼리 리라이팅 서비스

LLM 기반 쿼리 확장 및 키워드 추출
"""

import logging
import re
from typing import List

from app.tools.llm import get_chat_model

logger = logging.getLogger(__name__)

# 법률 도메인 키워드 목록
LEGAL_KEYWORDS = [
    "손해배상", "계약", "민법", "형법", "소송", "재판", "판결", "항소",
    "상고", "기각", "인용", "청구", "피고", "원고", "불법행위", "채무불이행",
    "이행청구", "손해", "과실", "고의", "책임", "면책", "시효", "소멸시효",
    "취득시효", "소유권", "점유권", "저당권", "담보", "보증", "연대보증",
    "임대차", "전세", "월세", "보증금", "명도", "퇴거", "사기", "횡령",
]


def rewrite_query(
    query: str,
    num_queries: int = 3,
    use_llm: bool = True,
) -> List[str]:
    """
    쿼리를 다양한 형태로 확장

    Args:
        query: 원본 검색 쿼리
        num_queries: 생성할 쿼리 수 (원본 포함)
        use_llm: LLM 사용 여부 (False면 키워드 기반 확장)

    Returns:
        확장된 쿼리 리스트 (원본 쿼리가 첫 번째)
    """
    queries = [query]

    if not use_llm:
        # LLM 미사용 시 키워드 기반 확장
        keywords = extract_legal_keywords(query)
        if keywords:
            expanded = f"{query} {' '.join(keywords)}"
            queries.append(expanded)
        return queries[:num_queries]

    try:
        model = get_chat_model(temperature=0.3)

        prompt = f"""다음 법률 검색 쿼리를 {num_queries - 1}개의 다른 표현으로 바꿔주세요.
각 쿼리는 같은 의미를 가지되 다른 단어나 표현을 사용해야 합니다.
법률 용어와 일상 용어를 적절히 혼용해주세요.

원본 쿼리: {query}

다음 형식으로 출력하세요 (번호와 쿼리만, 설명 없이):
1. [확장된 쿼리 1]
2. [확장된 쿼리 2]
..."""

        response = model.invoke([("user", prompt)])
        content = response.content if hasattr(response, "content") else str(response)

        # 응답에서 쿼리 추출
        rewritten = _parse_rewritten_queries(content)
        queries.extend(rewritten)

    except Exception as e:
        logger.warning("쿼리 리라이팅 실패 (LLM): %s", e)
        # 폴백: 키워드 기반 확장
        keywords = extract_legal_keywords(query)
        if keywords:
            queries.append(f"{query} {' '.join(keywords[:3])}")

    return queries[:num_queries]


def _parse_rewritten_queries(content: str) -> List[str]:
    """LLM 응답에서 쿼리 추출"""
    queries = []
    lines = content.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # "1. 쿼리" 또는 "- 쿼리" 형식 처리
        match = re.match(r"^[\d\-\.\)]+\s*(.+)$", line)
        if match:
            query = match.group(1).strip()
            # 대괄호 제거
            query = re.sub(r"^\[|\]$", "", query).strip()
            if query:
                queries.append(query)

    return queries


def extract_legal_keywords(query: str) -> List[str]:
    """
    쿼리에서 법률 관련 키워드 추출

    Args:
        query: 검색 쿼리

    Returns:
        추출된 법률 키워드 리스트
    """
    found_keywords = []

    for keyword in LEGAL_KEYWORDS:
        if keyword in query:
            found_keywords.append(keyword)

    # 관련 키워드 추가 (연관어 확장)
    expanded = _expand_related_keywords(found_keywords)

    return list(set(found_keywords + expanded))


def _expand_related_keywords(keywords: List[str]) -> List[str]:
    """키워드에 대한 연관어 확장"""
    related_map = {
        "손해배상": ["불법행위", "과실", "책임"],
        "계약": ["채무불이행", "이행청구", "해제"],
        "임대차": ["보증금", "월세", "명도", "퇴거"],
        "사기": ["횡령", "형사고소", "손해배상"],
        "소송": ["재판", "판결", "항소"],
    }

    expanded = []
    for keyword in keywords:
        if keyword in related_map:
            expanded.extend(related_map[keyword])

    return expanded
