"""
쿼리 리라이팅 서비스

LLM 기반 쿼리 확장, 키워드 추출
"""

import logging
from typing import List

from langsmith import traceable

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


@traceable(name="query_rewrite")
def rewrite_query(
    query: str,
    use_llm: bool = True,
) -> List[str]:
    """
    쿼리를 법률 검색에 최적화된 형태로 리라이팅

    Args:
        query: 원본 검색 쿼리
        use_llm: LLM 사용 여부 (False면 키워드 기반 확장)

    Returns:
        리라이팅된 쿼리 리스트 (1개)
    """
    if not use_llm:
        # LLM 미사용 시 키워드 기반 확장
        keywords = extract_legal_keywords(query)
        if keywords:
            return [f"{query} {' '.join(keywords)}"]
        return [query]

    try:
        model = get_chat_model(temperature=0.3)

        prompt = f"""사용자의 질문을 법률 문서 검색에 최적화된 검색 쿼리 1개로 변환하세요.

규칙:
- 일상 표현을 법률 용어로 변환 (예: "사기당했어" → "사기죄", "쫓겨날 것 같아" → "명도소송 퇴거")
- 관련 법률 개념을 추가 (예: "형사고소", "손해배상청구", "민사소송")
- 플랫폼/서비스명은 법적 행위로 변환 (예: "당근마켓" → "중고거래", "배민" → "배달 음식", "쿠팡" → "전자상거래")
- 불필요한 수식어, 감탄사 제거
- 키워드 나열이 아닌 자연스러운 문장 형태로 작성 (벡터 검색 최적화)
  좋은 예: "중고거래 사기 피해에 대한 형사고소 및 손해배상청구 절차"
  나쁜 예: "사기죄, 형사고소, 손해배상청구"

원본 질문: {query}

검색 쿼리 (설명 없이 쿼리만 출력):"""

        response = model.invoke([("user", prompt)])
        content = response.content if hasattr(response, "content") else str(response)
        rewritten = content.strip().lstrip("1.-) ").strip()

        if rewritten:
            return [rewritten]

    except Exception as e:
        logger.warning("쿼리 리라이팅 실패 (LLM): %s", e)
        # 폴백: 키워드 기반 확장
        keywords = extract_legal_keywords(query)
        if keywords:
            return [f"{query} {' '.join(keywords[:3])}"]

    return [query]


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
