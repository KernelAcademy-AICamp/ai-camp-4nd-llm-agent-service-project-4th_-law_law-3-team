"""
쿼리 리라이팅 서비스

LLM 기반 쿼리 확장, 키워드 추출
"""

import logging

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
) -> str:
    """쿼리를 법률 검색에 최적화된 형태로 리라이팅.

    Args:
        query: 원본 검색 쿼리
        use_llm: LLM 사용 여부 (False면 키워드 기반 확장)

    Returns:
        리라이팅된 쿼리 문자열
    """
    if not use_llm:
        keywords = extract_legal_keywords(query)
        if keywords:
            return f"{query} {' '.join(keywords)}"
        return query

    try:
        model = get_chat_model(temperature=0.3)

        prompt = f"""사용자의 질문을 법률 판례·법령 검색에 최적화된 검색 쿼리 1개로 변환하세요.

규칙:
1. 일상 표현을 법률 용어로 바꾸세요. 판례에 등장하지 않을 일상어는 법적 개념으로 변환하세요.
   예: "쫓겨날 것 같아" → "임대차 명도 퇴거", "사레 걸렸어" → "신체 피해 안전의무"
2. 적용될 수 있는 법률명·죄명을 추가하세요 (예: "크게 떠들어서 신고" → "경범죄처벌법 인근소란")
3. "민사소송", "법적 대응 절차" 같은 범용 표현은 넣지 마세요
4. 플랫폼/서비스명은 법적 행위로 변환 (예: "당근마켓" → "중고거래")
5. 감탄사·수식어 제거, 자연스러운 문장 형태로 작성

원본 질문: {query}

검색 쿼리 (설명 없이 쿼리만 출력):"""

        response = model.invoke([("user", prompt)])
        content = response.content if hasattr(response, "content") else str(response)
        rewritten = str(content).strip().lstrip("1.-) ").strip()

        if rewritten:
            return rewritten

    except (ValueError, RuntimeError) as e:
        logger.warning("쿼리 리라이팅 실패 (LLM): %s", e)
        keywords = extract_legal_keywords(query)
        if keywords:
            return f"{query} {' '.join(keywords[:3])}"

    return query


def extract_legal_keywords(query: str) -> list[str]:
    """쿼리에서 법률 관련 키워드 추출.

    Args:
        query: 검색 쿼리

    Returns:
        추출된 법률 키워드 리스트
    """
    found_keywords = []

    for keyword in LEGAL_KEYWORDS:
        if keyword in query:
            found_keywords.append(keyword)

    expanded = _expand_related_keywords(found_keywords)

    return list(set(found_keywords + expanded))


def _expand_related_keywords(keywords: list[str]) -> list[str]:
    """키워드에 대한 연관어 확장."""
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
