"""
쿼리 리라이팅 서비스

LLM 기반 쿼리 확장, 키워드 추출, 대화형 쿼리 리라이팅
"""

import logging
import re
from typing import List

from app.tools.llm import get_chat_model

logger = logging.getLogger(__name__)

# follow-up 감지용 키워드
_FOLLOWUP_KEYWORDS = frozenset([
    "더", "자세히", "그거", "그것", "계속", "알려줘", "설명해",
    "구체적", "예시", "어떻게", "왜", "뭐", "뭘", "이거",
    "아까", "방금", "위에", "말한", "그래서", "추가로",
])

# 8자 이하 메시지에서 법률 키워드가 없으면 follow-up으로 간주
_MIN_STANDALONE_LENGTH = 8

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


def _is_followup_query(message: str) -> bool:
    """키워드 기반 follow-up 질문 감지 (LLM 호출 없음).

    Args:
        message: 사용자 메시지

    Returns:
        follow-up 여부
    """
    stripped = message.strip()
    # 짧은 메시지 + 법률 키워드 없음 → follow-up
    if len(stripped) <= _MIN_STANDALONE_LENGTH:
        has_legal = any(kw in stripped for kw in LEGAL_KEYWORDS)
        if not has_legal:
            return True

    # follow-up 키워드 매칭 (2개 이상이면 확실)
    matched = sum(1 for kw in _FOLLOWUP_KEYWORDS if kw in stripped)
    if matched >= 2:
        return True

    # "더 자세히", "더 알려줘" 같은 패턴
    followup_patterns = [
        r"더\s*(자세히|알려|설명|구체적)",
        r"(그거|그것|이거|아까|방금).*(알려|설명|뭐)",
        r"(계속|추가로)\s*(알려|설명|해줘)",
    ]
    for pattern in followup_patterns:
        if re.search(pattern, stripped):
            return True

    return False


async def rewrite_conversational_query(
    message: str,
    history: list[dict[str, str]] | None = None,
) -> str:
    """대화 맥락을 반영하여 검색 쿼리를 리라이팅.

    follow-up이 아니면 원본 그대로 반환 (LLM 호출 없음).
    follow-up이면 최근 히스토리 + LLM으로 독립적 검색 쿼리 생성.

    Args:
        message: 현재 사용자 메시지
        history: 대화 히스토리 ``[{"role": "user"|"assistant", "content": "..."}]``

    Returns:
        검색에 사용할 쿼리 문자열
    """
    if not _is_followup_query(message):
        return message

    # 히스토리 없으면 리라이팅 불가 → 원본 반환
    if not history:
        return message

    # 최근 4개 메시지(약 2턴)만 사용
    recent = history[-4:]

    try:
        model = get_chat_model(temperature=0.0)

        conversation = "\n".join(
            f"{'사용자' if h['role'] == 'user' else 'AI'}: {h['content'][:200]}"
            for h in recent
        )

        prompt = f"""아래 대화 기록과 현재 질문을 보고, 벡터 검색에 적합한 독립적인 검색 쿼리 하나를 작성하세요.
대화 맥락을 반영하되, 검색 쿼리만 출력하세요. 설명이나 번호 없이 쿼리만 작성하세요.

대화 기록:
{conversation}

현재 질문: {message}

검색 쿼리:"""

        response = model.invoke([("user", prompt)])
        content = response.content if hasattr(response, "content") else str(response)
        rewritten = content.strip()

        if rewritten:
            logger.info(
                "쿼리 리라이팅: '%s' → '%s'", message, rewritten
            )
            return rewritten

    except Exception as e:
        logger.warning("대화형 쿼리 리라이팅 실패: %s", e)

    # 폴백: 히스토리에서 가장 최근 사용자 메시지 사용
    for h in reversed(recent):
        if h.get("role") == "user" and h.get("content", "").strip():
            fallback = h["content"].strip()
            logger.info(
                "쿼리 리라이팅 폴백: '%s' → '%s'", message, fallback
            )
            return fallback

    return message
