"""문서 유형 감지 - 정규식 패턴 매칭 기반"""

import re

DocType = str  # "criminal" | "civil" | "public" | "general"

DOC_TYPE_PATTERNS: dict[str, list[str]] = {
    "kakaotalk": [
        r"카카오톡 대화",
        r"\d{4}년\s*\d{1,2}월\s*\d{1,2}일\s*(오전|오후)\s*\d{1,2}:\d{2},",
        r"^\[.+?\]\s*\[(오전|오후)\s*\d{1,2}:\d{2}\]",
        r"저장한 날짜",
        r"님과 카카오톡 대화",
    ],
    "criminal": [
        r"공소사실",
        r"피고인\s*\w+",
        r"고합\d+",
        r"범행\s*일시",
        r"범\s*죄\s*사\s*실",
        r"피의자",
        r"형사",
        r"변론요지서",
    ],
    "civil": [
        r"의뢰인\s*상담",
        r"청구취지",
        r"원고.*피고",
        r"매매계약",
        r"상\s*담\s*내\s*용",
        r"상담일지",
        r"손해배상",
        r"소\s*장",
        r"임대차",
        r"토지",
        r"부동산",
        r"보증금",
        r"차임",
        r"소유권",
        r"채무",
        r"등기",
        r"대금",
        r"근저당",
    ],
    "public": [
        r"행정처분",
        r"등록취소",
        r"처분통지",
        r"법률상담일지",
        r"취소소송",
        r"행정소송",
        r"처분청",
        r"인허가",
        r"처분",
        r"사전통지",
        r"의견제출",
        r"체류자격",
        r"운전면허",
        r"선거",
        r"단속",
        r"학교폭력",
        r"명령서",
        r"행정심판",
    ],
}

_MATCH_THRESHOLD = 2


def detect_document_type(text: str) -> DocType:
    """텍스트에서 문서 유형을 감지한다.

    각 유형별 정규식 패턴을 매칭하여 2개 이상 일치하는 유형을 반환한다.
    여러 유형이 임계값을 초과하면 매칭 수가 가장 많은 유형을 선택한다.

    Args:
        text: 분석할 텍스트

    Returns:
        "criminal", "civil", "public", "general" 중 하나
    """
    scores: dict[str, int] = {}

    for doc_type, patterns in DOC_TYPE_PATTERNS.items():
        match_count = 0
        for pattern in patterns:
            if re.search(pattern, text):
                match_count += 1
        scores[doc_type] = match_count

    best_type = max(scores, key=lambda k: scores[k])
    if scores[best_type] >= _MATCH_THRESHOLD:
        return best_type

    return "general"
