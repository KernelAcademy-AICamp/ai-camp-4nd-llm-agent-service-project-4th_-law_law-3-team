"""
법률 안전 정책

법률 서비스 제공 시 필요한 면책 조항 및 안전 검사
"""

import re
from typing import List

# 법률 면책 조항
LEGAL_DISCLAIMER = """
⚠️ 이 서비스는 법률 정보 제공 목적이며, 실제 법률 조언을 대체하지 않습니다.
구체적인 법률 문제는 반드시 자격을 갖춘 변호사와 상담하시기 바랍니다.
""".strip()

# 위험 키워드 패턴 (불법 행위 조장 가능성)
DANGEROUS_PATTERNS: List[str] = [
    r"어떻게\s*(?:하면|해야).*(?:피할|회피|숨길|없앨)\s*수",
    r"(?:증거|서류).*(?:조작|위조|변조)",
    r"(?:탈세|세금\s*회피)",
    r"(?:불법|위법).*(?:방법|수단)",
]

# 민감한 법률 주제 (추가 주의 필요)
SENSITIVE_TOPICS: List[str] = [
    "형사",
    "고소",
    "고발",
    "구속",
    "체포",
    "사기",
    "횡령",
    "배임",
]


def check_input_safety(message: str) -> bool:
    """
    사용자 입력의 안전성 검사

    Args:
        message: 사용자 메시지

    Returns:
        True if 안전, False if 위험 키워드 감지
    """
    message_lower = message.lower()

    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, message_lower):
            return False

    return True


def add_disclaimer_if_needed(
    response: str,
    is_sensitive: bool = False,
) -> str:
    """
    필요시 응답에 면책 조항 추가

    Args:
        response: 원본 응답
        is_sensitive: 민감한 주제 여부

    Returns:
        면책 조항이 추가된 응답
    """
    # 이미 면책 조항이 있으면 추가하지 않음
    if "법률 조언을 대체하지 않습니다" in response:
        return response

    if is_sensitive:
        return f"{response}\n\n{LEGAL_DISCLAIMER}"

    return response


def contains_sensitive_topic(message: str) -> bool:
    """
    민감한 법률 주제 포함 여부 확인

    Args:
        message: 사용자 메시지

    Returns:
        True if 민감한 주제 포함
    """
    message_lower = message.lower()
    return any(topic in message_lower for topic in SENSITIVE_TOPICS)
