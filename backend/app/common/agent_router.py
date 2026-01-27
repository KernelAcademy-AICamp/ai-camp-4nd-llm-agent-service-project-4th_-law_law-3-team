"""
에이전트 라우터

사용자 메시지 분석 후 적절한 에이전트로 라우팅
"""

from enum import Enum
from typing import Any, Optional


class AgentType(str, Enum):
    """에이전트 타입"""
    LAWYER_FINDER = "lawyer_finder"
    CASE_SEARCH = "case_search"
    SMALL_CLAIMS = "small_claims"
    # 변호사 전용 에이전트 (추후 확장)
    CASE_ANALYSIS = "case_analysis"
    CLIENT_MANAGEMENT = "client_management"
    DOCUMENT_DRAFTING = "document_drafting"
    # 기본 에이전트
    GENERAL = "general"


class UserRole(str, Enum):
    """사용자 역할"""
    USER = "user"  # 일반 사용자
    LAWYER = "lawyer"  # 변호사


# 역할별 사용 가능한 에이전트
ROLE_AGENTS: dict[UserRole, list[AgentType]] = {
    UserRole.USER: [
        AgentType.LAWYER_FINDER,
        AgentType.CASE_SEARCH,
        AgentType.SMALL_CLAIMS,
        AgentType.GENERAL,
    ],
    UserRole.LAWYER: [
        AgentType.CASE_ANALYSIS,
        AgentType.CLIENT_MANAGEMENT,
        AgentType.DOCUMENT_DRAFTING,
        AgentType.CASE_SEARCH,
        AgentType.GENERAL,
    ],
}


# Intent 감지를 위한 키워드 패턴
INTENT_PATTERNS: dict[AgentType, list[str]] = {
    AgentType.LAWYER_FINDER: [
        "변호사 찾",
        "변호사를 찾",
        "변호사 추천",
        "변호사를 추천",
        "근처 변호사",
        "주변 변호사",
        "변호사 검색",
        "가까운 변호사",
        "내 근처",
        "내 주변",
        "현재 위치",
        "어디서 변호사",
    ],
    AgentType.CASE_SEARCH: [
        "판례",
        "사례",
        "유사한 사건",
        "판결",
        "법원 판결",
        "대법원",
        "재판",
        "선례",
    ],
    AgentType.SMALL_CLAIMS: [
        "소액소송",
        "내용증명",
        "지급명령",
        "사기",
        "환불",
        "손해배상",
        "보증금",
        "임대차",
        "월세",
        "계약 해제",
        "물건 안줌",
        "돈 안줌",
        "돈 못받",
        "중고거래",
    ],
    # 변호사 전용 패턴
    AgentType.CASE_ANALYSIS: [
        "판례 분석",
        "사건 분석",
        "법리 검토",
        "판결 분석",
    ],
    AgentType.CLIENT_MANAGEMENT: [
        "의뢰인",
        "사건 관리",
        "일정",
        "의뢰",
    ],
    AgentType.DOCUMENT_DRAFTING: [
        "문서 작성",
        "소장",
        "답변서",
        "준비서면",
        "계약서",
    ],
}


def detect_intent(
    message: str,
    user_role: UserRole = UserRole.USER,
    session_data: Optional[dict[str, Any]] = None,
) -> AgentType:
    """
    메시지에서 Intent 감지하여 적절한 에이전트 타입 반환

    Args:
        message: 사용자 메시지
        user_role: 사용자 역할
        session_data: 세션 데이터 (진행 중인 에이전트 정보)

    Returns:
        AgentType: 감지된 에이전트 타입
    """
    # 1. 진행 중인 세션이 있으면 해당 에이전트 유지
    if session_data and session_data.get("active_agent"):
        active_agent = session_data.get("active_agent")
        try:
            agent_type = AgentType(active_agent)
            # 역할에 허용된 에이전트인지 확인
            if agent_type in ROLE_AGENTS.get(user_role, []):
                return agent_type
        except ValueError:
            pass

    # 2. 키워드 기반 Intent 감지
    message_lower = message.lower()
    allowed_agents = ROLE_AGENTS.get(user_role, [AgentType.GENERAL])

    for agent_type, patterns in INTENT_PATTERNS.items():
        if agent_type not in allowed_agents:
            continue

        for pattern in patterns:
            if pattern in message_lower:
                return agent_type

    # 3. 기본 에이전트 (일반 사용자: 판례 검색, 변호사: 판례 분석)
    if user_role == UserRole.LAWYER:
        return AgentType.CASE_ANALYSIS
    return AgentType.CASE_SEARCH  # 기본적으로 판례 검색


def get_available_agents(user_role: UserRole) -> list[AgentType]:
    """역할별 사용 가능한 에이전트 목록 반환"""
    return ROLE_AGENTS.get(user_role, [AgentType.GENERAL])


def is_agent_available(agent_type: AgentType, user_role: UserRole) -> bool:
    """특정 에이전트가 해당 역할에서 사용 가능한지 확인"""
    return agent_type in ROLE_AGENTS.get(user_role, [])
