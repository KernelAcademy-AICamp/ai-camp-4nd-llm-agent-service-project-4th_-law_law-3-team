"""
규칙 기반 라우터

키워드 매칭을 통한 에이전트 라우팅
common/agent_router.py에서 리팩토링
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from app.multi_agent.schemas.plan import AgentPlan


class AgentType(str, Enum):
    """에이전트 타입"""
    LAWYER_FINDER = "lawyer_finder"
    CASE_SEARCH = "case_search"
    SMALL_CLAIMS = "small_claims"
    # 변호사 전용 에이전트
    CASE_ANALYSIS = "case_analysis"
    CLIENT_MANAGEMENT = "client_management"
    DOCUMENT_DRAFTING = "document_drafting"
    # 기본 에이전트
    GENERAL = "general"


class UserRole(str, Enum):
    """사용자 역할"""
    USER = "user"
    LAWYER = "lawyer"


# 역할별 사용 가능한 에이전트
ROLE_AGENTS: Dict[UserRole, List[AgentType]] = {
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


# Intent 감지를 위한 키워드 패턴 (신뢰도 점수 포함)
INTENT_PATTERNS: Dict[AgentType, List[tuple[str, float]]] = {
    AgentType.LAWYER_FINDER: [
        ("변호사 찾", 0.95),
        ("변호사를 찾", 0.95),
        ("변호사 추천", 0.9),
        ("변호사를 추천", 0.9),
        ("근처 변호사", 0.85),
        ("주변 변호사", 0.85),
        ("변호사 검색", 0.8),
        ("가까운 변호사", 0.8),
        ("내 근처", 0.7),
        ("내 주변", 0.7),
        ("현재 위치", 0.6),
    ],
    AgentType.CASE_SEARCH: [
        ("판례", 0.9),
        ("사례", 0.7),
        ("유사한 사건", 0.85),
        ("판결", 0.8),
        ("법원 판결", 0.85),
        ("대법원", 0.7),
        ("재판", 0.6),
        ("선례", 0.8),
    ],
    AgentType.SMALL_CLAIMS: [
        ("소액소송", 0.95),
        ("내용증명", 0.9),
        ("지급명령", 0.9),
        ("사기", 0.7),
        ("환불", 0.7),
        ("손해배상", 0.75),
        ("보증금", 0.7),
        ("임대차", 0.65),
        ("월세", 0.6),
        ("계약 해제", 0.7),
        ("돈 안줌", 0.8),
        ("돈 못받", 0.8),
        ("중고거래", 0.7),
    ],
    AgentType.CASE_ANALYSIS: [
        ("판례 분석", 0.95),
        ("사건 분석", 0.9),
        ("법리 검토", 0.9),
        ("판결 분석", 0.9),
    ],
    AgentType.CLIENT_MANAGEMENT: [
        ("의뢰인", 0.85),
        ("사건 관리", 0.8),
        ("일정", 0.6),
        ("의뢰", 0.7),
    ],
    AgentType.DOCUMENT_DRAFTING: [
        ("문서 작성", 0.85),
        ("소장", 0.8),
        ("답변서", 0.85),
        ("준비서면", 0.9),
        ("계약서", 0.7),
    ],
}


class RulesRouter:
    """규칙 기반 라우터"""

    def route(
        self,
        message: str,
        user_role: str = "user",
        session_data: Optional[Dict[str, Any]] = None,
    ) -> AgentPlan:
        """
        메시지에서 Intent 감지하여 에이전트 플랜 반환

        Args:
            message: 사용자 메시지
            user_role: 사용자 역할
            session_data: 세션 데이터

        Returns:
            AgentPlan
        """
        session_data = session_data or {}

        # UserRole 변환
        try:
            role = UserRole(user_role)
        except ValueError:
            role = UserRole.USER

        # 1. 진행 중인 세션이 있으면 해당 에이전트 유지
        if session_data.get("active_agent"):
            active_agent = session_data.get("active_agent")
            try:
                agent_type = AgentType(active_agent)
                if agent_type in ROLE_AGENTS.get(role, []):
                    return AgentPlan(
                        agent_type=agent_type.value,
                        use_rag=agent_type == AgentType.CASE_SEARCH,
                        confidence=0.9,
                        reason="세션 유지",
                    )
            except ValueError:
                pass

        # 2. 키워드 기반 Intent 감지 (신뢰도 점수 포함)
        message_lower = message.lower()
        allowed_agents = ROLE_AGENTS.get(role, [AgentType.GENERAL])

        best_match: Optional[tuple[AgentType, float]] = None

        for agent_type, patterns in INTENT_PATTERNS.items():
            if agent_type not in allowed_agents:
                continue

            for pattern, score in patterns:
                if pattern in message_lower:
                    if best_match is None or score > best_match[1]:
                        best_match = (agent_type, score)

        if best_match:
            agent_type, confidence = best_match
            return AgentPlan(
                agent_type=agent_type.value,
                use_rag=agent_type == AgentType.CASE_SEARCH,
                confidence=confidence,
                reason=f"키워드 매칭 (신뢰도: {confidence})",
            )

        # 3. 기본 에이전트
        if role == UserRole.LAWYER:
            return AgentPlan(
                agent_type=AgentType.CASE_ANALYSIS.value,
                use_rag=True,
                confidence=0.5,
                reason="기본 변호사 에이전트",
            )

        return AgentPlan(
            agent_type=AgentType.CASE_SEARCH.value,
            use_rag=True,
            confidence=0.5,
            reason="기본 판례 검색",
        )


def get_available_agents(user_role: UserRole) -> List[AgentType]:
    """역할별 사용 가능한 에이전트 목록 반환"""
    return ROLE_AGENTS.get(user_role, [AgentType.GENERAL])


def is_agent_available(agent_type: AgentType, user_role: UserRole) -> bool:
    """특정 에이전트가 해당 역할에서 사용 가능한지 확인"""
    return agent_type in ROLE_AGENTS.get(user_role, [])
