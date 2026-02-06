"""
에이전트 라우팅 모듈

규칙 기반 키워드 라우터 (RulesRouter)
"""

from enum import Enum
from typing import Any

from app.multi_agent.schemas.plan import AgentPlan


class AgentType(str, Enum):
    """에이전트 타입"""

    # 일반인 전용
    LAWYER_FINDER = "lawyer_finder"
    SMALL_CLAIMS = "small_claims"
    # 공통
    LEGAL_SEARCH = "legal_search"
    CASE_SEARCH = "case_search"  # 판례 명시 (agent_override용)
    LAW_SEARCH = "law_search"  # 법령 명시 (agent_override용)
    STORYBOARD = "storyboard"
    # 변호사 전용
    LAWYER_STATS = "lawyer_stats"
    LAW_STUDY = "law_study"
    # 폴백
    GENERAL = "general"


class UserRole(str, Enum):
    """사용자 역할"""

    USER = "user"
    LAWYER = "lawyer"


# 역할별 사용 가능한 에이전트
ROLE_AGENTS: dict[UserRole, list[AgentType]] = {
    UserRole.USER: [
        AgentType.LAWYER_FINDER,
        AgentType.SMALL_CLAIMS,
        AgentType.LEGAL_SEARCH,
        AgentType.CASE_SEARCH,
        AgentType.LAW_SEARCH,
        AgentType.STORYBOARD,
        AgentType.GENERAL,
    ],
    UserRole.LAWYER: [
        AgentType.LAWYER_STATS,
        AgentType.LAW_STUDY,
        AgentType.LEGAL_SEARCH,
        AgentType.CASE_SEARCH,
        AgentType.LAW_SEARCH,
        AgentType.STORYBOARD,
        AgentType.GENERAL,
    ],
}


# Intent 감지를 위한 키워드 패턴 (신뢰도 점수 포함)
INTENT_PATTERNS: dict[AgentType, list[tuple[str, float]]] = {
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
    AgentType.STORYBOARD: [
        ("타임라인", 0.9),
        ("스토리보드", 0.95),
        ("사건 정리", 0.85),
        ("시간순", 0.8),
        ("사건 경과", 0.8),
        ("사건 경위", 0.85),
    ],
    AgentType.LAWYER_STATS: [
        ("변호사 통계", 0.95),
        ("변호사 현황", 0.9),
        ("변호사 분포", 0.9),
        ("지역별 변호사", 0.85),
        ("전문분야 분포", 0.85),
    ],
    AgentType.LAW_STUDY: [
        ("공부", 0.8),
        ("학습", 0.85),
        ("시험", 0.75),
        ("로스쿨", 0.9),
        ("법학", 0.8),
        ("시험 문제", 0.75),
        ("법학 문제", 0.75),
    ],
}

# 법령 키워드 (detect_search_type 헬퍼용)
_LAW_KEYWORDS = ("법령", "법률", "조문", "시행령", "시행규칙", "법 제")


def detect_search_type(message: str) -> str:
    """메시지에서 법령/판례 검색 타입을 분류한다.

    Args:
        message: 사용자 메시지

    Returns:
        "law" 또는 "precedent"
    """
    if any(kw in message for kw in _LAW_KEYWORDS):
        return "law"
    return "precedent"


class RulesRouter:
    """규칙 기반 라우터"""

    def route(
        self,
        message: str,
        user_role: str = "user",
        session_data: dict[str, Any] | None = None,
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
                        use_rag=agent_type
                        in (
                            AgentType.CASE_SEARCH,
                            AgentType.LEGAL_SEARCH,
                            AgentType.LAW_SEARCH,
                            AgentType.LAW_STUDY,
                        ),
                        confidence=0.9,
                        reason="세션 유지",
                    )
            except ValueError:
                pass

        # 2. 키워드 기반 Intent 감지 (신뢰도 점수 포함)
        message_lower = message.lower()
        allowed_agents = ROLE_AGENTS.get(role, [AgentType.GENERAL])

        best_match: tuple[AgentType, float] | None = None

        for agent_type, patterns in INTENT_PATTERNS.items():
            if agent_type not in allowed_agents:
                continue

            for pattern, score in patterns:
                if pattern in message_lower:
                    if best_match is None or score > best_match[1]:
                        best_match = (agent_type, score)

        if best_match:
            agent_type, confidence = best_match
            use_rag = agent_type in (
                AgentType.CASE_SEARCH,
                AgentType.LEGAL_SEARCH,
                AgentType.LAW_SEARCH,
                AgentType.LAW_STUDY,
            )
            return AgentPlan(
                agent_type=agent_type.value,
                use_rag=use_rag,
                confidence=confidence,
                reason=f"키워드 매칭 (신뢰도: {confidence})",
            )

        # 3. 기본 에이전트: 양쪽 모두 LEGAL_SEARCH
        return AgentPlan(
            agent_type=AgentType.LEGAL_SEARCH.value,
            use_rag=True,
            confidence=0.5,
            reason="기본 법률 검색",
        )


__all__ = [
    # Enum
    "AgentType",
    "UserRole",
    # 상수
    "ROLE_AGENTS",
    "INTENT_PATTERNS",
    # 클래스
    "RulesRouter",
    # 함수
    "detect_search_type",
]
