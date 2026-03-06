"""
에이전트 라우팅 모듈

규칙 기반 키워드 라우터 (RulesRouter) + LLM 의도 분류 라우터 (LLMRouter)
"""

import json
import logging
from enum import Enum
from typing import Any

from app.multi_agent.schemas.plan import AgentPlan

logger = logging.getLogger(__name__)


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
    # 공통 (체험형)
    MOCK_TRIAL = "mock_trial"
    # 콘텐츠 마케팅
    CONTENT_MARKETING = "content_marketing"
    # 워크스페이스
    WORKSPACE = "workspace"
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
        AgentType.MOCK_TRIAL,
        AgentType.WORKSPACE,
        AgentType.GENERAL,
    ],
    UserRole.LAWYER: [
        AgentType.LAWYER_STATS,
        AgentType.LAW_STUDY,
        AgentType.LEGAL_SEARCH,
        AgentType.CASE_SEARCH,
        AgentType.LAW_SEARCH,
        AgentType.STORYBOARD,
        AgentType.MOCK_TRIAL,
        AgentType.CONTENT_MARKETING,
        AgentType.WORKSPACE,
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
    AgentType.MOCK_TRIAL: [
        ("모의재판", 0.95),
        ("모의 재판", 0.95),
        ("모의법정", 0.95),
        ("모의 법정", 0.95),
        ("재판 시뮬", 0.9),
        ("법정 체험", 0.9),
        ("법정 시뮬", 0.9),
        ("재판 연습", 0.85),
        ("재판 게임", 0.85),
        ("법정 게임", 0.85),
    ],
    AgentType.CONTENT_MARKETING: [
        ("트렌드", 0.85),
        ("법률 트렌드", 0.95),
        ("핫이슈", 0.85),
        ("핫한 법률", 0.9),
        ("유튜브 대본", 0.95),
        ("대본 생성", 0.95),
        ("대본 만들", 0.9),
        ("콘텐츠 마케팅", 0.95),
        ("영상 대본", 0.9),
        ("법률 이슈", 0.85),
        ("최근 이슈", 0.7),
    ],
    AgentType.LAW_SEARCH: [
        ("체계도", 0.95),
        ("법령 계층", 0.9),
        ("상위법", 0.9),
        ("하위법", 0.9),
        ("시행령 구조", 0.85),
        ("법률 체계", 0.8),
        ("법령 관계", 0.8),
    ],
    AgentType.WORKSPACE: [
        ("사건 목록", 0.9),
        ("워크스페이스", 0.95),
        ("내 사건", 0.85),
        ("사건 조회", 0.9),
        ("타임라인 재생성", 0.9),
        ("사건 정리", 0.85),
        ("진행 상황", 0.7),
    ],
    AgentType.GENERAL: [
        ("안녕", 0.85),
        ("안녕하세요", 0.9),
        ("감사합니다", 0.85),
        ("고마워", 0.85),
        ("고맙습니다", 0.85),
        ("도움이 됐", 0.8),
        ("반갑습니다", 0.85),
        ("ㅎㅇ", 0.8),
        ("ㅎㅎ", 0.7),
    ],
}

# 법령 키워드 (detect_search_type 헬퍼용)
_LAW_KEYWORDS = ("법령", "법률", "조문", "시행령", "시행규칙", "법 제")
# 세션 고정/agent_override를 해제할 최소 신뢰도
INTENT_OVERRIDE_CONFIDENCE = 0.9
# 세션 탈출 키워드 — active_agent 세션 고정을 해제
_SESSION_EXIT_KEYWORDS = frozenset({
    "그만", "종료", "끝", "다른질문", "다른거", "처음으로", "초기화",
})


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


_AGENT_DESCRIPTIONS: dict[AgentType, str] = {
    AgentType.LAWYER_FINDER: "lawyer_finder: 개별 변호사를 찾아서 연결. 상담할 변호사 찾기/추천/소개, 근처 변호사, 위치 기반 매칭. 사용자가 특정 변호사와 상담하고 싶을 때 사용",
    AgentType.SMALL_CLAIMS: "small_claims: 소액소송 가이드, 내용증명, 지급명령, 금전 분쟁, 사기 피해, 계약 해제, 보증금/임대차 분쟁",
    AgentType.CASE_SEARCH: "case_search: 판례 검색. 판결, 선례, 유사 사건, 대법원 판례, 판례 분석",
    AgentType.LAW_SEARCH: "law_search: 법령 검색. 법령 체계도, 상위법/하위법 관계, 시행령 구조, 법령 조문 해석",
    AgentType.STORYBOARD: "storyboard: 사건 타임라인/스토리보드 생성. 시간순 사건 정리, 사건 경위 구성",
    AgentType.LAWYER_STATS: "lawyer_stats: 변호사 시장 통계/데이터 분석. 지역별 변호사 수, 인구 대비 밀도, 향후 예측/전망, 개업 지역·전문분야 추천, 경쟁 분석. 숫자/통계/예측/현황/밀도/분포를 물을 때 사용",
    AgentType.LAW_STUDY: "law_study: 법학 학습/시험 준비. 로스쿨, 법학 문제 풀이, 법률 개념 학습",
    AgentType.MOCK_TRIAL: "mock_trial: 모의재판/모의법정 시뮬레이션 체험. 법정 역할극",
    AgentType.CONTENT_MARKETING: "content_marketing: 법률 트렌드 분석, 유튜브 대본 생성, 콘텐츠 마케팅",
    AgentType.WORKSPACE: "workspace: 워크스페이스 사건 관리. 사건 목록/조회, 타임라인 재생성, 진행 상황 확인",
    AgentType.GENERAL: "general: 어떤 에이전트에도 해당하지 않는 질문, 인사, 감사 등 일반 대화",
}

_LLM_ROUTER_SYSTEM_PROMPT = """당신은 법률 서비스 플랫폼의 의도 분류기입니다.
사용자 메시지를 분석하여 가장 적절한 에이전트를 하나 선택하세요.

## 에이전트 목록
{agent_descriptions}

## 핵심 구분
- "변호사 찾아줘/추천해줘/소개해줘" → lawyer_finder (개별 변호사 연결)
- "변호사 몇 명/밀도/예측/통계/현황/분포" → lawyer_stats (데이터 분석)
- "개업/창업/사무실 열기/전문분야 추천/경쟁 분석" → lawyer_stats (시장 데이터 분석)

## 규칙
1. 반드시 위 에이전트 목록의 이름 중 하나를 선택하세요.
2. 어떤 에이전트에 해당하는지 판단할 수 없으면 "general"을 선택하세요.
3. JSON만 반환하세요. 다른 텍스트를 포함하지 마세요.

## 출력 형식
{{"agent_type": "에이전트_이름", "confidence": 0.0~1.0, "reason": "선택 이유 한 줄"}}"""

_SESSION_CONTEXT_ADDENDUM = """
## 현재 세션 상태
현재 "{active_agent}" 에이전트로 대화 중입니다.
- 사용자가 같은 주제를 이어가면 현재 에이전트("{active_agent}")를 유지하세요.
- 명확히 다른 서비스를 요청하는 경우에만 다른 에이전트를 선택하세요."""


class RulesRouter:
    """규칙 기반 라우터"""

    @staticmethod
    def _find_best_keyword_match(
        message_nospace: str,
        allowed_agents: list[AgentType],
    ) -> tuple[AgentType, float] | None:
        """허용된 에이전트 범위에서 최고 점수 키워드 매치 탐색"""
        best_match: tuple[AgentType, float] | None = None

        for agent_type, patterns in INTENT_PATTERNS.items():
            if agent_type not in allowed_agents:
                continue

            for pattern, score in patterns:
                pattern_nospace = pattern.replace(" ", "")
                if pattern_nospace in message_nospace:
                    if best_match is None or score > best_match[1]:
                        best_match = (agent_type, score)

        return best_match

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

        # 공백 제거 버전: "소액 소송" → "소액소송" 등 띄어쓰기 변형 대응
        message_nospace = message.lower().replace(" ", "")
        allowed_agents = ROLE_AGENTS.get(role, [AgentType.GENERAL])
        best_match = self._find_best_keyword_match(
            message_nospace=message_nospace,
            allowed_agents=allowed_agents,
        )

        # 1. 진행 중인 세션이 있으면 해당 에이전트 유지
        #    단, 세션 탈출 키워드가 포함되면 세션 고정 해제
        if session_data.get("active_agent") and not any(
            kw in message_nospace for kw in _SESSION_EXIT_KEYWORDS
        ):
            active_agent = session_data.get("active_agent")
            try:
                agent_type = AgentType(active_agent)
                if agent_type in allowed_agents:
                    # 명시적 의도가 매우 강하면(active와 다를 때) 세션 고정 해제
                    if (
                        best_match
                        and best_match[1] >= INTENT_OVERRIDE_CONFIDENCE
                        and best_match[0] != agent_type
                    ):
                        matched_agent, confidence = best_match
                        use_rag = matched_agent in (
                            AgentType.CASE_SEARCH,
                            AgentType.LEGAL_SEARCH,
                            AgentType.LAW_SEARCH,
                            AgentType.LAW_STUDY,
                        )
                        return AgentPlan(
                            agent_type=matched_agent.value,
                            use_rag=use_rag,
                            confidence=confidence,
                            reason=(
                                "세션 전환: 명시적 의도 감지 "
                                f"(신뢰도: {confidence})"
                            ),
                        )

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
                logger.warning(
                    "유효하지 않은 active_agent 값: %s — 세션 유지 무시",
                    active_agent,
                )

        # 2. 키워드 기반 Intent 감지 (신뢰도 점수 포함)
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

        # 3. 기본 에이전트: 양쪽 모두 GENERAL (비용 절감)
        return AgentPlan(
            agent_type=AgentType.GENERAL.value,
            use_rag=False,
            confidence=0.3,
            reason="기본 일반 채팅",
        )


class LLMRouter:
    """LLM 기반 의도 분류 라우터"""

    _RAG_AGENTS = frozenset({
        AgentType.CASE_SEARCH, AgentType.LEGAL_SEARCH,
        AgentType.LAW_SEARCH, AgentType.LAW_STUDY,
    })

    def __init__(self) -> None:
        self._fallback = RulesRouter()

    async def route(
        self,
        message: str,
        user_role: str = "user",
        session_data: dict[str, Any] | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> AgentPlan:
        session_data = session_data or {}
        message_nospace = message.lower().replace(" ", "")

        # 1. 세션 탈출 키워드 → 세션 해제, LLM 분류로 진행
        is_exit = any(kw in message_nospace for kw in _SESSION_EXIT_KEYWORDS)

        # 2. active_agent 세션 유지 (LLM이 유지/전환 판단)
        active_agent = None if is_exit else session_data.get("active_agent")

        # 3. LLM 분류 시도
        plan = await self._classify(message, user_role, history, active_agent)
        if plan is not None:
            return plan

        # 4. LLM 실패 → RulesRouter fallback
        logger.warning("LLM 라우팅 실패, RulesRouter fallback")
        return self._fallback.route(message, user_role, session_data)

    async def _classify(
        self,
        message: str,
        user_role: str,
        history: list[dict[str, str]] | None,
        active_agent: str | None,
    ) -> AgentPlan | None:
        try:
            from app.tools.llm import get_chat_model

            model = get_chat_model(temperature=0, max_tokens=128)

            # 역할별 허용 에이전트만 프롬프트에 포함
            try:
                role = UserRole(user_role)
            except ValueError:
                role = UserRole.USER
            allowed = ROLE_AGENTS.get(role, [AgentType.GENERAL])

            agent_desc = "\n".join(
                f"- {_AGENT_DESCRIPTIONS[a]}"
                for a in allowed
                if a in _AGENT_DESCRIPTIONS
            )
            system = _LLM_ROUTER_SYSTEM_PROMPT.format(agent_descriptions=agent_desc)

            if active_agent:
                system += _SESSION_CONTEXT_ADDENDUM.format(active_agent=active_agent)

            # 최근 3턴(6메시지)만 포함
            user_prompt = message
            if history:
                recent = history[-6:]
                context = "\n".join(
                    f"{t.get('role', 'user')}: {t.get('content', '')[:200]}"
                    for t in recent
                )
                user_prompt = f"최근 대화:\n{context}\n\n현재 메시지: {message}"

            response = await model.ainvoke([
                ("system", system),
                ("user", user_prompt),
            ])

            # content 추출 (str | list[dict] 대응)
            content = response.content
            if isinstance(content, list):
                raw = "".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                ).strip()
            else:
                raw = str(content).strip()

            # 코드블록 제거
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            result = json.loads(raw)
            agent_type_str = result.get("agent_type", "general")
            confidence = min(1.0, max(0.0, float(result.get("confidence", 0.5))))
            reason = result.get("reason", "")

            # AgentType 유효성 검증
            try:
                agent_enum = AgentType(agent_type_str)
            except ValueError:
                logger.warning("LLM이 유효하지 않은 에이전트 반환: %s", agent_type_str)
                return None

            # 역할 권한 검증
            if agent_enum not in allowed:
                logger.warning(
                    "LLM이 허용되지 않은 에이전트 반환: %s (role=%s)",
                    agent_type_str, user_role,
                )
                agent_enum = AgentType.GENERAL
                confidence = 0.3
                reason = "역할 권한 외 → general fallback"

            return AgentPlan(
                agent_type=agent_enum.value,
                use_rag=agent_enum in self._RAG_AGENTS,
                confidence=confidence,
                reason=f"LLM 분류: {reason}",
            )

        except Exception:
            logger.warning("LLM 라우팅 실패", exc_info=True)
            return None


__all__ = [
    # Enum
    "AgentType",
    "UserRole",
    # 상수
    "ROLE_AGENTS",
    "INTENT_PATTERNS",
    "INTENT_OVERRIDE_CONFIDENCE",
    "_SESSION_EXIT_KEYWORDS",
    "_AGENT_DESCRIPTIONS",
    # 클래스
    "RulesRouter",
    "LLMRouter",
    # 함수
    "detect_search_type",
]
