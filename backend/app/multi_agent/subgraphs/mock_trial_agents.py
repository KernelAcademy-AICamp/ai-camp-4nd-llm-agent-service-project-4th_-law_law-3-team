"""
모의 법정 CourtAgent 클래스

SimCourt Profile/Memory/Strategy 패턴을 구현한 법정 에이전트
Design 문서 Section 6.1 기반
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from app.multi_agent.subgraphs.mock_trial_prompts import filter_llm_output

logger = logging.getLogger(__name__)

LLM_TIMEOUT_SECONDS = 30

VALID_EMOTIONS: frozenset[str] = frozenset({
    "neutral", "angry", "thinking", "sad", "confident", "stern", "recording", "judging"
})


@dataclass
class CourtAgent:
    """법정 에이전트 (Profile/Memory/Strategy 패턴)

    각 역할(판사, 검사, 변호사, 피고인, 서기)에 대해
    프로필, 기억, 전략 모듈을 가진 에이전트를 구현합니다.
    """

    # Profile Module
    role: str
    name: str
    system_prompt: str
    temperature: float = 0.5

    # Memory Module
    short_term: list[str] = field(default_factory=list)
    long_term: list[str] = field(default_factory=list)

    # Strategy Module
    strategy: str = ""

    # Tools
    tools: list[str] = field(default_factory=list)

    async def generate(
        self,
        stage: str,
        context: str,
        court_record: list[dict[str, Any]],
    ) -> tuple[str, str]:
        """에이전트 발언 생성

        Args:
            stage: 현재 재판 단계
            context: 사건 맥락 (case_summary + 현재 상황)
            court_record: 서기 기록 (이전 발언 내역)

        Returns:
            (생성된 발언 텍스트, 감정 태그) 튜플
        """
        from app.tools.llm import get_chat_model

        memory_context = self._build_memory_context()
        prompt = f"{self.system_prompt}\n\n"
        prompt += f"[전략] {self.strategy}\n\n" if self.strategy else ""
        prompt += f"[기억] {memory_context}\n\n"
        prompt += f"[현재 단계] {stage}\n\n"
        prompt += f"[사건 맥락] {context}\n\n"
        prompt += f"[법정 기록]\n{self._format_record(court_record)}\n\n"
        prompt += "위 맥락을 바탕으로 발언하세요."

        model = get_chat_model(temperature=self.temperature)
        try:
            response = await asyncio.wait_for(
                model.ainvoke([
                    ("system", self.system_prompt),
                    ("user", prompt),
                ]),
                timeout=LLM_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "LLM 타임아웃: agent=%s, stage=%s, timeout=%ds",
                self.name,
                stage,
                LLM_TIMEOUT_SECONDS,
            )
            return f"[{self.name}] (응답 생성 중 시간 초과. 잠시 후 다시 시도해주세요.)", "neutral"

        filtered = filter_llm_output(str(response.content))
        emotion, text = self._parse_emotion(filtered)
        self.short_term.append(text)
        return text, emotion

    @staticmethod
    def _parse_emotion(text: str) -> tuple[str, str]:
        """발언 첫 줄의 [EMOTION:태그]를 파싱합니다 (FR-51).

        Args:
            text: LLM 출력 텍스트

        Returns:
            (감정 태그, 태그가 제거된 텍스트) 튜플
        """
        match = re.match(r"\[EMOTION:(\w+)\]\s*", text)
        if match and match.group(1) in VALID_EMOTIONS:
            return match.group(1), text[match.end():]
        return "neutral", text

    def reflect(self) -> str:
        """단계 종료 시 기억 요약 (reflection)

        단기 기억을 요약하여 장기 기억으로 이동합니다.

        Returns:
            요약 문자열
        """
        summary = f"[{len(self.short_term)}개 발언 요약]: "
        summary += " / ".join(self.short_term[-3:])
        self.long_term.append(summary)
        self.short_term.clear()
        return summary

    def update_strategy(self, new_strategy: str) -> None:
        """다음 단계 전략 업데이트

        Args:
            new_strategy: 새 전략 문자열
        """
        self.strategy = new_strategy

    def to_state(self) -> dict[str, Any]:
        """AgentState TypedDict 호환 dict로 직렬화

        Returns:
            직렬화된 에이전트 상태
        """
        return {
            "profile": {
                "name": self.name,
                "role": self.role,
                "personality": "",
                "expertise": "",
            },
            "memory": {
                "short_term": list(self.short_term),
                "long_term": list(self.long_term),
            },
            "strategy": self.strategy,
        }

    @classmethod
    def from_state(
        cls,
        state: dict[str, Any],
        system_prompt: str,
        temperature: float = 0.5,
    ) -> "CourtAgent":
        """AgentState TypedDict에서 CourtAgent 복원

        Args:
            state: 에이전트 상태 dict
            system_prompt: 역할별 시스템 프롬프트
            temperature: LLM 온도

        Returns:
            복원된 CourtAgent 인스턴스
        """
        profile = state.get("profile", {})
        memory = state.get("memory", {})
        return cls(
            role=profile.get("role", ""),
            name=profile.get("name", ""),
            system_prompt=system_prompt,
            temperature=temperature,
            short_term=list(memory.get("short_term", [])),
            long_term=list(memory.get("long_term", [])),
            strategy=state.get("strategy", ""),
        )

    def _build_memory_context(self) -> str:
        """기억 컨텍스트 문자열 빌드"""
        parts: list[str] = []
        if self.long_term:
            parts.append("[이전 단계 요약]\n" + "\n".join(self.long_term[-3:]))
        if self.short_term:
            parts.append("[현재 단계 발언]\n" + "\n".join(self.short_term[-5:]))
        return "\n\n".join(parts) if parts else "(없음)"

    def _format_record(self, court_record: list[dict[str, Any]]) -> str:
        """법정 기록 포맷팅"""
        lines: list[str] = []
        for record in court_record[-10:]:
            lines.append(
                f"[{record.get('stage', '')}] "
                f"{record.get('speaker', '')}: "
                f"{record.get('content', '')}"
            )
        return "\n".join(lines) if lines else "(기록 없음)"
