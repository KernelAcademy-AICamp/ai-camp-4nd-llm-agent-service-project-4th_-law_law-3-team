"""
구조화 요약 생성기

대화 메시지에서 법률 사건 요약을 구조화된 형태로 추출한다.
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

SUMMARY_FIELDS = ["facts", "issues", "evidence", "open_questions", "next_steps"]

_SUMMARIZE_PROMPT = """다음 법률 상담 대화를 분석하여 구조화 요약을 생성하세요.

각 필드는 한국어로 짧은 문장 리스트로 작성합니다 (각 항목 50자 이내).

필드:
- facts: 확인된 사실관계 (사건 경위, 당사자, 날짜, 금액 등)
- issues: 법적 쟁점 (다툼이 되는 포인트)
- evidence: 증거 자료 (계약서, 사진, 녹음, 카카오톡 등)
- open_questions: 미확인 사항 (추가 확인이 필요한 부분)
- next_steps: 다음 단계 권고 (취해야 할 조치)

규칙:
- 각 필드 최대 5개 항목
- 정보가 없는 필드는 빈 배열
- JSON 객체만 반환 (설명 없이)

{existing_context}

대화:
{conversation}

JSON 객체만 반환:"""

_EXISTING_CONTEXT_TEMPLATE = """기존 요약 (증분 업데이트 — 기존 내용을 보완/수정):
{existing_summary}

위 기존 요약을 기반으로 새 대화 내용을 반영하여 업데이트하세요.
"""


class StructuredSummarizer:
    """구조화 요약 생성기"""

    @staticmethod
    async def generate_summary(
        messages: list[dict[str, str]],
        existing_summary: dict[str, Any] | None = None,
    ) -> dict[str, list[str]]:
        """대화 메시지에서 구조화 요약 생성

        Args:
            messages: [{"role": "user"|"assistant", "content": "..."}]
            existing_summary: 기존 요약 (증분 업데이트용)

        Returns:
            {"facts": [...], "issues": [...], "evidence": [...],
             "open_questions": [...], "next_steps": [...]}
        """
        if not messages:
            return {field: [] for field in SUMMARY_FIELDS}

        conversation_text = "\n".join(
            f"{'사용자' if m.get('role') == 'user' else '상담원'}: {m.get('content', '')}"
            for m in messages[-30:]  # 최근 30턴만
        )

        existing_context = ""
        if existing_summary:
            existing_context = _EXISTING_CONTEXT_TEMPLATE.format(
                existing_summary=json.dumps(existing_summary, ensure_ascii=False, indent=2)
            )

        prompt = _SUMMARIZE_PROMPT.format(
            existing_context=existing_context,
            conversation=conversation_text[:3000],
        )

        try:
            from app.tools.llm import get_chat_model

            model = get_chat_model(temperature=0)
            response = await model.ainvoke([("user", prompt)])

            raw_text = str(response.content).strip()
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
            raw_text = raw_text.strip()

            result = json.loads(raw_text)
            if not isinstance(result, dict):
                return {field: [] for field in SUMMARY_FIELDS}

            # 필드 정규화: 각 필드가 string 리스트인지 확인
            normalized: dict[str, list[str]] = {}
            for field in SUMMARY_FIELDS:
                items = result.get(field, [])
                if isinstance(items, list):
                    normalized[field] = [str(item)[:100] for item in items[:5]]
                else:
                    normalized[field] = []

            return normalized

        except (ValueError, KeyError, RuntimeError):
            logger.warning("구조화 요약 생성 실패 (무시)", exc_info=True)
            return existing_summary or {field: [] for field in SUMMARY_FIELDS}

    @staticmethod
    def build_resume_context(
        summary: dict[str, Any],
        recent_messages: list[dict[str, str]],
        token_budget: int = 2000,
    ) -> str:
        """이어가기용 컨텍스트 구성

        구조화 요약 + 토큰 예산 기반 최근 대화.

        Args:
            summary: 구조화 요약
            recent_messages: 최근 메시지 리스트
            token_budget: 대략적 토큰 예산

        Returns:
            이전 대화 컨텍스트 문자열
        """
        parts: list[str] = []

        # 구조화 요약 추가
        if summary:
            summary_lines: list[str] = ["[이전 대화 요약]"]
            for field in SUMMARY_FIELDS:
                items = summary.get(field, [])
                if items:
                    label = {
                        "facts": "사실관계",
                        "issues": "쟁점",
                        "evidence": "증거",
                        "open_questions": "미확인",
                        "next_steps": "다음단계",
                    }.get(field, field)
                    summary_lines.append(f"- {label}: {', '.join(str(i) for i in items)}")
            parts.append("\n".join(summary_lines))

        # 최근 대화 추가 (토큰 예산 내)
        if recent_messages:
            remaining = token_budget - len("\n".join(parts)) // 3
            msg_lines: list[str] = ["[최근 대화]"]
            for msg in reversed(recent_messages):
                line = f"{'사용자' if msg.get('role') == 'user' else '상담원'}: {msg.get('content', '')}"
                if len(line) // 3 > remaining:
                    break
                msg_lines.insert(1, line)
                remaining -= len(line) // 3
            if len(msg_lines) > 1:
                parts.append("\n".join(msg_lines))

        return "\n\n".join(parts)
