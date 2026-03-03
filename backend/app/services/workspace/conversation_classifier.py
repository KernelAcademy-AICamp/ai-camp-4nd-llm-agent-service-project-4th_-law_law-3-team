"""
대화 자동 분류기

대화 메시지에서 사건명/유형/고객명을 자동 분류한다.
"""

import json
import logging
import uuid
from typing import Any

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat_conversation import ChatConversation

logger = logging.getLogger(__name__)

_AUTO_CLASSIFY_INTERVAL = 5  # N턴마다 자동 분류

_CLASSIFY_PROMPT = """다음 법률 상담 대화를 분석하여 사건 정보를 분류하세요.

반환 필드:
- title: 대화 제목 (20자 이내, 핵심 사건 요약)
- case_type: 사건 유형 (아래 목록에서 선택)
- customer_name: 의뢰인 이름 (언급된 경우만, 없으면 null)

사건 유형 목록:
임대차, 이혼/가사, 교통사고, 산업재해, 의료사고, 명예훼손,
사기/횡령, 손해배상, 근로/노동, 상속, 부동산, 형사,
채권/채무, 소비자, 행정, 지식재산, 기타

규칙:
- 대화 초반이면 title은 "법률 상담"
- 사건 유형 불명확하면 "기타"
- JSON 객체만 반환 (설명 없이)

대화:
{conversation}

JSON 객체만 반환:"""


class ConversationClassifier:
    """대화 자동 분류기"""

    @staticmethod
    async def classify(
        messages: list[dict[str, str]],
        tagged_items: list[dict[str, Any]] | None = None,
    ) -> dict[str, str | None]:
        """사건명/유형/고객명 자동 분류

        Args:
            messages: 대화 메시지 리스트
            tagged_items: 태그 리스트 (참고용)

        Returns:
            {"title": "...", "case_type": "...", "customer_name": "..."|None}
        """
        if not messages:
            return {"title": "법률 상담", "case_type": "기타", "customer_name": None}

        conversation_text = "\n".join(
            f"{'사용자' if m.get('role') == 'user' else '상담원'}: {m.get('content', '')}"
            for m in messages[-20:]  # 최근 20턴
        )

        prompt = _CLASSIFY_PROMPT.format(conversation=conversation_text[:2000])

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
                return {"title": "법률 상담", "case_type": "기타", "customer_name": None}

            return {
                "title": str(result.get("title", "법률 상담"))[:50],
                "case_type": str(result.get("case_type", "기타"))[:20],
                "customer_name": (
                    str(result["customer_name"])[:50]
                    if result.get("customer_name")
                    else None
                ),
            }

        except (ValueError, KeyError, RuntimeError):
            logger.warning("대화 자동 분류 실패 (무시)", exc_info=True)
            return {"title": "법률 상담", "case_type": "기타", "customer_name": None}

    @staticmethod
    async def auto_classify_if_needed(
        db: AsyncSession,
        conversation_id: uuid.UUID,
        messages: list[dict[str, str]],
        is_title_manual: bool = False,
        tagged_items: list[dict[str, Any]] | None = None,
    ) -> None:
        """턴 수 기반 자동 분류 적용

        is_title_manual=True이면 title 덮어쓰기 방지.
        _AUTO_CLASSIFY_INTERVAL 턴마다 실행.
        """
        turn_count = len([m for m in messages if m.get("role") == "user"])
        if turn_count < 2 or turn_count % _AUTO_CLASSIFY_INTERVAL != 0:
            return

        result = await ConversationClassifier.classify(messages, tagged_items)

        update_values: dict[str, Any] = {}

        if not is_title_manual and result.get("title"):
            update_values["title"] = result["title"]

        if result.get("case_type"):
            update_values["case_type"] = result["case_type"]

        if result.get("customer_name"):
            update_values["customer_name"] = result["customer_name"]

        if update_values:
            from sqlalchemy import func

            update_values["updated_at"] = func.now()
            await db.execute(
                update(ChatConversation)
                .where(ChatConversation.id == conversation_id)
                .values(**update_values)
            )
