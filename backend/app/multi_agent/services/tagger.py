"""
태그 추출 서비스

대화 메시지에서 법률 관련 정보 태그를 LLM으로 추출한다.
모든 에이전트 노드에서 후처리로 호출되며, 실패 시 메인 응답에 영향 없이 무시된다.
"""

import json
import logging
from typing import Any

from app.multi_agent.schemas.tag import TaggedItem, TagType

logger = logging.getLogger(__name__)

MAX_TAGGED_ITEMS = 50

_TAG_EXTRACTION_PROMPT = """사용자 메시지에서 법률 사건 관련 정보를 추출하세요.

다음 태그 유형 중 해당하는 것만 JSON 배열로 반환하세요:
- lawyer: 변호사 관련 정보 (이름, 전문분야, 추천 결과)
- precedent: 판례 관련 정보 (판례번호, 판결 요지)
- evidence: 증거 (계약서, 사진, 카톡, 녹음 등)
- timeline: 시간순 사건 (날짜가 포함된 사건 경위)
- party: 당사자 정보 (피해자, 가해자, 관련인)
- amount: 금액 정보 (피해액, 청구액, 합의금)

규칙:
- content는 100자 이내 요약
- date_hint는 YYYY-MM-DD 형식 (날짜가 있을 때만)
- confidence는 0~1 (명확하면 0.8 이상)
- 해당 정보가 없으면 빈 배열 반환
- 일상 대화나 인사는 빈 배열

메시지: {message}

JSON 배열만 반환 (설명 없이):"""


async def extract_tags(
    message: str,
    agent_used: str,
    turn_index: int,
) -> list[TaggedItem]:
    """사용자 메시지에서 태그 추출 (LLM 기반)

    Args:
        message: 사용자 메시지
        agent_used: 현재 에이전트 이름
        turn_index: 대화 턴 번호

    Returns:
        추출된 TaggedItem 리스트 (실패 시 빈 리스트)
    """
    if not message or len(message.strip()) < 5:
        return []

    try:
        from app.tools.llm import get_chat_model

        model = get_chat_model(temperature=0)
        prompt = _TAG_EXTRACTION_PROMPT.format(message=message[:500])
        response = await model.ainvoke([("user", prompt)])

        raw_text = str(response.content).strip()
        # JSON 배열 파싱 (```json ... ``` 래핑 처리)
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
        raw_text = raw_text.strip()

        if not raw_text or raw_text == "[]":
            return []

        raw_items = json.loads(raw_text)
        if not isinstance(raw_items, list):
            return []

        tags: list[TaggedItem] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            tag_type_str = item.get("tag_type", "")
            if tag_type_str not in TagType.__members__.values():
                continue
            tags.append(
                TaggedItem(
                    tag_type=TagType(tag_type_str),
                    content=str(item.get("content", ""))[:100],
                    source_agent=agent_used,
                    turn_index=turn_index,
                    date_hint=item.get("date_hint"),
                    confidence=min(1.0, max(0.0, float(item.get("confidence", 0.5)))),
                )
            )
        return tags

    except Exception:
        logger.debug("태그 추출 실패 (무시)", exc_info=True)
        return []


def merge_tags(
    existing: list[dict[str, Any]],
    new_tags: list[TaggedItem],
    max_items: int = MAX_TAGGED_ITEMS,
) -> list[dict[str, Any]]:
    """기존 태그와 새 태그를 병합

    최대 개수를 초과하면 오래된(낮은 turn_index) 항목부터 제거한다.

    Args:
        existing: 기존 태그 dict 리스트
        new_tags: 새로 추출된 TaggedItem 리스트
        max_items: 최대 보관 개수

    Returns:
        병합된 태그 dict 리스트
    """
    merged = list(existing)
    for tag in new_tags:
        # 중복 방지: 같은 content + tag_type이면 스킵
        is_duplicate = any(
            item.get("content") == tag.content and item.get("tag_type") == tag.tag_type
            for item in merged
        )
        if not is_duplicate:
            merged.append(tag.model_dump())

    # 최대 개수 초과 시 오래된 것부터 제거
    if len(merged) > max_items:
        merged.sort(key=lambda x: x.get("turn_index", 0))
        merged = merged[-max_items:]

    return merged
