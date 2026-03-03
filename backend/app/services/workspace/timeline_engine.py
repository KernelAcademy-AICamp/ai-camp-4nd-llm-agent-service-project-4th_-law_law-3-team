"""
타임라인 재생성 엔진

태그 기반으로 사건 타임라인을 생성/갱신한다.
"""

import json
import logging
import uuid
from datetime import date
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.workspace_case import WorkspaceCaseTimelineItem

logger = logging.getLogger(__name__)

_TIMELINE_PROMPT = """다음 태그 목록에서 시간순 타임라인을 생성하세요.

규칙:
- 각 항목: date_text, title, description, category 포함
- date_text: 원본 날짜 표현 (예: "2025년 3월", "작년 여름")
- date_normalized: ISO 날짜 (YYYY-MM-DD), 불확실하면 null
- category: 계약, 사건발생, 분쟁, 소송, 합의, 증거, 기타 중 택 1
- 시간순 정렬 (날짜 불명확 시 문맥 기반 추정)
- JSON 배열만 반환 (설명 없이)

태그:
{tags}

JSON 배열만 반환:"""


class TimelineEngine:
    """태그 기반 타임라인 재생성 엔진"""

    @staticmethod
    def _extract_timeline_tags(
        tagged_items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """태그에서 타임라인 관련 항목 추출"""
        timeline_tags: list[dict[str, Any]] = []
        for tag in tagged_items:
            tag_type = tag.get("tag_type", "")
            # timeline, dates, events 유형 태그 추출
            if tag_type in ("timeline", "dates", "events", "key_facts"):
                timeline_tags.append(tag)
            elif tag.get("date_hint"):
                timeline_tags.append(tag)
        return timeline_tags

    @staticmethod
    async def rebuild(
        db: AsyncSession,
        case_id: uuid.UUID,
        tagged_items: list[dict[str, Any]],
        preserve_manual: bool = True,
    ) -> list[dict[str, Any]]:
        """태그에서 타임라인 재생성

        1. timeline 태그 추출
        2. LLM으로 시간순 정리 + 카테고리 분류
        3. 기존 manual 항목 보존 (preserve_manual=True 시)
        4. 결과 INSERT
        """
        timeline_tags = TimelineEngine._extract_timeline_tags(tagged_items)

        # manual 항목 보존
        manual_items: list[WorkspaceCaseTimelineItem] = []
        if preserve_manual:
            result = await db.execute(
                select(WorkspaceCaseTimelineItem).where(
                    WorkspaceCaseTimelineItem.case_id == case_id,
                    WorkspaceCaseTimelineItem.source_type == "manual",
                )
            )
            manual_items = list(result.scalars().all())

        # 기존 chat 소스 항목 삭제
        await db.execute(
            delete(WorkspaceCaseTimelineItem).where(
                WorkspaceCaseTimelineItem.case_id == case_id,
                WorkspaceCaseTimelineItem.source_type != "manual",
            )
        )

        if not timeline_tags:
            # 태그가 없으면 manual 항목만 유지
            return [
                {
                    "id": str(item.id),
                    "date_text": item.date_text,
                    "date_normalized": item.date_normalized.isoformat() if item.date_normalized else None,
                    "title": item.title,
                    "description": item.description,
                    "category": item.category,
                    "source_type": item.source_type,
                    "sort_order": item.sort_order,
                }
                for item in manual_items
            ]

        # LLM으로 타임라인 생성
        generated = await TimelineEngine._generate_with_llm(timeline_tags)

        # DB에 저장
        sort_offset = len(manual_items)
        result_items: list[dict[str, Any]] = []

        # manual 항목 먼저
        for item in manual_items:
            result_items.append({
                "id": str(item.id),
                "date_text": item.date_text,
                "date_normalized": item.date_normalized.isoformat() if item.date_normalized else None,
                "title": item.title,
                "description": item.description,
                "category": item.category,
                "source_type": "manual",
                "sort_order": item.sort_order,
            })

        # 생성된 항목
        for i, item_data in enumerate(generated):
            date_normalized = None
            raw_date = item_data.get("date_normalized")
            if raw_date:
                try:
                    date_normalized = date.fromisoformat(raw_date)
                except (ValueError, TypeError):
                    pass

            timeline_item = WorkspaceCaseTimelineItem(
                case_id=case_id,
                date_text=str(item_data.get("date_text", ""))[:50] or None,
                date_normalized=date_normalized,
                title=str(item_data.get("title", ""))[:200],
                description=str(item_data.get("description", ""))[:500] or None,
                category=str(item_data.get("category", "기타"))[:50],
                source_type="chat",
                sort_order=sort_offset + i,
            )
            db.add(timeline_item)
            await db.flush()

            result_items.append({
                "id": str(timeline_item.id),
                "date_text": timeline_item.date_text,
                "date_normalized": timeline_item.date_normalized.isoformat() if timeline_item.date_normalized else None,
                "title": timeline_item.title,
                "description": timeline_item.description,
                "category": timeline_item.category,
                "source_type": "chat",
                "sort_order": timeline_item.sort_order,
            })

        return result_items

    @staticmethod
    async def _generate_with_llm(
        tags: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """LLM을 사용하여 태그에서 타임라인 생성"""
        tags_text = json.dumps(tags[:30], ensure_ascii=False, indent=1)
        prompt = _TIMELINE_PROMPT.format(tags=tags_text[:3000])

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
            if not isinstance(result, list):
                return []

            return result[:20]  # 최대 20개

        except (ValueError, json.JSONDecodeError, RuntimeError):
            logger.warning("타임라인 LLM 생성 실패 (무시)", exc_info=True)
            return []
