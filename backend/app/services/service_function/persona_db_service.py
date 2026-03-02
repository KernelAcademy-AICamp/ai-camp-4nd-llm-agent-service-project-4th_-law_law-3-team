"""페르소나 DB 서비스 — CRUD + 피드백

Design 문서 §6 기반.
"""

import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.lawyer_persona import (
    LawyerPersonaFeedbackModel,
    LawyerPersonaModel,
)
from app.modules.content_marketing.schema import (
    ChannelStyle,
    LawyerPersona,
    PersonaFeedbackRequest,
    PersonaTone,
    PersonaUpdateRequest,
    TargetAudience,
    TrendCategory,
)

logger = logging.getLogger(__name__)


async def get_persona(
    db: AsyncSession,
    user_id: str,
) -> LawyerPersona | None:
    """user_id로 페르소나 조회"""
    result = await db.execute(
        select(LawyerPersonaModel).where(
            LawyerPersonaModel.user_id == user_id,
        ),
    )
    row = result.scalar_one_or_none()
    if row is None:
        return None
    return _model_to_schema(row)


async def get_persona_by_id(
    db: AsyncSession,
    persona_id: str,
) -> LawyerPersona | None:
    """persona_id로 페르소나 조회"""
    result = await db.execute(
        select(LawyerPersonaModel).where(
            LawyerPersonaModel.id == persona_id,
        ),
    )
    row = result.scalar_one_or_none()
    if row is None:
        return None
    return _model_to_schema(row)


async def create_persona(
    db: AsyncSession,
    persona: LawyerPersona,
) -> LawyerPersona:
    """페르소나 생성 또는 갱신 (upsert)

    user_id UNIQUE 제약조건으로 인해 동일 사용자에 대해
    INSERT 실패 시 기존 행을 UPDATE한다.
    """
    existing = await db.execute(
        select(LawyerPersonaModel).where(
            LawyerPersonaModel.user_id == persona.user_id,
        ),
    )
    row = existing.scalar_one_or_none()

    if row is not None:
        row.specialty_areas = [area.value for area in persona.specialty_areas]
        row.focus_topics = persona.focus_topics
        row.preferred_tone = persona.preferred_tone.value
        row.target_audience = persona.target_audience.value
        row.channel_style = (
            persona.channel_style.value if persona.channel_style else None
        )
        row.source = persona.source
        row.confidence = persona.confidence
        row.updated_at = datetime.now(tz=timezone.utc)
        await db.commit()
        await db.refresh(row)
        logger.info("페르소나 갱신: user_id=%s, id=%s", persona.user_id, row.id)
        return _model_to_schema(row)

    model = LawyerPersonaModel(
        id=persona.id,
        user_id=persona.user_id,
        specialty_areas=[area.value for area in persona.specialty_areas],
        focus_topics=persona.focus_topics,
        preferred_tone=persona.preferred_tone.value,
        target_audience=persona.target_audience.value,
        channel_style=(
            persona.channel_style.value if persona.channel_style else None
        ),
        source=persona.source,
        confidence=persona.confidence,
    )
    db.add(model)
    await db.commit()
    await db.refresh(model)
    logger.info("페르소나 생성: user_id=%s, id=%s", persona.user_id, persona.id)
    return _model_to_schema(model)


async def update_persona(
    db: AsyncSession,
    user_id: str,
    request: PersonaUpdateRequest,
) -> LawyerPersona | None:
    """페르소나 부분 수정"""
    updates: dict[str, object] = {}
    if request.specialty_areas is not None:
        updates["specialty_areas"] = [
            area.value for area in request.specialty_areas
        ]
    if request.focus_topics is not None:
        updates["focus_topics"] = request.focus_topics
    if request.preferred_tone is not None:
        updates["preferred_tone"] = request.preferred_tone.value
    if request.target_audience is not None:
        updates["target_audience"] = request.target_audience.value
    if request.channel_style is not None:
        updates["channel_style"] = request.channel_style.value

    if not updates:
        return await get_persona(db, user_id)

    updates["updated_at"] = datetime.now(tz=timezone.utc)

    await db.execute(
        update(LawyerPersonaModel)
        .where(LawyerPersonaModel.user_id == user_id)
        .values(**updates),
    )
    await db.commit()
    logger.info("페르소나 업데이트: user_id=%s", user_id)
    return await get_persona(db, user_id)


async def save_feedback(
    db: AsyncSession,
    request: PersonaFeedbackRequest,
) -> None:
    """피드백 저장"""
    model = LawyerPersonaFeedbackModel(
        id=str(uuid.uuid4()),
        persona_id=request.persona_id,
        script_id=request.script_id,
        rating=request.rating,
        feedback_type=request.feedback_type,
        feedback_text=request.feedback_text,
    )
    db.add(model)
    await db.commit()
    logger.info("피드백 저장: persona_id=%s", request.persona_id)


def _model_to_schema(model: LawyerPersonaModel) -> LawyerPersona:
    """ORM 모델 → Pydantic 스키마 변환"""
    specialty_list: list[object] = model.specialty_areas  # type: ignore[assignment]
    focus_list: list[str] = model.focus_topics or []  # type: ignore[assignment]
    channel = (
        ChannelStyle(str(model.channel_style))
        if model.channel_style
        else None
    )

    return LawyerPersona(
        id=str(model.id),
        user_id=str(model.user_id),
        specialty_areas=[
            TrendCategory(str(area)) for area in specialty_list
        ],
        focus_topics=focus_list,
        preferred_tone=PersonaTone(str(model.preferred_tone)),
        target_audience=TargetAudience(str(model.target_audience)),
        channel_style=channel,
        source=str(model.source),  # type: ignore[arg-type]
        confidence=float(model.confidence or 1.0),
        created_at=model.created_at,  # type: ignore[arg-type]
        updated_at=model.updated_at,  # type: ignore[arg-type]
    )
