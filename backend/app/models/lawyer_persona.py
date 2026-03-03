"""변호사 페르소나 ORM 모델 (v2.0)

Design 문서 §3.4 기반 — LawyerPersonaModel + LawyerPersonaFeedbackModel
"""

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Index,
    String,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB

from app.core.database import Base


class LawyerPersonaModel(Base):  # type: ignore[misc]
    """변호사 페르소나 ORM 모델"""

    __tablename__ = "lawyer_personas"

    id = Column(String(36), primary_key=True)
    user_id = Column(String(255), nullable=False, unique=True)
    specialty_areas = Column(JSONB, nullable=False)
    focus_topics = Column(JSONB, default=list)
    preferred_tone = Column(
        String(50), nullable=False, default="professional",
    )
    target_audience = Column(
        String(50), nullable=False, default="general_public",
    )
    channel_style = Column(String(50), nullable=True)
    source = Column(String(10), nullable=False)
    confidence = Column(Float, default=1.0)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(),
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        Index("idx_lawyer_personas_user_id", "user_id"),
    )


class LawyerPersonaFeedbackModel(Base):  # type: ignore[misc]
    """페르소나 피드백 ORM 모델"""

    __tablename__ = "lawyer_persona_feedback"

    id = Column(String(36), primary_key=True)
    persona_id = Column(String(36), nullable=False)
    script_id = Column(String(36), nullable=True)
    rating = Column(Float, nullable=False)
    feedback_type = Column(String(50), nullable=True)
    feedback_text = Column(String(1000), nullable=True)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(),
    )

    __table_args__ = (
        Index("idx_persona_feedback_persona_id", "persona_id"),
    )
