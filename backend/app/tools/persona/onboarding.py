"""OnboardingProcessor — Track 2 온보딩 결과 → LawyerPersona 생성

Design 문서 §5, §6 기반.
"""

import logging
import uuid
from datetime import datetime, timezone

from app.modules.content_marketing.schema import (
    LawyerPersona,
    PersonaOnboardingRequest,
)

logger = logging.getLogger(__name__)


class OnboardingProcessor:
    """Track 2: 온보딩 위저드 결과로 페르소나 생성"""

    def process(
        self,
        user_id: str,
        request: PersonaOnboardingRequest,
    ) -> LawyerPersona:
        """온보딩 데이터 → LawyerPersona 스키마 변환"""
        now = datetime.now(tz=timezone.utc)
        return LawyerPersona(
            id=str(uuid.uuid4()),
            user_id=user_id,
            specialty_areas=request.specialty_areas,
            focus_topics=request.focus_topics,
            preferred_tone=request.preferred_tone,
            target_audience=request.target_audience,
            channel_style=request.channel_style,
            source="active",
            confidence=1.0,
            created_at=now,
            updated_at=now,
        )
