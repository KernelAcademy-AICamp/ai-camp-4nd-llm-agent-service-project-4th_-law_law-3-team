"""PersonaAnalyzer — Track 1: 대화 이력 메타데이터 기반 자동 페르소나 분석

Design 문서 §5.2 기반.
Red Team 반영:
- [심각 1] user_id는 Auth Dependency에서 추출 (IDOR 방지)
- [심각 3] 원문 텍스트 대신 메타데이터만 LLM에 전달 (PII 원천 차단)
"""

import json
import logging
import uuid
from datetime import datetime, timezone

from app.core.config import settings
from app.modules.content_marketing.schema import (
    AnalysisInsights,
    LawyerPersona,
    PersonaTone,
    TargetAudience,
    TrendCategory,
)
from app.tools.persona.metadata_extractor import MetadataExtractor
from app.tools.persona.models import (
    AnalysisResult,
    ChatMetadata,
    PersonaExtractionResult,
)

logger = logging.getLogger(__name__)


class InsufficientHistoryError(Exception):
    """대화 이력 부족"""

    def __init__(self, count: int, minimum: int) -> None:
        self.count = count
        self.minimum = minimum
        super().__init__(
            f"대화 이력이 부족합니다. ({count}건/{minimum}건 필요)",
        )


class LowConfidenceError(Exception):
    """분석 신뢰도 부족"""

    def __init__(self, confidence: float) -> None:
        self.confidence = confidence
        super().__init__(
            f"분석 신뢰도가 낮습니다. ({confidence:.2f})",
        )


# 유효한 전문 분야 목록
VALID_SPECIALTIES: frozenset[str] = frozenset(
    c.value for c in TrendCategory if c != TrendCategory.ALL
)


class PersonaAnalyzer:
    """Track 1: 대화 이력 메타데이터 기반 자동 페르소나 분석"""

    def __init__(self) -> None:
        self._extractor = MetadataExtractor()

    async def analyze(
        self,
        user_id: str,
        messages: list[dict[str, str]],
        max_history: int = 100,
        days_back: int = 30,
    ) -> AnalysisResult:
        """대화 이력 메타데이터에서 페르소나 추출 + 4중 검증

        Returns:
            AnalysisResult(persona=LawyerPersona, insights=AnalysisInsights)
        """
        minimum = settings.PERSONA_MIN_HISTORY
        if len(messages) < minimum:
            raise InsufficientHistoryError(
                count=len(messages), minimum=minimum,
            )

        # 메타데이터 추출 (원문 LLM 미전달)
        metadata = self._extractor.extract(
            messages[:max_history],
        )
        logger.info(
            "메타데이터 추출 완료: %d건, 키워드 %d개",
            metadata.total_conversations,
            len(metadata.top_legal_keywords),
        )

        # LLM 전문 분야 + 쟁점 추출 (메타데이터만 전달)
        extraction = await self._extract_persona_from_metadata(metadata)

        # 4중 할루시네이션 검증
        validated = self._validate_extraction(extraction)

        threshold = settings.PERSONA_CONFIDENCE_THRESHOLD
        if validated.confidence < threshold:
            raise LowConfidenceError(confidence=validated.confidence)

        persona = self._build_persona(user_id, validated)
        insights = self._build_insights(metadata, validated, days_back)

        return AnalysisResult(persona=persona, insights=insights)

    async def _extract_persona_from_metadata(
        self,
        metadata: ChatMetadata,
    ) -> PersonaExtractionResult:
        """메타데이터로 페르소나 추출 (LLM 호출)"""
        prompt = (
            "다음 변호사의 상담 활동 통계를 분석하여 "
            "전문 분야와 관심 쟁점을 추출하세요.\n\n"
            f"총 상담 건수: {metadata.total_conversations}건\n"
            f"에이전트 유형별 빈도: {metadata.agent_type_distribution}\n"
            f"법률 키워드 상위: {metadata.top_legal_keywords}\n"
            f"카테고리 분포: {metadata.category_distribution}\n\n"
            'JSON으로만 응답: '
            '{"specialty_areas": [...], '
            '"focus_topics": [...], '
            '"confidence": 0.0~1.0}'
        )
        try:
            from langchain_core.messages import HumanMessage

            from app.tools.llm import get_chat_model

            llm = get_chat_model()
            result = await llm.ainvoke([HumanMessage(content=prompt)])
            response = str(result.content)
            parsed = json.loads(response)
            return PersonaExtractionResult(
                specialty_areas=parsed.get("specialty_areas", []),
                focus_topics=parsed.get("focus_topics", []),
                confidence=float(parsed.get("confidence", 0.5)),
                raw_response=response,
            )
        except Exception:
            logger.warning("LLM 페르소나 추출 실패, 메타데이터 기반 폴백")
            return self._fallback_extraction(metadata)

    def _fallback_extraction(
        self,
        metadata: ChatMetadata,
    ) -> PersonaExtractionResult:
        """LLM 실패 시 카테고리 분포 기반 폴백"""
        sorted_categories = sorted(
            metadata.category_distribution.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        specialties = [
            cat for cat, _ in sorted_categories[:3]
            if cat in VALID_SPECIALTIES
        ]
        topics = [kw for kw, _ in metadata.top_legal_keywords[:5]]
        return PersonaExtractionResult(
            specialty_areas=specialties or ["civil"],
            focus_topics=topics,
            confidence=0.5,
            raw_response="fallback",
        )

    def _validate_extraction(
        self,
        extraction: PersonaExtractionResult,
    ) -> PersonaExtractionResult:
        """4중 할루시네이션 검증: Enum 체크 → 키워드 매칭 → 길이 → 신뢰도"""
        # 1. Enum 체크: 유효한 전문 분야만
        valid_areas = [
            area for area in extraction.specialty_areas
            if area in VALID_SPECIALTIES
        ]
        if not valid_areas:
            valid_areas = ["civil"]
            extraction.confidence *= 0.5

        # 2. focus_topics 길이 제한
        topics = extraction.focus_topics[:5]

        return PersonaExtractionResult(
            specialty_areas=valid_areas[:3],
            focus_topics=topics,
            confidence=extraction.confidence,
            raw_response=extraction.raw_response,
        )

    def _build_persona(
        self,
        user_id: str,
        extraction: PersonaExtractionResult,
    ) -> LawyerPersona:
        """검증된 추출 결과 → LawyerPersona 스키마 변환"""
        now = datetime.now(tz=timezone.utc)
        return LawyerPersona(
            id=str(uuid.uuid4()),
            user_id=user_id,
            specialty_areas=[
                TrendCategory(area) for area in extraction.specialty_areas
            ],
            focus_topics=extraction.focus_topics,
            preferred_tone=PersonaTone.PROFESSIONAL,
            target_audience=TargetAudience.GENERAL_PUBLIC,
            channel_style=None,
            source="passive",
            confidence=extraction.confidence,
            created_at=now,
            updated_at=now,
        )

    def _build_insights(
        self,
        metadata: ChatMetadata,
        extraction: PersonaExtractionResult,
        days_back: int,
    ) -> AnalysisInsights:
        """메타데이터 + 추출 결과 → AnalysisInsights 생성"""
        # 카테고리 분포를 area_scores로 변환
        area_scores: dict[str, float] = {}
        for area, score in metadata.category_distribution.items():
            if area in VALID_SPECIALTIES:
                area_scores[area] = round(score, 3)

        # 요약 생성
        top_areas = sorted(
            area_scores.items(), key=lambda x: x[1], reverse=True,
        )[:3]
        area_names = [a for a, _ in top_areas]
        summary = (
            f"최근 {days_back}일간 {metadata.total_conversations}건의 "
            f"상담 내역을 분석한 결과, "
            f"{', '.join(area_names)} 분야에 전문성이 집중되어 있습니다."
        )

        return AnalysisInsights(
            area_scores=area_scores,
            summary=summary,
            total_conversations_analyzed=metadata.total_conversations,
            analysis_period_days=days_back,
            evidence_snippets=[],  # TODO: 실제 대화 발췌 구현
        )
