"""트렌드 스코어링 엔진 (하이브리드: 수치 + LLM)

v2.0: LegalGateScorer — Legal Gate + 5차원 가중합 스코어링
"""

import asyncio
import json
import logging
import math
import uuid

from langchain_core.messages import HumanMessage

from app.core.config import settings
from app.modules.content_marketing.schema import (
    LawyerPersona,
    LegalStage,
    TrendCategory,
)
from app.tools.llm import get_chat_model
from app.tools.trend.models import (
    RawTrendItem,
    ScoredIssue,
    ScoredIssueV2,
    UnifiedAnalysisResult,
)

logger = logging.getLogger(__name__)

LEGAL_KEYWORDS: frozenset[str] = frozenset(
    {
        "판결",
        "소송",
        "법원",
        "변호사",
        "검찰",
        "형사",
        "민사",
        "손해배상",
        "기소",
        "무죄",
        "유죄",
        "항소",
        "상고",
        "헌법",
        "법률",
        "법령",
        "조례",
        "판례",
        "재판",
        "공판",
        "고소",
        "고발",
        "구속",
        "보석",
        "가처분",
        "압류",
        "경매",
        "파산",
        "회생",
    }
)


class TrendScorer:
    """트렌드 스코어링 엔진 (하이브리드: 수치 + LLM)"""

    async def score(
        self,
        items: list[RawTrendItem],
    ) -> list[ScoredIssue]:
        """원시 항목을 그룹화 + 스코어링"""
        groups = self._group_by_topic(items)

        mention_weight = settings.TREND_MENTION_WEIGHT
        legal_weight = settings.TREND_LEGAL_WEIGHT

        scored: list[ScoredIssue] = []
        for group_title, group_items in groups.items():
            mention = self._calculate_mention_score(group_items, len(items))
            legal = await self._judge_legal_relevance(group_title, group_items)

            combined = (mention_weight * mention + legal_weight * legal) * 100

            scored.append(
                ScoredIssue(
                    id=str(uuid.uuid4()),
                    title=group_title,
                    raw_items=group_items,
                    mention_score=round(mention, 2),
                    legal_relevance_score=round(legal, 2),
                    combined_score=round(combined, 1),
                )
            )

        scored.sort(key=lambda x: x.combined_score, reverse=True)
        return scored

    def _group_by_topic(
        self,
        items: list[RawTrendItem],
    ) -> dict[str, list[RawTrendItem]]:
        """제목 유사도 기반 그룹화 (단어 집합 교집합 비율 >= 0.5)"""
        groups: dict[str, list[RawTrendItem]] = {}
        group_words: dict[str, set[str]] = {}

        for item in items:
            title_words = set(item.title.split())
            if not title_words:
                continue

            matched_group: str | None = None
            best_overlap = 0.0

            for group_title, words in group_words.items():
                if not words:
                    continue
                overlap = len(title_words & words) / min(len(title_words), len(words))
                if overlap >= 0.5 and overlap > best_overlap:
                    best_overlap = overlap
                    matched_group = group_title

            if matched_group is not None:
                groups[matched_group].append(item)
                group_words[matched_group] |= title_words
            else:
                groups[item.title] = [item]
                group_words[item.title] = title_words

        return groups

    def _calculate_mention_score(
        self,
        group_items: list[RawTrendItem],
        total_items: int,
    ) -> float:
        """언급 빈도 기반 점수 (0~1) — 로그 스케일로 소규모 그룹도 의미 있는 점수 부여"""
        count = len(group_items)
        divisor = max(total_items * 0.3, 2)
        return min(math.log2(count + 1) / math.log2(divisor + 1), 1.0)

    async def _judge_legal_relevance(
        self,
        title: str,
        items: list[RawTrendItem],
    ) -> float:
        """LLM 기반 법적 해석 가능성 판단 + 키워드 보정"""
        text = f"{title} {' '.join(i.snippet for i in items[:3])}"
        keyword_hits = sum(1 for kw in LEGAL_KEYWORDS if kw in text)
        keyword_bonus = min(keyword_hits * 0.05, 0.2)

        try:
            llm = get_chat_model(temperature=0.0)
            prompt = (
                "다음 뉴스 이슈가 법적 분석/해석이 가능한지 0.0~1.0 사이 "
                "숫자 하나만 응답하세요. 법률 관련성이 전혀 없으면 0.0, "
                "법적 쟁점이 명확하면 1.0입니다.\n\n"
                f"이슈: {title}\n"
                f"내용: {text[:500]}"
            )
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            raw = response.content.strip() if isinstance(response.content, str) else ""
            llm_score = float(raw)
            llm_score = max(0.0, min(llm_score, 1.0))
        except (ValueError, TypeError):
            logger.warning("LLM 법적 관련성 스코어 파싱 실패, 키워드 점수만 사용")
            llm_score = 0.5
        except Exception:
            logger.warning("LLM 호출 실패, 키워드 점수만 사용", exc_info=True)
            llm_score = 0.5

        return min(llm_score + keyword_bonus, 1.0)


# ── v2.0 Legal Gate Scorer ──

# 카테고리별 키워드 (fitness_score 계산용)
CATEGORY_KEYWORDS: dict[str, frozenset[str]] = {
    "criminal": frozenset({"형사", "검찰", "기소", "공판", "구속", "사기", "횡령", "폭행"}),
    "civil": frozenset({"손해배상", "채무", "계약", "부동산", "민사", "가처분"}),
    "family": frozenset({"이혼", "양육권", "위자료", "재산분할", "상속", "가사"}),
    "labor": frozenset({"해고", "임금", "퇴직금", "산재", "노동", "부당해고"}),
    "administrative": frozenset({"행정", "인허가", "처분", "취소", "행정소송"}),
    "corporate": frozenset({"회사", "주주", "이사", "파산", "회생", "M&A"}),
    "ip": frozenset({"특허", "상표", "저작권", "지식재산", "영업비밀"}),
}

# LegalStage 유효값
_VALID_STAGES: frozenset[str] = frozenset(s.value for s in LegalStage)
_VALID_CATEGORIES: frozenset[str] = frozenset(c.value for c in TrendCategory if c != TrendCategory.ALL)


_LEGAL_ADDITIVE_WEIGHT = 0.20  # legal_score 독립 가중치 (Gate 통과 시)


class LegalGateScorer(TrendScorer):
    """Legal Gate + 5차원 가중합 스코어링 엔진 (v2.1)

    점수 공식:
      Gate 통과 시: (0.20×L + 0.25×M + 0.20×C + 0.10×S + 0.25×F) × 100
                    최소 30점 보장
      Gate 미통과 시: (0.4×M + 0.3×C + 0.3×S) × 100 × 0.6
    """

    def __init__(self) -> None:
        super().__init__()
        self._legal_threshold: float = settings.TREND_LEGAL_THRESHOLD

    async def score_v2(
        self,
        items: list[RawTrendItem],
        persona: LawyerPersona | None = None,
    ) -> list[ScoredIssueV2]:
        """Legal Gate 필터 + 5차원 가중합 스코어링 (병렬 LLM 분석)"""
        groups = self._group_by_topic(items)
        scored: list[ScoredIssueV2] = []

        # Stage 1: 모든 그룹의 LLM 분석을 병렬 실행
        group_entries = list(groups.items())
        analyses = await asyncio.gather(
            *[self._analyze_unified(title, group_items) for title, group_items in group_entries]
        )

        for (group_title, group_items), analysis in zip(group_entries, analyses):
            # Stage 2: Legal Gate
            legal_gate_passed = analysis.legal_score >= self._legal_threshold

            # Stage 3: 개별 스코어 계산
            mention = self._calculate_mention_score(group_items, len(items))
            spread = self._calculate_spread_score(group_items)
            fitness = (
                self._calculate_fitness_score(group_title, analysis.category, persona)
                if persona
                else 0.5
            )

            # Stage 4: 가중합 (Legal Score = 독립 가중 차원)
            if legal_gate_passed:
                combined = (
                    _LEGAL_ADDITIVE_WEIGHT * analysis.legal_score
                    + settings.TREND_MENTION_WEIGHT * mention
                    + settings.TREND_CONTROVERSY_WEIGHT * analysis.controversy_score
                    + settings.TREND_SPREAD_WEIGHT * spread
                    + settings.TREND_FITNESS_WEIGHT * fitness
                ) * 100
                combined = max(combined, 30.0)  # Gate 통과 최소 30점 보장
            else:
                combined = (
                    0.4 * mention
                    + 0.3 * analysis.controversy_score
                    + 0.3 * spread
                ) * 100 * 0.6  # Gate 미통과 40% 페널티

            scored.append(
                ScoredIssueV2(
                    id=str(uuid.uuid4()),
                    title=group_title,
                    raw_items=group_items,
                    mention_score=round(mention, 2),
                    legal_score=round(analysis.legal_score, 2),
                    controversy_score=round(analysis.controversy_score, 2),
                    spread_score=round(spread, 2),
                    fitness_score=round(fitness, 2),
                    legal_stage=analysis.legal_stage,
                    legal_gate_passed=legal_gate_passed,
                    gate_rejection_reason=(
                        None
                        if legal_gate_passed
                        else (
                            f"법적 쟁점화 지표 {analysis.legal_score:.2f}로 "
                            f"법률 콘텐츠 기준({self._legal_threshold}) 미달"
                        )
                    ),
                    combined_score=round(combined, 1),
                    category=analysis.category,
                )
            )

        scored.sort(key=lambda x: (-x.legal_gate_passed, -x.combined_score))
        return scored

    async def _analyze_unified(
        self,
        title: str,
        items: list[RawTrendItem],
    ) -> UnifiedAnalysisResult:
        """통합 LLM 프롬프트 — 1회 호출로 4개 지표 추출

        3단계 JSON 폴백:
        1. 정상 JSON 파싱
        2. JSON 부분 추출 (```json ... ```)
        3. 키워드 기반 기본값 (LLM 실패 시)
        """
        text = f"{title} {' '.join(i.snippet for i in items[:3])}"
        prompt = (
            "다음 뉴스 이슈를 분석하세요. JSON으로만 응답:\n"
            '{"legal_score": 0.0~1.0, '
            '"legal_stage": "litigation|legislation|prosecution|dispute|mention", '
            '"controversy_ratio": 0.0~1.0, '
            '"category": "criminal|civil|labor|family|administrative|corporate|ip"}\n\n'
            "- legal_score: 법률적 분석 가능성 (관련 법령/판례가 존재할 수 있으면 0.3+, 직접 법적 쟁점이면 0.7+)\n"
            "- legal_stage: 현재 법적 단계\n"
            "- controversy_ratio: 논란/찬반 대립 정도\n"
            "- category: 가장 적합한 법률 카테고리\n\n"
            f"이슈: {title}\n내용: {text[:500]}"
        )

        try:
            llm = get_chat_model(temperature=0.0)
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            raw = response.content.strip() if isinstance(response.content, str) else ""
            return self._parse_unified_response(raw, title, items)
        except Exception:
            logger.warning("통합 LLM 분석 실패, 키워드 기반 기본값 사용", exc_info=True)
            return self._fallback_analysis(title, items)

    def _parse_unified_response(
        self,
        raw: str,
        title: str,
        items: list[RawTrendItem],
    ) -> UnifiedAnalysisResult:
        """3단계 JSON 폴백 파싱"""
        # Stage 1: 직접 JSON 파싱
        try:
            data = json.loads(raw)
            return self._extract_from_dict(data)
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

        # Stage 2: JSON 블록 추출 (```json ... ```)
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            data = json.loads(raw[start:end])
            return self._extract_from_dict(data)
        except (ValueError, json.JSONDecodeError, KeyError):
            pass

        # Stage 3: 키워드 기반 기본값
        logger.warning("JSON 파싱 3단계 모두 실패, 키워드 폴백: %s", title)
        return self._fallback_analysis(title, items)

    def _extract_from_dict(self, data: dict[str, object]) -> UnifiedAnalysisResult:
        """딕셔너리에서 분석 결과 추출 (값 클램핑 포함)"""
        raw_legal = data.get("legal_score", 0.5)
        raw_controversy = data.get("controversy_ratio", 0.5)
        legal_score = max(0.0, min(float(str(raw_legal)), 1.0))
        controversy = max(0.0, min(float(str(raw_controversy)), 1.0))

        stage = str(data.get("legal_stage", "mention"))
        if stage not in _VALID_STAGES:
            stage = "mention"

        category = str(data.get("category", "all"))
        if category not in _VALID_CATEGORIES:
            category = "all"

        return UnifiedAnalysisResult(
            legal_score=legal_score,
            legal_stage=stage,
            controversy_score=controversy,
            category=category,
        )

    def _fallback_analysis(
        self,
        title: str,
        items: list[RawTrendItem],
    ) -> UnifiedAnalysisResult:
        """키워드 기반 기본 분석 (LLM 실패 시 폴백)"""
        text = f"{title} {' '.join(i.snippet for i in items[:3])}"
        keyword_hits = sum(1 for kw in LEGAL_KEYWORDS if kw in text)
        legal_score = min(keyword_hits * 0.1, 0.8)

        # 카테고리 추정
        category = "all"
        best_count = 0
        for cat, keywords in CATEGORY_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in text)
            if count > best_count:
                best_count = count
                category = cat

        return UnifiedAnalysisResult(
            legal_score=round(legal_score, 2),
            legal_stage="mention",
            controversy_score=0.5,
            category=category,
        )

    def _calculate_spread_score(self, group_items: list[RawTrendItem]) -> float:
        """확산도 점수 — 소스 다양성 + 시간 분포"""
        unique_sources = len({item.source for item in group_items})
        source_diversity = min(unique_sources / 5.0, 1.0)  # 7개 소스 기준

        has_timestamps = [
            item for item in group_items if item.published_at is not None
        ]
        if len(has_timestamps) >= 2:
            timestamps = sorted(
                item.published_at for item in has_timestamps if item.published_at is not None
            )
            time_span = (timestamps[-1] - timestamps[0]).total_seconds()
            # 24시간 이상 지속이면 높은 확산도
            time_factor = min(time_span / 86400, 1.0)
        else:
            time_factor = 0.3

        return round(0.6 * source_diversity + 0.4 * time_factor, 2)

    def _calculate_fitness_score(
        self,
        title: str,
        category: str,
        persona: LawyerPersona,
    ) -> float:
        """채널 적합도 — 페르소나 전문분야와 이슈 카테고리 매칭"""
        # 1. 카테고리 매칭 (전문분야 일치 시 높은 점수)
        specialty_values = [area.value for area in persona.specialty_areas]
        category_match = 1.0 if category in specialty_values else 0.3

        # 2. 키워드 매칭 (focus_topics과 제목 겹침)
        keyword_match = 0.0
        if persona.focus_topics:
            matches = sum(1 for topic in persona.focus_topics if topic in title)
            keyword_match = min(matches / max(len(persona.focus_topics), 1), 1.0)

        return round(0.7 * category_match + 0.3 * keyword_match, 2)
