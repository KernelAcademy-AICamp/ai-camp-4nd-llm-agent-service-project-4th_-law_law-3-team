"""커뮤니티 인기글에서 LLM 기반 핵심 키워드를 추출하는 모듈

v2.1: extract_with_scores() 추가 - 4차원 스코어링 포함 키워드 추출
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections import Counter

from langchain_core.messages import HumanMessage, SystemMessage

from app.tools.llm import get_chat_model
from app.tools.trend.keyword_blacklist import filter_keywords
from app.tools.trend.models import (
    CommunityTopic,
    KeywordScore,
    RawTrendItem,
    ScoredKeyword,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "당신은 한국 트렌드 분석 전문가입니다. "
    "커뮤니티 인기글 목록에서 사회적으로 의미 있는 핵심 키워드를 추출합니다."
)

_EXTRACTION_PROMPT_TEMPLATE = (
    "아래는 한국 온라인 커뮤니티 인기글 목록입니다.\n\n"
    "{context}\n\n"
    "위 글들에서 사회적으로 의미 있는 핵심 키워드를 3~5개 추출하세요.\n"
    "규칙:\n"
    "- 각 키워드는 1~3단어로 구성\n"
    "- 사회 이슈, 논쟁, 정책, 경제, 문화 관련 키워드 우선\n"
    "- 게임, 유머, 엔터테인먼트, 스포츠 주제는 제외\n"
    "- 쉼표로 구분하여 키워드만 출력 (설명 없이)\n\n"
    "키워드:"
)

# 불용어 (폴백 키워드 추출 시 제외)
_STOPWORDS = frozenset({
    "의", "가", "이", "은", "는", "을", "를", "에", "에서", "로", "으로",
    "와", "과", "도", "만", "까지", "부터", "대한", "위한", "있는", "없는",
    "하는", "된", "할", "한", "것", "수", "등", "및", "또는", "그", "이",
    "저", "그것", "때문", "통해", "대해", "관한", "따른",
})

_MIN_KEYWORD_LENGTH = 2
_MAX_FALLBACK_KEYWORDS = 3
_MAX_CONTEXT_ITEMS = 15

# 4차원 스코어링 가중치 (virality, social_impact, legal_relevance, content_fitness)
_SCORE_WEIGHTS = (0.25, 0.25, 0.30, 0.20)

_SCORING_SYSTEM_PROMPT = (
    "당신은 한국 법률 콘텐츠 마케팅 전문가입니다. "
    "커뮤니티 인기글에서 법률 유튜브 콘텐츠로 활용 가능한 키워드를 추출하고 점수를 매깁니다."
)

_SCORING_PROMPT_TEMPLATE = (
    "아래는 한국 온라인 커뮤니티(DC갤러리, 에펨코리아, 더쿠, 보배드림, 인벤 등) 인기글 목록입니다.\n\n"
    "{context}\n\n"
    "위 글들에서 법률 유튜브 콘텐츠로 활용 가능한 사건/사고/이슈 키워드를 최대 {max_keywords}개 추출하고, "
    "각 키워드에 대해 4차원 점수를 매기세요.\n\n"
    "## 점수 기준 (각 0.0~1.0)\n"
    "- virality: 바이럴 잠재력 (언급 빈도, 화제성, 확산 가능성)\n"
    "- social_impact: 사회적 영향도 (사회 문제, 공공 관심도)\n"
    "- legal_relevance: 법적 연관성 (법적 쟁점, 소송 가능성, 법률 해석 필요도)\n"
    "- content_fitness: 콘텐츠 적합도 (영상 제작 가능성, 시청자 관심도)\n\n"
    "## 출력 형식 (JSON 배열만 출력, 설명 없이)\n"
    "```json\n"
    "[\n"
    '  {{"keyword": "키워드1", "context": "관련 맥락 1~2문장", '
    '"virality": 0.8, "social_impact": 0.7, "legal_relevance": 0.9, "content_fitness": 0.85, '
    '"score_reason": "점수 근거 1~2문장", "confidence": 0.9}},\n'
    "  ...\n"
    "]\n"
    "```\n\n"
    "규칙:\n"
    "- 키워드는 1~4단어로 구성된 사건/사고/이슈명\n"
    "- 유사 이슈는 하나로 통합\n"
    "- 게임, 유머, 엔터테인먼트, 스포츠 주제는 제외\n"
    "- legal_relevance가 0.3 미만인 키워드는 제외\n"
    "- JSON 배열만 출력\n"
)

_PROMPT_VERSION = "1.0"


class CommunityKeywordExtractor:
    """커뮤니티 인기글에서 핵심 키워드를 추출"""

    def _build_context_summary(self, items: list[RawTrendItem]) -> str:
        """제목+snippet을 번호 목록으로 조합"""
        lines: list[str] = []
        for i, item in enumerate(items[:_MAX_CONTEXT_ITEMS], start=1):
            snippet_part = f" - {item.snippet[:80]}" if item.snippet else ""
            lines.append(f"{i}. {item.title}{snippet_part}")
        return "\n".join(lines)

    async def _extract_keywords_via_llm(
        self, context: str,
    ) -> list[str]:
        """LLM 호출로 키워드 추출 (temperature=0.0)"""
        llm = get_chat_model(temperature=0.0)
        prompt = _EXTRACTION_PROMPT_TEMPLATE.format(context=context)

        response = await llm.ainvoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])

        raw_text = str(response.content).strip()
        # 쉼표 또는 줄바꿈으로 분리 후 정제
        keywords = [
            kw.strip().strip("- ·•\"'")
            for kw in re.split(r"[,\n]", raw_text)
            if kw.strip()
        ]
        # 빈 문자열, 너무 긴 키워드 필터링
        keywords = [kw for kw in keywords if _MIN_KEYWORD_LENGTH <= len(kw) <= 20]
        return keywords[:5]

    def _fallback_keywords(self, items: list[RawTrendItem]) -> list[str]:
        """LLM 실패 시 제목 단어 빈도 기반 폴백"""
        word_counter: Counter[str] = Counter()
        for item in items:
            words = item.title.split()
            for word in words:
                cleaned = word.strip(".,!?\"'()[]{}·…")
                if len(cleaned) >= _MIN_KEYWORD_LENGTH and cleaned not in _STOPWORDS:
                    word_counter[cleaned] += 1

        return [word for word, _ in word_counter.most_common(_MAX_FALLBACK_KEYWORDS)]

    async def extract(
        self, items: list[RawTrendItem],
    ) -> CommunityTopic:
        """커뮤니티 글 목록에서 키워드 추출 → CommunityTopic 반환"""
        if not items:
            return CommunityTopic(
                raw_items=[],
                extracted_keywords=[],
                context_summary="",
            )

        context_summary = self._build_context_summary(items)

        try:
            keywords = await self._extract_keywords_via_llm(context_summary)
            if not keywords:
                raise ValueError("LLM이 빈 키워드를 반환")
        except Exception:
            logger.warning("LLM 키워드 추출 실패, 폴백 사용", exc_info=True)
            keywords = self._fallback_keywords(items)

        logger.info("키워드 추출 완료: %s", keywords)
        return CommunityTopic(
            raw_items=items,
            extracted_keywords=keywords,
            context_summary=context_summary,
        )

    async def extract_with_scores(
        self,
        items: list[RawTrendItem],
        max_keywords: int = 10,
    ) -> list[ScoredKeyword]:
        """커뮤니티 글 목록에서 키워드 + 4차원 점수 추출

        Returns:
            ScoredKeyword 리스트 (total_score 내림차순, rank 부여)
        """
        if not items:
            return []

        context_summary = self._build_context_summary(items)

        try:
            scored = await self._extract_scored_keywords_via_llm(
                context_summary, max_keywords,
            )
            if not scored:
                raise ValueError("LLM이 빈 스코어드 키워드를 반환")
        except Exception:
            logger.warning("LLM 스코어링 추출 실패, 폴백 사용", exc_info=True)
            scored = self._fallback_scored_keywords(items, max_keywords)

        # Blacklist 필터 적용
        keyword_texts = [s.keyword for s in scored]
        filtered_texts = set(filter_keywords(keyword_texts))
        scored = [s for s in scored if s.keyword in filtered_texts]

        # total_score 내림차순 정렬 + rank 부여
        scored.sort(key=lambda s: s.total_score, reverse=True)
        for i, kw in enumerate(scored):
            kw.rank = i + 1

        logger.info("스코어링 키워드 추출 완료: %d개", len(scored))
        return scored

    async def _extract_scored_keywords_via_llm(
        self,
        context: str,
        max_keywords: int,
    ) -> list[ScoredKeyword]:
        """LLM으로 키워드 + 4차원 점수 추출"""
        llm = get_chat_model(temperature=0.0)
        prompt = _SCORING_PROMPT_TEMPLATE.format(
            context=context, max_keywords=max_keywords,
        )

        response = await llm.ainvoke([
            SystemMessage(content=_SCORING_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])

        raw_text = str(response.content).strip()

        # JSON 블록 추출 (```json ... ``` 또는 바로 배열)
        json_match = re.search(r"\[.*\]", raw_text, re.DOTALL)
        if not json_match:
            raise ValueError(f"JSON 배열을 찾을 수 없음: {raw_text[:200]}")

        parsed: list[dict[str, object]] = json.loads(json_match.group())

        def _to_float(val: object, default: float = 0.5) -> float:
            """object → float 변환 (mypy 안전)"""
            try:
                return float(str(val))
            except (ValueError, TypeError):
                return default

        results: list[ScoredKeyword] = []
        for item in parsed:
            keyword = str(item.get("keyword", "")).strip()
            if not keyword or len(keyword) < _MIN_KEYWORD_LENGTH:
                continue

            virality = _to_float(item.get("virality", 0.5))
            social_impact = _to_float(item.get("social_impact", 0.5))
            legal_relevance = _to_float(item.get("legal_relevance", 0.5))
            content_fitness = _to_float(item.get("content_fitness", 0.5))

            scores = KeywordScore(
                virality=min(max(virality, 0.0), 1.0),
                social_impact=min(max(social_impact, 0.0), 1.0),
                legal_relevance=min(max(legal_relevance, 0.0), 1.0),
                content_fitness=min(max(content_fitness, 0.0), 1.0),
            )

            total = (
                scores.virality * _SCORE_WEIGHTS[0]
                + scores.social_impact * _SCORE_WEIGHTS[1]
                + scores.legal_relevance * _SCORE_WEIGHTS[2]
                + scores.content_fitness * _SCORE_WEIGHTS[3]
            ) * 100

            results.append(ScoredKeyword(
                id=str(uuid.uuid4()),
                keyword=keyword,
                context=str(item.get("context", "")),
                source_posts=[],
                scores=scores,
                total_score=round(total, 1),
                rank=0,
                score_reason=str(item.get("score_reason", "")),
                confidence=min(max(_to_float(item.get("confidence", 0.0), 0.0), 0.0), 1.0),
            ))

        return results

    def _fallback_scored_keywords(
        self,
        items: list[RawTrendItem],
        max_keywords: int,
    ) -> list[ScoredKeyword]:
        """LLM 실패 시 단어 빈도 기반 폴백 (기본 점수 부여)"""
        fallback_keywords = self._fallback_keywords(items)
        filtered = filter_keywords(fallback_keywords)

        results: list[ScoredKeyword] = []
        for kw in filtered[:max_keywords]:
            scores = KeywordScore(
                virality=0.5,
                social_impact=0.5,
                legal_relevance=0.3,
                content_fitness=0.5,
            )
            total = (
                scores.virality * _SCORE_WEIGHTS[0]
                + scores.social_impact * _SCORE_WEIGHTS[1]
                + scores.legal_relevance * _SCORE_WEIGHTS[2]
                + scores.content_fitness * _SCORE_WEIGHTS[3]
            ) * 100

            results.append(ScoredKeyword(
                id=str(uuid.uuid4()),
                keyword=kw,
                context="",
                source_posts=[],
                scores=scores,
                total_score=round(total, 1),
                rank=0,
                score_reason="단어 빈도 기반 폴백 (LLM 추출 실패)",
                confidence=0.3,
            ))

        return results
