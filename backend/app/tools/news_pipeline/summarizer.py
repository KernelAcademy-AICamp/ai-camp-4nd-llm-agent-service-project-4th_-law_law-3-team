"""LLM 기반 구조화 요약 생성"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.tools.llm import get_chat_model
from app.tools.news_pipeline.config import NewsPipelineConfig
from app.tools.news_pipeline.exceptions import SummaryGenerationError
from app.tools.news_pipeline.models import ArticleSummary, CleanedArticle

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """당신은 법률 전문 기자입니다. 아래 기사를 법률 AI 보조 코퍼스용으로 요약해주세요.

## 출력 형식 (JSON만 출력, 다른 텍스트 없이)
{
  "one_liner": "핵심 한 문장 요약",
  "issues": ["쟁점1", "쟁점2"],
  "mentions": {
    "laws": ["관련 법령명"],
    "cases": ["관련 판례"],
    "institutions": ["관련 기관"]
  },
  "implications": ["시사점1"]
}

## 규칙
- 기사에 없는 판례/법령 조문 번호를 만들어내지 마세요
- "확정적 법률 결론"을 단정하지 마세요 (기사 요약임)
- 한국어로 작성하세요
- issues는 3~5개, implications는 1~3개"""


# 요약 JSON 품질 게이트 (Pydantic strict validation)
class MentionsSchema(BaseModel):
    """요약 내 참조 스키마"""

    laws: list[str] = Field(default_factory=list, max_length=20)
    cases: list[str] = Field(default_factory=list, max_length=20)
    institutions: list[str] = Field(default_factory=list, max_length=20)


class SummarySchema(BaseModel):
    """LLM 요약 출력 검증 스키마"""

    one_liner: str = Field(min_length=5, max_length=500)
    issues: list[str] = Field(min_length=1, max_length=10)
    mentions: MentionsSchema = Field(default_factory=MentionsSchema)
    implications: list[str] = Field(min_length=1, max_length=5)


class Summarizer:
    """LLM 기반 구조화 요약 생성기

    Primary: Upstage Solar-Pro2
    Fallback: OpenAI → 건너뛰기
    """

    def __init__(self, config: NewsPipelineConfig) -> None:
        self._config = config

    async def summarize(self, article: CleanedArticle) -> ArticleSummary:
        """기사를 구조화 요약으로 변환"""
        user_prompt = self._build_user_prompt(article)
        primary_err: Exception | None = None

        # Primary LLM 시도
        try:
            return await self._call_llm(
                provider=self._config.summary_provider,
                model=self._config.summary_model,
                user_prompt=user_prompt,
            )
        except Exception as exc:
            primary_err = exc
            logger.warning(
                "Primary LLM 실패 (%s/%s): %s",
                self._config.summary_provider,
                self._config.summary_model,
                exc,
            )

        # Fallback LLM 시도
        try:
            return await self._call_llm(
                provider=self._config.fallback_provider,
                model=None,
                user_prompt=user_prompt,
            )
        except Exception as fallback_exc:
            raise SummaryGenerationError(
                article.url,
                f"Primary + Fallback 모두 실패: {primary_err} / {fallback_exc}",
            ) from fallback_exc

    async def _call_llm(
        self,
        provider: str,
        model: str | None,
        user_prompt: str,
    ) -> ArticleSummary:
        """LLM 호출 및 JSON 파싱"""
        llm = get_chat_model(
            provider=provider,
            model=model,
            temperature=self._config.summary_temperature,
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        response = await llm.ainvoke(messages)
        raw_content = response.content
        if isinstance(raw_content, list):
            raw_text = str(raw_content[0]) if raw_content else ""
        else:
            raw_text = str(raw_content)

        return self._parse_response(raw_text)

    @staticmethod
    def _build_user_prompt(article: CleanedArticle) -> str:
        """사용자 프롬프트 생성"""
        pub_date = article.published_at.isoformat() if article.published_at else "불명"
        return (
            f"## 기사 원문\n"
            f"제목: {article.title}\n"
            f"매체: {article.publisher}\n"
            f"발행일: {pub_date}\n\n"
            f"{article.cleaned_text[:8000]}"
        )

    @staticmethod
    def _parse_response(text: str) -> ArticleSummary:
        """LLM 응답 JSON 파싱 + Pydantic 품질 게이트 검증"""
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        data: dict[str, Any] = json.loads(text)

        validated = SummarySchema.model_validate(data)
        mentions = validated.mentions

        return ArticleSummary(
            one_liner=validated.one_liner,
            issues=validated.issues,
            laws=mentions.laws if mentions else [],
            cases=mentions.cases if mentions else [],
            institutions=mentions.institutions if mentions else [],
            implications=validated.implications,
        )
