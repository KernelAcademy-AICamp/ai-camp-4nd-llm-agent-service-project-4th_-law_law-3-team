"""이슈 요약 + 관련 법령/판례 RAG 매칭

v2.0: ScoredIssueV2 지원 — score_detail, fitness_label 포함
"""

import asyncio
import logging
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage

from app.modules.content_marketing.schema import (
    LegalStage,
    RelatedCase,
    RelatedLaw,
    SourceArticle,
    TrendCategory,
    TrendIssue,
    TrendScoreDetail,
)
from app.services.rag.keyword_search import search_by_keyword
from app.tools.llm import get_chat_model
from app.tools.trend.models import ScoredIssue, ScoredIssueV2

logger = logging.getLogger(__name__)


class IssueSummarizer:
    """이슈 요약 + 관련 법령/판례 매칭"""

    def __init__(self) -> None:
        pass

    async def summarize(
        self,
        scored_issues: list[ScoredIssue],
        limit: int = 10,
    ) -> list[TrendIssue]:
        """스코어링된 이슈를 최종 TrendIssue로 변환 (v1.0 호환)"""
        issues: list[TrendIssue] = []

        for issue in scored_issues[:limit]:
            key_points = await self._generate_key_points(issue)
            related_laws, related_cases = self._find_related_legal(issue)
            summary = await self._generate_summary(issue)
            category = await self._classify_category(issue)

            unique_sources = list(dict.fromkeys(item.source for item in issue.raw_items))

            issues.append(
                TrendIssue(
                    id=issue.id,
                    title=issue.title,
                    summary=summary,
                    key_points=key_points,
                    score=issue.combined_score,
                    mention_score=issue.mention_score,
                    legal_relevance_score=issue.legal_relevance_score,
                    category=category,
                    sources=unique_sources,
                    source_articles=[
                        SourceArticle(
                            title=item.title,
                            url=item.url,
                            source=item.source,
                            published_at=item.published_at,
                            snippet=item.snippet,
                        )
                        for item in issue.raw_items
                    ],
                    related_laws=related_laws,
                    related_cases=related_cases,
                    collected_at=datetime.now(tz=timezone.utc),
                )
            )

        return issues

    async def summarize_v2(
        self,
        scored_issues: list[ScoredIssueV2],
        limit: int = 10,
    ) -> list[TrendIssue]:
        """v2.0 스코어링 이슈를 TrendIssue로 변환 (병렬 LLM + RAG 처리)"""
        issues_to_process = scored_issues[:limit]

        # 모든 이슈를 병렬 처리
        results = await asyncio.gather(
            *[self._process_single_issue_v2(issue) for issue in issues_to_process]
        )

        return list(results)

    async def _process_single_issue_v2(self, issue: ScoredIssueV2) -> TrendIssue:
        """단일 이슈의 LLM 요약을 병렬 실행 (법령/판례는 상세 조회 시 지연 검색)"""
        key_points_task = self._generate_key_points_v2(issue)
        summary_task = self._generate_summary_v2(issue)

        key_points, summary = await asyncio.gather(key_points_task, summary_task)

        related_laws: list[RelatedLaw] = []
        related_cases: list[RelatedCase] = []

        # 카테고리 변환 (str → TrendCategory)
        try:
            category = TrendCategory(issue.category)
        except ValueError:
            category = TrendCategory.ALL

        unique_sources = list(dict.fromkeys(item.source for item in issue.raw_items))

        # LegalStage 변환
        try:
            legal_stage = LegalStage(issue.legal_stage)
        except ValueError:
            legal_stage = LegalStage.MENTION

        score_detail = TrendScoreDetail(
            mention_score=issue.mention_score,
            legal_score=issue.legal_score,
            controversy_score=issue.controversy_score,
            spread_score=issue.spread_score,
            fitness_score=issue.fitness_score,
            legal_stage=legal_stage,
            legal_gate_passed=issue.legal_gate_passed,
            gate_rejection_reason=issue.gate_rejection_reason,
            combined_score=issue.combined_score,
        )

        fitness_label = (
            f"채널 적합도 {int(issue.fitness_score * 100)}%"
            if issue.legal_gate_passed
            else None
        )

        return TrendIssue(
            id=issue.id,
            title=issue.title,
            summary=summary,
            key_points=key_points,
            score=issue.combined_score,
            mention_score=issue.mention_score,
            legal_relevance_score=issue.legal_score,
            category=category,
            score_detail=score_detail,
            fitness_label=fitness_label,
            sources=unique_sources,
            source_articles=[
                SourceArticle(
                    title=item.title,
                    url=item.url,
                    source=item.source,
                    published_at=item.published_at,
                    snippet=item.snippet,
                )
                for item in issue.raw_items
            ],
            related_laws=related_laws,
            related_cases=related_cases,
            collected_at=datetime.now(tz=timezone.utc),
        )

    # ── v2.0 헬퍼 (ScoredIssueV2 용) ──

    async def _generate_key_points_v2(self, issue: ScoredIssueV2) -> list[str]:
        """LLM으로 핵심 쟁점 3줄 생성 (v2.0)"""
        context = "\n".join(
            f"- {item.title}: {item.snippet}" for item in issue.raw_items[:5]
        )
        prompt = (
            "다음 뉴스 이슈의 핵심 쟁점을 3줄로 요약하세요.\n"
            "각 줄은 30자 이내로, 핵심 논점만 작성합니다.\n"
            "줄바꿈으로 구분하여 3줄만 출력하세요.\n\n"
            f"이슈: {issue.title}\n"
            f"관련 기사:\n{context}"
        )
        response = await self._llm_generate(prompt)
        lines = [line.strip().lstrip("0123456789.-) ") for line in response.strip().split("\n") if line.strip()]
        return lines[:3] if lines else ["핵심 쟁점 분석 중"]

    def _find_related_legal_v2(
        self,
        issue: ScoredIssueV2,
    ) -> tuple[list[RelatedLaw], list[RelatedCase]]:
        """BM25 키워드 검색으로 관련 법령/판례 검색 (v2.1)"""
        query = f"{issue.title} {issue.raw_items[0].snippet[:200]}" if issue.raw_items else issue.title

        law_docs = search_by_keyword(query, n_results=3, doc_type="law")
        related_laws = [
            RelatedLaw(
                law_id=doc.get("metadata", {}).get("doc_id", ""),
                law_name=doc.get("metadata", {}).get("case_name", ""),
                relevance_score=round(doc.get("similarity", 0.5), 2),
            )
            for doc in law_docs
        ]

        case_docs = search_by_keyword(query, n_results=3, doc_type="precedent")
        related_cases = [
            RelatedCase(
                case_id=doc.get("metadata", {}).get("doc_id", ""),
                case_number=doc.get("metadata", {}).get("case_number", ""),
                case_name=doc.get("metadata", {}).get("case_name", ""),
                relevance_score=round(doc.get("similarity", 0.5), 2),
            )
            for doc in case_docs
        ]

        return related_laws, related_cases

    async def _generate_summary_v2(self, issue: ScoredIssueV2) -> str:
        """LLM 1~2문장 요약 (v2.0)"""
        context = "\n".join(
            f"- {item.title}: {item.snippet}" for item in issue.raw_items[:3]
        )
        prompt = (
            "다음 뉴스 이슈를 1~2문장으로 요약하세요.\n"
            "100자 이내로 간결하게 작성합니다.\n\n"
            f"이슈: {issue.title}\n"
            f"관련 기사:\n{context}"
        )
        return await self._llm_generate(prompt)

    # ── v1.0 헬퍼 (ScoredIssue 용) ──

    async def _generate_key_points(self, issue: ScoredIssue) -> list[str]:
        """LLM으로 핵심 쟁점 3줄 생성"""
        context = "\n".join(
            f"- {item.title}: {item.snippet}" for item in issue.raw_items[:5]
        )
        prompt = (
            "다음 뉴스 이슈의 핵심 쟁점을 3줄로 요약하세요.\n"
            "각 줄은 30자 이내로, 핵심 논점만 작성합니다.\n"
            "줄바꿈으로 구분하여 3줄만 출력하세요.\n\n"
            f"이슈: {issue.title}\n"
            f"관련 기사:\n{context}"
        )
        response = await self._llm_generate(prompt)
        lines = [line.strip().lstrip("0123456789.-) ") for line in response.strip().split("\n") if line.strip()]
        return lines[:3] if lines else ["핵심 쟁점 분석 중"]

    def _find_related_legal(
        self,
        issue: ScoredIssue,
    ) -> tuple[list[RelatedLaw], list[RelatedCase]]:
        """BM25 키워드 검색으로 관련 법령/판례 검색 (v2.1)"""
        query = f"{issue.title} {issue.raw_items[0].snippet[:200]}" if issue.raw_items else issue.title

        law_docs = search_by_keyword(query, n_results=3, doc_type="law")
        related_laws = [
            RelatedLaw(
                law_id=doc.get("metadata", {}).get("doc_id", ""),
                law_name=doc.get("metadata", {}).get("case_name", ""),
                relevance_score=round(doc.get("similarity", 0.5), 2),
            )
            for doc in law_docs
        ]

        case_docs = search_by_keyword(query, n_results=3, doc_type="precedent")
        related_cases = [
            RelatedCase(
                case_id=doc.get("metadata", {}).get("doc_id", ""),
                case_number=doc.get("metadata", {}).get("case_number", ""),
                case_name=doc.get("metadata", {}).get("case_name", ""),
                relevance_score=round(doc.get("similarity", 0.5), 2),
            )
            for doc in case_docs
        ]

        return related_laws, related_cases

    async def _generate_summary(self, issue: ScoredIssue) -> str:
        """LLM 1~2문장 요약"""
        context = "\n".join(
            f"- {item.title}: {item.snippet}" for item in issue.raw_items[:3]
        )
        prompt = (
            "다음 뉴스 이슈를 1~2문장으로 요약하세요.\n"
            "100자 이내로 간결하게 작성합니다.\n\n"
            f"이슈: {issue.title}\n"
            f"관련 기사:\n{context}"
        )
        return await self._llm_generate(prompt)

    async def _classify_category(self, issue: ScoredIssue) -> TrendCategory:
        """LLM으로 법률 카테고리 분류"""
        categories = ", ".join(c.value for c in TrendCategory if c != TrendCategory.ALL)
        prompt = (
            f"다음 법률 이슈의 카테고리를 하나만 선택하세요: {categories}\n"
            "카테고리 이름만 영어로 출력하세요.\n\n"
            f"이슈: {issue.title}"
        )
        response = await self._llm_generate(prompt)
        response_lower = response.strip().lower()
        for cat in TrendCategory:
            if cat.value == response_lower:
                return cat
        return TrendCategory.ALL

    async def _llm_generate(self, prompt: str) -> str:
        """LLM 텍스트 생성 헬퍼"""
        try:
            llm = get_chat_model(temperature=0.3)
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            return response.content.strip() if isinstance(response.content, str) else ""
        except Exception:
            logger.warning("LLM 생성 실패", exc_info=True)
            return ""
