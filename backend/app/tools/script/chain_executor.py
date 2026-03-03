"""3단계 RAG 프롬프트 체인 실행기

Design 문서 §5.4 기반.
Chain 1: 쟁점 추출 (LLM) + 자가 검증 (Red Team [보완 1])
Chain 2: 쟁점별 법령+판례 병렬 RAG 검색
Chain 3: 컨텍스트 구성 + 교차 검증
"""

import asyncio
import json
import logging
import time
from typing import Any

from langchain_core.messages import HumanMessage

from app.services.rag.pipeline import PipelineConfig, RAGPipeline
from app.tools.llm import get_chat_model
from app.tools.persona.models import ScriptContext

logger = logging.getLogger(__name__)

# Chain 1 캐시 TTL (초)
_CHAIN1_CACHE_TTL: int = 3600

# 타임아웃 (초)
_CHAIN1_TIMEOUT: int = 15
_CHAIN2_TIMEOUT: int = 45
_OVERALL_TIMEOUT: int = 60


class PromptChainExecutor:
    """3단계 RAG 프롬프트 체인

    Flow: Chain 1 → (자가 검증) → Chain 2 → Chain 3
    """

    def __init__(self) -> None:
        self._rag = RAGPipeline()
        # topic → (issues, timestamp) 캐시
        self._chain1_cache: dict[str, tuple[list[dict[str, str]], float]] = {}

    async def execute(
        self,
        topic: str,
        trend_context: str = "",
        trend_key_points: list[str] | None = None,
    ) -> ScriptContext:
        """Chain 1 → (자가 검증) → Chain 2 → Chain 3 실행

        각 체인에 개별 타임아웃 + 전체 타임아웃 적용.
        """
        try:
            return await asyncio.wait_for(
                self._execute_chains(topic, trend_context, trend_key_points),
                timeout=_OVERALL_TIMEOUT,
            )
        except TimeoutError:
            logger.warning("PromptChainExecutor 전체 타임아웃 (%ds)", _OVERALL_TIMEOUT)
            # 기본 쟁점으로 빈 컨텍스트 반환
            return ScriptContext(
                issues=[{"쟁점": topic, "검색_쿼리": topic}],
                laws_by_issue={},
                cases_by_issue={},
                cross_validated=False,
                chain_latency_ms={"timeout": _OVERALL_TIMEOUT * 1000},
            )

    async def _execute_chains(
        self,
        topic: str,
        trend_context: str = "",
        trend_key_points: list[str] | None = None,
    ) -> ScriptContext:
        """체인 실행 내부 로직 (타임아웃은 execute에서 관리)"""
        latency: dict[str, int] = {}

        # Chain 1: 쟁점 추출 (개별 타임아웃)
        start = time.monotonic()
        try:
            issues = await asyncio.wait_for(
                self._chain1_extract_issues(topic, trend_context),
                timeout=_CHAIN1_TIMEOUT,
            )
        except TimeoutError:
            logger.warning("Chain 1 타임아웃 (%ds), 기본 쟁점 사용", _CHAIN1_TIMEOUT)
            issues = [{"쟁점": topic, "검색_쿼리": topic}]
        latency["chain_1"] = int((time.monotonic() - start) * 1000)
        logger.info("Chain 1 완료: %d건 쟁점 추출 (%dms)", len(issues), latency["chain_1"])

        # Chain 1.5: 자가 검증 (Red Team [보완 1])
        if trend_key_points:
            issues = self._validate_chain1_against_key_points(issues, trend_key_points)
            logger.info("Chain 1.5 검증 후: %d건 유효", len(issues))

        # Chain 2: RAG 심화 검색 (개별 타임아웃)
        start = time.monotonic()
        try:
            laws_by_issue, cases_by_issue = await asyncio.wait_for(
                self._chain2_rag_search(issues),
                timeout=_CHAIN2_TIMEOUT,
            )
        except TimeoutError:
            logger.warning("Chain 2 타임아웃 (%ds), 빈 RAG 결과 사용", _CHAIN2_TIMEOUT)
            laws_by_issue = {}
            cases_by_issue = {}
        latency["chain_2"] = int((time.monotonic() - start) * 1000)
        total_laws = sum(len(v) for v in laws_by_issue.values())
        total_cases = sum(len(v) for v in cases_by_issue.values())
        logger.info(
            "Chain 2 완료: 법령 %d건, 판례 %d건 (%dms)",
            total_laws,
            total_cases,
            latency["chain_2"],
        )

        # Chain 3: 컨텍스트 구성 + 교차 검증
        start = time.monotonic()
        context = self._chain3_build_context(issues, laws_by_issue, cases_by_issue, latency)
        latency["chain_3"] = int((time.monotonic() - start) * 1000)
        logger.info("Chain 3 완료: cross_validated=%s (%dms)", context.cross_validated, latency["chain_3"])

        return context

    # ── Chain 1: 쟁점 추출 ──

    async def _chain1_extract_issues(
        self,
        topic: str,
        trend_context: str,
    ) -> list[dict[str, str]]:
        """LLM으로 법적 쟁점 추출 (캐시 확인)"""
        cache_key = f"{topic}:{trend_context[:100]}"

        # 캐시 확인
        cached = self._chain1_cache.get(cache_key)
        if cached is not None:
            issues, ts = cached
            if time.time() - ts < _CHAIN1_CACHE_TTL:
                logger.info("Chain 1 캐시 히트: %s", topic[:50])
                return issues
            del self._chain1_cache[cache_key]

        context_section = f"\n\n트렌드 맥락:\n{trend_context}" if trend_context else ""
        prompt = (
            "다음 법률 주제에서 핵심 법적 쟁점 2~4개를 추출하세요.\n"
            "각 쟁점마다 관련 법령/판례 검색에 사용할 쿼리도 작성하세요.\n\n"
            "JSON 배열로만 응답:\n"
            '[{"쟁점": "쟁점 설명", "검색_쿼리": "RAG 검색용 쿼리"}, ...]\n\n'
            f"주제: {topic}{context_section}"
        )

        try:
            llm = get_chat_model(provider="upstage", temperature=0.2)
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            raw = response.content.strip() if isinstance(response.content, str) else ""
            issues = self._parse_issues_json(raw)
        except Exception:
            logger.warning("Chain 1 LLM 호출 실패, 기본 쟁점 생성", exc_info=True)
            issues = [{"쟁점": topic, "검색_쿼리": topic}]

        # 캐시 저장
        self._chain1_cache[cache_key] = (issues, time.time())
        return issues

    def _parse_issues_json(self, raw: str) -> list[dict[str, str]]:
        """3단계 JSON 폴백 파싱"""
        # Stage 1: 직접 파싱
        try:
            data = json.loads(raw)
            if isinstance(data, list) and data:
                return [self._normalize_issue(item) for item in data[:4]]
        except (json.JSONDecodeError, KeyError):
            pass

        # Stage 2: JSON 블록 추출
        try:
            start = raw.index("[")
            end = raw.rindex("]") + 1
            data = json.loads(raw[start:end])
            if isinstance(data, list) and data:
                return [self._normalize_issue(item) for item in data[:4]]
        except (ValueError, json.JSONDecodeError, KeyError):
            pass

        # Stage 3: 라인별 추출
        logger.warning("Chain 1 JSON 파싱 실패, 라인별 추출 시도")
        lines = [line.strip().lstrip("0123456789.-) ") for line in raw.split("\n") if line.strip()]
        return [{"쟁점": line, "검색_쿼리": line} for line in lines[:3]] if lines else []

    def _normalize_issue(self, item: Any) -> dict[str, str]:
        """쟁점 항목 정규화"""
        if isinstance(item, dict):
            return {
                "쟁점": str(item.get("쟁점", item.get("issue", ""))),
                "검색_쿼리": str(item.get("검색_쿼리", item.get("query", item.get("쟁점", "")))),
            }
        return {"쟁점": str(item), "검색_쿼리": str(item)}

    # ── Chain 1.5: 자가 검증 (Red Team [보완 1]) ──

    def _validate_chain1_against_key_points(
        self,
        issues: list[dict[str, str]],
        key_points: list[str],
    ) -> list[dict[str, str]]:
        """Chain 1 쟁점을 트렌드 key_points와 유사도 비교하여 할루시네이션 필터링"""
        validated: list[dict[str, str]] = []

        for issue in issues:
            issue_text = issue.get("쟁점", "")
            # 키워드 겹침으로 간단 검증 (임베딩 유사도는 v2.1에서)
            overlap = sum(
                1
                for kp in key_points
                if kp in issue_text or issue_text in kp
            )
            if overlap > 0 or len(issues) <= 2:
                validated.append(issue)

        # 전부 탈락 시 상위 2개 유지
        return validated if validated else issues[:2]

    # ── Chain 2: RAG 심화 검색 ──

    async def _chain2_rag_search(
        self,
        issues: list[dict[str, str]],
    ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
        """각 쟁점별 법령+판례 병렬 RAG 검색"""
        law_config = PipelineConfig(
            n_results=10,
            doc_type="law",
            enable_rerank=False,
        )
        case_config = PipelineConfig(
            n_results=10,
            doc_type="precedent",
            enable_rerank=False,
        )

        # 병렬 실행 준비
        tasks: list[asyncio.Task[Any]] = []
        issue_keys: list[str] = []

        for issue in issues:
            query = issue.get("검색_쿼리", issue.get("쟁점", ""))
            if not query:
                continue
            issue_keys.append(issue.get("쟁점", query))
            tasks.append(
                asyncio.create_task(self._rag.execute_async(query, law_config))
            )
            tasks.append(
                asyncio.create_task(self._rag.execute_async(query, case_config))
            )

        if not tasks:
            return {}, {}

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 결과 매핑 (쟁점별로 law, case 교대)
        laws_by_issue: dict[str, list[dict[str, Any]]] = {}
        cases_by_issue: dict[str, list[dict[str, Any]]] = {}

        for i, key in enumerate(issue_keys):
            law_idx = i * 2
            case_idx = i * 2 + 1

            law_result = results[law_idx] if law_idx < len(results) else None
            if law_result is not None and not isinstance(law_result, BaseException):
                laws_by_issue[key] = law_result.documents
            else:
                laws_by_issue[key] = []

            case_result = results[case_idx] if case_idx < len(results) else None
            if case_result is not None and not isinstance(case_result, BaseException):
                cases_by_issue[key] = case_result.documents
            else:
                cases_by_issue[key] = []

        return laws_by_issue, cases_by_issue

    # ── Chain 3: 컨텍스트 구성 + 교차 검증 ──

    def _chain3_build_context(
        self,
        issues: list[dict[str, str]],
        laws_by_issue: dict[str, list[dict[str, Any]]],
        cases_by_issue: dict[str, list[dict[str, Any]]],
        latency: dict[str, int],
    ) -> ScriptContext:
        """컨텍스트 구성 + 교차 검증"""
        # 교차 검증: 모든 쟁점에 최소 1개 이상의 법령 또는 판례가 있는지
        cross_validated = all(
            len(laws_by_issue.get(issue.get("쟁점", ""), [])) > 0
            or len(cases_by_issue.get(issue.get("쟁점", ""), [])) > 0
            for issue in issues
        )

        # ScriptContext용 데이터 변환
        issues_data: list[dict[str, object]] = [
            {"쟁점": issue.get("쟁점", ""), "검색_쿼리": issue.get("검색_쿼리", "")}
            for issue in issues
        ]
        laws_data: dict[str, list[object]] = {
            k: [dict(doc) for doc in v] for k, v in laws_by_issue.items()
        }
        cases_data: dict[str, list[object]] = {
            k: [dict(doc) for doc in v] for k, v in cases_by_issue.items()
        }

        return ScriptContext(
            issues=issues_data,
            laws_by_issue=laws_data,
            cases_by_issue=cases_data,
            cross_validated=cross_validated,
            chain_latency_ms=latency,
        )
