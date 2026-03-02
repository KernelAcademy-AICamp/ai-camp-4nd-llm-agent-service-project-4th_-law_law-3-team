"""3단 구조 대본 생성기 (SSE 스트리밍)

v2.0: PromptChainExecutor 연동, PersonaTone 지원, stage_update 이벤트
v2.1: SSE heartbeat, 비동기 RAG 폴백, 타임아웃 안전장치
"""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

from langchain_core.messages import HumanMessage

from app.modules.content_marketing.schema import (
    PERSONA_TYPE_TO_TONE,
    MetadataRequest,
    PersonaTone,
    PersonaType,
    ScriptMetadata,
    ScriptRequest,
    ScriptStreamEvent,
    SectionType,
)
from app.services.rag.pipeline import PipelineConfig, RAGPipeline
from app.tools.llm import get_chat_model
from app.tools.persona.models import ScriptContext
from app.tools.script.chain_executor import PromptChainExecutor
from app.tools.script.templates import METADATA_PROMPT, get_section_prompt

logger = logging.getLogger(__name__)

# v1.0 (하위호환)
WORDS_PER_MINUTE: dict[PersonaType, int] = {
    PersonaType.PROFESSIONAL: 250,
    PersonaType.CASUAL: 300,
}

# v2.0 PersonaTone별 분당 글자 수
WORDS_PER_TONE: dict[PersonaTone, int] = {
    PersonaTone.PROFESSIONAL: 250,
    PersonaTone.CASUAL: 300,
    PersonaTone.STORYTELLING: 280,
    PersonaTone.EDUCATIONAL: 260,
}

SECTIONS_CONFIG: list[tuple[SectionType, str, float]] = [
    (SectionType.HOOKING, "도입 (Hooking)", 0.15),
    (SectionType.ANALYSIS, "본론 (Legal Analysis)", 0.65),
    (SectionType.ADVICE_CTA, "결론 (Advice & CTA)", 0.20),
]


class ScriptGenerator:
    """3단 구조 대본 생성기 (SSE 스트리밍)

    v2.0: PromptChainExecutor 연동, PersonaTone 지원
    """

    def __init__(self) -> None:
        self._rag = RAGPipeline()
        self._chain_executor = PromptChainExecutor()

    async def generate_stream(
        self,
        request: ScriptRequest,
        persona_tone: PersonaTone | None = None,
    ) -> AsyncGenerator[ScriptStreamEvent, None]:
        """대본을 섹션별로 SSE 스트리밍 생성

        v2.0: persona_tone 우선, PromptChainExecutor로 RAG 심화 검색
        v2.1: RAG 검색 중 heartbeat 전송으로 SSE 연결 유지
        """
        # PersonaTone 결정: 인자 > PERSONA_TYPE_TO_TONE 매핑
        resolved_tone = persona_tone or PERSONA_TYPE_TO_TONE.get(request.persona)

        # 목표 글자 수 계산
        if resolved_tone is not None:
            target_words = WORDS_PER_TONE[resolved_tone] * request.duration
        else:
            target_words = WORDS_PER_MINUTE[request.persona] * request.duration

        # stage_update: RAG 심화 검색 시작
        yield ScriptStreamEvent(
            event="stage_update",
            stage="rag_search",
            status="started",
            detail="법령/판례 RAG 심화 검색 중...",
        )

        # RAG 검색을 백그라운드로 실행하면서 3초마다 heartbeat 전송
        rag_task: asyncio.Task[dict[str, Any]] = asyncio.create_task(
            self._build_rag_context(request)
        )
        while not rag_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(rag_task), timeout=3.0)
            except TimeoutError:
                # 아직 진행 중 → heartbeat 전송
                yield ScriptStreamEvent(
                    event="stage_update",
                    stage="rag_search",
                    status="in_progress",
                    detail="법령/판례 검색 진행 중...",
                )

        rag_context = rag_task.result()

        yield ScriptStreamEvent(
            event="stage_update",
            stage="rag_search",
            status="completed",
            detail="RAG 검색 완료",
        )

        # 섹션별 스트리밍 생성 (대본 내용 누적)
        accumulated_content: list[str] = []

        for section_type, _title, ratio in SECTIONS_CONFIG:
            section_words = int(target_words * ratio)

            yield ScriptStreamEvent(
                event="section_start",
                section=section_type,
            )

            prompt = get_section_prompt(
                section_type=section_type,
                topic=request.topic,
                persona=request.persona,
                target_words=section_words,
                rag_context=rag_context,
                persona_tone=resolved_tone,
            )

            async for chunk in self._llm_stream(prompt):
                accumulated_content.append(chunk)
                yield ScriptStreamEvent(
                    event="content",
                    section=section_type,
                    content=chunk,
                )

            yield ScriptStreamEvent(
                event="section_end",
                section=section_type,
            )

        # done을 먼저 전송하여 프론트엔드 UI 즉시 해제
        yield ScriptStreamEvent(event="done")

        # 메타데이터는 done 이후 비동기 생성 (SSE 스트림은 아직 열려 있음)
        full_script = "".join(accumulated_content)
        try:
            metadata = await self._generate_metadata_from_request(request, full_script)
            yield ScriptStreamEvent(
                event="metadata",
                metadata=metadata,
            )
        except Exception:
            logger.warning("메타데이터 생성 실패 (done 이후), 무시", exc_info=True)

    # ── RAG 컨텍스트 구성 ──

    def _build_trend_context_from_news(self, request: ScriptRequest) -> str:
        """선택된 뉴스 기사를 트렌드 컨텍스트 문자열로 변환"""
        if not request.news_articles:
            return ""

        parts: list[str] = []
        for i, article in enumerate(request.news_articles, 1):
            snippet = article.snippet[:200] if article.snippet else ""
            source_info = f" ({article.source})" if article.source else ""
            parts.append(f"[뉴스 {i}] {article.title}{source_info}\n{snippet}")

        return "\n\n".join(parts)

    async def _build_rag_context(self, request: ScriptRequest) -> dict[str, Any]:
        """v2.0: PromptChainExecutor로 심화 RAG 검색, 실패 시 기존 방식 폴백
        v2.1: 폴백도 비동기로 실행 (이벤트 루프 블로킹 방지)
        """
        trend_context = self._build_trend_context_from_news(request)
        try:
            context = await self._chain_executor.execute(
                topic=request.topic,
                trend_context=trend_context,
            )
            return self._script_context_to_rag_dict(context)
        except Exception:
            logger.warning(
                "PromptChainExecutor 실패, 기존 RAG 검색으로 폴백", exc_info=True
            )
            return await asyncio.to_thread(self._search_legal_context, request)

    def _script_context_to_rag_dict(
        self,
        context: ScriptContext,
    ) -> dict[str, Any]:
        """ScriptContext → 템플릿용 dict 변환"""
        laws: list[object] = []
        cases: list[object] = []

        for law_docs in context.laws_by_issue.values():
            for doc in law_docs:
                laws.append(doc)

        for case_docs in context.cases_by_issue.values():
            for doc in case_docs:
                cases.append(doc)

        return {"laws": laws, "cases": cases}

    def _search_legal_context(self, request: ScriptRequest) -> dict[str, Any]:
        """기존 RAG 검색 (v1.0 폴백)"""
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

        law_result = self._rag.execute(request.topic, law_config)
        case_result = self._rag.execute(request.topic, case_config)

        return {
            "laws": law_result.documents,
            "cases": case_result.documents,
        }

    # ── LLM / 메타데이터 ──

    # 청크 간 최대 대기 시간 (LLM API 무응답 감지)
    _CHUNK_TIMEOUT: int = 45
    # 섹션별 최대 생성 시간
    _SECTION_TIMEOUT: int = 150

    async def _llm_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """LLM 스트리밍 생성 (Upstage Solar Pro2)

        v2.1: 청크별 타임아웃 + 섹션별 타임아웃으로 무한 대기 방지
        """
        llm = get_chat_model(provider="upstage", temperature=0.7)
        stream = llm.astream([HumanMessage(content=prompt)])
        aiter = stream.__aiter__()
        deadline = asyncio.get_event_loop().time() + self._SECTION_TIMEOUT
        chunk_count = 0

        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                logger.warning(
                    "LLM 섹션 타임아웃 (%ds), %d 청크 생성 후 중단",
                    self._SECTION_TIMEOUT, chunk_count,
                )
                break

            try:
                chunk = await asyncio.wait_for(
                    aiter.__anext__(),
                    timeout=min(self._CHUNK_TIMEOUT, remaining),
                )
                if hasattr(chunk, "content") and chunk.content:
                    chunk_count += 1
                    yield str(chunk.content)
            except StopAsyncIteration:
                break
            except TimeoutError:
                logger.warning(
                    "LLM 청크 타임아웃 (%ds), %d 청크 생성 후 중단",
                    self._CHUNK_TIMEOUT, chunk_count,
                )
                break

    async def _generate_metadata_from_request(
        self,
        request: ScriptRequest,
        script_content: str = "",
    ) -> ScriptMetadata:
        """대본 요청 정보로 메타데이터 생성"""
        content = script_content if len(script_content) >= 100 else (
            f"주제: {request.topic}\n\n{script_content}"
        ).ljust(100, " ")
        metadata_request = MetadataRequest(
            script_content=content,
            topic=request.topic,
            persona=request.persona,
        )
        return await generate_metadata(metadata_request)


async def generate_metadata(request: MetadataRequest) -> ScriptMetadata:
    """대본 메타데이터 LLM 생성"""
    script_summary = request.script_content[:500]
    prompt = METADATA_PROMPT.format(
        topic=request.topic,
        script_summary=script_summary,
    )

    try:
        llm = get_chat_model(provider="upstage", temperature=0.3)
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        raw = response.content.strip() if isinstance(response.content, str) else ""

        # JSON 파싱 (코드블록 제거)
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        data = json.loads(raw)
        return ScriptMetadata(
            description=data.get(
                "description", f"{request.topic}에 대한 법률 분석 영상입니다."
            ),
            tags=data.get("tags", [request.topic, "법률", "변호사"]),
            cta_text=data.get(
                "cta_text",
                "법률 상담이 필요하시면 아래 링크로 무료 상담을 신청하세요.",
            ),
            hashtags=data.get("hashtags", ["#법률", "#변호사", "#법률상식"]),
        )
    except Exception:
        logger.warning("메타데이터 LLM 생성 실패, 기본값 반환", exc_info=True)
        return ScriptMetadata(
            description=f"{request.topic}에 대한 법률 분석 영상입니다.",
            tags=[request.topic, "법률", "변호사", "법률상식", "판례분석"],
            cta_text="법률 상담이 필요하시면 아래 링크로 무료 상담을 신청하세요.",
            hashtags=["#법률", "#변호사", "#법률상식", "#판례", "#법률상담"],
        )
