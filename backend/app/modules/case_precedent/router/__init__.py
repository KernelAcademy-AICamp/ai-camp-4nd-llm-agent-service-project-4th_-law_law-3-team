"""
판례 추천 모듈 - 업무 사례 기반 관련 판례 제공
RAG 기반으로 사용자 상황에 맞는 판례 검색 및 변호사 추천
"""
import datetime
import logging
from typing import Any, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel
from sqlalchemy import select

from app.core.database import async_session_factory
from app.core.errors import EmbeddingModelNotFoundError
from app.core.rate_limit import AI_RATE_LIMIT, limiter
from app.models.law_article import LawArticle
from app.models.law_document import LawDocument
from app.modules.case_precedent.schema import (
    AIQuestionResponse,
    AskQuestionRequest,
    ChatRequest,
    ChatResponse,
    ChatSource,
    PrecedentDetailResponse,
    PrecedentItem,
    PrecedentListResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from app.services.rag import search_relevant_documents_async
from app.services.service_function.law_filter_service import (
    get_law_filter_service,
)
from app.services.service_function.precedent_service import (
    fetch_precedent_details,
    get_precedent_service,
)
from app.tools.graph.pg_graph_service import get_pg_graph_service
from app.tools.vectorstore import get_vector_store


def _map_data_type(data_type: str) -> str:
    """LanceDB data_type을 표준 doc_type으로 변환"""
    mapping = {
        "판례": "precedent",
        "법령": "law",
        "헌법재판소": "constitutional",
    }
    return mapping.get(data_type, data_type.lower() if data_type else "")

logger = logging.getLogger(__name__)

router = APIRouter()


# API 엔드포인트
@router.post("/chat", response_model=ChatResponse)
@limiter.limit(AI_RATE_LIMIT)
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    """
    RAG 기반 법률 챗봇

    사용자 메시지를 받아 관련 판례를 검색하고 AI 응답 생성.
    통합 채팅 API(/api/chat)와 별개로, 판례 모듈 전용 간이 챗봇.
    """
    try:
        from app.tools.llm import get_chat_model

        # 1. RAG 검색 (판례 + 법령)
        search_results = await search_relevant_documents_async(
            query=body.message,
            n_results=5,
        )

        # 2. 판례 상세 정보 조회
        source_ids = [
            doc.get("metadata", {}).get("doc_id")
            for doc in search_results
            if doc.get("metadata", {}).get("doc_id")
        ]
        precedent_details = await fetch_precedent_details(source_ids) if source_ids else {}

        # 3. 컨텍스트 구성
        context_parts: list[str] = []
        for doc in search_results:
            metadata = doc.get("metadata", {})
            doc_id = metadata.get("doc_id", "")
            detail = precedent_details.get(doc_id, {})
            content = detail.get("reasoning") or doc.get("content", "")
            case_name = detail.get("case_name") or metadata.get("case_name", "")
            case_number = detail.get("case_number") or metadata.get("case_number", "")
            if content:
                context_parts.append(
                    f"[{metadata.get('doc_type', '')}] {case_name} ({case_number})\n{content[:1000]}"
                )

        context_text = "\n\n---\n\n".join(context_parts) if context_parts else "관련 문서를 찾지 못했습니다."

        # 4. LLM 응답 생성
        chat_model = get_chat_model()
        messages = [
            {
                "role": "system",
                "content": (
                    "당신은 한국 법률 전문 AI 어시스턴트입니다. "
                    "제공된 판례와 법령을 참고하여 정확하고 이해하기 쉽게 답변해주세요. "
                    "법률 용어는 쉽게 풀어 설명하고, 구체적인 법률 상담은 변호사에게 의뢰하도록 안내하세요."
                ),
            },
            {
                "role": "user",
                "content": f"[참고 자료]\n{context_text}\n\n[질문]\n{body.message}",
            },
        ]
        if body.history:
            # 대화 기록을 system과 user 사이에 삽입
            history_messages = [{"role": msg.role, "content": msg.content} for msg in body.history]
            messages = [messages[0]] + history_messages + [messages[1]]

        ai_response = await chat_model.ainvoke(messages)
        content = ai_response.content if hasattr(ai_response, "content") else str(ai_response)
        response_text = content if isinstance(content, str) else str(content)

        # 5. 소스 정보 구성
        sources = []
        for doc in search_results:
            metadata = doc.get("metadata", {})
            content = doc.get("content", "")
            sources.append(
                ChatSource(
                    case_name=metadata.get("case_name"),
                    case_number=metadata.get("case_number"),
                    doc_type=metadata.get("doc_type", ""),
                    similarity=doc.get("similarity", 0),
                    content=content[:500] if content else None,
                    summary=content[:300] + "..." if content and len(content) > 300 else content,
                )
            )

        return ChatResponse(response=response_text, sources=sources)
    except EmbeddingModelNotFoundError as e:
        logger.error("임베딩 모델 없음: %s", e)
        raise HTTPException(
            status_code=503,
            detail="임베딩 모델이 준비되지 않았습니다. 서버 관리자에게 문의하세요.",
        )
    except Exception as e:
        logger.error("챗봇 응답 생성 실패: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="챗봇 응답 생성 중 오류가 발생했습니다")


@router.post("/search", response_model=SearchResponse)
@limiter.limit(AI_RATE_LIMIT)
async def search(request: Request, body: SearchRequest) -> SearchResponse:
    """
    법률 문서 유사도 검색

    쿼리와 유사한 판례/법률 문서 검색
    """
    try:
        results = await search_relevant_documents_async(
            query=body.query,
            n_results=body.n_results or 5,
            doc_type=body.doc_type,
        )

        search_results = [
            SearchResult(
                id=doc["id"],
                content=doc["content"][:500],  # 내용 제한
                case_name=doc.get("metadata", {}).get("case_name", ""),
                case_number=doc.get("metadata", {}).get("case_number", ""),
                doc_type=doc.get("metadata", {}).get("doc_type", ""),
                similarity=round(doc.get("similarity", 0), 3),
            )
            for doc in results
        ]

        return SearchResponse(query=body.query, results=search_results)
    except EmbeddingModelNotFoundError as e:
        logger.error(f"임베딩 모델 없음: {e}")
        raise HTTPException(
            status_code=503,
            detail="임베딩 모델이 준비되지 않았습니다. 서버 관리자에게 문의하세요.",
        )
    except Exception as e:
        logger.error(f"검색 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="검색 중 오류가 발생했습니다")


@router.post("/analyze", deprecated=True)
@limiter.limit(AI_RATE_LIMIT)
async def analyze_case(request: Request, description: str) -> dict[str, Any]:
    """사용자 상황 분석 및 관련 판례 검색"""
    return {
        "analysis": "사용자 상황 분석 결과",
        "description": description,
        "related_precedents": [],
        "recommended_lawyers": [],
    }


# ============================================
# 판례 필터 검색 API (PostgreSQL 직접 쿼리)
# ============================================


class FilteredPrecedentItem(BaseModel):
    """필터 검색 결과 아이템"""
    id: str
    serial_number: str
    case_name: Optional[str] = None
    case_number: Optional[str] = None
    case_type: Optional[str] = None
    court_name: Optional[str] = None
    decision_date: Optional[str] = None
    summary: Optional[str] = None


class FilteredPrecedentListResponse(BaseModel):
    """필터 검색 응답"""
    keyword: str
    total: int
    offset: int
    limit: int
    precedents: List[FilteredPrecedentItem]


class CaseTypesResponse(BaseModel):
    """사건종류 목록 응답"""
    case_types: List[str]


@router.get("/precedents/filter", response_model=FilteredPrecedentListResponse)
async def filter_precedents(
    keyword: str = Query("", description="검색 키워드 (사건명, 판시사항, 사건번호, 판결요지 ILIKE)"),
    case_type: Optional[str] = Query(None, description="사건종류명 (예: 민사, 형사)"),
    date_from: Optional[datetime.date] = Query(None, description="선고일 시작 (YYYY-MM-DD)"),
    date_to: Optional[datetime.date] = Query(None, description="선고일 종료 (YYYY-MM-DD)"),
    sort: str = Query("relevance", description="정렬 기준 (relevance | latest)"),
    offset: int = Query(0, ge=0, description="페이지 오프셋"),
    limit: int = Query(20, ge=1, le=100, description="결과 수"),
) -> FilteredPrecedentListResponse:
    """
    판례 필터 검색 (PostgreSQL 직접 쿼리)

    사건종류, 기간, 키워드로 판례를 필터링합니다.
    사이드바 직접 진입 시 사용됩니다.
    """
    try:
        service = get_precedent_service()
        result = await service.search_by_filter(
            keyword=keyword,
            case_type=case_type,
            date_from=date_from,
            date_to=date_to,
            sort=sort,
            offset=offset,
            limit=limit,
        )
        return FilteredPrecedentListResponse(
            keyword=keyword,
            total=result["total"],
            offset=offset,
            limit=limit,
            precedents=[
                FilteredPrecedentItem(**p) for p in result["precedents"]
            ],
        )
    except Exception as e:
        logger.error("판례 필터 검색 실패: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="판례 필터 검색 중 오류가 발생했습니다")


@router.get("/precedents/case-types", response_model=CaseTypesResponse)
async def get_case_types() -> CaseTypesResponse:
    """
    사건종류명 목록 조회

    필터 드롭다운에 표시할 DISTINCT 사건종류 목록을 반환합니다.
    """
    try:
        service = get_precedent_service()
        case_types = await service.get_case_types()
        return CaseTypesResponse(case_types=case_types)
    except Exception as e:
        logger.error("사건종류 목록 조회 실패: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="사건종류 목록 조회 중 오류가 발생했습니다")


@router.get("/precedents", response_model=PrecedentListResponse)
async def search_precedents(
    keyword: str = Query(..., description="검색 키워드"),
    doc_type: Optional[str] = Query(None, description="문서 유형 필터 (precedent, constitutional)"),
    court: Optional[str] = Query(None, description="법원 필터"),
    limit: int = Query(20, ge=1, le=100, description="결과 수"),
) -> PrecedentListResponse:
    """
    판례 키워드 검색

    RAG 기반으로 키워드와 관련된 판례를 검색합니다.
    """
    try:
        results = await search_relevant_documents_async(
            query=keyword,
            n_results=limit,
            doc_type=doc_type,
        )

        precedents = []
        for doc in results:
            metadata = doc.get("metadata", {})
            # court_name 또는 court 필드 사용 (search_relevant_documents는 court_name으로 반환)
            court_value = metadata.get("court_name") or metadata.get("court", "")
            # 법원 필터 적용
            if court and court_value != court:
                continue
            precedents.append(
                PrecedentItem(
                    id=doc["id"],
                    case_name=metadata.get("case_name", ""),
                    case_number=metadata.get("case_number", ""),
                    doc_type=metadata.get("doc_type", ""),
                    court=court_value,
                    date=metadata.get("date"),
                    summary=doc["content"][:300] + "..." if len(doc["content"]) > 300 else doc["content"],
                    similarity=round(doc.get("similarity", 0), 3),
                )
            )

        return PrecedentListResponse(
            keyword=keyword,
            total=len(precedents),
            precedents=precedents,
        )
    except EmbeddingModelNotFoundError as e:
        logger.error(f"임베딩 모델 없음: {e}")
        raise HTTPException(
            status_code=503,
            detail="임베딩 모델이 준비되지 않았습니다. 서버 관리자에게 문의하세요.",
        )
    except Exception as e:
        logger.error(f"판례 검색 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="판례 검색 중 오류가 발생했습니다")


@router.get("/precedents/{precedent_id}", response_model=PrecedentDetailResponse)
async def get_precedent_detail(precedent_id: str) -> PrecedentDetailResponse:
    """
    판례 상세 정보 조회

    특정 판례의 전체 내용을 조회합니다.
    LanceDB에서 기본 정보 조회 후, PostgreSQL에서 상세 필드를 보강합니다.
    """
    try:
        store = get_vector_store()
        result = store.get_by_id(precedent_id)

        # RAG 검색 결과는 source_id를 id로 사용하므로, 청크 id로 못 찾으면 source_id로 폴백
        if not result and hasattr(store, "get_by_source_id"):
            source_result = store.get_by_source_id(precedent_id)
            if source_result and source_result.get("ids"):
                result = {
                    "id": source_result["ids"][0],
                    "content": source_result["documents"][0] if source_result.get("documents") else "",
                    "metadata": source_result["metadatas"][0] if source_result.get("metadatas") else {},
                }

        if not result:
            raise HTTPException(status_code=404, detail="판례를 찾을 수 없습니다")

        metadata = result.get("metadata", {})
        content = result.get("content", "")

        # LanceDB 필드명 → 표준 필드명 매핑
        case_name = metadata.get("title", "") or metadata.get("case_name", "")
        doc_type = _map_data_type(metadata.get("data_type", "")) or metadata.get("doc_type", "")
        court = metadata.get("source_name", "") or metadata.get("court", "")
        case_number = metadata.get("case_number", "")

        # 기본 응답 데이터 (LanceDB 기반)
        response_data: dict[str, Any] = {
            "id": precedent_id,
            "case_name": case_name,
            "case_number": case_number,
            "doc_type": doc_type,
            "court": court,
            "date": metadata.get("date"),
            "content": content,
            "summary": "",  # PostgreSQL에서 가져옴 (LanceDB content 사용 안함)
        }

        # 법령인 경우 doc_id, law_name 추가 (프론트엔드 LawDetailLawyer용)
        if doc_type == "law":
            law_source_id = metadata.get("source_id", "") or precedent_id
            response_data["doc_id"] = law_source_id
            response_data["law_name"] = case_name

        # PostgreSQL에서 상세 필드 조회 (source_id가 serial_number에 매핑)
        source_id = metadata.get("source_id", "")
        if source_id:
            details = await fetch_precedent_details([source_id])
            if source_id in details:
                precedent = details[source_id]
                response_data.update({
                    "ruling": precedent.get("ruling") or None,
                    "claim": precedent.get("claim") or None,
                    "reasoning": precedent.get("reasoning") or None,
                    "full_reason": precedent.get("full_reason") or None,
                    "full_text": precedent.get("full_text") or None,
                    "reference_provisions": precedent.get("reference_provisions") or None,
                    "reference_cases": precedent.get("reference_cases") or None,
                    "decision_date": precedent.get("decision_date") or None,
                    "court_name": precedent.get("court_name") or None,
                    # PostgreSQL 데이터로 기본 필드 보강
                    "case_name": precedent.get("case_name") or case_name,
                    "case_number": precedent.get("case_number") or case_number,
                    "court": precedent.get("court_name") or court,
                    # summary를 판시사항으로 대체 (PostgreSQL만 사용, LanceDB fallback 없음)
                    "summary": precedent.get("summary") or "",
                })

        return PrecedentDetailResponse(**response_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"판례 조회 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="판례 조회 중 오류가 발생했습니다")


# ============================================
# 법령 전문 조회 API (Law Full Text)
# ============================================


class LawArticleItem(BaseModel):
    """조문 단위 응답"""
    article_number: str
    article_title: Optional[str] = None
    article_content: str


class LawFullTextResponse(BaseModel):
    """법령 전문 응답 (조문 단위)"""
    law_id: str
    law_name: str
    law_type: Optional[str] = None
    ministry: Optional[str] = None
    ai_summary: Optional[str] = None
    supplementary: Optional[str] = None
    articles: List[LawArticleItem]
    total_articles: int
    enforcement_date: Optional[str] = None
    promulgation_date: Optional[str] = None
    promulgation_no: Optional[str] = None


@router.get("/laws/{law_id}/full-text", response_model=LawFullTextResponse)
async def get_law_full_text(law_id: str) -> LawFullTextResponse:
    """
    법령 전문 조회 API

    law_id로 법령의 조문 목록, 부칙, AI 요약을 반환합니다.
    law_articles 테이블에서 조문 단위로 조회합니다.
    """
    try:
        async with async_session_factory() as session:
            # 법령 기본 정보 조회
            law_result = await session.execute(
                select(LawDocument).where(LawDocument.law_id == law_id)
            )
            law = law_result.scalar_one_or_none()

            if not law:
                raise HTTPException(status_code=404, detail="법령을 찾을 수 없습니다")

            # 조문 목록 조회 (조문번호 순)
            articles_result = await session.execute(
                select(LawArticle)
                .where(LawArticle.law_id == law_id)
                .order_by(LawArticle.id)
            )
            articles = articles_result.scalars().all()

        return LawFullTextResponse(
            law_id=str(law.law_id),
            law_name=str(law.law_name),
            law_type=str(law.law_type) if law.law_type is not None else None,
            ministry=str(law.ministry) if law.ministry is not None else None,
            ai_summary=str(law.ai_summary) if law.ai_summary is not None else None,
            supplementary=str(law.supplementary) if law.supplementary is not None else None,
            articles=[
                LawArticleItem(
                    article_number=str(a.article_number),
                    article_title=str(a.article_title) if a.article_title is not None else None,
                    article_content=str(a.article_content),
                )
                for a in articles
            ],
            total_articles=len(articles),
            enforcement_date=(
                law.enforcement_date.isoformat()
                if law.enforcement_date
                else None
            ),
            promulgation_date=str(law.promulgation_date) if law.promulgation_date is not None else None,
            promulgation_no=str(law.promulgation_no) if law.promulgation_no is not None else None,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("법령 전문 조회 실패: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500, detail="법령 전문 조회 중 오류가 발생했습니다"
        )


# ============================================
# 법령 계층도 API (Statute Hierarchy)
# ============================================


class StatuteNodeResponse(BaseModel):
    """법령 노드 응답"""
    id: str
    name: str
    type: str
    abbreviation: Optional[str] = None
    citation_count: int = 0
    content: Optional[str] = None
    supplementary: Optional[str] = None
    ai_summary: Optional[str] = None


class StatuteSearchResponse(BaseModel):
    """법령 검색 응답"""
    query: str
    results: List[StatuteNodeResponse]


class StatuteHierarchyResponse(BaseModel):
    """법령 계층 응답"""
    root: Optional[StatuteNodeResponse] = None
    upper: List[StatuteNodeResponse]
    lower: List[StatuteNodeResponse]
    related: List[StatuteNodeResponse]


class StatuteChildrenResponse(BaseModel):
    """법령 하위 목록 응답"""
    statute_id: str
    children: List[StatuteNodeResponse]


class CitingCaseItem(BaseModel):
    """법령을 인용한 판례 항목"""
    serial_number: Optional[str] = None
    case_number: Optional[str] = None
    case_name: Optional[str] = None
    decision_date: Optional[str] = None
    court_name: Optional[str] = None


class CitingCasesResponse(BaseModel):
    """법령을 인용한 판례 목록 응답"""
    statute_id: str
    total: int
    cases: List[CitingCaseItem]


class GraphNodeResponse(BaseModel):
    """그래프 노드"""
    id: str
    name: str
    type: str
    abbreviation: Optional[str] = None
    citation_count: int = 0


class GraphLinkResponse(BaseModel):
    """그래프 링크"""
    source: str
    target: str
    relation: str


class StatuteGraphResponse(BaseModel):
    """법령 그래프 응답 (Force-directed용)"""
    nodes: List[GraphNodeResponse]
    links: List[GraphLinkResponse]


@router.get("/statutes/search", response_model=StatuteSearchResponse)
async def search_statutes(
    query: str = Query(..., description="검색어 (법령명, 약칭)"),
    limit: int = Query(10, ge=1, le=50, description="결과 수"),
) -> StatuteSearchResponse:
    """
    법령 검색 API

    법령명, 공식 약칭, 비공식 약칭으로 법령을 검색합니다.
    """
    try:
        pg = get_pg_graph_service()
        results = await pg.search_statutes(query, limit)
        return StatuteSearchResponse(
            query=query,
            results=[
                StatuteNodeResponse(
                    id=r["id"] or "",
                    name=r["name"] or "",
                    type=r["type"] or "",
                    abbreviation=r.get("abbreviation"),
                    citation_count=r.get("citation_count") or 0,
                )
                for r in results
            ],
        )
    except Exception as e:
        logger.error(f"법령 검색 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="법령 검색 중 오류가 발생했습니다")


@router.get("/statutes/hierarchy/{statute_id}", response_model=StatuteHierarchyResponse)
async def get_statute_hierarchy(statute_id: str) -> StatuteHierarchyResponse:
    """
    법령 계층 조회 API

    특정 법령의 상위/하위 계급 및 관련 법령을 조회합니다.
    """
    try:
        pg = get_pg_graph_service()
        detail = await pg.get_statute_hierarchy_detail(statute_id)
        if not detail:
            raise HTTPException(status_code=404, detail="법령을 찾을 수 없습니다")

        def _to_node(d: dict[str, Any]) -> StatuteNodeResponse:
            return StatuteNodeResponse(
                id=d["id"] or "",
                name=d["name"] or "",
                type=d["type"] or "",
                abbreviation=d.get("abbreviation"),
                citation_count=d.get("citation_count") or 0,
                content=d.get("content"),
                supplementary=d.get("supplementary"),
                ai_summary=d.get("ai_summary"),
            )

        return StatuteHierarchyResponse(
            root=_to_node(detail["root"]),
            upper=[_to_node(n) for n in detail["upper"]],
            lower=[_to_node(n) for n in detail["lower"]],
            related=[_to_node(n) for n in detail["related"]],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"법령 계층 조회 실패: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="법령 계층 조회 중 오류가 발생했습니다"
        )


@router.get("/statutes/{statute_id}/children", response_model=StatuteChildrenResponse)
async def get_statute_children(
    statute_id: str,
    limit: int = Query(20, ge=1, le=100, description="결과 수"),
) -> StatuteChildrenResponse:
    """
    법령 하위 법령 조회 API (지연 로딩용)

    특정 법령의 하위 법령 목록을 조회합니다.
    """
    try:
        pg = get_pg_graph_service()
        pg_children = await pg.get_statute_children(statute_id, limit)
        return StatuteChildrenResponse(
            statute_id=statute_id,
            children=[
                StatuteNodeResponse(
                    id=r["id"] or "",
                    name=r["name"] or "",
                    type=r["type"] or "",
                    abbreviation=r.get("abbreviation"),
                    citation_count=r.get("citation_count") or 0,
                )
                for r in pg_children
            ],
        )
    except Exception as e:
        logger.error(f"하위 법령 조회 실패: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="하위 법령 조회 중 오류가 발생했습니다"
        )


@router.get("/statutes/{statute_id}/citing-cases", response_model=CitingCasesResponse)
async def get_citing_cases(
    statute_id: str,
    limit: int = Query(10, ge=1, le=50, description="결과 수"),
) -> CitingCasesResponse:
    """
    법령을 인용한 판례 목록 조회

    특정 법령을 참조조문으로 인용한 판례들을 반환합니다.
    """
    try:
        pg = get_pg_graph_service()
        cases = await pg.get_cases_citing_statute(statute_id, limit)
        return CitingCasesResponse(
            statute_id=statute_id,
            total=len(cases),
            cases=[
                CitingCaseItem(
                    serial_number=c.get("serial_number"),
                    case_number=c.get("case_number"),
                    case_name=c.get("case_name"),
                    decision_date=c.get("decision_date"),
                    court_name=c.get("court_name"),
                )
                for c in cases
            ],
        )
    except Exception as e:
        logger.error(f"인용 판례 조회 실패: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="인용 판례 조회 중 오류가 발생했습니다"
        )


@router.get("/statutes/graph", response_model=StatuteGraphResponse)
async def get_statute_graph(
    center_id: Optional[str] = Query(None, description="중심 법령 ID"),
    depth: int = Query(2, ge=1, le=3, description="탐색 깊이"),
    limit: int = Query(100, ge=10, le=500, description="최대 노드 수"),
) -> StatuteGraphResponse:
    """
    법령 그래프 데이터 조회 (Force-directed 시각화용)

    중심 법령 기준으로 연결된 법령들의 그래프 데이터를 반환합니다.
    center_id가 없으면 인용수 상위 법령들로 시작합니다.
    """
    try:
        pg = get_pg_graph_service()
        graph_data = await pg.get_statute_graph(center_id, depth, limit)
        return StatuteGraphResponse(
            nodes=[
                GraphNodeResponse(
                    id=n["id"] or "",
                    name=n["name"] or "",
                    type=n["type"] or "",
                    abbreviation=n.get("abbreviation"),
                    citation_count=n.get("citation_count") or 0,
                )
                for n in graph_data.get("nodes", [])
            ],
            links=[
                GraphLinkResponse(
                    source=link["source"],
                    target=link["target"],
                    relation=link["relation"],
                )
                for link in graph_data.get("links", [])
            ],
        )
    except Exception as e:
        logger.error(f"법령 그래프 조회 실패: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="법령 그래프 조회 중 오류가 발생했습니다"
        )


@router.post("/precedents/{precedent_id}/ask", response_model=AIQuestionResponse)
@limiter.limit(AI_RATE_LIMIT)
async def ask_about_precedent(request: Request, precedent_id: str, body: AskQuestionRequest) -> AIQuestionResponse:
    """
    특정 판례에 대해 AI에게 질문

    선택한 판례의 컨텍스트를 기반으로 질문에 답변합니다.
    그래프 컨텍스트로 인용 법령 및 유사 판례 정보를 추가합니다.
    """
    try:
        from app.tools.llm import get_chat_model

        # 판례 내용 조회
        store = get_vector_store()
        precedent = store.get_by_id(precedent_id)

        if not precedent:
            raise HTTPException(status_code=404, detail="판례를 찾을 수 없습니다")

        metadata = precedent.get("metadata", {})
        content = precedent.get("content", "")
        case_number = metadata.get("case_number", "")

        # 그래프 컨텍스트 조회 (선택적)
        cited_statutes = None
        similar_cases = None
        graph_context_text = ""

        try:
            pg = get_pg_graph_service()
            case_context = await pg.enrich_case_context(case_number)
            if case_context.get("cited_statutes"):
                cited_statutes = [
                    s.get("name", "") for s in case_context["cited_statutes"][:5]
                ]
                graph_context_text += f"\n- 인용 법령: {', '.join(cited_statutes)}"

            if case_context.get("similar_cases"):
                similar_cases = [
                    s.get("case_number", "")
                    for s in case_context["similar_cases"][:3]
                ]
                graph_context_text += f"\n- 유사 판례: {', '.join(similar_cases)}"
        except Exception as e:
            logger.debug(f"그래프 컨텍스트 조회 실패: {e}")

        # AI 응답 생성 (LLM 추상화 레이어 사용)
        system_prompt = """당신은 한국 법률 전문 AI 어시스턴트입니다.
제공된 판례 내용을 바탕으로 사용자의 질문에 정확하고 도움이 되는 답변을 제공합니다.
답변 시 판례의 핵심 내용을 인용하고, 쉬운 언어로 설명해주세요."""

        user_prompt = f"""다음 판례에 대해 질문이 있습니다.

[판례 정보]
- 사건명: {metadata.get('case_name', '')}
- 사건번호: {case_number}
- 법원: {metadata.get('court', '')}
- 판결일: {metadata.get('date', '')}{graph_context_text}

[판례 내용]
{content[:3000]}

[질문]
{body.question}

위 판례 내용을 바탕으로 질문에 답변해주세요."""

        llm = get_chat_model(temperature=0.7)
        ai_response = await llm.ainvoke([
            ("system", system_prompt),
            ("user", user_prompt),
        ])

        answer = str(ai_response.content) if ai_response.content else None
        if not answer:
            raise HTTPException(status_code=503, detail="AI 응답이 없습니다")

        return AIQuestionResponse(
            answer=answer or "",
            sources=[
                ChatSource(
                    case_name=metadata.get("case_name"),
                    case_number=case_number or None,
                    doc_type=metadata.get("doc_type", ""),
                    similarity=1.0,
                    summary=(
                        content[:300] + "..." if len(content) > 300 else content
                    ),
                    content=content[:500] if content else None,
                    cited_statutes=cited_statutes,
                    similar_cases=similar_cases,
                )
            ],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"질문 처리 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="질문 처리 중 오류가 발생했습니다")


# ============================================
# 법령 필터 검색 API (PostgreSQL 직접 쿼리)
# ============================================


class FilteredLawItem(BaseModel):
    """법령 필터 검색 결과 아이템"""
    id: str
    law_name: str
    law_type: Optional[str] = None
    ministry: Optional[str] = None
    enforcement_date: Optional[str] = None
    promulgation_date: Optional[str] = None
    abbreviation: Optional[str] = None
    ai_summary: Optional[str] = None


class FilteredLawListResponse(BaseModel):
    """법령 필터 검색 응답"""
    keyword: str
    total: int
    offset: int
    limit: int
    laws: List[FilteredLawItem]


class LawFilterOptionsResponse(BaseModel):
    """법령 필터 옵션 응답"""
    law_types: List[str]
    ministries: List[str]


@router.get("/laws/filter", response_model=FilteredLawListResponse)
async def filter_laws(
    keyword: str = Query("", description="검색 키워드 (법령명, 조문 BM25+ILIKE)"),
    law_type: Optional[str] = Query(None, description="법령 유형 (예: 법률, 시행령)"),
    ministry: Optional[str] = Query(None, description="소관부처 (예: 법무부)"),
    promulgation_from: Optional[str] = Query(None, description="공포일자 시작 (YYYYMMDD)"),
    promulgation_to: Optional[str] = Query(None, description="공포일자 종료 (YYYYMMDD)"),
    enforcement_from: Optional[datetime.date] = Query(None, description="시행일자 시작 (YYYY-MM-DD)"),
    enforcement_to: Optional[datetime.date] = Query(None, description="시행일자 종료 (YYYY-MM-DD)"),
    sort: str = Query("relevance", description="정렬 기준 (relevance | latest)"),
    offset: int = Query(0, ge=0, description="페이지 오프셋"),
    limit: int = Query(20, ge=1, le=100, description="결과 수"),
) -> FilteredLawListResponse:
    """
    법령 필터 검색 (PostgreSQL 직접 쿼리, BM25+ILIKE 하이브리드)

    법령유형, 소관부처, 공포일자, 시행일자, 키워드로 법령을 필터링합니다.
    """
    try:
        service = get_law_filter_service()
        result = await service.search_by_filter(
            keyword=keyword,
            law_type=law_type,
            ministry=ministry,
            promulgation_from=promulgation_from,
            promulgation_to=promulgation_to,
            enforcement_from=enforcement_from,
            enforcement_to=enforcement_to,
            sort=sort,
            offset=offset,
            limit=limit,
        )
        return FilteredLawListResponse(
            keyword=keyword,
            total=result["total"],
            offset=offset,
            limit=limit,
            laws=[FilteredLawItem(**law) for law in result["laws"]],
        )
    except Exception as e:
        logger.error("법령 필터 검색 실패: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="법령 필터 검색 중 오류가 발생했습니다")


@router.get("/laws/filter-options", response_model=LawFilterOptionsResponse)
async def get_law_filter_options() -> LawFilterOptionsResponse:
    """
    법령 필터 옵션 조회

    필터 드롭다운에 표시할 법령유형, 소관부처 DISTINCT 목록을 반환합니다.
    """
    try:
        service = get_law_filter_service()
        options = await service.get_filter_options()
        return LawFilterOptionsResponse(**options)
    except Exception as e:
        logger.error("법령 필터 옵션 조회 실패: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="법령 필터 옵션 조회 중 오류가 발생했습니다")
