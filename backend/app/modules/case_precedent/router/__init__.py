"""
판례 추천 모듈 - 업무 사례 기반 관련 판례 제공
RAG 기반으로 사용자 상황에 맞는 판례 검색 및 변호사 추천
"""
import logging
from typing import Any, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.core.config import settings
from app.core.errors import EmbeddingModelNotFoundError
from app.services.rag import search_relevant_documents_async
from app.services.service_function.precedent_service import fetch_precedent_details
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


# Request/Response 스키마
class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = None


class ChatSource(BaseModel):
    """
    챗봇 응답의 출처 정보

    판례와 법령 모두 지원 (필드가 각각 다름)
    """

    # 판례 필드 (법령일 때는 없음)
    case_name: Optional[str] = None
    case_number: Optional[str] = None

    # 법령 필드 (판례일 때는 없음)
    law_name: Optional[str] = None
    law_type: Optional[str] = None

    # 공통 필드
    doc_type: str
    similarity: float
    summary: Optional[str] = None
    content: Optional[str] = None

    # 그래프 보강 정보 (optional)
    cited_statutes: Optional[List[str]] = None
    similar_cases: Optional[List[str]] = None


class ChatResponse(BaseModel):
    response: str
    sources: List[ChatSource]


class SearchRequest(BaseModel):
    query: str
    n_results: Optional[int] = 5
    doc_type: Optional[str] = None


class SearchResult(BaseModel):
    id: str
    content: str
    case_name: str
    case_number: str
    doc_type: str
    similarity: float


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]


# 판례 검색 전용 스키마
class PrecedentItem(BaseModel):
    id: str
    case_name: str
    case_number: str
    doc_type: str
    court: Optional[str] = None
    date: Optional[str] = None
    summary: str
    similarity: float


class PrecedentListResponse(BaseModel):
    keyword: str
    total: int
    precedents: List[PrecedentItem]


class PrecedentDetailResponse(BaseModel):
    id: str
    case_name: str
    case_number: str
    doc_type: str
    court: Optional[str] = None
    date: Optional[str] = None
    content: str
    summary: str
    # 판례 상세 필드 (PostgreSQL 조회)
    ruling: Optional[str] = None  # 주문
    claim: Optional[str] = None  # 청구취지
    reasoning: Optional[str] = None  # 판결요지
    full_reason: Optional[str] = None  # 이유
    full_text: Optional[str] = None  # 전문
    reference_provisions: Optional[str] = None  # 참조조문
    reference_cases: Optional[str] = None  # 참조판례
    court_name: Optional[str] = None  # 법원명
    decision_date: Optional[str] = None  # 선고일


class AskQuestionRequest(BaseModel):
    question: str


class AIQuestionResponse(BaseModel):
    answer: str
    sources: List[ChatSource]


# API 엔드포인트
@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    RAG 기반 법률 챗봇

    사용자 메시지를 받아 관련 판례를 검색하고 AI 응답 생성.
    통합 채팅 API(/api/chat)와 별개로, 판례 모듈 전용 간이 챗봇.
    """
    try:
        from app.tools.llm import get_chat_model

        # 1. RAG 검색 (판례 + 법령)
        search_results = await search_relevant_documents_async(
            query=request.message,
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
                "content": f"[참고 자료]\n{context_text}\n\n[질문]\n{request.message}",
            },
        ]
        if request.history:
            # 대화 기록을 system과 user 사이에 삽입
            history_messages = [{"role": msg.role, "content": msg.content} for msg in request.history]
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
async def search(request: SearchRequest) -> SearchResponse:
    """
    법률 문서 유사도 검색

    쿼리와 유사한 판례/법률 문서 검색
    """
    try:
        results = await search_relevant_documents_async(
            query=request.query,
            n_results=request.n_results or 5,
            doc_type=request.doc_type,
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

        return SearchResponse(query=request.query, results=search_results)
    except EmbeddingModelNotFoundError as e:
        logger.error(f"임베딩 모델 없음: {e}")
        raise HTTPException(
            status_code=503,
            detail="임베딩 모델이 준비되지 않았습니다. 서버 관리자에게 문의하세요.",
        )
    except Exception as e:
        logger.error(f"검색 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="검색 중 오류가 발생했습니다")


@router.post("/analyze")
async def analyze_case(description: str) -> dict[str, Any]:
    """사용자 상황 분석 및 관련 판례 검색"""
    return {
        "analysis": "사용자 상황 분석 결과",
        "description": description,
        "related_precedents": [],
        "recommended_lawyers": [],
    }


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
        response_data = {
            "id": precedent_id,
            "case_name": case_name,
            "case_number": case_number,
            "doc_type": doc_type,
            "court": court,
            "date": metadata.get("date"),
            "content": content,
            "summary": "",  # PostgreSQL에서 가져옴 (LanceDB content 사용 안함)
        }

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


def _get_graph_service():
    """GraphService lazy 로드 (Neo4j 연결 실패 시에도 동작)"""
    try:
        from app.tools.graph import get_graph_service

        service = get_graph_service()
        if service.is_connected:
            return service
    except Exception as e:
        logger.debug(f"GraphService 로드 실패: {e}")
    return None


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
):
    """
    법령 검색 API

    법령명, 공식 약칭, 비공식 약칭으로 법령을 검색합니다.
    """
    if settings.USE_PG_GRAPH:
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
            logger.error(f"법령 검색 실패 (PG): {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="법령 검색 중 오류가 발생했습니다")

    graph = _get_graph_service()
    if not graph:
        raise HTTPException(status_code=503, detail="그래프 DB가 연결되지 않았습니다")

    try:
        with graph.session() as session:
            # 통합 검색: 정식명, 공식 약칭, 비공식 약칭 (Alias)
            # UNION으로 Statute 직접 검색 + Alias를 통한 검색 결합
            cypher = """
            CALL {
                // 1. 법령명/공식약칭 검색
                CALL db.index.fulltext.queryNodes("ft_statute_search", $search_term)
                YIELD node, score
                WHERE node:Statute
                RETURN node as s, score, null as alias_name

                UNION

                // 2. 비공식 약칭(Alias) 검색
                CALL db.index.fulltext.queryNodes("ft_alias_search", $search_term)
                YIELD node, score
                MATCH (node)-[:ALIAS_OF]->(s:Statute)
                RETURN s, score, node.name as alias_name
            }
            WITH s, max(score) as best_score, collect(alias_name)[0] as matched_alias
            RETURN s.id as id, s.name as name, s.type as type,
                   COALESCE(matched_alias, s.abbreviation) as abbreviation,
                   COALESCE(s.citation_count, 0) as citation_count
            ORDER BY best_score DESC
            LIMIT $limit
            """
            result = session.run(cypher, search_term=query, limit=limit)
            search_results: list[StatuteNodeResponse] = [
                StatuteNodeResponse(
                    id=r["id"] or "",
                    name=r["name"] or "",
                    type=r["type"] or "",
                    abbreviation=r["abbreviation"],
                    citation_count=r["citation_count"] or 0,
                )
                for r in result
            ]

            # 결과가 없으면 LIKE 검색 fallback (Alias 포함)
            if not search_results:
                fallback_cypher = """
                CALL {
                    MATCH (s:Statute)
                    WHERE s.name CONTAINS $search_term OR s.abbreviation CONTAINS $search_term
                    RETURN s, s.abbreviation as matched_alias

                    UNION

                    MATCH (a:Alias)-[:ALIAS_OF]->(s:Statute)
                    WHERE a.name CONTAINS $search_term
                    RETURN s, a.name as matched_alias
                }
                WITH s, collect(matched_alias)[0] as abbreviation
                RETURN DISTINCT s.id as id, s.name as name, s.type as type,
                       abbreviation,
                       COALESCE(s.citation_count, 0) as citation_count
                ORDER BY citation_count DESC
                LIMIT $limit
                """
                result = session.run(fallback_cypher, search_term=query, limit=limit)
                search_results = [
                    StatuteNodeResponse(
                        id=r["id"] or "",
                        name=r["name"] or "",
                        type=r["type"] or "",
                        abbreviation=r["abbreviation"],
                        citation_count=r["citation_count"] or 0,
                    )
                    for r in result
                ]

            return StatuteSearchResponse(query=query, results=search_results)
    except Exception as e:
        logger.error(f"법령 검색 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="법령 검색 중 오류가 발생했습니다")


@router.get("/statutes/hierarchy/{statute_id}", response_model=StatuteHierarchyResponse)
async def get_statute_hierarchy(statute_id: str):
    """
    법령 계층 조회 API

    특정 법령의 상위/하위 계급 및 관련 법령을 조회합니다.
    """
    if settings.USE_PG_GRAPH:
        try:
            pg = get_pg_graph_service()
            detail = await pg.get_statute_hierarchy_detail(statute_id)
            if not detail:
                raise HTTPException(status_code=404, detail="법령을 찾을 수 없습니다")

            def _to_node(d: dict) -> StatuteNodeResponse:
                return StatuteNodeResponse(
                    id=d["id"] or "",
                    name=d["name"] or "",
                    type=d["type"] or "",
                    abbreviation=d.get("abbreviation"),
                    citation_count=d.get("citation_count") or 0,
                    content=d.get("content"),
                    supplementary=d.get("supplementary"),
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
            logger.error(f"법령 계층 조회 실패 (PG): {e}", exc_info=True)
            raise HTTPException(
                status_code=500, detail="법령 계층 조회 중 오류가 발생했습니다"
            )

    graph = _get_graph_service()
    if not graph:
        raise HTTPException(status_code=503, detail="그래프 DB가 연결되지 않았습니다")

    try:
        with graph.session() as session:
            # 법령 정보 + 상위/하위 계급 + 관련 법령 조회
            cypher = """
            MATCH (s:Statute {id: $statute_id})
            OPTIONAL MATCH (s)-[:HIERARCHY_OF]->(upper:Statute)
            OPTIONAL MATCH (lower:Statute)-[:HIERARCHY_OF]->(s)
            OPTIONAL MATCH (s)-[:RELATED_TO]-(related:Statute)
                RETURN s.id as id, s.name as name, s.type as type,
                       s.abbreviation as abbreviation,
                       s.content as content,
                       s.supplementary as supplementary,
                       COALESCE(s.citation_count, 0) as citation_count,
                       collect(DISTINCT {
                           id: upper.id, name: upper.name, type: upper.type,
                           abbreviation: upper.abbreviation,
                           content: upper.content,
                           supplementary: upper.supplementary,
                           citation_count: COALESCE(upper.citation_count, 0)
                       }) as upper_list,
                       collect(DISTINCT {
                           id: lower.id, name: lower.name, type: lower.type,
                           abbreviation: lower.abbreviation,
                           content: lower.content,
                           supplementary: lower.supplementary,
                           citation_count: COALESCE(lower.citation_count, 0)
                       }) as lower_list,
                       collect(DISTINCT {
                           id: related.id, name: related.name, type: related.type,
                           abbreviation: related.abbreviation,
                           content: related.content,
                           supplementary: related.supplementary,
                           citation_count: COALESCE(related.citation_count, 0)
                       }) as related_list
                """
            result = session.run(cypher, statute_id=statute_id)
            record = result.single()

            if not record or not record["id"]:
                raise HTTPException(status_code=404, detail="법령을 찾을 수 없습니다")

            root = StatuteNodeResponse(
                id=record["id"],
                name=record["name"] or "",
                type=record["type"] or "",
                abbreviation=record["abbreviation"],
                citation_count=record["citation_count"] or 0,
                content=record["content"],
                supplementary=record["supplementary"],
            )

            def filter_valid(items):
                return [
                    StatuteNodeResponse(
                        id=item["id"],
                        name=item["name"] or "",
                        type=item["type"] or "",
                        abbreviation=item["abbreviation"],
                        content=item.get("content"),
                        supplementary=item.get("supplementary"),
                        citation_count=item["citation_count"] or 0,
                    )
                    for item in items
                    if item.get("id") is not None
                ]

            return StatuteHierarchyResponse(
                root=root,
                upper=filter_valid(record["upper_list"]),
                lower=filter_valid(record["lower_list"]),
                related=filter_valid(record["related_list"]),
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"법령 계층 조회 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="법령 계층 조회 중 오류가 발생했습니다")


@router.get("/statutes/{statute_id}/children", response_model=StatuteChildrenResponse)
async def get_statute_children(
    statute_id: str,
    limit: int = Query(20, ge=1, le=100, description="결과 수"),
):
    """
    법령 하위 법령 조회 API (지연 로딩용)

    특정 법령의 하위 법령 목록을 조회합니다.
    """
    if settings.USE_PG_GRAPH:
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
            logger.error(f"하위 법령 조회 실패 (PG): {e}", exc_info=True)
            raise HTTPException(
                status_code=500, detail="하위 법령 조회 중 오류가 발생했습니다"
            )

    graph = _get_graph_service()
    if not graph:
        raise HTTPException(status_code=503, detail="그래프 DB가 연결되지 않았습니다")

    try:
        with graph.session() as session:
            cypher = """
            MATCH (lower:Statute)-[:HIERARCHY_OF]->(s:Statute {id: $statute_id})
            RETURN lower.id as id, lower.name as name, lower.type as type,
                   lower.abbreviation as abbreviation,
                   COALESCE(lower.citation_count, 0) as citation_count
            ORDER BY lower.citation_count DESC
            LIMIT $limit
            """
            result = session.run(cypher, statute_id=statute_id, limit=limit)
            child_nodes = [
                StatuteNodeResponse(
                    id=r["id"] or "",
                    name=r["name"] or "",
                    type=r["type"] or "",
                    abbreviation=r["abbreviation"],
                    citation_count=r["citation_count"] or 0,
                )
                for r in result
            ]

            return StatuteChildrenResponse(
                statute_id=statute_id,
                children=child_nodes,
            )
    except Exception as e:
        logger.error(f"하위 법령 조회 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="하위 법령 조회 중 오류가 발생했습니다")


@router.get("/statutes/graph", response_model=StatuteGraphResponse)
async def get_statute_graph(
    center_id: Optional[str] = Query(None, description="중심 법령 ID"),
    depth: int = Query(2, ge=1, le=3, description="탐색 깊이"),
    limit: int = Query(100, ge=10, le=500, description="최대 노드 수"),
):
    """
    법령 그래프 데이터 조회 (Force-directed 시각화용)

    중심 법령 기준으로 연결된 법령들의 그래프 데이터를 반환합니다.
    center_id가 없으면 인용수 상위 법령들로 시작합니다.
    """
    if settings.USE_PG_GRAPH:
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
            logger.error(f"법령 그래프 조회 실패 (PG): {e}", exc_info=True)
            raise HTTPException(
                status_code=500, detail="법령 그래프 조회 중 오류가 발생했습니다"
            )

    graph = _get_graph_service()
    if not graph:
        raise HTTPException(status_code=503, detail="그래프 DB가 연결되지 않았습니다")

    try:
        with graph.session() as session:
            if center_id:
                # 특정 법령 중심 그래프
                cypher = """
                MATCH (center:Statute {id: $center_id})
                CALL {
                    WITH center
                    MATCH path = (center)-[:HIERARCHY_OF|RELATED_TO*1..2]-(connected:Statute)
                    RETURN DISTINCT connected
                    LIMIT $limit
                }
                WITH center, collect(DISTINCT connected) as connected_nodes
                UNWIND ([center] + connected_nodes) as node
                WITH collect(DISTINCT node) as all_nodes
                UNWIND all_nodes as n1
                UNWIND all_nodes as n2
                OPTIONAL MATCH (n1)-[r:HIERARCHY_OF]->(n2)
                OPTIONAL MATCH (n1)-[r2:RELATED_TO]-(n2) WHERE id(n1) < id(n2)
                WITH all_nodes,
                     collect(DISTINCT CASE WHEN r IS NOT NULL THEN {source: n1.id, target: n2.id, relation: 'HIERARCHY_OF'} END) as hier_links,
                     collect(DISTINCT CASE WHEN r2 IS NOT NULL THEN {source: n1.id, target: n2.id, relation: 'RELATED_TO'} END) as rel_links
                RETURN all_nodes,
                       [link in hier_links WHERE link IS NOT NULL] + [link in rel_links WHERE link IS NOT NULL] as links
                """
                result = session.run(cypher, center_id=center_id, limit=limit)
            else:
                # 인용수 상위 법령들로 시작하는 그래프
                cypher = """
                MATCH (s:Statute)
                WHERE s.citation_count > 0
                WITH s ORDER BY s.citation_count DESC LIMIT 50
                WITH collect(s) as top_statutes
                UNWIND top_statutes as center
                CALL {
                    WITH center
                    MATCH (center)-[:HIERARCHY_OF|RELATED_TO]-(connected:Statute)
                    RETURN connected
                    LIMIT 3
                }
                WITH top_statutes, collect(DISTINCT connected) as connected_nodes
                WITH top_statutes + connected_nodes as all_list
                UNWIND all_list as node
                WITH collect(DISTINCT node) as all_nodes
                UNWIND all_nodes as n1
                UNWIND all_nodes as n2
                OPTIONAL MATCH (n1)-[r:HIERARCHY_OF]->(n2)
                OPTIONAL MATCH (n1)-[r2:RELATED_TO]-(n2) WHERE id(n1) < id(n2)
                WITH all_nodes,
                     collect(DISTINCT CASE WHEN r IS NOT NULL THEN {source: n1.id, target: n2.id, relation: 'HIERARCHY_OF'} END) as hier_links,
                     collect(DISTINCT CASE WHEN r2 IS NOT NULL THEN {source: n1.id, target: n2.id, relation: 'RELATED_TO'} END) as rel_links
                RETURN all_nodes,
                       [link in hier_links WHERE link IS NOT NULL] + [link in rel_links WHERE link IS NOT NULL] as links
                """
                result = session.run(cypher)

            record = result.single()
            if not record:
                return StatuteGraphResponse(nodes=[], links=[])

            nodes = [
                GraphNodeResponse(
                    id=n["id"] or "",
                    name=n["name"] or "",
                    type=n["type"] or "",
                    abbreviation=n.get("abbreviation"),
                    citation_count=n.get("citation_count") or 0,
                )
                for n in record["all_nodes"]
                if n and n.get("id")
            ]

            links = [
                GraphLinkResponse(
                    source=link["source"],
                    target=link["target"],
                    relation=link["relation"],
                )
                for link in record["links"]
                if link and link.get("source") and link.get("target")
            ]

            return StatuteGraphResponse(nodes=nodes, links=links)
    except Exception as e:
        logger.error(f"법령 그래프 조회 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="법령 그래프 조회 중 오류가 발생했습니다")


@router.post("/precedents/{precedent_id}/ask", response_model=AIQuestionResponse)
async def ask_about_precedent(precedent_id: str, request: AskQuestionRequest) -> AIQuestionResponse:
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

        if graph_service := _get_graph_service():
            try:
                case_context = graph_service.enrich_case_context(case_number)
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
{request.question}

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
