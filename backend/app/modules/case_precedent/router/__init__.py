"""
판례 추천 모듈 - 업무 사례 기반 관련 판례 제공
RAG 기반으로 사용자 상황에 맞는 판례 검색 및 변호사 추천
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List

from app.common.chat_service import generate_chat_response, search_relevant_documents

router = APIRouter()


# Request/Response 스키마
class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = None


class ChatSource(BaseModel):
    case_name: str
    case_number: str
    doc_type: str
    similarity: float


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


class AskQuestionRequest(BaseModel):
    question: str


class AIQuestionResponse(BaseModel):
    answer: str
    sources: List[ChatSource]


# API 엔드포인트
@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    RAG 기반 법률 챗봇

    사용자 메시지를 받아 관련 판례를 검색하고 AI 응답 생성
    """
    try:
        # 대화 기록 변환
        history = None
        if request.history:
            history = [{"role": msg.role, "content": msg.content} for msg in request.history]

        # 응답 생성
        result = generate_chat_response(
            user_message=request.message,
            chat_history=history,
        )

        return ChatResponse(
            response=result["response"],
            sources=[ChatSource(**s) for s in result["sources"]],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"챗봇 응답 생성 실패: {str(e)}")


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    법률 문서 유사도 검색

    쿼리와 유사한 판례/법률 문서 검색
    """
    try:
        results = search_relevant_documents(
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"검색 실패: {str(e)}")


@router.post("/analyze")
async def analyze_case(description: str):
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
):
    """
    판례 키워드 검색

    RAG 기반으로 키워드와 관련된 판례를 검색합니다.
    """
    try:
        results = search_relevant_documents(
            query=keyword,
            n_results=limit,
            doc_type=doc_type,
        )

        precedents = []
        for doc in results:
            metadata = doc.get("metadata", {})
            # 법원 필터 적용
            if court and metadata.get("court") != court:
                continue
            precedents.append(
                PrecedentItem(
                    id=doc["id"],
                    case_name=metadata.get("case_name", ""),
                    case_number=metadata.get("case_number", ""),
                    doc_type=metadata.get("doc_type", ""),
                    court=metadata.get("court"),
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"판례 검색 실패: {str(e)}")


@router.get("/precedents/{precedent_id}", response_model=PrecedentDetailResponse)
async def get_precedent_detail(precedent_id: str):
    """
    판례 상세 정보 조회

    특정 판례의 전체 내용을 조회합니다.
    """
    try:
        from app.common.vectorstore import VectorStore

        store = VectorStore()
        result = store.get_by_id(precedent_id)

        if not result:
            raise HTTPException(status_code=404, detail="판례를 찾을 수 없습니다")

        metadata = result.get("metadata", {})

        return PrecedentDetailResponse(
            id=precedent_id,
            case_name=metadata.get("case_name", ""),
            case_number=metadata.get("case_number", ""),
            doc_type=metadata.get("doc_type", ""),
            court=metadata.get("court"),
            date=metadata.get("date"),
            content=result.get("content", ""),
            summary=result.get("content", "")[:500] + "..." if len(result.get("content", "")) > 500 else result.get("content", ""),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"판례 조회 실패: {str(e)}")


@router.post("/precedents/{precedent_id}/ask", response_model=AIQuestionResponse)
async def ask_about_precedent(precedent_id: str, request: AskQuestionRequest):
    """
    특정 판례에 대해 AI에게 질문

    선택한 판례의 컨텍스트를 기반으로 질문에 답변합니다.
    """
    try:
        from app.common.vectorstore import VectorStore
        from openai import OpenAI
        from app.core.config import settings

        # 판례 내용 조회
        store = VectorStore()
        precedent = store.get_by_id(precedent_id)

        if not precedent:
            raise HTTPException(status_code=404, detail="판례를 찾을 수 없습니다")

        metadata = precedent.get("metadata", {})
        content = precedent.get("content", "")

        # AI 응답 생성
        client = OpenAI(api_key=settings.OPENAI_API_KEY)

        system_prompt = """당신은 한국 법률 전문 AI 어시스턴트입니다.
제공된 판례 내용을 바탕으로 사용자의 질문에 정확하고 도움이 되는 답변을 제공합니다.
답변 시 판례의 핵심 내용을 인용하고, 쉬운 언어로 설명해주세요."""

        user_prompt = f"""다음 판례에 대해 질문이 있습니다.

[판례 정보]
- 사건명: {metadata.get('case_name', '')}
- 사건번호: {metadata.get('case_number', '')}
- 법원: {metadata.get('court', '')}
- 판결일: {metadata.get('date', '')}

[판례 내용]
{content[:3000]}

[질문]
{request.question}

위 판례 내용을 바탕으로 질문에 답변해주세요."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=1000,
        )

        answer = response.choices[0].message.content

        return AIQuestionResponse(
            answer=answer or "",
            sources=[
                ChatSource(
                    case_name=metadata.get("case_name", ""),
                    case_number=metadata.get("case_number", ""),
                    doc_type=metadata.get("doc_type", ""),
                    similarity=1.0,
                )
            ],
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"질문 처리 실패: {str(e)}")
