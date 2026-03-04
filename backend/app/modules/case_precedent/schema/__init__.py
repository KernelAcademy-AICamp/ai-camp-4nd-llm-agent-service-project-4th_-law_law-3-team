"""
판례 추천 모듈 스키마 정의

Request/Response Pydantic 모델
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str = Field(max_length=5000)
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
    query: str = Field(max_length=500)
    n_results: Optional[int] = Field(default=5, ge=1, le=50)
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
    question: str = Field(max_length=2000)


class AIQuestionResponse(BaseModel):
    answer: str
    sources: List[ChatSource]
