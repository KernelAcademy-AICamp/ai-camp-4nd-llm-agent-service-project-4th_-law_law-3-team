"""변호사시험 기록형 연습 - Pydantic 스키마"""

from pydantic import BaseModel, Field


class ExamFile(BaseModel):
    """시험 문제 파일 메타데이터"""

    category: str = Field(description="카테고리 (CIVIL, CRIMINAL, PUBLIC)")
    year: int = Field(description="시험 연도 (2012~)")
    session: int = Field(description="회차 (1~15)")
    filename: str = Field(description="원본 파일명")
    title: str = Field(description="시험 제목")


class ExamListResponse(BaseModel):
    """시험 문제 목록 응답"""

    exams: list[ExamFile]
    total: int


class ExamContentResponse(BaseModel):
    """시험 문제 전문 응답"""

    category: str
    year: int
    session: int
    title: str
    content: str = Field(description="마크다운 형식 문제 전문")
    total_chars: int


class ReferenceSearchRequest(BaseModel):
    """참조 검색 요청"""

    query: str = Field(min_length=1, max_length=500)
    doc_type: str | None = Field(default=None, description="precedent | law")
    n_results: int = Field(default=5, ge=1, le=20)


class ReferenceSearchResult(BaseModel):
    """참조 검색 결과 항목"""

    id: str
    doc_type: str
    title: str
    case_number: str | None = None
    summary: str
    similarity: float


class ReferenceSearchResponse(BaseModel):
    """참조 검색 응답"""

    query: str
    results: list[ReferenceSearchResult]


class AnswerFeedbackRequest(BaseModel):
    """답안 피드백 요청"""

    answer_text: str = Field(min_length=1, max_length=50000)


class AnswerFeedbackResponse(BaseModel):
    """AI 피드백 응답"""

    feedback: str = Field(description="마크다운 형식 피드백")
