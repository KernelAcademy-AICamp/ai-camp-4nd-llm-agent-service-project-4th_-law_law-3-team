"""
소액소송 모듈 스키마 정의

Pydantic 모델: 요청/응답 + Phase 3 인터뷰/증거/가이드
"""

from typing import Any, List, Optional  # noqa: F401

from pydantic import BaseModel

# ── 증거 체크리스트 ──

class EvidenceItem(BaseModel):
    id: str
    label: str
    required: bool
    description: str


class EvidenceChecklistResponse(BaseModel):
    dispute_type: str
    description: str
    items: List[EvidenceItem]


# ── 사건 정보 ──

class CaseInfo(BaseModel):
    dispute_type: str
    plaintiff_name: str
    plaintiff_address: str
    plaintiff_phone: Optional[str] = None
    defendant_name: str
    defendant_address: Optional[str] = None
    defendant_phone: Optional[str] = None
    amount: int
    description: str
    incident_date: Optional[str] = None


# ── 서류 생성 ──

class DocumentGenerateRequest(BaseModel):
    document_type: str  # "demand_letter" | "payment_order" | "complaint"
    case_info: CaseInfo


class DocumentResponse(BaseModel):
    document_type: str
    title: str
    content: str
    template_sections: dict[str, Any]
    pdf_url: Optional[str] = None
    docx_url: Optional[str] = None


class DocumentRegenerateRequest(BaseModel):
    """편집된 텍스트로 PDF/DOCX 재생성 요청 (Phase 2B)"""
    document_type: str
    title: str
    content: str
    formats: List[str] = ["pdf", "docx"]


class DocumentRegenerateResponse(BaseModel):
    """재생성된 파일 URL 응답 (Phase 2B)"""
    pdf_url: Optional[str] = None
    docx_url: Optional[str] = None


# ── 관련 판례 ──

class RelatedCaseItem(BaseModel):
    id: str
    case_name: str
    case_number: str
    summary: str
    similarity: float
    relevance: str
    ruling: Optional[str] = None
    reasoning: Optional[str] = None


class RelatedCasesResponse(BaseModel):
    dispute_type: str
    cases: List[RelatedCaseItem]


# ── 증거 업로드 ──

class EvidenceUploadFile(BaseModel):
    file_id: str
    original_name: str
    file_type: str
    file_size: int


class EvidenceUploadResponse(BaseModel):
    uploaded_files: List[EvidenceUploadFile]


# ── Phase 3: 인터뷰 ──

class InterviewStartRequest(BaseModel):
    case_type: str


class InterviewQuestion(BaseModel):
    question_index: int
    question: str
    total_questions: int
    field_hint: Optional[str] = None


class InterviewAnswerRequest(BaseModel):
    answer: str


class InterviewResponse(BaseModel):
    session_id: str
    is_complete: bool
    current_question: Optional[InterviewQuestion] = None
    filled_case_info: Optional[CaseInfo] = None


# ── Phase 3: 증거 타임라인 ──

class EvidenceTimelineItem(BaseModel):
    file_id: str
    original_name: str
    date: Optional[str] = None
    summary: Optional[str] = None
    category: Optional[str] = None


class EvidenceOrganizeResponse(BaseModel):
    session_id: str
    timeline: List[EvidenceTimelineItem]
    pdf_url: Optional[str] = None


# ── Phase 3: 가이드 ──

class GuideStep(BaseModel):
    step: int
    title: str
    description: str
    duration: Optional[str] = None
    tips: Optional[List[str]] = None


class LawsuitGuideResponse(BaseModel):
    case_type: str
    title: str
    steps: List[GuideStep]
