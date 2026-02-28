"""
소액 소송 에이전트 모듈 - 나홀로 소송 지원
중고거래 사기, 떼인 알바비, 층간소음 등 소액 사건 처리 지원
"""

import logging
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.state import get_session_store
from app.modules.small_claims.schema import (
    CaseInfo,
    DocumentGenerateRequest,
    DocumentRegenerateRequest,
    DocumentRegenerateResponse,
    DocumentResponse,
    EvidenceChecklistResponse,
    EvidenceItem,
    EvidenceOrganizeResponse,
    EvidenceTimelineItem,
    EvidenceUploadFile,
    EvidenceUploadResponse,
    GuideStep,
    InterviewAnswerRequest,
    InterviewQuestion,
    InterviewResponse,
    InterviewStartRequest,
    LawsuitGuideResponse,
    RelatedCaseItem,
    RelatedCasesResponse,
)
from app.services.document_service import DocumentService
from app.services.rag import search_relevant_documents_async
from app.services.service_function.small_claims_service import (
    EVIDENCE_CHECKLISTS,
    INTERVIEW_QUESTIONS,
    LAWSUIT_GUIDES,
    SMALL_CLAIMS_TEMPLATES,
    detect_dispute_type,
    extract_amount,
    render_template_for_case,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# ── 인터뷰 세션 키 접두사 ──
_INTERVIEW_PREFIX = "interview:"

# ── 날짜 파싱 패턴 ──
_DATE_PATTERNS = [
    re.compile(r"(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})"),  # 2025.12.15, 2025-12-15
    re.compile(r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일"),   # 2025년 12월 15일
    re.compile(r"(\d{2})[.\-/](\d{1,2})[.\-/](\d{1,2})"),   # 25.12.15
]


def _extract_date_from_text(text: str) -> Optional[str]:
    """텍스트에서 날짜 문자열을 추출합니다 (YYYY-MM-DD 형식 반환)."""
    for pattern in _DATE_PATTERNS:
        match = pattern.search(text)
        if match:
            groups = match.groups()
            year = int(groups[0])
            if year < 100:
                year += 2000
            month = int(groups[1])
            day = int(groups[2])
            return f"{year:04d}-{month:02d}-{day:02d}"
    return None


def _extract_from_answer(
    question_index: int,
    answer: str,
    existing: dict[str, Any],
) -> dict[str, Any]:
    """인터뷰 답변에서 사건 정보를 간단한 키워드 매칭으로 추출합니다."""
    field_hint = INTERVIEW_QUESTIONS[question_index]["field_hint"]
    updated = dict(existing)

    if field_hint == "dispute_type":
        detected = detect_dispute_type(answer)
        updated["dispute_type"] = detected or answer.strip()[:50]

    elif field_hint == "defendant_name":
        updated["defendant_name"] = answer.strip()[:100]

    elif field_hint == "amount":
        amount = extract_amount(answer)
        if amount is not None:
            updated["amount"] = amount

    elif field_hint == "incident_date":
        date = _extract_date_from_text(answer)
        updated["incident_date"] = date or answer.strip()[:30]

    elif field_hint == "description":
        updated["description"] = answer.strip()[:500]

    return updated


# ── Phase 3: 인터뷰 엔드포인트 ──

@router.post("/interview/start", response_model=InterviewResponse)
async def start_interview(request: InterviewStartRequest) -> InterviewResponse:
    """자연어 인터뷰 시작 — 첫 번째 질문을 반환하고 세션을 생성합니다."""
    session_id = str(uuid.uuid4())
    store = get_session_store()

    store.set(
        f"{_INTERVIEW_PREFIX}{session_id}",
        {
            "case_type": request.case_type,
            "question_index": 0,
            "collected": {
                "dispute_type": request.case_type,
                "plaintiff_name": "원고",
                "plaintiff_address": "",
                "defendant_name": "",
                "amount": 0,
                "description": "",
            },
        },
    )

    first_q = INTERVIEW_QUESTIONS[0]
    return InterviewResponse(
        session_id=session_id,
        is_complete=False,
        current_question=InterviewQuestion(
            question_index=0,
            question=first_q["question"],
            total_questions=len(INTERVIEW_QUESTIONS),
            field_hint=first_q["field_hint"],
        ),
    )


@router.post("/interview/{session_id}/answer", response_model=InterviewResponse)
async def submit_answer(
    session_id: str,
    request: InterviewAnswerRequest,
) -> InterviewResponse:
    """인터뷰 답변 제출 — 다음 질문 또는 완료 시 수집된 사건 정보를 반환합니다."""
    store = get_session_store()
    session_key = f"{_INTERVIEW_PREFIX}{session_id}"
    session_data = store.get(session_key)

    if not session_data:
        raise HTTPException(
            status_code=404,
            detail=f"세션을 찾을 수 없습니다: {session_id}",
        )

    question_index: int = session_data["question_index"]
    collected: dict[str, Any] = session_data["collected"]

    # 현재 질문의 답변으로 정보 추출
    collected = _extract_from_answer(question_index, request.answer, collected)
    next_index = question_index + 1

    if next_index >= len(INTERVIEW_QUESTIONS):
        # 인터뷰 완료 — 세션 삭제 후 CaseInfo 반환
        store.delete(session_key)

        filled_case_info = CaseInfo(
            dispute_type=str(collected.get("dispute_type", "")),
            plaintiff_name=str(collected.get("plaintiff_name", "원고")),
            plaintiff_address=str(collected.get("plaintiff_address", "")),
            defendant_name=str(collected.get("defendant_name", "")),
            amount=int(collected.get("amount", 0)),
            description=str(collected.get("description", "")),
            incident_date=collected.get("incident_date"),
        )
        return InterviewResponse(
            session_id=session_id,
            is_complete=True,
            filled_case_info=filled_case_info,
        )

    # 다음 질문 반환
    store.set(
        session_key,
        {
            **session_data,
            "question_index": next_index,
            "collected": collected,
        },
    )
    next_q = INTERVIEW_QUESTIONS[next_index]
    return InterviewResponse(
        session_id=session_id,
        is_complete=False,
        current_question=InterviewQuestion(
            question_index=next_index,
            question=next_q["question"],
            total_questions=len(INTERVIEW_QUESTIONS),
            field_hint=next_q["field_hint"],
        ),
    )


@router.post("/documents/generate")
async def generate_documents(
    session_id: str,
    document_types: List[str],
) -> dict[str, Any]:
    """법률 서류 자동 생성 (미구현)"""
    raise HTTPException(status_code=501, detail="서류 자동 생성 기능은 현재 준비 중입니다")


ALLOWED_EXTENSIONS = {
    ".pdf", ".hwp", ".hwpx", ".doc", ".docx",
    ".jpg", ".jpeg", ".png", ".gif", ".webp",
    ".xls", ".xlsx", ".txt",
}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


@router.post("/evidence/upload", response_model=EvidenceUploadResponse)
async def upload_evidence(
    evidence_item_id: str = "general",
    session_id: str = "default",
    files: List[UploadFile] = File(...),
) -> EvidenceUploadResponse:
    """
    증거 자료 업로드

    파일을 서버에 저장하고 메타데이터를 반환합니다.
    지원 형식: PDF, HWP, DOC, 이미지, XLS, TXT (최대 10MB)
    """
    upload_dir = Path("data/uploads/small_claims") / session_id / evidence_item_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    uploaded: list[EvidenceUploadFile] = []

    for file in files:
        if not file.filename:
            continue

        # 확장자 검증
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"지원하지 않는 파일 형식입니다: {ext}. "
                       f"지원 형식: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
            )

        # 파일 크기 검증
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"파일 크기가 10MB를 초과합니다: {file.filename} ({len(content) / 1024 / 1024:.1f}MB)",
            )

        file_id = str(uuid.uuid4())
        saved_name = f"{file_id}{ext}"
        file_path = upload_dir / saved_name

        file_path.write_bytes(content)

        uploaded.append(
            EvidenceUploadFile(
                file_id=file_id,
                original_name=file.filename,
                file_type=ext.lstrip("."),
                file_size=len(content),
            )
        )

    if not uploaded:
        raise HTTPException(status_code=400, detail="업로드할 파일이 없습니다")

    return EvidenceUploadResponse(uploaded_files=uploaded)


@router.post("/evidence/{session_id}/organize", response_model=EvidenceOrganizeResponse)
async def organize_evidence(session_id: str) -> EvidenceOrganizeResponse:
    """
    증거 자료 타임라인 정리

    data/uploads/small_claims/{session_id}/ 하위 파일 목록을 조회하고
    파일명에서 날짜를 추출하여 시간순으로 정렬한 타임라인을 반환합니다.
    """
    upload_base = Path("data/uploads/small_claims") / session_id

    if not upload_base.exists():
        return EvidenceOrganizeResponse(session_id=session_id, timeline=[])

    timeline_items: list[EvidenceTimelineItem] = []

    for file_path in upload_base.rglob("*"):
        if not file_path.is_file():
            continue

        original_name = file_path.name
        # UUID 파일명(저장명)에서 날짜 추출 시도 → 파일명에서는 보통 없으므로
        # 파일 수정 시각을 fallback으로 사용하여 정렬
        extracted_date = _extract_date_from_text(original_name)
        if extracted_date is None:
            # mtime 기반 날짜
            mtime = file_path.stat().st_mtime
            extracted_date = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")

        ext = file_path.suffix.lower().lstrip(".")
        category = _guess_evidence_category(ext)

        timeline_items.append(
            EvidenceTimelineItem(
                file_id=file_path.stem,
                original_name=original_name,
                date=extracted_date,
                category=category,
            )
        )

    # 날짜순 정렬 (None을 마지막으로)
    timeline_items.sort(key=lambda x: x.date or "9999-99-99")

    return EvidenceOrganizeResponse(session_id=session_id, timeline=timeline_items)


def _guess_evidence_category(ext: str) -> str:
    """파일 확장자로 증거 유형을 추정합니다."""
    image_exts = {"jpg", "jpeg", "png", "gif", "webp"}
    doc_exts = {"pdf", "hwp", "hwpx", "doc", "docx"}
    spreadsheet_exts = {"xls", "xlsx"}

    if ext in image_exts:
        return "이미지"
    if ext in doc_exts:
        return "문서"
    if ext in spreadsheet_exts:
        return "스프레드시트"
    if ext == "txt":
        return "텍스트"
    return "기타"


@router.get("/guide/{case_type}", response_model=LawsuitGuideResponse)
async def get_lawsuit_guide(case_type: str) -> LawsuitGuideResponse:
    """
    소송 절차 가이드 조회

    case_type: product_payment | fraud | deposit | service_payment | wage
    """
    guide_data = LAWSUIT_GUIDES.get(case_type)
    if not guide_data:
        raise HTTPException(
            status_code=404,
            detail=f"지원하지 않는 사건 유형입니다: {case_type}. "
                   f"지원 유형: {', '.join(LAWSUIT_GUIDES.keys())}",
        )

    steps = [
        GuideStep(
            step=s["step"],
            title=s["title"],
            description=s["description"],
            duration=s.get("duration"),
            tips=s.get("tips"),
        )
        for s in guide_data["steps"]
    ]

    return LawsuitGuideResponse(
        case_type=case_type,
        title=guide_data["title"],
        steps=steps,
    )


# 새로운 엔드포인트
@router.get("/evidence-checklist/{dispute_type}", response_model=EvidenceChecklistResponse)
async def get_evidence_checklist(dispute_type: str) -> EvidenceChecklistResponse:
    """
    분쟁 유형별 증거 체크리스트 조회

    dispute_type: product_payment | fraud | deposit | service_payment | wage
    """
    checklist = EVIDENCE_CHECKLISTS.get(dispute_type)
    if not checklist:
        raise HTTPException(
            status_code=404,
            detail=f"지원하지 않는 분쟁 유형입니다: {dispute_type}. "
                   f"지원 유형: {', '.join(EVIDENCE_CHECKLISTS.keys())}",
        )

    return EvidenceChecklistResponse(
        dispute_type=checklist["dispute_type"],
        description=checklist["description"],
        items=[EvidenceItem(**item) for item in checklist["items"]],
    )


@router.get("/dispute-types")
async def get_dispute_types() -> dict[str, Any]:
    """지원하는 분쟁 유형 목록 조회"""
    return {
        "dispute_types": [
            {"id": key, "name": val["dispute_type"], "description": val["description"]}
            for key, val in EVIDENCE_CHECKLISTS.items()
        ]
    }


@router.post("/generate-document", response_model=DocumentResponse)
async def generate_document(request: DocumentGenerateRequest) -> DocumentResponse:
    """
    서류 생성

    document_type:
    - demand_letter: 내용증명
    - payment_order: 지급명령신청서
    - complaint: 소액심판청구서
    """
    try:
        from app.tools.llm import get_chat_model

        case_info = request.case_info
        document_type = request.document_type
        today = datetime.now().strftime("%Y년 %m월 %d일")

        if document_type not in SMALL_CLAIMS_TEMPLATES:
            raise HTTPException(
                status_code=400,
                detail=f"지원하지 않는 서류 유형입니다: {document_type}. "
                       f"지원 유형: {', '.join(SMALL_CLAIMS_TEMPLATES.keys())}",
            )

        template = render_template_for_case(case_info, today, document_type)

        # AI로 본문 생성 (LLM 추상화 레이어 사용)
        llm = get_chat_model(temperature=0.5)
        ai_response = await llm.ainvoke([
            ("system", "당신은 한국 법률 문서 작성 전문가입니다. 사용자가 제공한 정보를 바탕으로 법적 효력이 있는 문서를 작성합니다."),
            ("user", template["ai_prompt"]),
        ])

        generated_body = str(ai_response.content) if ai_response.content else None
        if not generated_body:
            raise HTTPException(status_code=503, detail="AI 응답이 없습니다")

        # 템플릿 섹션에 생성된 본문 추가
        template_sections = template["template_sections"].copy()
        if document_type == "demand_letter":
            template_sections["body"] = generated_body or ""
        else:
            template_sections["reason"] = generated_body or ""

        # 전체 내용 조합
        if document_type == "demand_letter":
            full_content = f"""{template_sections['header']}

{template_sections['recipient']}
{template_sections['sender']}

{template_sections['body']}

{template_sections['footer']}

발신인: {case_info.plaintiff_name} (인)
"""
        else:
            full_content = f"""{template_sections['header']}

{template_sections['court']}

{template_sections['parties']}

{template_sections['claim']}

청구원인:
{template_sections['reason']}

{template_sections['footer']}

신청인(원고): {case_info.plaintiff_name} (인)
"""

        # PDF 생성
        pdf_url: str | None = None
        docx_url: str | None = None
        base_dir = Path("data/media/documents")
        base_dir.mkdir(parents=True, exist_ok=True)

        try:
            doc_service = DocumentService()
            filename = f"{document_type}_{uuid.uuid4()}.pdf"
            output_path = base_dir / filename
            doc_service.generate_pdf_from_text(full_content, str(output_path))
            pdf_url = f"/media/documents/{filename}"
        except Exception as e:
            logger.error("PDF 생성 실패: %s", e)

        # DOCX 생성 (워드 파일 - 한글에서도 열림)
        try:
            doc_service = DocumentService()
            docx_filename = f"{document_type}_{uuid.uuid4()}.docx"
            docx_output_path = base_dir / docx_filename
            doc_service.generate_docx_from_text(full_content, str(docx_output_path))
            docx_url = f"/media/documents/{docx_filename}"
        except Exception as e:
            logger.error("DOCX 생성 실패: %s", e)

        return DocumentResponse(
            document_type=document_type,
            title=template["title"],
            content=full_content,
            template_sections=template_sections,
            pdf_url=pdf_url,
            docx_url=docx_url,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("서류 생성 실패: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="서류 생성 중 오류가 발생했습니다")


@router.post("/regenerate-document", response_model=DocumentRegenerateResponse)
async def regenerate_document(request: DocumentRegenerateRequest) -> DocumentRegenerateResponse:
    """
    편집된 텍스트로 PDF/DOCX 재생성 (Phase 2B)

    사용자가 수정한 서류 내용을 받아 PDF/DOCX 파일로 재생성합니다.
    formats: ["pdf", "docx"] (기본값: 둘 다 생성)
    """
    try:
        base_dir = Path("data/media/documents")
        base_dir.mkdir(parents=True, exist_ok=True)

        doc_service = DocumentService()
        pdf_url: str | None = None
        docx_url: str | None = None

        if "pdf" in request.formats:
            try:
                filename = f"{request.document_type}_{uuid.uuid4()}.pdf"
                output_path = base_dir / filename
                doc_service.generate_pdf_from_text(request.content, str(output_path))
                pdf_url = f"/media/documents/{filename}"
            except Exception as e:
                logger.error("PDF 재생성 실패: %s", e)

        if "docx" in request.formats:
            try:
                docx_filename = f"{request.document_type}_{uuid.uuid4()}.docx"
                docx_output_path = base_dir / docx_filename
                doc_service.generate_docx_from_text(request.content, str(docx_output_path))
                docx_url = f"/media/documents/{docx_filename}"
            except Exception as e:
                logger.error("DOCX 재생성 실패: %s", e)

        if pdf_url is None and docx_url is None:
            raise HTTPException(status_code=500, detail="파일 재생성에 실패했습니다")

        return DocumentRegenerateResponse(pdf_url=pdf_url, docx_url=docx_url)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("서류 재생성 실패: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="서류 재생성 중 오류가 발생했습니다")


@router.get("/related-cases/{dispute_type}", response_model=RelatedCasesResponse)
async def get_related_cases(dispute_type: str) -> RelatedCasesResponse:
    """
    분쟁 유형별 관련 판례 조회

    RAG를 통해 해당 분쟁 유형과 관련된 판례를 검색합니다.
    """
    try:
        # 분쟁 유형별 검색 쿼리
        search_queries = {
            "product_payment": "물품대금 매매대금 청구 판결",
            "fraud": "중고거래 사기 손해배상 판결",
            "deposit": "임대차 보증금 반환 판결",
            "service_payment": "용역대금 채무불이행 판결",
            "wage": "임금 체불 급여 청구 판결",
        }

        query = search_queries.get(dispute_type)
        if not query:
            raise HTTPException(
                status_code=404,
                detail=f"지원하지 않는 분쟁 유형입니다: {dispute_type}",
            )

        results = await search_relevant_documents_async(
            query=query, n_results=5, exclude_doc_types=["법령"],
        )

        # 관련성 설명 생성
        relevance_descriptions = {
            "product_payment": "물품대금 청구와 관련된 판례로, 유사한 사안의 법원 판단을 참고할 수 있습니다.",
            "fraud": "사기 피해 및 손해배상 청구와 관련된 판례입니다.",
            "deposit": "임대차 보증금 반환 청구와 관련된 판례입니다.",
            "service_payment": "용역대금 청구와 관련된 판례입니다.",
            "wage": "임금 체불 및 급여 청구와 관련된 판례입니다.",
        }

        # RAG 결과에서 source_id 추출 후 PostgreSQL에서 판결/판결요지 조회
        source_ids = [doc["id"] for doc in results if doc.get("id")]
        precedent_details: dict[str, dict[str, str]] = {}
        if source_ids:
            try:
                from app.services.service_function.precedent_service import (
                    get_precedent_service,
                )

                service = get_precedent_service()
                precedent_details = await service.get_details(source_ids)
            except Exception as e:
                logger.warning("판례 상세 조회 실패 (계속 진행): %s", e)

        cases = []
        for doc in results:
            metadata = doc.get("metadata", {})
            doc_id = doc["id"]
            detail = precedent_details.get(doc_id, {})
            cases.append(
                RelatedCaseItem(
                    id=doc_id,
                    case_name=metadata.get("case_name", "") or detail.get("case_name", ""),
                    case_number=metadata.get("case_number", "") or detail.get("case_number", ""),
                    summary=doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"],
                    similarity=round(doc.get("similarity", 0), 3),
                    relevance=relevance_descriptions.get(dispute_type, ""),
                    doc_type=metadata.get("data_type", "판례"),
                    ruling=detail.get("ruling"),
                    reasoning=detail.get("reasoning"),
                )
            )

        # 유사도 내림차순 정렬 (하이브리드 검색의 RRF 병합 순서와 similarity 값이 불일치할 수 있음)
        cases.sort(key=lambda c: c.similarity, reverse=True)

        return RelatedCasesResponse(
            dispute_type=EVIDENCE_CHECKLISTS.get(dispute_type, {}).get("dispute_type", dispute_type),
            cases=cases,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("관련 판례 조회 실패: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="관련 판례 조회 중 오류가 발생했습니다")
