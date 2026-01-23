"""
소액 소송 에이전트 모듈 - 나홀로 소송 지원
중고거래 사기, 떼인 알바비, 층간소음 등 소액 사건 처리 지원
"""

import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.common.chat_service import search_relevant_documents
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


# 서류 템플릿 정의 (동적 값은 {placeholder} 형식)
SMALL_CLAIMS_TEMPLATES = {
    "demand_letter": {
        "title": "내용증명",
        "template_sections": {
            "header": "내용증명\n\n발신일: {today}",
            "recipient": "수신: {defendant_name}",
            "sender": "발신: {plaintiff_name}\n주소: {plaintiff_address}",
            "body": "",
            "footer": "위와 같이 내용증명 우편으로 통지합니다.\n\n14일 이내에 이행하지 않을 경우 법적 조치를 취할 것임을 알려드립니다.",
        },
        "ai_prompt": """한국어 내용증명 본문을 작성해주세요.

분쟁 유형: {dispute_type}
청구 금액: {amount_formatted}원
분쟁 경위: {description}
발생일: {incident_date}

요구사항:
1. 법적 효력이 있는 공식적인 문체 사용
2. 사실관계를 명확히 기술
3. 청구 금액과 지급 기한(14일) 명시
4. 불이행 시 법적 조치 경고
5. 500자 내외로 작성""",
    },
    "payment_order": {
        "title": "지급명령신청서",
        "template_sections": {
            "header": "지급명령신청서\n\n{today}",
            "court": "○○지방법원 귀중",
            "parties": "채권자(신청인): {plaintiff_name}\n주소: {plaintiff_address}\n\n채무자(피신청인): {defendant_name}\n주소: {defendant_address}",
            "claim": "청구금액: 금 {amount_formatted}원",
            "reason": "",
            "evidence": "",
            "footer": "위와 같이 지급명령을 신청합니다.",
        },
        "ai_prompt": """지급명령신청서의 '청구원인' 부분을 작성해주세요.

분쟁 유형: {dispute_type}
청구 금액: {amount_formatted}원
분쟁 경위: {description}
발생일: {incident_date}

요구사항:
1. 채권 발생 원인을 명확히 기술
2. 변제기(지급 기한) 명시
3. 법률적 근거 포함
4. 간결하고 명확한 문체
5. 400자 내외로 작성""",
    },
    "complaint": {
        "title": "소액심판 청구서",
        "template_sections": {
            "header": "소액사건심판 청구서\n\n{today}",
            "court": "○○지방법원 귀중",
            "parties": "원고: {plaintiff_name}\n주소: {plaintiff_address}\n\n피고: {defendant_name}\n주소: {defendant_address}",
            "claim": "청구취지: 피고는 원고에게 금 {amount_formatted}원 및 이에 대하여 이 사건 소장 부본 송달 다음날부터 다 갚는 날까지 연 12%의 비율로 계산한 돈을 지급하라.",
            "reason": "",
            "evidence": "",
            "footer": "위와 같이 청구합니다.",
        },
        "ai_prompt": """소액심판 청구서의 '청구원인' 부분을 작성해주세요.

분쟁 유형: {dispute_type}
청구 금액: {amount_formatted}원
분쟁 경위: {description}
발생일: {incident_date}

요구사항:
1. 사실관계를 시간순으로 명확히 기술
2. 원고의 권리 발생 근거 설명
3. 피고의 의무 불이행 사실 명시
4. 법률적 청구 근거 포함
5. 500자 내외로 작성""",
    },
}


def render_template_for_case(
    case_info: "CaseInfo",
    today: str,
    document_type: str,
) -> dict:
    """
    템플릿을 케이스 정보로 렌더링

    Args:
        case_info: 사건 정보
        today: 오늘 날짜 문자열
        document_type: 서류 유형 (demand_letter, payment_order, complaint)

    Returns:
        렌더링된 템플릿 (title, template_sections, ai_prompt)
    """
    template = SMALL_CLAIMS_TEMPLATES.get(document_type)
    if not template:
        return {}

    # 템플릿 변수
    variables = {
        "today": today,
        "plaintiff_name": case_info.plaintiff_name,
        "plaintiff_address": case_info.plaintiff_address,
        "defendant_name": case_info.defendant_name,
        "defendant_address": case_info.defendant_address or "(주소 조사 필요)",
        "amount_formatted": f"{case_info.amount:,}",
        "dispute_type": case_info.dispute_type,
        "description": case_info.description,
        "incident_date": case_info.incident_date or "미상",
    }

    # template_sections 렌더링
    rendered_sections = {}
    for key, value in template["template_sections"].items():
        rendered_sections[key] = value.format(**variables)

    return {
        "title": template["title"],
        "template_sections": rendered_sections,
        "ai_prompt": template["ai_prompt"].format(**variables),
    }


# 증거 체크리스트 데이터
EVIDENCE_CHECKLISTS = {
    "product_payment": {
        "dispute_type": "물품대금",
        "description": "물품을 판매했으나 대금을 받지 못한 경우",
        "items": [
            {"id": "contract", "label": "매매계약서 또는 거래 내역서", "required": True, "description": "판매 조건이 명시된 문서"},
            {"id": "delivery", "label": "배송 완료 증빙", "required": True, "description": "택배 송장, 수령 확인서 등"},
            {"id": "invoice", "label": "세금계산서 또는 영수증", "required": False, "description": "거래 금액 증빙"},
            {"id": "communication", "label": "거래 관련 대화 기록", "required": False, "description": "카카오톡, 문자, 이메일 등"},
            {"id": "payment_request", "label": "대금 지급 요청 내역", "required": False, "description": "독촉 메시지, 통화 기록 등"},
        ],
    },
    "fraud": {
        "dispute_type": "중고거래 사기",
        "description": "중고거래에서 물건을 받지 못했거나 상품이 설명과 다른 경우",
        "items": [
            {"id": "chat_capture", "label": "거래 대화 캡처", "required": True, "description": "판매자와의 대화 내용 전체"},
            {"id": "transfer", "label": "계좌이체 내역", "required": True, "description": "송금 확인 화면 또는 거래 내역서"},
            {"id": "product_info", "label": "상품 게시글/사진", "required": True, "description": "판매 게시글 캡처"},
            {"id": "seller_info", "label": "판매자 정보", "required": True, "description": "연락처, 계좌번호, ID 등"},
            {"id": "received_product", "label": "수령한 상품 사진", "required": False, "description": "하자가 있는 경우 사진 증거"},
        ],
    },
    "deposit": {
        "dispute_type": "임대차 보증금",
        "description": "전세/월세 보증금을 돌려받지 못한 경우",
        "items": [
            {"id": "lease_contract", "label": "임대차계약서", "required": True, "description": "계약서 원본 또는 사본"},
            {"id": "deposit_proof", "label": "보증금 입금 내역", "required": True, "description": "최초 보증금 지급 증빙"},
            {"id": "move_out_proof", "label": "퇴거 증빙", "required": True, "description": "전입세대 열람원, 이사 영수증 등"},
            {"id": "termination_notice", "label": "계약 해지/종료 통지", "required": False, "description": "내용증명 등"},
            {"id": "property_photos", "label": "퇴거 시 주거 상태 사진", "required": False, "description": "원상복구 증빙"},
        ],
    },
    "service_payment": {
        "dispute_type": "용역대금",
        "description": "용역(서비스)을 제공했으나 대금을 받지 못한 경우",
        "items": [
            {"id": "service_contract", "label": "용역계약서", "required": True, "description": "계약 조건이 명시된 문서"},
            {"id": "work_completion", "label": "작업 완료 증빙", "required": True, "description": "완료 사진, 납품 확인서 등"},
            {"id": "communication", "label": "업무 관련 대화 기록", "required": False, "description": "작업 지시, 수정 요청 등"},
            {"id": "invoice", "label": "청구서/견적서", "required": False, "description": "금액이 명시된 문서"},
            {"id": "payment_request", "label": "대금 지급 요청 내역", "required": False, "description": "독촉 기록"},
        ],
    },
    "wage": {
        "dispute_type": "임금 체불",
        "description": "근무했으나 급여/알바비를 받지 못한 경우",
        "items": [
            {"id": "employment_proof", "label": "근로계약서 또는 채용 확인", "required": True, "description": "문자, 카톡 채용 확인도 가능"},
            {"id": "work_record", "label": "출퇴근 기록", "required": True, "description": "타임카드, 근무표, 문자 기록 등"},
            {"id": "payment_record", "label": "기존 급여 지급 내역", "required": False, "description": "이전에 받은 급여 증빙"},
            {"id": "company_info", "label": "사업장 정보", "required": True, "description": "상호명, 대표자, 주소"},
            {"id": "communication", "label": "급여 요청 대화 기록", "required": False, "description": "사장/담당자와의 대화"},
        ],
    },
}


# Pydantic 스키마
class EvidenceItem(BaseModel):
    id: str
    label: str
    required: bool
    description: str


class EvidenceChecklistResponse(BaseModel):
    dispute_type: str
    description: str
    items: List[EvidenceItem]


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


class DocumentGenerateRequest(BaseModel):
    document_type: str  # "demand_letter" | "payment_order" | "complaint"
    case_info: CaseInfo


class DocumentResponse(BaseModel):
    document_type: str
    title: str
    content: str
    template_sections: dict


class RelatedCaseItem(BaseModel):
    id: str
    case_name: str
    case_number: str
    summary: str
    similarity: float
    relevance: str


class RelatedCasesResponse(BaseModel):
    dispute_type: str
    cases: List[RelatedCaseItem]


# 기존 엔드포인트
@router.post("/interview/start")
async def start_interview(case_type: str):
    """자연어 인터뷰 시작"""
    return {
        "session_id": "interview_session_id",
        "case_type": case_type,
        "next_question": "어떤 상황인지 자세히 말씀해 주세요.",
    }


@router.post("/interview/{session_id}/answer")
async def submit_answer(session_id: str, answer: str):
    """인터뷰 답변 제출"""
    return {
        "session_id": session_id,
        "next_question": "상대방의 연락처나 계좌 정보를 알고 계신가요?",
        "collected_info": {},
        "is_complete": False,
    }


@router.post("/documents/generate")
async def generate_documents(
    session_id: str,
    document_types: List[str],
):
    """법률 서류 자동 생성 (내용증명, 지급명령신청서, 소액심판청구서)"""
    return {
        "session_id": session_id,
        "documents": [],
    }


@router.post("/evidence/upload")
async def upload_evidence(
    session_id: str,
    files: List[UploadFile] = File(...),
    evidence_type: str = "chat_log",
):
    """증거 자료 업로드"""
    return {
        "session_id": session_id,
        "uploaded_files": [],
    }


@router.post("/evidence/{session_id}/organize")
async def organize_evidence(session_id: str):
    """증거 자료 타임라인 정리 및 PDF 변환"""
    return {
        "session_id": session_id,
        "organized_pdf_url": "",
        "timeline": [],
    }


@router.get("/guide/{case_type}")
async def get_lawsuit_guide(case_type: str):
    """소송 절차 가이드 조회"""
    return {
        "case_type": case_type,
        "steps": [
            {"step": 1, "title": "내용증명 발송", "description": "..."},
            {"step": 2, "title": "지급명령 신청", "description": "..."},
            {"step": 3, "title": "소액심판 청구", "description": "..."},
        ],
    }


# 새로운 엔드포인트
@router.get("/evidence-checklist/{dispute_type}", response_model=EvidenceChecklistResponse)
async def get_evidence_checklist(dispute_type: str):
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
async def get_dispute_types():
    """지원하는 분쟁 유형 목록 조회"""
    return {
        "dispute_types": [
            {"id": key, "name": val["dispute_type"], "description": val["description"]}
            for key, val in EVIDENCE_CHECKLISTS.items()
        ]
    }


@router.post("/generate-document", response_model=DocumentResponse)
async def generate_document(request: DocumentGenerateRequest):
    """
    서류 생성

    document_type:
    - demand_letter: 내용증명
    - payment_order: 지급명령신청서
    - complaint: 소액심판청구서
    """
    try:
        from openai import OpenAI

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

        # AI로 본문 생성
        client = OpenAI(api_key=settings.OPENAI_API_KEY)

        response = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "당신은 한국 법률 문서 작성 전문가입니다. 사용자가 제공한 정보를 바탕으로 법적 효력이 있는 문서를 작성합니다.",
                },
                {"role": "user", "content": template["ai_prompt"]},
            ],
            temperature=0.5,
            max_tokens=800,
        )

        generated_body = response.choices[0].message.content

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

        return DocumentResponse(
            document_type=document_type,
            title=template["title"],
            content=full_content,
            template_sections=template_sections,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"서류 생성 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="서류 생성 중 오류가 발생했습니다")


@router.get("/related-cases/{dispute_type}", response_model=RelatedCasesResponse)
async def get_related_cases(dispute_type: str):
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

        results = search_relevant_documents(query=query, n_results=5)

        # 관련성 설명 생성
        relevance_descriptions = {
            "product_payment": "물품대금 청구와 관련된 판례로, 유사한 사안의 법원 판단을 참고할 수 있습니다.",
            "fraud": "사기 피해 및 손해배상 청구와 관련된 판례입니다.",
            "deposit": "임대차 보증금 반환 청구와 관련된 판례입니다.",
            "service_payment": "용역대금 청구와 관련된 판례입니다.",
            "wage": "임금 체불 및 급여 청구와 관련된 판례입니다.",
        }

        cases = []
        for doc in results:
            metadata = doc.get("metadata", {})
            cases.append(
                RelatedCaseItem(
                    id=doc["id"],
                    case_name=metadata.get("case_name", ""),
                    case_number=metadata.get("case_number", ""),
                    summary=doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"],
                    similarity=round(doc.get("similarity", 0), 3),
                    relevance=relevance_descriptions.get(dispute_type, ""),
                )
            )

        return RelatedCasesResponse(
            dispute_type=EVIDENCE_CHECKLISTS.get(dispute_type, {}).get("dispute_type", dispute_type),
            cases=cases,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"관련 판례 조회 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="관련 판례 조회 중 오류가 발생했습니다")
