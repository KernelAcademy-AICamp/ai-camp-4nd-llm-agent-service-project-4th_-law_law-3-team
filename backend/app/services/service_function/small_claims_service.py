"""
소액소송 서비스

소액소송 절차 안내, 분쟁 유형 감지, 서류 템플릿, 증거 체크리스트, 가이드
"""

import logging
import re
from typing import Any, Dict, List, Optional

from app.services.rag.format_utils import (
    format_precedent_context,
    format_precedent_sources,
)
from app.services.rag.pipeline import PipelineConfig, search_with_pipeline_async

logger = logging.getLogger(__name__)

# ── 소액소송 한도 (단일 정의) ──
SMALL_CLAIMS_LIMIT = 30_000_000


# ── 분쟁 유형 키워드 (통합: 에이전트 5종 + 서비스 6종) ──
DISPUTE_TYPE_KEYWORDS: Dict[str, List[str]] = {
    "물품대금": ["물건", "물품", "상품", "대금", "매매", "물건값"],
    "중고거래": ["중고", "당근", "번개", "거래", "사기"],
    "임대차": ["보증금", "월세", "전세", "임대", "집주인", "세입자", "임대인"],
    "용역대금": ["용역", "서비스", "작업", "수리", "공사", "인테리어"],
    "임금체불": ["임금", "급여", "알바", "체불", "월급"],
    "대여금": ["빌려", "빌린", "대여", "꿔"],
    "손해배상": ["손해", "배상", "피해", "보상"],
}


# ── 분쟁 유형별 RAG 쿼리 템플릿 ──

DISPUTE_QUERY_TEMPLATES: Dict[str, str] = {
    "물품대금": "{description} 물품대금 청구 판례",
    "중고거래": "{description} 중고거래 사기 손해배상 판례",
    "임대차": "{description} 임대차 보증금 반환 청구",
    "용역대금": "{description} 용역대금 채무불이행",
    "임금체불": "{description} 임금체불 근로기준법",
    "대여금": "{description} 대여금 반환 청구",
    "손해배상": "{description} 손해배상 청구 판례",
}


async def search_for_dispute_type(
    dispute_type: str,
    description: str,
    limit: int = 3,
) -> tuple[list[dict[str, Any]], str, list[dict[str, Any]]]:
    """분쟁 유형에 맞는 편향 쿼리로 판례를 검색합니다.

    mock_trial_service.search_for_role() 패턴 참조.

    Args:
        dispute_type: 분쟁 유형 (물품대금, 중고거래, 임대차 등)
        description: 사건 경위 (쿼리 생성에 사용)
        limit: 리랭킹 후 반환할 결과 수

    Returns:
        (판례 결과 리스트, LLM 컨텍스트 문자열, 프론트엔드 소스 리스트) 튜플
    """
    template = DISPUTE_QUERY_TEMPLATES.get(dispute_type, "{description} 소액소송 판례")
    query = template.format(description=description[:200])

    config = PipelineConfig(
        n_results=10,
        doc_type="precedent",
        enable_rerank=True,
        rerank_top_k=limit,
        enable_rewrite=False,
    )

    try:
        result = await search_with_pipeline_async(query, config)
        documents = result.documents
        context = format_precedent_context(documents)
        sources = format_precedent_sources(documents)
        return documents, context, sources
    except (ValueError, RuntimeError) as e:
        logger.warning("소액소송 RAG 검색 실패 (dispute_type=%s): %s", dispute_type, e)
        return [], "", []


def detect_dispute_type(message: str) -> Optional[str]:
    """
    메시지에서 분쟁 유형 감지 (통합 버전)

    에이전트와 서비스 양쪽에서 이 함수를 사용합니다.
    """
    for dispute_type, keywords in DISPUTE_TYPE_KEYWORDS.items():
        if any(kw in message for kw in keywords):
            return dispute_type
    return None


def extract_amount(message: str) -> Optional[int]:
    """메시지에서 금액 추출"""
    patterns = [
        r"(\d+)\s*만\s*원",
        r"(\d{1,3}(?:,\d{3})*)\s*원",
        r"(\d+)\s*원",
    ]

    for pattern in patterns:
        match = re.search(pattern, message)
        if match:
            amount_str = match.group(1).replace(",", "")
            amount = int(amount_str)
            if "만" in pattern:
                amount *= 10000
            return amount
    return None


# ── 서류 템플릿 ──

SMALL_CLAIMS_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "demand_letter": {
        "title": "내용증명",
        "template_sections": {
            "header": "내용증명\n\n발신일: {today}",
            "recipient": "수신: {defendant_name}",
            "sender": "발신: {plaintiff_name}\n주소: {plaintiff_address}",
            "body": "",
            "footer": (
                "위와 같이 내용증명 우편으로 통지합니다.\n\n"
                "14일 이내에 이행하지 않을 경우 법적 조치를 취할 것임을 알려드립니다."
            ),
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
            "parties": (
                "채권자(신청인): {plaintiff_name}\n주소: {plaintiff_address}\n\n"
                "채무자(피신청인): {defendant_name}\n주소: {defendant_address}"
            ),
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
            "parties": (
                "원고: {plaintiff_name}\n주소: {plaintiff_address}\n\n"
                "피고: {defendant_name}\n주소: {defendant_address}"
            ),
            "claim": (
                "청구취지: 피고는 원고에게 금 {amount_formatted}원 및 이에 대하여 "
                "이 사건 소장 부본 송달 다음날부터 다 갚는 날까지 연 12%의 비율로 "
                "계산한 돈을 지급하라."
            ),
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
    case_info: Any,
    today: str,
    document_type: str,
) -> dict[str, Any]:
    """
    템플릿을 케이스 정보로 렌더링

    Args:
        case_info: CaseInfo Pydantic 모델 또는 동일 속성의 객체
        today: 오늘 날짜 문자열
        document_type: 서류 유형 (demand_letter, payment_order, complaint)

    Returns:
        렌더링된 템플릿 (title, template_sections, ai_prompt)
    """
    template = SMALL_CLAIMS_TEMPLATES.get(document_type)
    if not template:
        return {}

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

    rendered_sections = {}
    for key, value in template["template_sections"].items():
        rendered_sections[key] = value.format(**variables)

    return {
        "title": template["title"],
        "template_sections": rendered_sections,
        "ai_prompt": template["ai_prompt"].format(**variables),
    }


# ── 증거 체크리스트 ──

EVIDENCE_CHECKLISTS: Dict[str, Dict[str, Any]] = {
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


# ── Phase 3: 가이드 데이터 ──

LAWSUIT_GUIDES: Dict[str, Dict[str, Any]] = {
    "product_payment": {
        "title": "물품대금 청구 소송 가이드",
        "steps": [
            {
                "step": 1,
                "title": "내용증명 발송",
                "description": "상대방에게 대금 지급을 공식 요청합니다. 14일 이내 이행을 요구하세요.",
                "duration": "1~2일",
                "tips": ["우체국 방문 또는 전자내용증명(e-그린우편) 이용", "발송 후 배달증명 보관"],
            },
            {
                "step": 2,
                "title": "지급명령 신청",
                "description": "내용증명에 응답이 없으면 법원에 지급명령을 신청합니다.",
                "duration": "1~2주",
                "tips": ["전자소송(ecfs.scourt.go.kr) 이용 가능", "인지대: 청구금액의 0.5%"],
            },
            {
                "step": 3,
                "title": "소액심판 청구",
                "description": "지급명령에 이의신청이 들어오면 소액심판으로 전환됩니다.",
                "duration": "1~2개월",
                "tips": ["1회 변론으로 판결", "변호사 없이 본인 출석 가능"],
            },
        ],
    },
    "fraud": {
        "title": "중고거래 사기 피해 구제 가이드",
        "steps": [
            {
                "step": 1,
                "title": "경찰 신고 및 증거 수집",
                "description": "사기 피해를 경찰에 신고하고, 거래 기록을 모두 보존합니다.",
                "duration": "즉시",
                "tips": ["사이버수사대 온라인 신고 가능", "거래 플랫폼에 신고"],
            },
            {
                "step": 2,
                "title": "내용증명 발송",
                "description": "판매자에게 환불을 공식 요구합니다.",
                "duration": "1~2일",
                "tips": ["상대방 실명/주소 확인 필요", "계좌 지급정지 신청 검토"],
            },
            {
                "step": 3,
                "title": "소액심판 청구",
                "description": "환불이 이뤄지지 않으면 법원에 손해배상을 청구합니다.",
                "duration": "1~2개월",
                "tips": ["형사 고소와 민사 소송 병행 가능", "판매 게시글 캡처가 핵심 증거"],
            },
        ],
    },
    "deposit": {
        "title": "임대차 보증금 반환 소송 가이드",
        "steps": [
            {
                "step": 1,
                "title": "내용증명 발송",
                "description": "임대인에게 보증금 반환을 공식 요구합니다.",
                "duration": "1~2일",
                "tips": ["임대차 종료 후 14일 이내 발송 권장", "전입세대 열람 발급"],
            },
            {
                "step": 2,
                "title": "임차권등기명령 신청",
                "description": "이사 후에도 보증금 우선변제권을 유지합니다.",
                "duration": "1~2주",
                "tips": ["관할 법원에 신청", "등기 비용 약 5만원"],
            },
            {
                "step": 3,
                "title": "지급명령 또는 소액심판",
                "description": "보증금 3천만원 이하 시 소액심판, 초과 시 일반 민사소송.",
                "duration": "1~3개월",
                "tips": ["주택임대차보호법 적용", "확정일자 여부 확인"],
            },
        ],
    },
    "service_payment": {
        "title": "용역대금 청구 소송 가이드",
        "steps": [
            {
                "step": 1,
                "title": "내용증명 발송",
                "description": "발주자에게 용역대금 지급을 공식 요구합니다.",
                "duration": "1~2일",
                "tips": ["작업 완료 증빙 첨부", "계약서상 지급 조건 명시"],
            },
            {
                "step": 2,
                "title": "지급명령 신청",
                "description": "내용증명 불응 시 법원에 지급명령을 신청합니다.",
                "duration": "1~2주",
                "tips": ["계약서 + 작업완료 증빙이 핵심", "구두 계약도 소송 가능"],
            },
            {
                "step": 3,
                "title": "소액심판 청구",
                "description": "이의신청 시 소액심판으로 전환됩니다.",
                "duration": "1~2개월",
                "tips": ["하자보수 항변에 대비", "작업 과정 기록 보관"],
            },
        ],
    },
    "wage": {
        "title": "임금 체불 구제 가이드",
        "steps": [
            {
                "step": 1,
                "title": "고용노동부 진정",
                "description": "관할 지방고용노동청에 임금체불 진정을 접수합니다. (무료)",
                "duration": "2~4주",
                "tips": ["온라인 민원(minwon.moel.go.kr) 가능", "근로감독관 조사 후 시정명령"],
            },
            {
                "step": 2,
                "title": "체불임금 소액심판",
                "description": "노동청 조정 실패 시 법원에 소액심판을 청구합니다.",
                "duration": "1~2개월",
                "tips": ["근로기준법 위반으로 형사 고소 병행 가능", "지연이자 연 20% 청구 가능"],
            },
            {
                "step": 3,
                "title": "체불임금 간이대지급금 신청",
                "description": "사업주 지급 불능 시 정부가 대신 지급합니다.",
                "duration": "1~2개월",
                "tips": ["근로복지공단에 신청", "최대 1,000만원 한도"],
            },
        ],
    },
}


# ── Phase 3: 인터뷰 질문 ──

INTERVIEW_QUESTIONS: List[Dict[str, str]] = [
    {
        "question": "어떤 유형의 분쟁인지 간단히 설명해주세요. (예: 중고거래 사기, 보증금 미반환 등)",
        "field_hint": "dispute_type",
    },
    {
        "question": "분쟁 상대방의 이름(또는 업체명)을 알려주세요.",
        "field_hint": "defendant_name",
    },
    {
        "question": "청구하실 금액은 얼마인가요? (예: 50만원, 300만원)",
        "field_hint": "amount",
    },
    {
        "question": "사건이 발생한 날짜를 알려주세요. (예: 2025년 12월 15일)",
        "field_hint": "incident_date",
    },
    {
        "question": "사건 경위를 자세히 설명해주세요. 무슨 일이 있었는지 시간 순서대로 알려주시면 좋습니다.",
        "field_hint": "description",
    },
]


# ── 소액소송 절차 단계 ──

SMALL_CLAIMS_STEPS: Dict[str, Dict[str, Any]] = {
    "check_eligibility": {
        "step": 1,
        "title": "소액소송 대상 확인",
        "description": "청구 금액이 3,000만원 이하인 민사 사건인지 확인합니다.",
        "checklist": [
            "청구 금액이 3,000만원 이하인가요?",
            "민사 사건인가요? (형사 사건 제외)",
            "상대방의 주소를 알고 있나요?",
        ],
    },
    "prepare_documents": {
        "step": 2,
        "title": "서류 준비",
        "description": "소장 작성 및 필요 서류를 준비합니다.",
        "documents": [
            "소장 (법원 양식)",
            "증거 서류 (계약서, 영수증, 카톡 대화 등)",
            "주민등록등본 (원고)",
            "상대방 주소 확인 서류",
        ],
    },
    "file_lawsuit": {
        "step": 3,
        "title": "소장 제출",
        "description": "관할 법원에 소장을 제출합니다.",
        "tips": [
            "피고 주소지 관할 법원 또는 의무이행지 관할 법원",
            "전자소송 (ecourt.go.kr) 이용 가능",
            "인지대, 송달료 납부 필요",
        ],
    },
    "attend_hearing": {
        "step": 4,
        "title": "변론기일 출석",
        "description": "지정된 날짜에 법원에 출석합니다.",
        "tips": [
            "1회 변론기일에 판결 선고 원칙",
            "증거 자료 원본 지참",
            "변호사 선임 없이 본인 소송 가능",
        ],
    },
    "get_judgment": {
        "step": 5,
        "title": "판결 확인",
        "description": "판결문을 수령하고 집행합니다.",
        "tips": [
            "승소 시 강제집행 신청 가능",
            "패소 시 2주 내 항소 가능",
        ],
    },
}


class SmallClaimsService:
    """소액소송 서비스 클래스"""

    def get_step_guide(self, step: str) -> Optional[Dict[str, Any]]:
        """소액소송 단계별 가이드 반환"""
        return SMALL_CLAIMS_STEPS.get(step)

    def get_all_steps(self) -> List[Dict[str, Any]]:
        """모든 소액소송 단계 반환"""
        steps = list(SMALL_CLAIMS_STEPS.values())
        steps.sort(key=lambda x: x["step"])
        return steps

    def detect_dispute_type(self, message: str) -> Optional[str]:
        """메시지에서 분쟁 유형 감지 (인스턴스 메서드 — 모듈 함수로 위임)"""
        return detect_dispute_type(message)

    def check_eligibility(self, amount: int) -> Dict[str, Any]:
        """소액소송 대상 여부 확인"""
        if amount <= SMALL_CLAIMS_LIMIT:
            return {
                "eligible": True,
                "reason": f"청구 금액 {amount:,}원은 소액소송 대상입니다.",
                "court_fee": self._calculate_court_fee(amount),
            }
        else:
            return {
                "eligible": False,
                "reason": f"청구 금액 {amount:,}원은 소액소송 한도(3,000만원)를 초과합니다.",
                "alternative": "일반 민사소송을 진행해야 합니다.",
            }

    def _calculate_court_fee(self, amount: int) -> Dict[str, int]:
        """법원 비용 계산 (인지대, 송달료) — 간이 계산"""
        stamp_fee = max(int(amount * 0.01), 1_000)
        service_fee = 5_200 * 2  # 원고, 피고 각 1회
        return {
            "stamp_fee": stamp_fee,
            "service_fee": service_fee,
            "total": stamp_fee + service_fee,
        }

    def get_document_checklist(self, dispute_type: Optional[str] = None) -> List[str]:
        """필요 서류 체크리스트 반환"""
        base_documents = [
            "소장 (법원 양식 또는 자유 양식)",
            "인감증명서 또는 본인서명사실확인서",
            "주민등록등본 (원고)",
        ]

        type_specific: Dict[str, List[str]] = {
            "금전채무": ["차용증 또는 계약서", "송금 내역 (계좌이체 확인증)", "독촉 문자/카톡 대화"],
            "중고거래사기": ["거래 화면 캡처", "송금 내역", "판매자 정보 (ID, 연락처)", "경찰 신고 접수증 (있는 경우)"],
            "임대차": ["임대차계약서", "보증금 송금 내역", "내용증명 발송 내역 (있는 경우)"],
            "손해배상": ["피해 증빙 자료", "견적서 또는 수리비 영수증", "사진 증거"],
        }

        additional = type_specific.get(dispute_type, []) if dispute_type else []
        return base_documents + additional


_small_claims_service: Optional[SmallClaimsService] = None


def get_small_claims_service() -> SmallClaimsService:
    """SmallClaimsService 싱글톤 인스턴스 반환"""
    global _small_claims_service
    if _small_claims_service is None:
        _small_claims_service = SmallClaimsService()
    return _small_claims_service
