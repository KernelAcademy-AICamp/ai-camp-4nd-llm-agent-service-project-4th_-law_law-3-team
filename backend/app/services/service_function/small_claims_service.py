"""
소액소송 서비스

소액소송 절차 안내 및 분쟁 유형 감지
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 소액소송 절차 단계
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

# 분쟁 유형 키워드
DISPUTE_TYPE_KEYWORDS: Dict[str, List[str]] = {
    "금전채무": ["돈", "빌려준", "갚지", "못받", "대여금", "차용"],
    "중고거래사기": ["중고", "거래", "사기", "배송", "물건", "입금"],
    "임대차": ["전세", "월세", "보증금", "집주인", "임대인", "세입자"],
    "용역대금": ["용역", "공사", "인테리어", "수리", "대금"],
    "매매대금": ["매매", "물건값", "물품대금", "구입"],
    "손해배상": ["손해", "배상", "피해", "보상"],
}


class SmallClaimsService:
    """소액소송 서비스 클래스"""

    def get_step_guide(self, step: str) -> Optional[Dict[str, Any]]:
        """
        소액소송 단계별 가이드 반환

        Args:
            step: 단계 키 (check_eligibility, prepare_documents, 등)

        Returns:
            단계 정보 또는 None
        """
        return SMALL_CLAIMS_STEPS.get(step)

    def get_all_steps(self) -> List[Dict[str, Any]]:
        """모든 소액소송 단계 반환"""
        steps = list(SMALL_CLAIMS_STEPS.values())
        steps.sort(key=lambda x: x["step"])
        return steps

    def detect_dispute_type(self, message: str) -> Optional[str]:
        """
        메시지에서 분쟁 유형 감지

        Args:
            message: 사용자 메시지

        Returns:
            분쟁 유형 또는 None
        """
        message_lower = message.lower()

        for dispute_type, keywords in DISPUTE_TYPE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in message_lower:
                    return dispute_type

        return None

    def check_eligibility(self, amount: int) -> Dict[str, Any]:
        """
        소액소송 대상 여부 확인

        Args:
            amount: 청구 금액 (원)

        Returns:
            {"eligible": bool, "reason": str}
        """
        threshold = 30_000_000  # 3천만원

        if amount <= threshold:
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
        """
        법원 비용 계산 (인지대, 송달료)

        간이 계산 - 실제 비용은 법원에서 확인 필요
        """
        # 소액사건 인지대 계산 (간이)
        if amount <= 10_000_000:
            stamp_fee = max(int(amount * 0.01), 1_000)
        elif amount <= 20_000_000:
            stamp_fee = int(amount * 0.01)
        else:
            stamp_fee = int(amount * 0.01)

        # 송달료 (기본 1회 송달 기준)
        service_fee = 5_200 * 2  # 원고, 피고 각 1회

        return {
            "stamp_fee": stamp_fee,
            "service_fee": service_fee,
            "total": stamp_fee + service_fee,
        }

    def get_document_checklist(self, dispute_type: Optional[str] = None) -> List[str]:
        """
        필요 서류 체크리스트 반환

        Args:
            dispute_type: 분쟁 유형 (선택)

        Returns:
            필요 서류 목록
        """
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

        additional = type_specific.get(dispute_type, [])
        return base_documents + additional


_small_claims_service: Optional[SmallClaimsService] = None


def get_small_claims_service() -> SmallClaimsService:
    """SmallClaimsService 싱글톤 인스턴스 반환"""
    global _small_claims_service
    if _small_claims_service is None:
        _small_claims_service = SmallClaimsService()
    return _small_claims_service
