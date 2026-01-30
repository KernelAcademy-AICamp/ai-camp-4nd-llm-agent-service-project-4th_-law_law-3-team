"""
소액소송 에이전트

단계별 소액소송 가이드 및 서류 작성 지원
"""

import logging
import re
from typing import Any

from app.common.agent_base import ActionType, AgentResponse, BaseAgent, ChatAction
from app.common.chat_service import fetch_precedent_details, search_relevant_documents

logger = logging.getLogger(__name__)


class SmallClaimsStep:
    """소액소송 진행 단계"""
    INIT = "init"
    GATHER_INFO = "gather_info"
    EVIDENCE = "evidence"
    DEMAND_LETTER = "demand_letter"
    COURT = "court"
    COMPLETE = "complete"


# 단계별 안내 메시지
STEP_MESSAGES = {
    SmallClaimsStep.INIT: """
소액소송 절차를 안내해드리겠습니다.

**소액소송이란?**
소송 목적의 값이 3,000만원 이하인 민사사건을 간단하고 신속하게 처리하는 제도입니다.

먼저 상황을 파악하기 위해 몇 가지 질문을 드릴게요.

**어떤 유형의 분쟁인가요?**
1. 물품 대금 미지급
2. 중고거래 사기
3. 임대차 보증금
4. 용역/서비스 대금
5. 기타 금전 분쟁
""",

    SmallClaimsStep.GATHER_INFO: """
감사합니다. 다음 정보를 알려주세요:

1. **분쟁 금액**: 얼마를 청구하시나요?
2. **상대방 정보**: 이름, 연락처, 주소 중 아는 정보가 있나요?
3. **분쟁 발생 시기**: 언제 일이 발생했나요?
4. **간단한 경위**: 어떤 상황인지 간단히 설명해주세요.
""",

    SmallClaimsStep.EVIDENCE: """
**증거 자료 정리가 필요합니다.**

다음 자료들을 준비해주세요:
- 계약서, 영수증, 거래내역
- 카카오톡/문자 대화 캡처
- 계좌이체 내역
- 사진, 영상 등

**이미 가지고 있는 증거가 있나요?**
(있는 증거를 알려주시면, 추가로 필요한 것을 안내해드립니다)
""",

    SmallClaimsStep.DEMAND_LETTER: """
**내용증명 발송 단계**

소송 전에 내용증명을 보내면:
1. 상대방에게 심리적 압박
2. 소송 시 증거로 활용
3. 합의 가능성 높임

**내용증명 작성 요령:**
- 사실관계 명확히 기술
- 요구사항 구체적으로 명시
- 이행 기한 설정 (보통 7~14일)
- 불이행 시 법적 조치 예고

내용증명 초안을 작성해드릴까요?
""",

    SmallClaimsStep.COURT: """
**소액소송 제기 단계**

**필요 서류:**
1. 소장 (법원 양식)
2. 증거 서류 사본
3. 주민등록등본 (원고)

**제기 방법:**
- 온라인: 대법원 전자소송 (ecfs.scourt.go.kr)
- 방문: 관할 법원 민원실

**비용:**
- 인지대: 청구금액의 0.5%
- 송달료: 약 5,000원

자세한 소장 작성 방법을 안내해드릴까요?
""",
}


def extract_amount(message: str) -> int | None:
    """메시지에서 금액 추출"""
    # 숫자 + 만원/원 패턴
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


def detect_dispute_type(message: str) -> str | None:
    """분쟁 유형 감지"""
    type_keywords = {
        "물품대금": ["물건", "물품", "상품", "대금"],
        "중고거래": ["중고", "당근", "번개", "거래", "사기"],
        "임대차": ["보증금", "월세", "전세", "임대", "집주인", "세입자"],
        "용역대금": ["용역", "서비스", "작업", "수리"],
        "대여금": ["빌려", "빌린", "대여", "꿔"],
    }

    for dispute_type, keywords in type_keywords.items():
        if any(kw in message for kw in keywords):
            return dispute_type
    return None


class SmallClaimsAgent(BaseAgent):
    """소액소송 에이전트"""

    @property
    def name(self) -> str:
        return "small_claims"

    @property
    def description(self) -> str:
        return "단계별 소액소송 가이드 및 서류 작성 지원"

    async def process(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        session_data: dict[str, Any] | None = None,
        user_location: dict[str, float] | None = None,
    ) -> AgentResponse:
        """소액소송 가이드 처리"""
        session_data = session_data or {}
        current_step = session_data.get("step", SmallClaimsStep.INIT)

        # 세션 데이터 복사
        new_session = {
            **session_data,
            "active_agent": self.name,
        }

        # 분쟁 유형 감지
        dispute_type = detect_dispute_type(message)
        if dispute_type:
            new_session["dispute_type"] = dispute_type

        # 금액 추출
        amount = extract_amount(message)
        if amount:
            new_session["claim_amount"] = amount

        # 관련 판례 검색 (컨텍스트 제공용)
        related_docs = []
        if dispute_type or amount:
            search_query = f"{dispute_type or ''} 소액소송 손해배상"
            try:
                related_docs = search_relevant_documents(search_query, n_results=3)
            except Exception as e:
                logger.warning(f"관련 문서 검색 실패: {e}")

        # 단계별 처리
        if current_step == SmallClaimsStep.INIT:
            # 초기 단계 - 분쟁 유형 파악
            if dispute_type:
                new_session["step"] = SmallClaimsStep.GATHER_INFO
                response = f"**{dispute_type}** 관련 분쟁이시군요.\n\n"
                response += STEP_MESSAGES[SmallClaimsStep.GATHER_INFO]
            else:
                response = STEP_MESSAGES[SmallClaimsStep.INIT]
                new_session["step"] = SmallClaimsStep.INIT

            actions = [
                ChatAction(
                    type=ActionType.BUTTON,
                    label="물품 대금",
                    action="dispute_type_goods",
                ),
                ChatAction(
                    type=ActionType.BUTTON,
                    label="중고거래 사기",
                    action="dispute_type_fraud",
                ),
                ChatAction(
                    type=ActionType.BUTTON,
                    label="임대차 보증금",
                    action="dispute_type_deposit",
                ),
            ]

        elif current_step == SmallClaimsStep.GATHER_INFO:
            # 정보 수집 단계
            if amount:
                if amount > 30000000:
                    response = (
                        f"청구 금액이 **{amount:,}원**이시군요.\n\n"
                        "소액소송은 3,000만원 이하만 가능합니다. "
                        "금액이 이를 초과하면 일반 민사소송을 진행해야 합니다.\n\n"
                        "그래도 소액소송 범위 내에서 진행하시겠습니까?"
                    )
                else:
                    response = (
                        f"청구 금액: **{amount:,}원** (소액소송 가능)\n\n"
                        "다음은 증거 자료를 정리해야 합니다.\n\n"
                        + STEP_MESSAGES[SmallClaimsStep.EVIDENCE]
                    )
                    new_session["step"] = SmallClaimsStep.EVIDENCE
            else:
                response = STEP_MESSAGES[SmallClaimsStep.GATHER_INFO]

            actions = []

        elif current_step == SmallClaimsStep.EVIDENCE:
            # 증거 수집 단계
            response = (
                "증거 자료를 확인했습니다.\n\n"
                "다음 단계는 내용증명 발송입니다.\n\n"
                + STEP_MESSAGES[SmallClaimsStep.DEMAND_LETTER]
            )
            new_session["step"] = SmallClaimsStep.DEMAND_LETTER

            actions = [
                ChatAction(
                    type=ActionType.BUTTON,
                    label="내용증명 작성 도움",
                    action="draft_demand_letter",
                ),
                ChatAction(
                    type=ActionType.BUTTON,
                    label="바로 소송 진행",
                    action="skip_to_court",
                ),
            ]

        elif current_step == SmallClaimsStep.DEMAND_LETTER:
            # 내용증명 단계
            response = (
                "내용증명 발송 후 응답이 없거나 거부당하면, "
                "소송을 제기할 수 있습니다.\n\n"
                + STEP_MESSAGES[SmallClaimsStep.COURT]
            )
            new_session["step"] = SmallClaimsStep.COURT

            actions = [
                ChatAction(
                    type=ActionType.LINK,
                    label="전자소송 바로가기",
                    url="https://ecfs.scourt.go.kr",
                ),
                ChatAction(
                    type=ActionType.BUTTON,
                    label="소장 작성 도움",
                    action="draft_complaint",
                ),
            ]

        else:
            # 기본 응답
            response = STEP_MESSAGES[SmallClaimsStep.COURT]
            actions = [
                ChatAction(
                    type=ActionType.BUTTON,
                    label="처음부터 다시",
                    action="reset_session",
                ),
            ]

        # 관련 판례 정보 추가 (역할별 차등 표시용 상세 필드 포함)
        sources = []
        if related_docs:
            # 판례 상세 정보 조회
            source_ids = [
                doc.get("metadata", {}).get("doc_id")
                for doc in related_docs
                if doc.get("metadata", {}).get("doc_id")
            ]
            precedent_details = fetch_precedent_details(source_ids) if source_ids else {}

            for doc in related_docs:
                metadata = doc.get("metadata", {})
                source_id = metadata.get("doc_id", "")

                source_item = {
                    "case_name": metadata.get("case_name", ""),
                    "case_number": metadata.get("case_number", ""),
                    "doc_type": metadata.get("doc_type", ""),
                    "court_name": metadata.get("court_name", ""),
                    "similarity": round(doc.get("similarity", 0), 3),
                    "content": doc.get("content", ""),
                }

                # 판례 상세 정보 추가
                if source_id in precedent_details:
                    details = precedent_details[source_id]
                    source_item["ruling"] = details.get("ruling", "")
                    source_item["claim"] = details.get("claim", "")
                    source_item["reasoning"] = details.get("reasoning", "")
                    source_item["full_reason"] = details.get("full_reason", "")

                sources.append(source_item)

        return AgentResponse(
            message=response,
            sources=sources,
            actions=actions,
            session_data=new_session,
        )

    def can_handle(self, message: str) -> bool:
        """소액소송 관련 키워드 확인"""
        keywords = ["소액소송", "내용증명", "지급명령", "사기", "환불", "손해배상"]
        return any(kw in message for kw in keywords)
