"""
모의 법정 Pydantic 스키마

Design 문서 Section 3, 4.3 기반 요청/응답 모델 정의
"""

from typing import Any

from pydantic import BaseModel, Field

# ── 사건 유형 / 역할 관련 ──


class CaseCategory(BaseModel):
    """사건 세부 유형"""

    id: str
    name: str
    description: str


class CaseTypeInfo(BaseModel):
    """사건 유형 (형사/민사)"""

    id: str
    name: str
    categories: list[CaseCategory]


class CaseTypesResponse(BaseModel):
    """GET /api/mock-trial/case-types 응답"""

    case_types: list[CaseTypeInfo]


class RoleInfo(BaseModel):
    """선택 가능 역할"""

    id: str
    name: str
    description: str


class RolesResponse(BaseModel):
    """GET /api/mock-trial/roles/{case_type} 응답"""

    case_type: str
    roles: list[RoleInfo]


# ── 증거 검색 관련 ──


class EvidenceSearchRequest(BaseModel):
    """POST /api/mock-trial/search-evidence 요청"""

    query: str = Field(..., min_length=1, max_length=500)
    search_type: str = Field(default="all", pattern=r"^(all|cases|articles)$")
    limit: int = Field(default=5, ge=1, le=20)


class EvidenceCaseItem(BaseModel):
    """판례 검색 결과 항목"""

    id: str
    title: str
    summary: str
    relevance_score: float
    source: str


class EvidenceArticleItem(BaseModel):
    """법령 검색 결과 항목"""

    id: str
    title: str
    content: str
    relevance_score: float
    source: str


class EvidenceSearchResponse(BaseModel):
    """POST /api/mock-trial/search-evidence 응답"""

    cases: list[EvidenceCaseItem]
    articles: list[EvidenceArticleItem]


# ── 단계 정보 관련 ──


class StageInfo(BaseModel):
    """재판 단계 정보"""

    id: str
    name: str
    order: int
    legal_basis: str
    description: str
    user_action: str
    duration_hint: str


class StageInfoResponse(BaseModel):
    """GET /api/mock-trial/stage-info/{case_type} 응답"""

    case_type: str
    stages: list[StageInfo]


# ── 서브그래프 보조 타입 (TypedDict가 아닌 Pydantic 직렬화용) ──


class AgentProfileSchema(BaseModel):
    """에이전트 프로필"""

    name: str
    role: str
    personality: str = ""
    expertise: str = ""


class AgentMemorySchema(BaseModel):
    """에이전트 기억"""

    short_term: list[str] = Field(default_factory=list)
    long_term: list[str] = Field(default_factory=list)


class AgentStateSchema(BaseModel):
    """에이전트 전체 상태"""

    profile: AgentProfileSchema
    memory: AgentMemorySchema
    strategy: str = ""


class CourtRecordSchema(BaseModel):
    """서기 기록 엔트리"""

    stage: str
    speaker: str
    content: str
    timestamp: str


class MockTrialSetupRequest(BaseModel):
    """모의재판 설정 요청 (프론트엔드 → interrupt resume)"""

    case_type: str = Field(..., pattern=r"^(criminal|civil)$")
    case_category: str
    user_role: str
    case_summary: str = Field(..., min_length=1, max_length=500)


# ── 상수 데이터 ──


CASE_TYPES: list[dict[str, Any]] = [
    {
        "id": "criminal",
        "name": "형사 재판",
        "categories": [
            {
                "id": "criminal_assault",
                "name": "폭행/상해",
                "description": "폭행죄, 상해죄 등",
            },
            {
                "id": "criminal_fraud",
                "name": "사기",
                "description": "사기죄, 횡령죄 등",
            },
            {
                "id": "criminal_theft",
                "name": "절도",
                "description": "절도죄, 강도죄 등",
            },
            {
                "id": "criminal_embezzlement",
                "name": "횡령/배임",
                "description": "횡령죄, 배임죄 등",
            },
            {
                "id": "criminal_other",
                "name": "기타",
                "description": "기타 형사 사건",
            },
        ],
    },
    {
        "id": "civil",
        "name": "민사 재판",
        "categories": [
            {
                "id": "civil_damages",
                "name": "손해배상",
                "description": "불법행위, 채무불이행 등",
            },
            {
                "id": "civil_contract",
                "name": "계약 분쟁",
                "description": "계약 해제, 이행 청구 등",
            },
            {
                "id": "civil_property",
                "name": "부동산",
                "description": "임대차, 소유권 분쟁 등",
            },
            {
                "id": "civil_other",
                "name": "기타",
                "description": "기타 민사 사건",
            },
        ],
    },
]


CRIMINAL_ROLES: list[dict[str, str]] = [
    {
        "id": "prosecutor",
        "name": "검사",
        "description": "공소 유지 및 범죄 입증",
    },
    {
        "id": "attorney",
        "name": "변호사",
        "description": "피고인 방어 및 무죄/감형 논증",
    },
]

CIVIL_ROLES: list[dict[str, str]] = [
    {
        "id": "prosecutor",
        "name": "원고측 대리인",
        "description": "청구원인 입증 및 손해 논증",
    },
    {
        "id": "attorney",
        "name": "피고측 대리인",
        "description": "청구 기각 및 항변 제시",
    },
]


CRIMINAL_STAGES: list[dict[str, Any]] = [
    {
        "id": "identity",
        "name": "인정신문",
        "order": 1,
        "legal_basis": "형사소송법 §284",
        "description": "재판장이 피고인 인적사항 확인 및 진술거부권 고지",
        "user_action": "자동 진행 (관전)",
        "duration_hint": "1-2분",
    },
    {
        "id": "opening",
        "name": "모두진술",
        "order": 2,
        "legal_basis": "형사소송법 §285~§286",
        "description": "검사 공소사실 요지 진술, 피고인/변호인 의견 진술",
        "user_action": "역할에 따라 진술 입력",
        "duration_hint": "3-5분",
    },
    {
        "id": "evidence",
        "name": "증거조사",
        "order": 3,
        "legal_basis": "형사소송법 §290~§313",
        "description": "판례/법령 검색, 증거 제출",
        "user_action": "증거 선택/제출",
        "duration_hint": "5-10분",
    },
    {
        "id": "examination",
        "name": "피고인신문",
        "order": 4,
        "legal_basis": "형사소송법 §296-2",
        "description": "검사/변호인이 피고인에게 질문",
        "user_action": "질문 입력",
        "duration_hint": "3-5분",
    },
    {
        "id": "closing",
        "name": "최종변론",
        "order": 5,
        "legal_basis": "형사소송법 §302~§303",
        "description": "검사 구형, 변호인 최후변론, 피고인 최후진술",
        "user_action": "최후변론 입력",
        "duration_hint": "3-5분",
    },
    {
        "id": "verdict",
        "name": "판결선고",
        "order": 6,
        "legal_basis": "형사소송법 §318-4",
        "description": "AI 판사 판결문 낭독",
        "user_action": "관전",
        "duration_hint": "2-3분",
    },
]


CIVIL_STAGES: list[dict[str, Any]] = [
    {
        "id": "pretrial",
        "name": "변론준비",
        "order": 1,
        "legal_basis": "민사소송법 §258~§268",
        "description": "쟁점 정리, 증거 목록 확인",
        "user_action": "자동 진행",
        "duration_hint": "1-2분",
    },
    {
        "id": "claims",
        "name": "주장/답변",
        "order": 2,
        "legal_basis": "민사소송법 §256~§257",
        "description": "원고 청구원인, 피고 답변",
        "user_action": "역할에 따라 입력",
        "duration_hint": "3-5분",
    },
    {
        "id": "evidence",
        "name": "증거조사",
        "order": 3,
        "legal_basis": "민사소송법 §288~§344",
        "description": "판례/법령 검색, 서증 제출",
        "user_action": "증거 선택/제출",
        "duration_hint": "5-10분",
    },
    {
        "id": "argument",
        "name": "변론",
        "order": 4,
        "legal_basis": "민사소송법 §134~§148",
        "description": "양측 주장/반박 교환",
        "user_action": "주장 입력 (2-3 라운드)",
        "duration_hint": "5-10분",
    },
    {
        "id": "closing",
        "name": "변론종결",
        "order": 5,
        "legal_basis": "민사소송법 §200",
        "description": "양측 최종 주장 정리",
        "user_action": "최종 주장 입력",
        "duration_hint": "2-3분",
    },
    {
        "id": "verdict",
        "name": "판결선고",
        "order": 6,
        "legal_basis": "민사소송법 §206~§208",
        "description": "AI 판사 판결문 낭독",
        "user_action": "관전",
        "duration_hint": "2-3분",
    },
]
