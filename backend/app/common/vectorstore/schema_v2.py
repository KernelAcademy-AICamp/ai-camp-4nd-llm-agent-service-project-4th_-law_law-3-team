"""
LanceDB 스키마 v2 - 단일 테이블 + NULL 허용 (메모리 최적화)

설계 원칙:
- 모든 필드를 개별 컬럼으로 정의
- 해당하지 않는 필드는 NULL
- data_type으로 문서 유형 구분
- ruling, claim, reasoning은 PostgreSQL에 저장하여 메모리 효율화

문서 타입:
- data_type = "법령": 법률, 시행령, 시행규칙 등
- data_type = "판례": 대법원, 고등법원, 지방법원 판결

컬럼 그룹 (총 20개):
- 공통: 10개 (id, source_id, data_type, title, content, vector, date, source_name, chunk_index, total_chunks)
- 법령 전용: 4개 (promulgation_date, promulgation_no, law_type, article_no)
- 판례 전용: 6개 (case_number, case_type, judgment_type, judgment_status, reference_provisions, reference_cases)

검색 흐름:
1. LanceDB 벡터 검색 → source_id 추출
2. PostgreSQL에서 원본 조회 (ruling, claim, reasoning 등)
"""

import pyarrow as pa
from pydantic import BaseModel
from typing import Optional


# =============================================================================
# 상수
# =============================================================================

VECTOR_DIM = 1024  # 임베딩 차원
TABLE_NAME = "legal_chunks"


# =============================================================================
# PyArrow 스키마 (LanceDB 테이블 생성용)
# =============================================================================

LEGAL_CHUNKS_SCHEMA = pa.schema([
    # ========== 공통 필드 ==========
    pa.field("id", pa.utf8()),              # 청크 고유 ID (예: "010719_0")
    pa.field("source_id", pa.utf8()),       # 원본 문서 ID (예: "010719")
    pa.field("data_type", pa.utf8()),       # "법령" | "판례"
    pa.field("title", pa.utf8()),           # 제목 (법령명 / 사건명)
    pa.field("content", pa.utf8()),         # 청크 텍스트 (prefix 포함)
    pa.field("vector", pa.list_(pa.float32(), VECTOR_DIM)),  # 임베딩 벡터
    pa.field("date", pa.utf8()),            # 날짜 (법령: 시행일, 판례: 선고일)
    pa.field("source_name", pa.utf8()),     # 출처 (법령: 소관부처, 판례: 법원명)
    pa.field("chunk_index", pa.int32()),    # 청크 인덱스
    pa.field("total_chunks", pa.int32()),   # 해당 문서의 총 청크 수

    # ========== 법령 전용 (판례는 NULL) ==========
    pa.field("promulgation_date", pa.utf8()),   # 공포일자 (예: "20230808")
    pa.field("promulgation_no", pa.utf8()),     # 공포번호 (예: "19592")
    pa.field("law_type", pa.utf8()),            # 법령 유형 (법률/시행령/시행규칙)
    pa.field("article_no", pa.utf8()),          # 조문 번호 (예: "제750조")

    # ========== 판례 전용 (법령은 NULL) ==========
    # NOTE: ruling, claim, reasoning은 PostgreSQL precedent_documents 테이블에서 조회
    # LanceDB에는 검색용 메타데이터만 저장하여 메모리 효율화
    pa.field("case_number", pa.utf8()),         # 사건번호 (예: "84나3990")
    pa.field("case_type", pa.utf8()),           # 사건 유형 (민사/형사/행정)
    pa.field("judgment_type", pa.utf8()),       # 판결 법원부 (예: "제11민사부판결")
    pa.field("judgment_status", pa.utf8()),     # 판결 상태 (확정/미확정)
    pa.field("reference_provisions", pa.utf8()),# 참조 조문 (예: "민법 제750조, 제756조")
    pa.field("reference_cases", pa.utf8()),     # 참조 판례
])


# =============================================================================
# 컬럼 그룹 정의 (문서화 및 유효성 검사용)
# =============================================================================

COMMON_COLUMNS = [
    "id", "source_id", "data_type", "title", "content",
    "vector", "date", "source_name", "chunk_index", "total_chunks"
]

LAW_COLUMNS = [
    "promulgation_date", "promulgation_no", "law_type", "article_no"
]

PRECEDENT_COLUMNS = [
    "case_number", "case_type", "judgment_type",
    "judgment_status", "reference_provisions", "reference_cases"
]

ALL_COLUMNS = COMMON_COLUMNS + LAW_COLUMNS + PRECEDENT_COLUMNS


# =============================================================================
# Pydantic 모델 (데이터 검증 및 타입 힌트)
# =============================================================================

class LegalChunk(BaseModel):
    """법률 청크 모델 (법령 + 판례 통합)"""

    # === 공통 필드 ===
    id: str                                 # 청크 고유 ID
    source_id: str                          # 원본 문서 ID
    data_type: str                          # "법령" | "판례"
    title: str                              # 제목
    content: str                            # 청크 텍스트
    vector: list[float]                     # 임베딩 벡터
    date: str                               # 날짜
    source_name: str                        # 출처
    chunk_index: int = 0                    # 청크 인덱스
    total_chunks: int = 1                   # 총 청크 수

    # === 법령 전용 (판례는 None) ===
    promulgation_date: Optional[str] = None # 공포일자
    promulgation_no: Optional[str] = None   # 공포번호
    law_type: Optional[str] = None          # 법령 유형
    article_no: Optional[str] = None        # 조문 번호

    # === 판례 전용 (법령은 None) ===
    # NOTE: ruling, claim, reasoning은 PostgreSQL precedent_documents 테이블에서 조회
    case_number: Optional[str] = None       # 사건번호
    case_type: Optional[str] = None         # 사건 유형
    judgment_type: Optional[str] = None     # 판결 법원부
    judgment_status: Optional[str] = None   # 판결 상태
    reference_provisions: Optional[str] = None  # 참조 조문
    reference_cases: Optional[str] = None   # 참조 판례

    def to_dict(self) -> dict:
        """LanceDB 삽입용 딕셔너리 변환"""
        return self.model_dump()

    def validate_by_type(self) -> bool:
        """data_type에 따른 필드 유효성 검사"""
        if self.data_type == "법령":
            # 판례 필드는 None이어야 함
            precedent_fields = [
                self.case_number, self.case_type, self.judgment_type,
                self.judgment_status, self.reference_provisions, self.reference_cases
            ]
            if any(f is not None for f in precedent_fields):
                raise ValueError("법령 데이터에 판례 필드가 설정되어 있습니다.")

        elif self.data_type == "판례":
            # 법령 필드는 None이어야 함
            law_fields = [
                self.promulgation_date, self.promulgation_no,
                self.law_type, self.article_no
            ]
            if any(f is not None for f in law_fields):
                raise ValueError("판례 데이터에 법령 필드가 설정되어 있습니다.")

        return True


# =============================================================================
# 헬퍼 함수
# =============================================================================

def create_law_chunk(
    source_id: str,
    chunk_index: int,
    title: str,
    content: str,
    vector: list[float],
    enforcement_date: str,
    department: str,
    total_chunks: int = 1,
    promulgation_date: str = None,
    promulgation_no: str = None,
    law_type: str = None,
    article_no: str = None,
) -> dict:
    """
    법령 청크 생성

    Args:
        source_id: 원본 문서 ID
        chunk_index: 청크 인덱스 (0부터 시작)
        title: 법령명
        content: 청크 텍스트 (prefix 포함 권장)
        vector: 임베딩 벡터
        enforcement_date: 시행일 (YYYY-MM-DD)
        department: 소관부처
        total_chunks: 해당 문서의 총 청크 수
        promulgation_date: 공포일자
        promulgation_no: 공포번호
        law_type: 법령 유형 (법률/시행령/시행규칙)
        article_no: 조문 번호

    Returns:
        LanceDB 삽입용 딕셔너리
    """
    return {
        # 공통
        "id": f"{source_id}_{chunk_index}",
        "source_id": source_id,
        "data_type": "법령",
        "title": title,
        "content": content,
        "vector": vector,
        "date": enforcement_date,
        "source_name": department,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        # 법령 전용
        "promulgation_date": promulgation_date,
        "promulgation_no": promulgation_no,
        "law_type": law_type,
        "article_no": article_no,
        # 판례 전용 (NULL)
        "case_number": None,
        "case_type": None,
        "judgment_type": None,
        "judgment_status": None,
        "reference_provisions": None,
        "reference_cases": None,
    }


def create_precedent_chunk(
    source_id: str,
    chunk_index: int,
    title: str,
    content: str,
    vector: list[float],
    decision_date: str,
    court_name: str,
    total_chunks: int = 1,
    case_number: str = None,
    case_type: str = None,
    judgment_type: str = None,
    judgment_status: str = None,
    reference_provisions: str = None,
    reference_cases: str = None,
) -> dict:
    """
    판례 청크 생성

    NOTE: ruling, claim, reasoning은 PostgreSQL precedent_documents 테이블에서 조회
          LanceDB에는 검색용 메타데이터만 저장하여 메모리 효율화

    Args:
        source_id: 원본 문서 ID (PostgreSQL precedent_documents.serial_number)
        chunk_index: 청크 인덱스 (0부터 시작)
        title: 사건명
        content: 청크 텍스트 (prefix 포함 권장)
        vector: 임베딩 벡터
        decision_date: 선고일 (YYYY-MM-DD)
        court_name: 법원명
        total_chunks: 해당 문서의 총 청크 수
        case_number: 사건번호
        case_type: 사건 유형 (민사/형사/행정)
        judgment_type: 판결 법원부
        judgment_status: 판결 상태 (확정/미확정)
        reference_provisions: 참조 조문
        reference_cases: 참조 판례

    Returns:
        LanceDB 삽입용 딕셔너리
    """
    return {
        # 공통
        "id": f"{source_id}_{chunk_index}",
        "source_id": source_id,
        "data_type": "판례",
        "title": title,
        "content": content,
        "vector": vector,
        "date": decision_date,
        "source_name": court_name,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        # 법령 전용 (NULL)
        "promulgation_date": None,
        "promulgation_no": None,
        "law_type": None,
        "article_no": None,
        # 판례 전용
        "case_number": case_number,
        "case_type": case_type,
        "judgment_type": judgment_type,
        "judgment_status": judgment_status,
        "reference_provisions": reference_provisions,
        "reference_cases": reference_cases,
    }


# =============================================================================
# 검색 헬퍼 함수
# =============================================================================

def get_law_columns() -> list[str]:
    """법령 조회 시 반환할 컬럼 목록"""
    return COMMON_COLUMNS + LAW_COLUMNS


def get_precedent_columns() -> list[str]:
    """판례 조회 시 반환할 컬럼 목록"""
    return COMMON_COLUMNS + PRECEDENT_COLUMNS


# =============================================================================
# 사용 예시
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("스키마 v2 테스트")
    print("=" * 60)

    # 스키마 출력
    print("\n[PyArrow 스키마]")
    for field in LEGAL_CHUNKS_SCHEMA:
        print(f"  {field.name}: {field.type}")

    # 법령 청크 생성 예시
    print("\n[법령 청크 예시]")
    law = create_law_chunk(
        source_id="010719",
        chunk_index=0,
        title="민법",
        content="[법령] 민법 제750조: 고의 또는 과실로 인한 위법행위로 타인에게 손해를 가한 자는 그 손해를 배상할 책임이 있다.",
        vector=[0.1] * VECTOR_DIM,
        enforcement_date="2023-08-08",
        department="법무부",
        promulgation_date="20230808",
        promulgation_no="19592",
        law_type="법률",
        article_no="제750조",
        total_chunks=1,
    )
    for k, v in law.items():
        if k != "vector":
            print(f"  {k}: {v}")

    # 판례 청크 생성 예시
    print("\n[판례 청크 예시]")
    precedent = create_precedent_chunk(
        source_id="84나3990",
        chunk_index=0,
        title="손해배상청구사건",
        content="[판례] 손해배상청구사건: 수련의에게 마취를 담당케 하여 의료사고가 발생한 경우의 책임",
        vector=[0.2] * VECTOR_DIM,
        decision_date="1986-01-15",
        court_name="서울고법",
        case_number="84나3990",
        case_type="민사",
        judgment_type="제11민사부판결",
        judgment_status="확정",
        reference_provisions="민법 제750조, 제756조",
        total_chunks=1,
    )
    for k, v in precedent.items():
        if k != "vector":
            print(f"  {k}: {v}")

    print("\n[컬럼 그룹]")
    print(f"  공통: {COMMON_COLUMNS}")
    print(f"  법령 전용: {LAW_COLUMNS}")
    print(f"  판례 전용: {PRECEDENT_COLUMNS}")
