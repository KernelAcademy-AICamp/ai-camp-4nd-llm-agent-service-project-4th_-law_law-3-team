"""스토리보드 타임라인 추출 평가 스키마"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DatePrecision(str, Enum):
    """날짜 정밀도"""

    EXACT = "exact"  # 2012-01-05
    MONTH = "month"  # 2012-01
    YEAR = "year"  # 2012
    APPROXIMATE = "approximate"  # 1995년경, 2000년 여름
    RELATIVE = "relative"  # 그 뒤에도, 며칠 후


class NarrativeOrder(str, Enum):
    """서술 순서와 시간 순서의 관계"""

    CHRONOLOGICAL = "chronological"  # 서술 순서 == 시간 순서
    NON_CHRONOLOGICAL = "non_chronological"  # 서술 순서 != 시간 순서


class TimelineEventGT(BaseModel):
    """타임라인 이벤트 정답 (Ground Truth)"""

    event_id: str = Field(..., description="이벤트 ID (E01, E02, ...)")
    date_raw: str = Field(..., description="원본 날짜 표현 (예: '2012. 1. 5.')")
    date_normalized: str | None = Field(
        None, description="정규화된 날짜 (예: '2012-01-05'), None이면 불명확"
    )
    date_precision: DatePrecision = Field(..., description="날짜 정밀도")
    title: str = Field(..., description="이벤트 제목")
    key_participants: list[str] = Field(
        default_factory=list, description="핵심 참여자 이름 목록"
    )
    participant_roles: dict[str, str] = Field(
        default_factory=dict,
        description="참여자별 역할 매핑 (예: {'박대원': 'other', '신영수': 'authority'})",
    )
    location_hint: str | None = Field(None, description="장소 힌트")
    legal_significance_keywords: list[str] = Field(
        default_factory=list, description="법적 의미 키워드"
    )
    is_key_event: bool = Field(True, description="핵심 이벤트 여부")


class TimelineGroundTruth(BaseModel):
    """타임라인 정답 데이터"""

    expected_event_count: int = Field(..., description="예상 이벤트 수")
    chronological_order: list[str] = Field(
        ..., description="시간순 정렬된 event_id 목록"
    )
    events: list[TimelineEventGT] = Field(..., description="이벤트 목록")
    key_participants: list[str] = Field(
        default_factory=list, description="전체 핵심 참여자 목록"
    )
    narrative_vs_chronological: NarrativeOrder = Field(
        ..., description="서술 순서와 시간 순서의 관계"
    )


class TimelineEvalCase(BaseModel):
    """타임라인 평가 개별 케이스"""

    id: str = Field(..., description="케이스 ID (예: CIVIL-001)")
    source_file: str = Field(..., description="원본 파일명 (예: [CIVIL]1.md)")
    doc_type: str = Field(..., description="문서 유형 (civil, criminal, public)")
    input_text: str = Field(..., description="핵심 섹션 발췌 텍스트")
    ground_truth: TimelineGroundTruth = Field(..., description="정답 데이터")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="부가 메타데이터 (difficulty, participant_count 등)",
    )


class TimelineEvalDataset(BaseModel):
    """타임라인 평가 데이터셋"""

    version: str = Field(..., description="데이터셋 버전")
    name: str = Field(..., description="데이터셋 이름")
    cases: list[TimelineEvalCase] = Field(..., description="평가 케이스 목록")


class TimelineCaseResult(BaseModel):
    """개별 케이스 평가 결과"""

    case_id: str = Field(..., description="케이스 ID")
    event_count_accuracy: float = Field(..., description="이벤트 수 정확도")
    chronological_order_score: float = Field(..., description="시간 순서 점수")
    participant_recall: float = Field(..., description="참여자 재현율")
    role_accuracy: float = Field(..., description="역할 정확도")
    key_event_coverage: float = Field(..., description="핵심 이벤트 포함율")
    composite_score: float = Field(..., description="종합 점수")


class TimelineEvalResult(BaseModel):
    """전체 평가 결과"""

    dataset_name: str = Field(..., description="데이터셋 이름")
    model: str = Field(..., description="사용 모델명")
    prompt_version: str = Field(..., description="프롬프트 버전")
    case_results: list[TimelineCaseResult] = Field(
        ..., description="케이스별 결과"
    )
    aggregate_by_doc_type: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="문서 유형별 집계"
    )
    overall: dict[str, float] = Field(
        default_factory=dict, description="전체 평균 메트릭"
    )
