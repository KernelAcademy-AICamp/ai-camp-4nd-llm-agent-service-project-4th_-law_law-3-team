"""변호사 통계 모듈 - Pydantic 스키마"""

from typing import Literal

from pydantic import BaseModel


class StatusCount(BaseModel):
    """상태별 변호사 수"""

    status: str
    count: int


class OverviewResponse(BaseModel):
    """전체 현황 요약 응답"""

    total_lawyers: int
    status_counts: list[StatusCount]
    coord_rate: float
    specialty_rate: float


class RegionStat(BaseModel):
    """지역별 변호사 수"""

    region: str
    count: int


class RegionStatResponse(BaseModel):
    """지역별 통계 응답"""

    data: list[RegionStat]


class DensityStat(BaseModel):
    """지역별 변호사 밀도"""

    region: str
    count: int
    population: int
    density: float  # 인구 10만명당 변호사 수
    density_current: float | None = None  # 예측 모드에서 현재 기준 밀도
    change_percent: float | None = None  # 예측 모드에서 현재 대비 변화율


class DensityStatResponse(BaseModel):
    """지역별 밀도 통계 응답"""

    data: list[DensityStat]


class SpecialtyDetail(BaseModel):
    """세부 전문분야 카운트"""

    name: str
    count: int


class SpecialtyStat(BaseModel):
    """전문분야별 변호사 수"""

    category_id: str
    category_name: str
    count: int
    specialties: list[SpecialtyDetail]


class SpecialtyStatResponse(BaseModel):
    """전문분야별 통계 응답"""

    data: list[SpecialtyStat]


class CrossAnalysisCell(BaseModel):
    """교차 분석 셀"""

    region: str
    category_id: str
    category_name: str
    count: int


class CrossAnalysisResponse(BaseModel):
    """교차 분석 응답"""

    data: list[CrossAnalysisCell]
    regions: list[str]
    categories: list[str]


class CrossAnalysisRequest(BaseModel):
    """교차 분석 요청 (선택된 지역 목록)"""

    regions: list[str]


# =============================================================================
# 수요 통계 스키마
# =============================================================================
class DemandStat(BaseModel):
    """지역별 사건 수요 통계"""

    region: str
    case_count: int
    lawyer_count: int
    burden_index: float  # case_count / lawyer_count
    court_name: str  # 관할법원명


class DemandStatResponse(BaseModel):
    """수요 통계 응답"""

    data: list[DemandStat]
    year: int
    category: str
    available_years: list[int]
    available_categories: list[str]


# =============================================================================
# 에이전트 의도 분석 스키마
# =============================================================================
QueryType = Literal[
    "recommend_market",
    "recommend_specialty",
    "recommend_region",
    "overview",
    "region",
    "density",
    "specialty",
    "cross",
    "demand",
    "prediction",
]


class StatsIntent(BaseModel):
    """LLM이 파싱한 통계 요청 의도"""

    query_type: QueryType
    regions: list[str] = []
    province: str | None = None
    specialty_interest: str | None = None
    view_mode: str | None = None
    indicator_group: str | None = None
    active_tab: str | None = None
    prediction_year: int | None = None
    demand_category: str | None = None
    demand_year: int | None = None
