"""변호사 통계 모듈 - Pydantic 스키마"""

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
    density_2024: float | None = None  # 예측 모드에서 2024년 기준 밀도
    change_percent: float | None = None  # 예측 모드에서 2024년 대비 변화율


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
