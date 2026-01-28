"""변호사 통계 모듈 - API 라우터"""

from fastapi import APIRouter

from app.modules.lawyer_stat.schema import (
    CrossAnalysisResponse,
    OverviewResponse,
    RegionStat,
    RegionStatResponse,
    SpecialtyStat,
    SpecialtyStatResponse,
)
from app.modules.lawyer_stat.service import (
    calculate_by_region,
    calculate_by_specialty,
    calculate_cross_analysis,
    calculate_overview,
)

router = APIRouter()


@router.get("/overview", response_model=OverviewResponse)
async def get_overview() -> OverviewResponse:
    """전체 현황 요약 조회."""
    data = calculate_overview()
    return OverviewResponse(**data)


@router.get("/by-region", response_model=RegionStatResponse)
async def get_by_region() -> RegionStatResponse:
    """지역별 변호사 수 조회."""
    data = calculate_by_region()
    return RegionStatResponse(data=[RegionStat(**item) for item in data])


@router.get("/by-specialty", response_model=SpecialtyStatResponse)
async def get_by_specialty() -> SpecialtyStatResponse:
    """전문분야(12대분류)별 변호사 수 조회."""
    data = calculate_by_specialty()
    return SpecialtyStatResponse(data=[SpecialtyStat(**item) for item in data])


@router.get("/cross-analysis", response_model=CrossAnalysisResponse)
async def get_cross_analysis() -> CrossAnalysisResponse:
    """지역 × 전문분야 교차 분석 조회."""
    data = calculate_cross_analysis()
    return CrossAnalysisResponse(**data)
