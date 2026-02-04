"""변호사 통계 모듈 - API 라우터"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.modules.lawyer_stats.schema import (
    CrossAnalysisRequest,
    CrossAnalysisResponse,
    DensityStat,
    DensityStatResponse,
    OverviewResponse,
    RegionStat,
    RegionStatResponse,
    SpecialtyStat,
    SpecialtyStatResponse,
)
from app.services.service_function.lawyer_stats_service import (
    calculate_by_region,
    calculate_by_specialty,
    calculate_cross_analysis,
    calculate_cross_analysis_by_province,
    calculate_cross_analysis_by_regions,
    calculate_density_by_region,
    calculate_overview,
    calculate_specialty_by_region,
)

router = APIRouter()


@router.get("/overview", response_model=OverviewResponse)
async def get_overview(db: AsyncSession = Depends(get_db)) -> OverviewResponse:
    """전체 현황 요약 조회."""
    if settings.USE_DB_LAWYERS:
        from app.services.service_function.lawyer_stats_db_service import (
            calculate_overview_db,
        )
        data = await calculate_overview_db(db)
    else:
        data = calculate_overview()
    return OverviewResponse(**data)


@router.get("/by-region", response_model=RegionStatResponse)
async def get_by_region(db: AsyncSession = Depends(get_db)) -> RegionStatResponse:
    """지역별 변호사 수 조회."""
    if settings.USE_DB_LAWYERS:
        from app.services.service_function.lawyer_stats_db_service import (
            calculate_by_region_db,
        )
        data = await calculate_by_region_db(db)
    else:
        data = calculate_by_region()
    return RegionStatResponse(data=[RegionStat(**item) for item in data])


@router.get("/density-by-region", response_model=DensityStatResponse)
async def get_density_by_region(
    year: str = Query(
        default="current",
        description="인구 데이터 연도 ('current', '2030', '2035', '2040')",
    ),
    include_change: bool = Query(
        default=False,
        description="현재 대비 변화율 포함 여부",
    ),
    db: AsyncSession = Depends(get_db),
) -> DensityStatResponse:
    """지역별 인구 대비 변호사 밀도 조회."""
    if settings.USE_DB_LAWYERS:
        from app.services.service_function.lawyer_stats_db_service import (
            calculate_density_by_region_db,
        )
        data = await calculate_density_by_region_db(db, year=year, include_change=include_change)
    else:
        data = calculate_density_by_region(year=year, include_change=include_change)
    return DensityStatResponse(data=[DensityStat(**item) for item in data])


@router.get("/by-specialty", response_model=SpecialtyStatResponse)
async def get_by_specialty(db: AsyncSession = Depends(get_db)) -> SpecialtyStatResponse:
    """전문분야(12대분류)별 변호사 수 조회."""
    if settings.USE_DB_LAWYERS:
        from app.services.service_function.lawyer_stats_db_service import (
            calculate_by_specialty_db,
        )
        data = await calculate_by_specialty_db(db)
    else:
        data = calculate_by_specialty()
    return SpecialtyStatResponse(data=[SpecialtyStat(**item) for item in data])


@router.get("/cross-analysis", response_model=CrossAnalysisResponse)
async def get_cross_analysis(db: AsyncSession = Depends(get_db)) -> CrossAnalysisResponse:
    """지역 × 전문분야 교차 분석 조회."""
    if settings.USE_DB_LAWYERS:
        from app.services.service_function.lawyer_stats_db_service import (
            calculate_cross_analysis_db,
        )
        data = await calculate_cross_analysis_db(db)
    else:
        data = calculate_cross_analysis()
    return CrossAnalysisResponse(**data)


@router.get("/region/{region}/specialties", response_model=SpecialtyStatResponse)
async def get_region_specialties(
    region: str, db: AsyncSession = Depends(get_db),
) -> SpecialtyStatResponse:
    """특정 지역의 전문분야별 변호사 수 조회."""
    if settings.USE_DB_LAWYERS:
        from app.services.service_function.lawyer_stats_db_service import (
            calculate_specialty_by_region_db,
        )
        data = await calculate_specialty_by_region_db(db, region)
    else:
        data = calculate_specialty_by_region(region)
    return SpecialtyStatResponse(data=[SpecialtyStat(**item) for item in data])


@router.get("/cross-analysis/{province}", response_model=CrossAnalysisResponse)
async def get_cross_analysis_by_province(
    province: str, db: AsyncSession = Depends(get_db),
) -> CrossAnalysisResponse:
    """특정 시/도 내 지역 × 전문분야 교차 분석 조회."""
    if settings.USE_DB_LAWYERS:
        from app.services.service_function.lawyer_stats_db_service import (
            calculate_cross_analysis_by_province_db,
        )
        data = await calculate_cross_analysis_by_province_db(db, province)
    else:
        data = calculate_cross_analysis_by_province(province)
    return CrossAnalysisResponse(**data)


@router.post("/cross-analysis/regions", response_model=CrossAnalysisResponse)
async def get_cross_analysis_by_regions(
    request: CrossAnalysisRequest,
    db: AsyncSession = Depends(get_db),
) -> CrossAnalysisResponse:
    """선택된 지역 목록에 대한 교차 분석 조회."""
    if settings.USE_DB_LAWYERS:
        from app.services.service_function.lawyer_stats_db_service import (
            calculate_cross_analysis_by_regions_db,
        )
        data = await calculate_cross_analysis_by_regions_db(db, request.regions)
    else:
        data = calculate_cross_analysis_by_regions(request.regions)
    return CrossAnalysisResponse(**data)
