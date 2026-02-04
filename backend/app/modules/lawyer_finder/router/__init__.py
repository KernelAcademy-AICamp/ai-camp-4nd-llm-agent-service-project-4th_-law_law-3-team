"""
변호사 찾기 모듈 - API 라우터
위치 기반 변호사 검색, 클러스터링, 상세 조회
"""
import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.services.service_function.lawyer_service import (
    find_nearby_lawyers,
    get_categories,
    get_clusters,
    get_lawyer_by_id,
    get_zoom_grid_size,
    load_lawyers_data,
    search_lawyers,
)

from ..schema import (
    ClusterItem,
    ClusterResponse,
    LawyerResponse,
    NearbySearchResponse,
    SearchResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/categories")
async def get_specialty_categories() -> dict[str, Any]:
    """
    전문분야 12대분류 목록 조회

    사용자에게 표시할 12개의 전문분야 대분류를 반환합니다.
    """
    categories = get_categories()
    return {
        "categories": categories,
        "count": len(categories),
    }


@router.get("/nearby", response_model=NearbySearchResponse)
async def get_nearby_lawyers(
    latitude: float = Query(..., ge=-90, le=90, description="위도"),
    longitude: float = Query(..., ge=-180, le=180, description="경도"),
    radius: int = Query(5000, ge=100, le=50000, description="반경 (미터)"),
    limit: int = Query(500, ge=1, le=5000, description="최대 결과 수 (기본 500)"),
    category: Optional[str] = Query(None, description="전문분야 카테고리 ID"),
    specialty: Optional[str] = Query(None, description="특정 전문분야 (예: 이혼, 형사법)"),
    db: AsyncSession = Depends(get_db),
) -> NearbySearchResponse:
    """
    사용자 위치 기반 주변 변호사 검색

    - **latitude**: 위도 (-90 ~ 90)
    - **longitude**: 경도 (-180 ~ 180)
    - **radius**: 검색 반경 (미터, 기본 5km)
    - **limit**: 최대 결과 수 (기본 500, 최대 5000)
    - **category**: 전문분야 카테고리 ID (예: "criminal", "civil-family")
    - **specialty**: 특정 전문분야 키워드 (예: "이혼", "형사법") - category보다 우선 적용
    """
    if settings.USE_DB_LAWYERS:
        from app.services.service_function.lawyer_db_service import (
            find_nearby_lawyers_db,
        )
        result = await find_nearby_lawyers_db(
            db=db,
            latitude=latitude,
            longitude=longitude,
            radius_m=radius,
            limit=limit,
            category=category,
            specialty=specialty,
        )
        lawyers_data = result["lawyers"]
        total_count = result["total_count"]
    else:
        result = find_nearby_lawyers(
            latitude=latitude,
            longitude=longitude,
            radius_m=radius,
            limit=limit,
            category=category,
            specialty=specialty,
        )
        lawyers_data = result["lawyers"]
        total_count = result["total_count"]

    lawyers = [LawyerResponse(**lawyer) for lawyer in lawyers_data]
    return NearbySearchResponse(
        lawyers=lawyers,
        total_count=total_count,
        center={"lat": latitude, "lng": longitude},
        radius=radius,
    )


@router.get("/clusters", response_model=ClusterResponse)
async def get_lawyer_clusters(
    min_lat: float = Query(..., description="최소 위도"),
    max_lat: float = Query(..., description="최대 위도"),
    min_lng: float = Query(..., description="최소 경도"),
    max_lng: float = Query(..., description="최대 경도"),
    zoom: int = Query(10, ge=1, le=21, description="줌 레벨"),
    category: Optional[str] = Query(None, description="전문분야 카테고리 ID"),
    specialty: Optional[str] = Query(None, description="특정 전문분야 (예: 이혼, 형사법)"),
    db: AsyncSession = Depends(get_db),
) -> ClusterResponse:
    """
    줌 레벨에 따른 클러스터 데이터 반환

    지도 뷰포트 내의 변호사들을 그리드로 묶어 클러스터 정보 제공
    """
    grid_size = get_zoom_grid_size(zoom)

    if settings.USE_DB_LAWYERS:
        from app.services.service_function.lawyer_db_service import get_clusters_db
        clusters_data = await get_clusters_db(
            db, min_lat, max_lat, min_lng, max_lng, grid_size,
            category=category, specialty=specialty,
        )
    else:
        clusters_data = get_clusters(
            min_lat, max_lat, min_lng, max_lng, grid_size,
            category=category, specialty=specialty,
        )

    clusters = [ClusterItem(**c) for c in clusters_data]
    total_count = sum(c.count for c in clusters)

    return ClusterResponse(
        clusters=clusters,
        zoom_level=zoom,
        total_count=total_count,
    )


@router.get("/search", response_model=SearchResponse)
async def search_lawyers_endpoint(
    name: Optional[str] = Query(None, description="변호사 이름"),
    office: Optional[str] = Query(None, description="사무소명"),
    district: Optional[str] = Query(None, description="지역 (구/군)"),
    category: Optional[str] = Query(None, description="전문분야 카테고리 ID"),
    specialty: Optional[str] = Query(None, description="특정 전문분야 (예: 이혼, 형사법)"),
    latitude: Optional[float] = Query(None, ge=-90, le=90, description="위치 필터 - 위도"),
    longitude: Optional[float] = Query(None, ge=-180, le=180, description="위치 필터 - 경도"),
    radius: int = Query(5000, ge=100, le=50000, description="위치 필터 - 반경 (미터)"),
    limit: int = Query(500, ge=1, le=5000, description="최대 결과 수 (기본 500)"),
    db: AsyncSession = Depends(get_db),
) -> SearchResponse:
    """
    변호사 검색 (이름/사무소/지역/전문분야 + 선택적 위치 필터)

    - **name**: 이름에 포함된 문자열
    - **office**: 사무소명에 포함된 문자열
    - **district**: 주소에 포함된 구/군 (예: "강남구", "송파구")
    - **category**: 전문분야 카테고리 ID (예: "criminal", "civil-family")
    - **specialty**: 특정 전문분야 키워드 (예: "이혼", "형사법") - category보다 우선 적용
    - **latitude**: 위치 필터 - 위도 (선택, longitude와 함께 사용)
    - **longitude**: 위치 필터 - 경도 (선택, latitude와 함께 사용)
    - **radius**: 위치 필터 - 반경 (미터, 기본 5km)
    - **limit**: 최대 결과 수 (기본 500, 최대 5000)
    """
    if not any([name, office, district, category, specialty]):
        raise HTTPException(
            status_code=400,
            detail="최소 하나의 검색 조건이 필요합니다 (name, office, district, category, specialty)"
        )

    try:
        if settings.USE_DB_LAWYERS:
            from app.services.service_function.lawyer_db_service import (
                search_lawyers_db,
            )
            result = await search_lawyers_db(
                db=db,
                name=name,
                office=office,
                district=district,
                category=category,
                specialty=specialty,
                latitude=latitude,
                longitude=longitude,
                radius_m=radius,
                limit=limit,
            )
            lawyers_data = result["lawyers"]
            total_count = result["total_count"]
        else:
            result = search_lawyers(
                name=name,
                office=office,
                district=district,
                category=category,
                specialty=specialty,
                latitude=latitude,
                longitude=longitude,
                radius_m=radius,
                limit=limit,
            )
            lawyers_data = result["lawyers"]
            total_count = result["total_count"]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    validated = [LawyerResponse(**lawyer) for lawyer in lawyers_data]
    return SearchResponse(lawyers=validated, total_count=total_count)


@router.get("/stats")
async def get_stats(db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """데이터 통계 조회"""
    if settings.USE_DB_LAWYERS:
        from app.services.service_function.lawyer_db_service import get_lawyer_stats_db
        return await get_lawyer_stats_db(db)

    data = load_lawyers_data()
    lawyers = data.get("lawyers", [])
    metadata = data.get("metadata", {})

    # 좌표가 있는 변호사 수
    with_coords = sum(1 for lawyer in lawyers if lawyer.get("latitude") is not None)

    # 전문분야가 있는 변호사 수
    with_specialties = sum(
        1 for lawyer in lawyers
        if lawyer.get("specialties") and len(lawyer.get("specialties", [])) > 0
    )

    return {
        "total_lawyers": len(lawyers),
        "with_coordinates": with_coords,
        "without_coordinates": len(lawyers) - with_coords,
        "with_specialties": with_specialties,
        "source": metadata.get("source"),
        "crawled_at": metadata.get("crawled_at"),
        "geocoded_at": metadata.get("geocoded_at"),
        "specialty_crawl_date": metadata.get("specialty_crawl_date"),
    }


@router.get("/{lawyer_id}", response_model=LawyerResponse)
async def get_lawyer_detail(
    lawyer_id: int,
    db: AsyncSession = Depends(get_db),
) -> LawyerResponse:
    """
    변호사 상세 정보 조회

    - **lawyer_id**: 변호사 ID (인덱스)
    """
    if settings.USE_DB_LAWYERS:
        from app.services.service_function.lawyer_db_service import get_lawyer_by_id_db
        lawyer = await get_lawyer_by_id_db(db, lawyer_id)
    else:
        lawyer = get_lawyer_by_id(lawyer_id)

    if not lawyer:
        raise HTTPException(status_code=404, detail="변호사를 찾을 수 없습니다")

    return LawyerResponse(**lawyer)
