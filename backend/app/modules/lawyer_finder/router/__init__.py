"""
변호사 찾기 모듈 - API 라우터
위치 기반 변호사 검색, 클러스터링, 상세 조회
"""
from fastapi import APIRouter, Query, HTTPException
from typing import Optional

from ..schema import (
    LawyerResponse,
    NearbySearchResponse,
    ClusterResponse,
)
from ..service import (
    find_nearby_lawyers,
    get_lawyer_by_id,
    search_lawyers,
    get_clusters,
    get_zoom_grid_size,
    load_lawyers_data,
    get_categories,
)

router = APIRouter()


@router.get("/categories")
async def get_specialty_categories():
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
    limit: Optional[int] = Query(None, ge=1, description="최대 결과 수 (미지정 시 전체)"),
    category: Optional[str] = Query(None, description="전문분야 카테고리 ID"),
    specialty: Optional[str] = Query(None, description="특정 전문분야 (예: 이혼, 형사법)"),
):
    """
    Search for lawyers near a geographic point using the provided location and filters.
    
    Parameters:
        latitude (float): Latitude of the search center in degrees (-90 to 90).
        longitude (float): Longitude of the search center in degrees (-180 to 180).
        radius (int): Search radius in meters.
        limit (Optional[int]): Maximum number of results to return; when None, return all matching results.
        category (Optional[str]): Specialty category ID to filter results.
        specialty (Optional[str]): Specific specialty keyword to filter results; when provided, it takes precedence over `category`.
    
    Returns:
        dict: A response object containing:
            - "lawyers" (list): List of matching lawyer records.
            - "total_count" (int): Number of lawyers in the "lawyers" list.
            - "center" (dict): Center coordinates with keys "lat" and "lng".
            - "radius" (int): The search radius in meters.
    """
    lawyers = find_nearby_lawyers(
        latitude=latitude,
        longitude=longitude,
        radius_m=radius,
        limit=limit,
        category=category,
        specialty=specialty
    )

    return {
        "lawyers": lawyers,
        "total_count": len(lawyers),
        "center": {"lat": latitude, "lng": longitude},
        "radius": radius,
    }


@router.get("/clusters", response_model=ClusterResponse)
async def get_lawyer_clusters(
    min_lat: float = Query(..., description="최소 위도"),
    max_lat: float = Query(..., description="최대 위도"),
    min_lng: float = Query(..., description="최소 경도"),
    max_lng: float = Query(..., description="최대 경도"),
    zoom: int = Query(10, ge=1, le=21, description="줌 레벨"),
):
    """
    줌 레벨에 따른 클러스터 데이터 반환

    지도 뷰포트 내의 변호사들을 그리드로 묶어 클러스터 정보 제공
    """
    grid_size = get_zoom_grid_size(zoom)
    clusters = get_clusters(min_lat, max_lat, min_lng, max_lng, grid_size)

    total_count = sum(c["count"] for c in clusters)

    return {
        "clusters": clusters,
        "zoom_level": zoom,
        "total_count": total_count,
    }


@router.get("/search")
async def search_lawyers_endpoint(
    name: Optional[str] = Query(None, description="변호사 이름"),
    office: Optional[str] = Query(None, description="사무소명"),
    district: Optional[str] = Query(None, description="지역 (구/군)"),
    category: Optional[str] = Query(None, description="전문분야 카테고리 ID"),
    specialty: Optional[str] = Query(None, description="특정 전문분야 (예: 이혼, 형사법)"),
    latitude: Optional[float] = Query(None, ge=-90, le=90, description="위치 필터 - 위도"),
    longitude: Optional[float] = Query(None, ge=-180, le=180, description="위치 필터 - 경도"),
    radius: int = Query(5000, ge=100, le=50000, description="위치 필터 - 반경 (미터)"),
    limit: Optional[int] = Query(None, ge=1, description="최대 결과 수 (미지정 시 전체)"),
):
    """
    Searches lawyers by name, office, district, category, or specialty with optional location filtering.
    
    Parameters:
        name (Optional[str]): Substring to match in lawyer names.
        office (Optional[str]): Substring to match in office names.
        district (Optional[str]): District (e.g., "Gangnam-gu") to filter by address.
        category (Optional[str]): Specialty category ID.
        specialty (Optional[str]): Specific specialty keyword (e.g., "divorce", "criminal"); if provided, it takes precedence over `category`.
        latitude (Optional[float]): Latitude for location filtering; used together with `longitude`.
        longitude (Optional[float]): Longitude for location filtering; used together with `latitude`.
        radius (int): Search radius in meters (default 5000).
        limit (Optional[int]): Maximum number of results to return; if None, returns all matching results.
    
    Returns:
        dict: {
            "lawyers": list of matching lawyer records,
            "total_count": int number of returned lawyers
        }
    
    Raises:
        HTTPException: If no search criteria are provided (HTTP 400) or if input validation fails and `search_lawyers` raises a ValueError (converted to HTTP 400).
    """
    if not any([name, office, district, category, specialty]):
        raise HTTPException(
            status_code=400,
            detail="최소 하나의 검색 조건이 필요합니다 (name, office, district, category, specialty)"
        )

    try:
        lawyers = search_lawyers(
            name=name,
            office=office,
            district=district,
            category=category,
            specialty=specialty,
            latitude=latitude,
            longitude=longitude,
            radius_m=radius,
            limit=limit
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "lawyers": lawyers,
        "total_count": len(lawyers),
    }


@router.get("/stats")
async def get_stats():
    """데이터 통계 조회"""
    data = load_lawyers_data()
    lawyers = data.get("lawyers", [])
    metadata = data.get("metadata", {})

    # 좌표가 있는 변호사 수
    with_coords = sum(1 for l in lawyers if l.get("latitude") is not None)

    # 전문분야가 있는 변호사 수
    with_specialties = sum(
        1 for l in lawyers
        if l.get("specialties") and len(l.get("specialties", [])) > 0
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
async def get_lawyer_detail(lawyer_id: int):
    """
    변호사 상세 정보 조회

    - **lawyer_id**: 변호사 ID (인덱스)
    """
    lawyer = get_lawyer_by_id(lawyer_id)

    if not lawyer:
        raise HTTPException(status_code=404, detail="변호사를 찾을 수 없습니다")

    return lawyer