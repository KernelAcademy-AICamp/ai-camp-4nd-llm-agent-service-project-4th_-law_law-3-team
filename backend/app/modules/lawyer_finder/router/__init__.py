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
)

router = APIRouter()


@router.get("/nearby", response_model=NearbySearchResponse)
async def get_nearby_lawyers(
    latitude: float = Query(..., ge=-90, le=90, description="위도"),
    longitude: float = Query(..., ge=-180, le=180, description="경도"),
    radius: int = Query(5000, ge=100, le=50000, description="반경 (미터)"),
    limit: int = Query(50, ge=1, le=200, description="최대 결과 수"),
):
    """
    사용자 위치 기반 주변 변호사 검색

    - **latitude**: 위도 (-90 ~ 90)
    - **longitude**: 경도 (-180 ~ 180)
    - **radius**: 검색 반경 (미터, 기본 5km)
    - **limit**: 최대 결과 수 (기본 50)
    """
    lawyers = find_nearby_lawyers(latitude, longitude, radius, limit)

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
    latitude: Optional[float] = Query(None, ge=-90, le=90, description="위치 필터 - 위도"),
    longitude: Optional[float] = Query(None, ge=-180, le=180, description="위치 필터 - 경도"),
    radius: int = Query(5000, ge=100, le=50000, description="위치 필터 - 반경 (미터)"),
    limit: int = Query(50, ge=1, le=200, description="최대 결과 수"),
):
    """
    변호사 검색 (이름/사무소/지역 + 선택적 위치 필터)

    - **name**: 이름에 포함된 문자열
    - **office**: 사무소명에 포함된 문자열
    - **district**: 주소에 포함된 구/군 (예: "강남구", "송파구")
    - **latitude**: 위치 필터 - 위도 (선택, longitude와 함께 사용)
    - **longitude**: 위치 필터 - 경도 (선택, latitude와 함께 사용)
    - **radius**: 위치 필터 - 반경 (미터, 기본 5km)
    """
    if not any([name, office, district]):
        raise HTTPException(
            status_code=400,
            detail="최소 하나의 검색 조건이 필요합니다 (name, office, district)"
        )

    try:
        lawyers = search_lawyers(name, office, district, latitude, longitude, radius, limit)
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

    return {
        "total_lawyers": len(lawyers),
        "with_coordinates": with_coords,
        "without_coordinates": len(lawyers) - with_coords,
        "source": metadata.get("source"),
        "crawled_at": metadata.get("crawled_at"),
        "geocoded_at": metadata.get("geocoded_at"),
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
