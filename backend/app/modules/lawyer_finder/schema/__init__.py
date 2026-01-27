"""변호사 찾기 모듈 - Pydantic 스키마"""
from typing import Any, List, Optional

from pydantic import BaseModel, Field


class LawyerBase(BaseModel):
    """변호사 기본 정보"""
    name: str
    status: Optional[str] = None
    photo_url: Optional[str] = None
    office_name: Optional[str] = None
    address: Optional[str] = None
    phone: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class LawyerResponse(LawyerBase):
    """변호사 응답 모델"""
    id: int
    detail_id: Optional[str] = None
    birth_year: Optional[str] = None
    distance: Optional[float] = Field(None, description="사용자 위치로부터 거리 (km)")
    specialties: Optional[List[str]] = Field(None, description="전문분야 목록")

    class Config:
        from_attributes = True


class NearbySearchRequest(BaseModel):
    """주변 검색 요청"""
    latitude: float = Field(..., ge=-90, le=90, description="위도")
    longitude: float = Field(..., ge=-180, le=180, description="경도")
    radius: int = Field(5000, ge=100, le=50000, description="반경 (미터)")
    limit: int = Field(50, ge=1, le=200, description="최대 결과 수")


class NearbySearchResponse(BaseModel):
    """주변 검색 응답"""
    lawyers: List[LawyerResponse]
    total_count: int
    center: dict[str, Any]  # {"lat": ..., "lng": ...}
    radius: int


class SearchRequest(BaseModel):
    """변호사 검색 요청"""
    name: Optional[str] = None
    office: Optional[str] = None
    district: Optional[str] = None  # 구/군


class ClusterItem(BaseModel):
    """클러스터 아이템"""
    latitude: float
    longitude: float
    count: int


class ClusterResponse(BaseModel):
    """클러스터 응답"""
    clusters: List[ClusterItem]
    zoom_level: int
    total_count: int
