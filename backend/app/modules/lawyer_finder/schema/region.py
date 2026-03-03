"""변호사 찾기 모듈 - 지역 데이터 스키마"""

from pydantic import BaseModel


class DistrictInfo(BaseModel):
    """시/군/구 정보"""

    name: str
    center_lat: float
    center_lng: float
    count: int


class ProvinceInfo(BaseModel):
    """시/도 정보"""

    name: str
    center_lat: float
    center_lng: float
    count: int
    districts: list[DistrictInfo]


class RegionResponse(BaseModel):
    """지역 데이터 응답"""

    provinces: list[ProvinceInfo]
