"""
변호사 찾기 모듈 - 위치 기반 변호사 추천
카카오맵 API를 사용하여 사용자 반경 내 변호사 검색
"""
from fastapi import APIRouter

router = APIRouter()


@router.get("/nearby")
async def get_nearby_lawyers(
    latitude: float,
    longitude: float,
    radius: int = 5000,
    specialty: str = None,
):
    """사용자 위치 기반 주변 변호사 검색"""
    return {
        "message": "주변 변호사 검색",
        "location": {"lat": latitude, "lng": longitude},
        "radius": radius,
        "specialty": specialty,
    }


@router.get("/{lawyer_id}")
async def get_lawyer_detail(lawyer_id: int):
    """변호사 상세 정보 조회"""
    return {"lawyer_id": lawyer_id}
