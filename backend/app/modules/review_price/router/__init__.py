"""
후기/가격 비교 모듈 - 상담 후기 및 가격 정보 취합
사용자들의 상담 후기와 가격 정보를 평균 내어 추천
"""
from fastapi import APIRouter
from typing import Optional

router = APIRouter()


@router.get("/lawyers/{lawyer_id}/reviews")
async def get_lawyer_reviews(
    lawyer_id: int,
    sort_by: str = "recent",
    limit: int = 20,
):
    """변호사별 상담 후기 조회"""
    return {
        "lawyer_id": lawyer_id,
        "reviews": [],
        "average_rating": 0,
    }


@router.post("/lawyers/{lawyer_id}/reviews")
async def create_review(lawyer_id: int, rating: int, content: str, price: int):
    """상담 후기 작성"""
    return {
        "message": "후기가 등록되었습니다",
        "lawyer_id": lawyer_id,
    }


@router.get("/price-comparison")
async def get_price_comparison(
    category: str,
    region: Optional[str] = None,
):
    """분야별/지역별 상담료 비교"""
    return {
        "category": category,
        "region": region,
        "average_price": 0,
        "min_price": 0,
        "max_price": 0,
        "lawyers": [],
    }
