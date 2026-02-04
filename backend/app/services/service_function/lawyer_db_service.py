"""
변호사 DB 서비스 (PostgreSQL 기반)

JSON 파일 대신 PostgreSQL lawyers 테이블에서 데이터를 조회합니다.
settings.USE_DB_LAWYERS=True 일 때 사용됩니다.
"""

import logging
from typing import Any, Optional

from sqlalchemy import Float, Numeric, cast, func, literal_column
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.models.lawyer import Lawyer
from app.services.service_function.lawyer_service import (
    get_bounding_box,
    get_specialties_by_category,
)

logger = logging.getLogger(__name__)

# 지구 반지름 (km)
EARTH_RADIUS_KM = 6371.0


# =============================================================================
# Haversine SQL 표현식
# =============================================================================
def _haversine_distance_sql(lat: float, lng: float) -> Any:
    """
    SQL Haversine 거리 계산 표현식 (PostGIS 불필요).

    Returns:
        거리(km)를 계산하는 SQLAlchemy 표현식
    """
    return (
        literal_column(str(EARTH_RADIUS_KM))
        * func.acos(
            func.cos(func.radians(literal_column(str(lat))))
            * func.cos(func.radians(Lawyer.latitude))
            * func.cos(
                func.radians(Lawyer.longitude) - func.radians(literal_column(str(lng)))
            )
            + func.sin(func.radians(literal_column(str(lat))))
            * func.sin(func.radians(Lawyer.latitude))
        )
    )


# =============================================================================
# 변호사 검색 함수
# =============================================================================
async def find_nearby_lawyers_db(
    db: AsyncSession,
    latitude: float,
    longitude: float,
    radius_m: int = 5000,
    limit: Optional[int] = None,
    category: Optional[str] = None,
    specialty: Optional[str] = None,
) -> dict[str, Any]:
    """
    반경 내 변호사 검색 (DB 기반).

    1단계: 바운딩 박스로 1차 필터링
    2단계: Haversine SQL로 정확한 거리 계산
    3단계: 전문분야 필터링

    Returns:
        {"lawyers": [...], "total_count": int} - total_count는 limit 적용 전 전체 건수
    """
    radius_km = radius_m / 1000
    min_lat, max_lat, min_lng, max_lng = get_bounding_box(latitude, longitude, radius_km)

    # 거리 표현식
    distance_expr = _haversine_distance_sql(latitude, longitude)

    base_query = (
        select(Lawyer, distance_expr.label("distance"))
        .where(
            Lawyer.latitude.isnot(None),
            Lawyer.longitude.isnot(None),
            Lawyer.latitude.between(min_lat, max_lat),
            Lawyer.longitude.between(min_lng, max_lng),
        )
        .where(distance_expr <= radius_km)
    )

    # 전문분야 필터
    base_query = _apply_specialty_filter(base_query, category, specialty)

    # 전체 건수 조회 (limit 적용 전)
    count_query = select(func.count()).select_from(base_query.subquery())
    total_count = await db.scalar(count_query) or 0

    # 정렬 + 제한
    query = base_query.order_by("distance")
    if limit:
        query = query.limit(limit)

    result = await db.execute(query)
    rows = result.all()

    lawyers = [
        {**_lawyer_to_dict(lawyer), "distance": round(distance, 2)}
        for lawyer, distance in rows
    ]
    return {"lawyers": lawyers, "total_count": total_count}


async def search_lawyers_db(
    db: AsyncSession,
    name: Optional[str] = None,
    office: Optional[str] = None,
    district: Optional[str] = None,
    category: Optional[str] = None,
    specialty: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    radius_m: int = 5000,
    limit: Optional[int] = None,
) -> dict[str, Any]:
    """
    이름/사무소/지역/전문분야로 검색 (DB 기반).

    Returns:
        {"lawyers": [...], "total_count": int} - total_count는 limit 적용 전 전체 건수
    """
    # 위치 필터링 입력 검증
    has_latitude = latitude is not None
    has_longitude = longitude is not None
    if has_latitude != has_longitude:
        missing = "longitude" if has_latitude else "latitude"
        provided = "latitude" if has_latitude else "longitude"
        raise ValueError(
            f"위치 필터링을 사용하려면 latitude와 longitude가 모두 필요합니다. "
            f"{provided}만 제공되었고 {missing}가 누락되었습니다."
        )

    base_query = select(Lawyer)
    has_location = has_latitude and has_longitude

    # 이름 또는 사무소 검색 (OR 조건)
    if name or office:
        conditions = []
        if name:
            conditions.append(Lawyer.name.ilike(f"%{name}%"))
        if office:
            conditions.append(Lawyer.office_name.ilike(f"%{office}%"))

        from sqlalchemy import or_
        base_query = base_query.where(or_(*conditions))

    # 지역 검색 (AND 조건)
    if district:
        base_query = base_query.where(Lawyer.address.ilike(f"%{district}%"))

    # 전문분야 필터
    base_query = _apply_specialty_filter(base_query, category, specialty)

    # 위치 필터링
    distance_expr = None
    if has_location and latitude is not None and longitude is not None:
        radius_km = radius_m / 1000
        min_lat, max_lat, min_lng, max_lng = get_bounding_box(
            latitude, longitude, radius_km,
        )
        distance_expr = _haversine_distance_sql(latitude, longitude)

        base_query = base_query.where(
            Lawyer.latitude.isnot(None),
            Lawyer.longitude.isnot(None),
            Lawyer.latitude.between(min_lat, max_lat),
            Lawyer.longitude.between(min_lng, max_lng),
        ).where(distance_expr <= radius_km)

    # 전체 건수 조회 (limit 적용 전)
    count_query = select(func.count()).select_from(base_query.subquery())
    total_count = await db.scalar(count_query) or 0

    # 쿼리에 거리 컬럼 추가
    if distance_expr is not None:
        base_query = base_query.add_columns(distance_expr.label("distance"))
        base_query = base_query.order_by("distance")
    else:
        base_query = base_query.add_columns(literal_column("NULL").label("distance"))

    if limit:
        base_query = base_query.limit(limit)

    result = await db.execute(base_query)
    rows = result.all()

    results = []
    for lawyer, distance in rows:
        item = _lawyer_to_dict(lawyer)
        if distance is not None:
            item["distance"] = round(distance, 2)
        results.append(item)

    return {"lawyers": results, "total_count": total_count}


async def get_clusters_db(
    db: AsyncSession,
    min_lat: float,
    max_lat: float,
    min_lng: float,
    max_lng: float,
    grid_size: float = 0.01,
    category: Optional[str] = None,
    specialty: Optional[str] = None,
) -> list[dict[str, Any]]:
    """뷰포트 내 변호사를 그리드로 클러스터링 (DB 기반)."""
    grid_lat = func.round(
        cast(cast(Lawyer.latitude / grid_size, Float) * grid_size, Numeric), 6,
    )
    grid_lng = func.round(
        cast(cast(Lawyer.longitude / grid_size, Float) * grid_size, Numeric), 6,
    )

    query = (
        select(
            grid_lat.label("latitude"),
            grid_lng.label("longitude"),
            func.count().label("count"),
        )
        .where(
            Lawyer.latitude.isnot(None),
            Lawyer.longitude.isnot(None),
            Lawyer.latitude.between(min_lat, max_lat),
            Lawyer.longitude.between(min_lng, max_lng),
        )
    )

    # 전문분야 필터
    query = _apply_specialty_filter(query, category, specialty)

    query = query.group_by(grid_lat, grid_lng)

    result = await db.execute(query)
    rows = result.all()

    return [
        {"latitude": float(row.latitude), "longitude": float(row.longitude), "count": row.count}
        for row in rows
    ]


async def get_lawyer_by_id_db(
    db: AsyncSession, lawyer_id: int,
) -> Optional[dict[str, Any]]:
    """ID로 변호사 조회 (DB 기반)."""
    result = await db.execute(select(Lawyer).where(Lawyer.id == lawyer_id))
    lawyer = result.scalar_one_or_none()
    if lawyer is None:
        return None
    return _lawyer_to_dict(lawyer)


async def get_lawyer_stats_db(db: AsyncSession) -> dict[str, Any]:
    """데이터 통계 조회 (DB 기반)."""
    total = await db.scalar(select(func.count(Lawyer.id))) or 0
    with_coords = await db.scalar(
        select(func.count(Lawyer.id)).where(
            Lawyer.latitude.isnot(None),
            Lawyer.longitude.isnot(None),
        )
    ) or 0
    with_specialties = await db.scalar(
        select(func.count(Lawyer.id)).where(
            func.array_length(Lawyer.specialties, 1) > 0,
        )
    ) or 0

    return {
        "total_lawyers": total,
        "with_coordinates": with_coords,
        "without_coordinates": total - with_coords,
        "with_specialties": with_specialties,
        "source": "PostgreSQL",
        "crawled_at": None,
        "geocoded_at": None,
        "specialty_crawl_date": None,
    }


# =============================================================================
# 내부 헬퍼 함수
# =============================================================================
def _apply_specialty_filter(
    query: Any,
    category: Optional[str],
    specialty: Optional[str],
) -> Any:
    """전문분야/카테고리 필터 적용."""
    if specialty:
        # ARRAY @> 연산: specialties가 [specialty]를 포함하는지
        query = query.where(Lawyer.specialties.any(specialty))
    elif category:
        category_specs = get_specialties_by_category(category)
        if category_specs:
            # ARRAY && 연산: specialties와 category_specs에 교집합이 있는지
            query = query.where(
                Lawyer.specialties.overlap(list(category_specs))
            )
    return query


def _lawyer_to_dict(lawyer: Lawyer) -> dict[str, Any]:
    """Lawyer ORM 객체를 dict로 변환 (JSON 서비스와 동일한 형식)."""
    return {
        "id": lawyer.id,
        "detail_id": lawyer.detail_id,
        "name": lawyer.name,
        "status": lawyer.status,
        "birth_year": lawyer.birth_year,
        "photo_url": lawyer.photo_url,
        "office_name": lawyer.office_name,
        "address": lawyer.address,
        "phone": lawyer.phone,
        "fax": lawyer.fax,
        "email": lawyer.email,
        "birthdate": lawyer.birthdate,
        "local_bar": lawyer.local_bar,
        "qualification": lawyer.qualification,
        "klaw_url": lawyer.klaw_url,
        "latitude": lawyer.latitude,
        "longitude": lawyer.longitude,
        "specialties": list(lawyer.specialties) if lawyer.specialties else [],
    }
