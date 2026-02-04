"""
변호사 통계 DB 서비스 (PostgreSQL 기반)

JSON 파일 대신 PostgreSQL lawyers 테이블에서 통계를 계산합니다.
settings.USE_DB_LAWYERS=True 일 때 사용됩니다.
"""

import logging
from collections import defaultdict
from typing import Any

from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.models.lawyer import Lawyer
from app.services.service_function.lawyer_service import SPECIALTY_CATEGORIES
from app.services.service_function.lawyer_stats_service import (
    get_category_for_specialty,
    get_population_data,
)

logger = logging.getLogger(__name__)


# =============================================================================
# 통계 계산 함수 (DB 기반)
# =============================================================================
async def calculate_overview_db(db: AsyncSession) -> dict[str, Any]:
    """전체 현황 요약 계산 (DB 기반)."""
    total = await db.scalar(select(func.count(Lawyer.id))) or 0

    if total == 0:
        return {
            "total_lawyers": 0,
            "status_counts": [],
            "coord_rate": 0.0,
            "specialty_rate": 0.0,
        }

    # 상태별 집계
    status_result = await db.execute(
        select(Lawyer.status, func.count(Lawyer.id).label("cnt"))
        .group_by(Lawyer.status)
        .order_by(func.count(Lawyer.id).desc())
    )
    status_counts = [
        {"status": row.status or "알 수 없음", "count": row.cnt}
        for row in status_result.all()
    ]

    # 좌표 보유 건수
    coord_count = await db.scalar(
        select(func.count(Lawyer.id)).where(
            Lawyer.latitude.isnot(None),
            Lawyer.longitude.isnot(None),
        )
    ) or 0

    # 전문분야 보유 건수
    specialty_count = await db.scalar(
        select(func.count(Lawyer.id)).where(
            func.array_length(Lawyer.specialties, 1) > 0,
        )
    ) or 0

    return {
        "total_lawyers": total,
        "status_counts": status_counts,
        "coord_rate": round(coord_count / total * 100, 1),
        "specialty_rate": round(specialty_count / total * 100, 1),
    }


async def calculate_by_region_db(db: AsyncSession) -> list[dict[str, Any]]:
    """지역별 변호사 수 계산 (DB 기반)."""
    result = await db.execute(
        select(Lawyer.region, func.count(Lawyer.id).label("cnt"))
        .where(Lawyer.region.isnot(None))
        .group_by(Lawyer.region)
        .order_by(func.count(Lawyer.id).desc())
    )
    return [
        {"region": row.region, "count": row.cnt}
        for row in result.all()
    ]


async def calculate_density_by_region_db(
    db: AsyncSession,
    year: int | str = "current",
    include_change: bool = False,
) -> list[dict[str, Any]]:
    """지역별 인구 대비 변호사 밀도 계산 (DB 기반)."""
    region_stats = await calculate_by_region_db(db)
    population_data = get_population_data(year)
    population_current = get_population_data("current") if include_change else None

    result = []
    for stat in region_stats:
        region = stat["region"]
        count = stat["count"]
        population = population_data.get(region)

        if population and population > 0:
            density = round(count / population * 100000, 2)
            item: dict[str, Any] = {
                "region": region,
                "count": count,
                "population": population,
                "density": density,
            }

            if include_change and population_current:
                pop_current = population_current.get(region, population)
                if pop_current and pop_current > 0:
                    density_current = count / pop_current * 100000
                    change_percent = (
                        round((density - density_current) / density_current * 100, 1)
                        if density_current > 0
                        else 0.0
                    )
                    item["density_current"] = round(density_current, 2)
                    item["change_percent"] = change_percent

            result.append(item)

    return sorted(result, key=lambda x: -x["density"])


async def calculate_by_specialty_db(db: AsyncSession) -> list[dict[str, Any]]:
    """전문분야(12대분류)별 변호사 수 계산 (DB 기반)."""
    # unnest(specialties)로 개별 전문분야를 행으로 펼침
    unnest_query = (
        select(
            Lawyer.id,
            func.unnest(Lawyer.specialties).label("specialty"),
        )
        .where(func.array_length(Lawyer.specialties, 1) > 0)
        .subquery()
    )

    result = await db.execute(
        select(
            unnest_query.c.specialty,
            func.count(func.distinct(unnest_query.c.id)).label("cnt"),
        )
        .group_by(unnest_query.c.specialty)
    )
    specialty_counts = {row.specialty: row.cnt for row in result.all()}

    # Python에서 카테고리 매핑
    category_counter: dict[str, set[int]] = defaultdict(set)
    specialty_detail: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # 각 전문분야를 카테고리에 매핑
    for spec, count in specialty_counts.items():
        cat_id = get_category_for_specialty(spec)
        if cat_id:
            specialty_detail[cat_id][spec] = count

    # 카테고리별 변호사 수 (중복 제거를 위해 별도 쿼리)
    for cat_id, cat_info in SPECIALTY_CATEGORIES.items():
        specs = cat_info["specialties"]
        if not specs:
            continue

        cat_count = await db.scalar(
            select(func.count(func.distinct(Lawyer.id))).where(
                Lawyer.specialties.overlap(specs)
            )
        ) or 0
        category_counter[cat_id] = cat_count  # type: ignore[assignment]

    output = []
    for cat_id, cat_info in SPECIALTY_CATEGORIES.items():
        count = category_counter.get(cat_id, 0)
        if isinstance(count, set):
            count = len(count)
        spec_details = [
            {"name": name, "count": cnt}
            for name, cnt in sorted(
                specialty_detail[cat_id].items(),
                key=lambda x: -x[1],
            )
        ]
        output.append({
            "category_id": cat_id,
            "category_name": cat_info["name"],
            "count": count,
            "specialties": spec_details,
        })

    return sorted(output, key=lambda x: -x["count"])


async def calculate_specialty_by_region_db(
    db: AsyncSession, region: str,
) -> list[dict[str, Any]]:
    """특정 지역의 전문분야별 변호사 수 계산 (DB 기반)."""
    # 해당 지역의 변호사들의 전문분야 펼침
    unnest_query = (
        select(
            Lawyer.id,
            func.unnest(Lawyer.specialties).label("specialty"),
        )
        .where(
            Lawyer.region == region,
            func.array_length(Lawyer.specialties, 1) > 0,
        )
        .subquery()
    )

    result = await db.execute(
        select(
            unnest_query.c.specialty,
            func.count(func.distinct(unnest_query.c.id)).label("cnt"),
        )
        .group_by(unnest_query.c.specialty)
    )
    specialty_counts = {row.specialty: row.cnt for row in result.all()}

    # 카테고리별 매핑
    specialty_detail: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for spec, count in specialty_counts.items():
        cat_id = get_category_for_specialty(spec)
        if cat_id:
            specialty_detail[cat_id][spec] = count

    # 카테고리별 변호사 수 (중복 제거)
    category_counts: dict[str, int] = {}
    for cat_id, cat_info in SPECIALTY_CATEGORIES.items():
        specs = cat_info["specialties"]
        if not specs:
            continue
        cat_count = await db.scalar(
            select(func.count(func.distinct(Lawyer.id))).where(
                Lawyer.region == region,
                Lawyer.specialties.overlap(specs),
            )
        ) or 0
        category_counts[cat_id] = cat_count

    output = []
    for cat_id, cat_info in SPECIALTY_CATEGORIES.items():
        count = category_counts.get(cat_id, 0)
        if count == 0:
            continue
        spec_details = [
            {"name": name, "count": cnt}
            for name, cnt in sorted(
                specialty_detail[cat_id].items(),
                key=lambda x: -x[1],
            )
        ]
        output.append({
            "category_id": cat_id,
            "category_name": cat_info["name"],
            "count": count,
            "specialties": spec_details,
        })

    return sorted(output, key=lambda x: -x["count"])


async def calculate_cross_analysis_db(db: AsyncSession) -> dict[str, Any]:
    """지역 × 전문분야 교차 분석 계산 (DB 기반)."""
    return await _cross_analysis_impl(db, top_n=15)


async def calculate_cross_analysis_by_regions_db(
    db: AsyncSession, regions: list[str],
) -> dict[str, Any]:
    """선택된 지역 목록에 대한 교차 분석 (DB 기반)."""
    return await _cross_analysis_impl(db, filter_regions=regions)


async def calculate_cross_analysis_by_province_db(
    db: AsyncSession, province: str,
) -> dict[str, Any]:
    """특정 시/도 내 교차 분석 (DB 기반)."""
    return await _cross_analysis_impl(db, province_prefix=province)


# =============================================================================
# 교차 분석 내부 구현
# =============================================================================
async def _cross_analysis_impl(
    db: AsyncSession,
    top_n: int | None = None,
    filter_regions: list[str] | None = None,
    province_prefix: str | None = None,
) -> dict[str, Any]:
    """교차 분석 공통 구현."""
    # unnest로 (lawyer_id, region, specialty) 플랫 테이블 생성
    base_query = (
        select(
            Lawyer.id,
            Lawyer.region,
            func.unnest(Lawyer.specialties).label("specialty"),
        )
        .where(
            Lawyer.region.isnot(None),
            func.array_length(Lawyer.specialties, 1) > 0,
        )
    )

    if filter_regions:
        base_query = base_query.where(Lawyer.region.in_(filter_regions))
    if province_prefix:
        base_query = base_query.where(Lawyer.region.startswith(province_prefix))

    subq = base_query.subquery()

    # specialty → category 매핑은 Python에서 수행
    result = await db.execute(
        select(
            subq.c.region,
            subq.c.specialty,
            func.count(func.distinct(subq.c.id)).label("cnt"),
        )
        .group_by(subq.c.region, subq.c.specialty)
    )

    # (region, category) → 변호사 수 (중복 제거는 category 단위에서)
    # 간단히 전문분야 카운트를 카테고리로 매핑
    cross_counter: dict[tuple[str, str], set[str]] = defaultdict(set)
    region_set: set[str] = set()

    for row in result.all():
        cat_id = get_category_for_specialty(row.specialty)
        if cat_id:
            cross_counter[(row.region, cat_id)].add(row.specialty)
            region_set.add(row.region)

    # 카테고리별 정확한 카운트 (overlap 쿼리)
    # 효율을 위해 region별 카테고리별 카운트를 한 번에 계산
    region_cat_counts: dict[tuple[str, str], int] = defaultdict(int)

    if region_set:
        for cat_id, cat_info in SPECIALTY_CATEGORIES.items():
            specs = cat_info["specialties"]
            if not specs:
                continue

            cat_query = (
                select(
                    Lawyer.region,
                    func.count(Lawyer.id).label("cnt"),
                )
                .where(
                    Lawyer.region.isnot(None),
                    Lawyer.specialties.overlap(specs),
                )
            )
            if filter_regions:
                cat_query = cat_query.where(Lawyer.region.in_(filter_regions))
            if province_prefix:
                cat_query = cat_query.where(Lawyer.region.startswith(province_prefix))

            cat_query = cat_query.group_by(Lawyer.region)
            cat_result = await db.execute(cat_query)

            for row in cat_result.all():
                region_cat_counts[(row.region, cat_id)] = row.cnt

    # 지역별 총 변호사 수 정렬
    region_totals: dict[str, int] = defaultdict(int)
    for (region, _), count in region_cat_counts.items():
        region_totals[region] += count

    sorted_regions = sorted(region_totals.keys(), key=lambda r: -region_totals[r])
    if top_n:
        sorted_regions = sorted_regions[:top_n]

    # 결과 데이터 생성
    cells = []
    for region in sorted_regions:
        for cat_id, cat_info in SPECIALTY_CATEGORIES.items():
            count = region_cat_counts.get((region, cat_id), 0)
            cells.append({
                "region": region,
                "category_id": cat_id,
                "category_name": cat_info["name"],
                "count": count,
            })

    category_names = [cat["name"] for cat in SPECIALTY_CATEGORIES.values()]

    return {
        "data": cells,
        "regions": sorted_regions,
        "categories": category_names,
    }
