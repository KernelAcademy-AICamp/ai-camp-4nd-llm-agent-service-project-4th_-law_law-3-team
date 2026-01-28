"""변호사 통계 모듈 - 통계 계산 서비스"""

import re
from collections import defaultdict
from functools import lru_cache
from typing import Any

from app.modules.lawyer_finder.service import SPECIALTY_CATEGORIES, load_lawyers_data

# 주소에서 지역(시군구) 추출 패턴
REGION_PATTERN = re.compile(r"^(\S+)\s+(\S+구|\S+시|\S+군)")


def extract_region(address: str | None) -> str | None:
    """
    주소에서 시군구 단위 지역 추출.

    Args:
        address: 전체 주소 문자열

    Returns:
        지역명 (예: "서울 강남구") 또는 None
    """
    if not address:
        return None
    match = REGION_PATTERN.match(address)
    if match:
        return f"{match.group(1)} {match.group(2)}"
    return None


def get_category_for_specialty(specialty: str) -> str | None:
    """전문분야가 속한 카테고리 ID 반환."""
    for cat_id, cat_info in SPECIALTY_CATEGORIES.items():
        if specialty in cat_info["specialties"]:
            return cat_id
    return None


@lru_cache(maxsize=1)
def calculate_overview() -> dict[str, Any]:
    """
    전체 현황 요약 계산.

    Returns:
        - total_lawyers: 전체 변호사 수
        - status_counts: 상태별 변호사 수
        - coord_rate: 좌표 보유율 (%)
        - specialty_rate: 전문분야 보유율 (%)
    """
    data = load_lawyers_data()
    lawyers = data.get("lawyers", [])

    total = len(lawyers)
    if total == 0:
        return {
            "total_lawyers": 0,
            "status_counts": [],
            "coord_rate": 0.0,
            "specialty_rate": 0.0,
        }

    # 상태별 집계
    status_counter: dict[str, int] = defaultdict(int)
    coord_count = 0
    specialty_count = 0

    for lawyer in lawyers:
        # 상태 집계
        status = lawyer.get("status", "알 수 없음")
        status_counter[status] += 1

        # 좌표 보유 여부
        if lawyer.get("latitude") is not None and lawyer.get("longitude") is not None:
            coord_count += 1

        # 전문분야 보유 여부
        specialties = lawyer.get("specialties", [])
        if isinstance(specialties, list) and len(specialties) > 0:
            specialty_count += 1

    status_counts = [
        {"status": status, "count": count}
        for status, count in sorted(status_counter.items(), key=lambda x: -x[1])
    ]

    return {
        "total_lawyers": total,
        "status_counts": status_counts,
        "coord_rate": round(coord_count / total * 100, 1),
        "specialty_rate": round(specialty_count / total * 100, 1),
    }


@lru_cache(maxsize=1)
def calculate_by_region() -> list[dict[str, Any]]:
    """
    지역별 변호사 수 계산.

    Returns:
        지역별 카운트 리스트 (내림차순 정렬)
    """
    data = load_lawyers_data()
    lawyers = data.get("lawyers", [])

    region_counter: dict[str, int] = defaultdict(int)

    for lawyer in lawyers:
        region = extract_region(lawyer.get("address"))
        if region:
            region_counter[region] += 1

    return [
        {"region": region, "count": count}
        for region, count in sorted(region_counter.items(), key=lambda x: -x[1])
    ]


@lru_cache(maxsize=1)
def calculate_by_specialty() -> list[dict[str, Any]]:
    """
    전문분야(12대분류)별 변호사 수 계산.

    Returns:
        카테고리별 카운트 리스트 (내림차순 정렬)
        - specialties: 세부 전문분야별 카운트 포함
    """
    data = load_lawyers_data()
    lawyers = data.get("lawyers", [])

    # 카테고리별 변호사 수 (한 변호사가 여러 카테고리에 속할 수 있음)
    category_counter: dict[str, int] = defaultdict(int)
    # 세부 전문분야별 카운트: {category_id: {specialty_name: count}}
    specialty_counter: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for lawyer in lawyers:
        specialties = lawyer.get("specialties", [])
        if not isinstance(specialties, list):
            continue

        # 해당 변호사가 속한 카테고리 집합 (중복 카운트 방지)
        lawyer_categories: set[str] = set()
        for spec in specialties:
            cat_id = get_category_for_specialty(spec)
            if cat_id:
                lawyer_categories.add(cat_id)
                # 세부 전문분야 카운트
                specialty_counter[cat_id][spec] += 1

        for cat_id in lawyer_categories:
            category_counter[cat_id] += 1

    result = []
    for cat_id, cat_info in SPECIALTY_CATEGORIES.items():
        count = category_counter.get(cat_id, 0)
        # 세부 전문분야 리스트 생성 (카운트 내림차순 정렬)
        spec_details = [
            {"name": name, "count": cnt}
            for name, cnt in sorted(
                specialty_counter[cat_id].items(),
                key=lambda x: -x[1]
            )
        ]
        result.append({
            "category_id": cat_id,
            "category_name": cat_info["name"],
            "count": count,
            "specialties": spec_details,
        })

    return sorted(result, key=lambda x: -x["count"])


@lru_cache(maxsize=1)
def calculate_cross_analysis() -> dict[str, Any]:
    """
    지역 × 전문분야 교차 분석 계산.

    Returns:
        - data: 셀 데이터 리스트
        - regions: 지역 목록
        - categories: 카테고리 목록
    """
    data = load_lawyers_data()
    lawyers = data.get("lawyers", [])

    # (지역, 카테고리) -> 변호사 수
    cross_counter: dict[tuple[str, str], int] = defaultdict(int)
    regions_set: set[str] = set()

    for lawyer in lawyers:
        region = extract_region(lawyer.get("address"))
        if not region:
            continue

        specialties = lawyer.get("specialties", [])
        if not isinstance(specialties, list):
            continue

        # 해당 변호사가 속한 카테고리 집합
        lawyer_categories: set[str] = set()
        for spec in specialties:
            cat_id = get_category_for_specialty(spec)
            if cat_id:
                lawyer_categories.add(cat_id)

        for cat_id in lawyer_categories:
            cross_counter[(region, cat_id)] += 1
            regions_set.add(region)

    # 지역별 총 변호사 수로 정렬하여 상위 지역만 선택
    region_totals: dict[str, int] = defaultdict(int)
    for (region, _), count in cross_counter.items():
        region_totals[region] += count

    top_regions = sorted(region_totals.keys(), key=lambda r: -region_totals[r])[:15]

    # 결과 데이터 생성
    cells = []
    for region in top_regions:
        for cat_id, cat_info in SPECIALTY_CATEGORIES.items():
            count = cross_counter.get((region, cat_id), 0)
            cells.append({
                "region": region,
                "category_id": cat_id,
                "category_name": cat_info["name"],
                "count": count,
            })

    category_names = [cat["name"] for cat in SPECIALTY_CATEGORIES.values()]

    return {
        "data": cells,
        "regions": top_regions,
        "categories": category_names,
    }
