"""변호사 통계 모듈 - 통계 계산 서비스"""

import json
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.modules.lawyer_finder.service import SPECIALTY_CATEGORIES, load_lawyers_data

# 인구 데이터 JSON 파일 경로 (data/population.json)
POPULATION_JSON_PATH = Path(__file__).parent.parent.parent.parent.parent.parent / "data" / "population.json"

# 주소에서 지역(시군구) 추출 패턴
REGION_PATTERN = re.compile(r"^(\S+)\s+(\S+구|\S+시|\S+군)")

# 시/도명 정규화 매핑
PROVINCE_NORMALIZE_MAP: dict[str, str] = {
    "서울특별시": "서울",
    "서울시": "서울",
    "부산광역시": "부산",
    "부산시": "부산",
    "대구광역시": "대구",
    "대구시": "대구",
    "인천광역시": "인천",
    "인천시": "인천",
    "광주광역시": "광주",
    "광주시": "광주",
    "대전광역시": "대전",
    "대전시": "대전",
    "울산광역시": "울산",
    "울산시": "울산",
    "세종특별자치시": "세종",
    "세종시": "세종",
    "경기도": "경기",
    "강원도": "강원",
    "강원특별자치도": "강원",
    "충청북도": "충북",
    "충북": "충북",
    "충청남도": "충남",
    "충남": "충남",
    "전라북도": "전북",
    "전북특별자치도": "전북",
    "전북": "전북",
    "전라남도": "전남",
    "전남": "전남",
    "경상북도": "경북",
    "경북": "경북",
    "경상남도": "경남",
    "경남": "경남",
    "제주특별자치도": "제주",
    "제주도": "제주",
    "제주시": "제주",
}

# 인구 데이터 캐시
_population_cache: dict[str, Any] | None = None


def _load_population_json() -> dict[str, Any]:
    """
    인구 데이터 JSON 파일 로드 (캐시 사용).

    Returns:
        인구 데이터 딕셔너리
    """
    global _population_cache
    if _population_cache is None:
        if not POPULATION_JSON_PATH.exists():
            raise FileNotFoundError(f"인구 데이터 파일이 없습니다: {POPULATION_JSON_PATH}")
        with open(POPULATION_JSON_PATH, encoding="utf-8") as f:
            _population_cache = json.load(f)
    return _population_cache


def get_population_data(year: int | str = "current") -> dict[str, int]:
    """
    연도별 인구 데이터 반환.

    Args:
        year: "current" 또는 연도 (2030, 2035, 2040)

    Returns:
        {지역명: 인구수} 딕셔너리
    """
    data = _load_population_json()
    year_key = str(year) if year != "current" else "current"

    result: dict[str, int] = {}
    for region_name, region_data in data["data"].items():
        if year_key in region_data:
            result[region_name] = region_data[year_key]
    return result


def get_population_meta() -> dict[str, Any]:
    """인구 데이터 메타 정보 반환."""
    data = _load_population_json()
    return data.get("meta", {})


def normalize_province(province: str) -> str:
    """시/도명을 표준 형식으로 정규화."""
    return PROVINCE_NORMALIZE_MAP.get(province, province)


def extract_region(address: str | None) -> str | None:
    """
    주소에서 시군구 단위 지역 추출 및 정규화.

    Args:
        address: 전체 주소 문자열

    Returns:
        지역명 (예: "서울 강남구") 또는 None
    """
    if not address:
        return None
    match = REGION_PATTERN.match(address)
    if match:
        province = normalize_province(match.group(1))
        district = match.group(2)
        return f"{province} {district}"
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


def calculate_density_by_region(
    year: int | str = "current",
    include_change: bool = False,
) -> list[dict[str, Any]]:
    """
    지역별 인구 대비 변호사 밀도 계산.

    Args:
        year: 인구 데이터 연도 ("current", 2030, 2035, 2040)
        include_change: 현재 연도 대비 변화율 포함 여부

    Returns:
        지역별 밀도 리스트 (밀도 내림차순 정렬)
        - region: 지역명
        - count: 변호사 수
        - population: 인구 수
        - density: 인구 10만명당 변호사 수
        - density_current: 현재 연도 기준 밀도 (include_change=True일 때)
        - change_percent: 현재 연도 대비 변화율 (include_change=True일 때)
    """
    region_stats = calculate_by_region()
    population_data = get_population_data(year)
    population_current = get_population_data("current") if include_change else None

    result = []
    for stat in region_stats:
        region = stat["region"]
        count = stat["count"]
        population = population_data.get(region)

        if population and population > 0:
            # 인구 10만명당 변호사 수
            density = round(count / population * 100000, 2)
            item: dict[str, Any] = {
                "region": region,
                "count": count,
                "population": population,
                "density": density,
            }

            # 변화율 계산 (예측 모드)
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

    # 밀도 내림차순 정렬
    return sorted(result, key=lambda x: -x["density"])


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


def calculate_specialty_by_region(region: str) -> list[dict[str, Any]]:
    """
    특정 지역의 전문분야별 변호사 수 계산 (세부 전문분야 포함).

    Args:
        region: 지역명 (예: "서울 강남구")

    Returns:
        카테고리별 카운트 리스트 (내림차순 정렬, 세부 전문분야 포함)
    """
    data = load_lawyers_data()
    lawyers = data.get("lawyers", [])

    category_counter: dict[str, int] = defaultdict(int)
    specialty_counter: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for lawyer in lawyers:
        lawyer_region = extract_region(lawyer.get("address"))
        if lawyer_region != region:
            continue

        specialties = lawyer.get("specialties", [])
        if not isinstance(specialties, list):
            continue

        lawyer_categories: set[str] = set()
        for spec in specialties:
            cat_id = get_category_for_specialty(spec)
            if cat_id:
                lawyer_categories.add(cat_id)
                specialty_counter[cat_id][spec] += 1

        for cat_id in lawyer_categories:
            category_counter[cat_id] += 1

    result = []
    for cat_id, cat_info in SPECIALTY_CATEGORIES.items():
        count = category_counter.get(cat_id, 0)
        if count == 0:
            continue
        spec_details = [
            {"name": name, "count": cnt}
            for name, cnt in sorted(specialty_counter[cat_id].items(), key=lambda x: -x[1])
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


def calculate_cross_analysis_by_regions(regions: list[str]) -> dict[str, Any]:
    """
    선택된 지역 목록에 대한 교차 분석 계산.

    Args:
        regions: 지역명 리스트 (예: ["서울 강남구", "경기 수원시", "부산 해운대구"])

    Returns:
        - data: 셀 데이터 리스트
        - regions: 선택된 지역 목록 (변호사 수 내림차순)
        - categories: 카테고리 목록
    """
    data = load_lawyers_data()
    lawyers = data.get("lawyers", [])

    regions_set = set(regions)

    # (지역, 카테고리) -> 변호사 수
    cross_counter: dict[tuple[str, str], int] = defaultdict(int)

    for lawyer in lawyers:
        region = extract_region(lawyer.get("address"))
        if not region or region not in regions_set:
            continue

        specialties = lawyer.get("specialties", [])
        if not isinstance(specialties, list):
            continue

        lawyer_categories: set[str] = set()
        for spec in specialties:
            cat_id = get_category_for_specialty(spec)
            if cat_id:
                lawyer_categories.add(cat_id)

        for cat_id in lawyer_categories:
            cross_counter[(region, cat_id)] += 1

    # 지역별 총 변호사 수로 정렬
    region_totals: dict[str, int] = defaultdict(int)
    for (region, _), count in cross_counter.items():
        region_totals[region] += count

    sorted_regions = sorted(region_totals.keys(), key=lambda r: -region_totals[r])

    # 결과 데이터 생성
    cells = []
    for region in sorted_regions:
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
        "regions": sorted_regions,
        "categories": category_names,
    }


def calculate_cross_analysis_by_province(province: str) -> dict[str, Any]:
    """
    특정 시/도 내 지역 × 전문분야 교차 분석 계산.

    Args:
        province: 시/도명 (예: "서울", "경기")

    Returns:
        - data: 셀 데이터 리스트
        - regions: 해당 시/도 내 지역 목록
        - categories: 카테고리 목록
    """
    data = load_lawyers_data()
    lawyers = data.get("lawyers", [])

    # (지역, 카테고리) -> 변호사 수
    cross_counter: dict[tuple[str, str], int] = defaultdict(int)

    for lawyer in lawyers:
        region = extract_region(lawyer.get("address"))
        if not region:
            continue

        # 해당 시/도 지역만 필터링
        if not region.startswith(province):
            continue

        specialties = lawyer.get("specialties", [])
        if not isinstance(specialties, list):
            continue

        lawyer_categories: set[str] = set()
        for spec in specialties:
            cat_id = get_category_for_specialty(spec)
            if cat_id:
                lawyer_categories.add(cat_id)

        for cat_id in lawyer_categories:
            cross_counter[(region, cat_id)] += 1

    # 지역별 총 변호사 수로 정렬
    region_totals: dict[str, int] = defaultdict(int)
    for (region, _), count in cross_counter.items():
        region_totals[region] += count

    sorted_regions = sorted(region_totals.keys(), key=lambda r: -region_totals[r])

    # 결과 데이터 생성
    cells = []
    for region in sorted_regions:
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
        "regions": sorted_regions,
        "categories": category_names,
    }
