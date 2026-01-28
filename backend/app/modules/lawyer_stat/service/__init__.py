"""변호사 통계 모듈 - 통계 계산 서비스"""

import re
from collections import defaultdict
from functools import lru_cache
from typing import Any

from app.modules.lawyer_finder.service import SPECIALTY_CATEGORIES, load_lawyers_data

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


# 시/군/구별 인구 데이터 (2024년 주민등록인구 기준, 단위: 명)
POPULATION_DATA: dict[str, int] = {
    # 서울
    "서울 종로구": 151776, "서울 중구": 133360, "서울 용산구": 229431,
    "서울 성동구": 303589, "서울 광진구": 352450, "서울 동대문구": 352720,
    "서울 중랑구": 404155, "서울 성북구": 440986, "서울 강북구": 308367,
    "서울 도봉구": 323974, "서울 노원구": 522284, "서울 은평구": 479653,
    "서울 서대문구": 319257, "서울 마포구": 380426, "서울 양천구": 454974,
    "서울 강서구": 584109, "서울 구로구": 425690, "서울 금천구": 246215,
    "서울 영등포구": 397389, "서울 동작구": 398951, "서울 관악구": 502540,
    "서울 서초구": 423075, "서울 강남구": 545348, "서울 송파구": 668579,
    "서울 강동구": 453616,
    # 부산
    "부산 중구": 41523, "부산 서구": 106468, "부산 동구": 87591,
    "부산 영도구": 109267, "부산 부산진구": 359632, "부산 동래구": 272049,
    "부산 남구": 270213, "부산 북구": 299523, "부산 해운대구": 408276,
    "부산 사하구": 318153, "부산 금정구": 239693, "부산 강서구": 133451,
    "부산 연제구": 209295, "부산 수영구": 175596, "부산 사상구": 221884,
    "부산 기장군": 193986,
    # 대구
    "대구 중구": 76857, "대구 동구": 344158, "대구 서구": 179089,
    "대구 남구": 152544, "대구 북구": 432754, "대구 수성구": 425721,
    "대구 달서구": 561084, "대구 달성군": 262091,
    # 인천
    "인천 중구": 121619, "인천 동구": 66096, "인천 미추홀구": 408031,
    "인천 연수구": 370984, "인천 남동구": 531019, "인천 부평구": 509315,
    "인천 계양구": 311254, "인천 서구": 567168, "인천 강화군": 65824,
    "인천 옹진군": 20456, "인천 남구": 408031,
    # 광주
    "광주 동구": 95605, "광주 서구": 303116, "광주 남구": 217949,
    "광주 북구": 439070, "광주 광산구": 406621,
    # 대전
    "대전 동구": 229078, "대전 중구": 249693, "대전 서구": 469872,
    "대전 유성구": 356877, "대전 대덕구": 183946,
    # 울산
    "울산 중구": 228073, "울산 남구": 335693, "울산 동구": 161096,
    "울산 북구": 197633, "울산 울주군": 226012,
    # 세종
    "세종 세종시": 387112,
    # 경기
    "경기 수원시": 1185697, "경기 성남시": 927822, "경기 의정부시": 457487,
    "경기 안양시": 548499, "경기 부천시": 816589, "경기 광명시": 306364,
    "경기 평택시": 579594, "경기 동두천시": 95681, "경기 안산시": 657948,
    "경기 고양시": 1072762, "경기 과천시": 71707, "경기 구리시": 197466,
    "경기 남양주시": 730317, "경기 오산시": 230931, "경기 시흥시": 519913,
    "경기 군포시": 270532, "경기 의왕시": 163049, "경기 하남시": 305399,
    "경기 용인시": 1091989, "경기 파주시": 494392, "경기 이천시": 221915,
    "경기 안성시": 192070, "경기 김포시": 508250, "경기 화성시": 950456,
    "경기 광주시": 390649, "경기 양주시": 239879, "경기 포천시": 143994,
    "경기 여주시": 112282, "경기 연천군": 43023, "경기 가평군": 62753,
    "경기 양평군": 121558,
    # 강원
    "강원 춘천시": 283671, "강원 원주시": 357691, "강원 강릉시": 213486,
    "강원 동해시": 89596, "강원 태백시": 40508, "강원 속초시": 82486,
    "강원 삼척시": 65122, "강원 홍천군": 68885, "강원 횡성군": 46291,
    "강원 영월군": 38741, "강원 평창군": 41273, "강원 정선군": 34715,
    "강원 철원군": 44061, "강원 화천군": 25103, "강원 양구군": 23769,
    "강원 인제군": 32028, "강원 고성군": 27195, "강원 양양군": 27869,
    # 충북
    "충북 청주시": 847282, "충북 충주시": 211055, "충북 제천시": 133071,
    "충북 보은군": 31715, "충북 옥천군": 50720, "충북 영동군": 47042,
    "충북 증평군": 37632, "충북 진천군": 90063, "충북 괴산군": 36177,
    "충북 음성군": 97173, "충북 단양군": 29076,
    # 충남
    "충남 천안시": 661298, "충남 공주시": 106097, "충남 보령시": 99040,
    "충남 아산시": 343782, "충남 서산시": 177261, "충남 논산시": 117135,
    "충남 계룡시": 44279, "충남 당진시": 168461, "충남 금산군": 51887,
    "충남 부여군": 63364, "충남 서천군": 51428, "충남 청양군": 30126,
    "충남 홍성군": 98063, "충남 예산군": 78494, "충남 태안군": 64261,
    # 전북
    "전북 전주시": 656424, "전북 군산시": 267622, "전북 익산시": 287229,
    "전북 정읍시": 108295, "전북 남원시": 81153, "전북 김제시": 83286,
    "전북 완주군": 95785, "전북 진안군": 24844, "전북 무주군": 23746,
    "전북 장수군": 21635, "전북 임실군": 27372, "전북 순창군": 27197,
    "전북 고창군": 53711, "전북 부안군": 51809,
    # 전남
    "전남 목포시": 223155, "전남 여수시": 282041, "전남 순천시": 280665,
    "전남 나주시": 118125, "전남 광양시": 152999, "전남 담양군": 46584,
    "전남 곡성군": 27612, "전남 구례군": 25178, "전남 고흥군": 61193,
    "전남 보성군": 39293, "전남 화순군": 62935, "전남 장흥군": 35922,
    "전남 강진군": 33187, "전남 해남군": 68337, "전남 영암군": 54726,
    "전남 무안군": 92119, "전남 함평군": 31279, "전남 영광군": 51118,
    "전남 장성군": 44891, "전남 완도군": 47016, "전남 진도군": 30549,
    "전남 신안군": 38686,
    # 경북
    "경북 포항시": 500814, "경북 경주시": 254028, "경북 김천시": 138969,
    "경북 안동시": 157853, "경북 구미시": 411018, "경북 영주시": 102657,
    "경북 영천시": 102191, "경북 상주시": 95993, "경북 문경시": 70789,
    "경북 경산시": 275188, "경북 군위군": 22984, "경북 의성군": 51073,
    "경북 청송군": 24572, "경북 영양군": 16350, "경북 영덕군": 35001,
    "경북 청도군": 42203, "경북 고령군": 33049, "경북 성주군": 43746,
    "경북 칠곡군": 117046, "경북 예천군": 55736, "경북 봉화군": 30919,
    "경북 울진군": 48312, "경북 울릉군": 9042,
    # 경남
    "경남 창원시": 1024268, "경남 진주시": 349447, "경남 통영시": 127364,
    "경남 사천시": 111585, "경남 김해시": 538267, "경남 밀양시": 105862,
    "경남 거제시": 238453, "경남 양산시": 366831, "경남 의령군": 26316,
    "경남 함안군": 63933, "경남 창녕군": 61153, "경남 고성군": 51324,
    "경남 남해군": 42412, "경남 하동군": 45043, "경남 산청군": 34527,
    "경남 함양군": 38819, "경남 거창군": 61690, "경남 합천군": 43605,
    # 제주
    "제주 제주시": 494804, "제주 서귀포시": 181553,
}


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


@lru_cache(maxsize=1)
def calculate_density_by_region() -> list[dict[str, Any]]:
    """
    지역별 인구 대비 변호사 밀도 계산.

    Returns:
        지역별 밀도 리스트 (밀도 내림차순 정렬)
        - region: 지역명
        - count: 변호사 수
        - population: 인구 수
        - density: 인구 10만명당 변호사 수
    """
    region_stats = calculate_by_region()

    result = []
    for stat in region_stats:
        region = stat["region"]
        count = stat["count"]
        population = POPULATION_DATA.get(region)

        if population and population > 0:
            # 인구 10만명당 변호사 수
            density = round(count / population * 100000, 2)
            result.append({
                "region": region,
                "count": count,
                "population": population,
                "density": density,
            })

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
