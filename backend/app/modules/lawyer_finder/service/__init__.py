"""변호사 찾기 모듈 - 서비스 레이어"""
import json
from math import radians, cos, sin, asin, sqrt
from pathlib import Path
from typing import List, Optional, Tuple, Set
from functools import lru_cache

# 전문분야 12대분류 (사용자에게는 이것만 표시)
SPECIALTY_CATEGORIES: dict[str, dict] = {
    "civil-family": {
        "name": "민사·가사",
        "icon": "👨‍👩‍👧",
        "description": "개인 간 분쟁 / 가족 관계",
        "specialties": ["민사법", "손해배상", "민사집행", "가사법", "이혼", "상속", "성년후견", "소년법"],
    },
    "criminal": {
        "name": "형사",
        "icon": "⚖️",
        "description": "범죄, 수사, 재판",
        "specialties": ["형사법", "군형법"],
    },
    "real-estate": {
        "name": "부동산·건설",
        "icon": "🏗️",
        "description": "부동산 거래·개발·분쟁",
        "specialties": ["부동산", "건설", "임대차관련법", "재개발·재건축", "수용 및 보상", "등기·경매"],
    },
    "labor": {
        "name": "노동·산재",
        "icon": "👷",
        "description": "근로관계, 산업재해",
        "specialties": ["노동법", "산재"],
    },
    "corporate": {
        "name": "기업·상사",
        "icon": "🏢",
        "description": "기업 운영·거래·분쟁",
        "specialties": ["회사법", "상사법", "인수합병", "영업비밀", "채권추심"],
    },
    "finance": {
        "name": "금융·자본시장",
        "icon": "💰",
        "description": "금융 규제, 자본, 구조조정",
        "specialties": ["금융", "증권", "보험", "도산"],
    },
    "tax": {
        "name": "조세·관세",
        "icon": "🧾",
        "description": "세금·통관",
        "specialties": ["조세법", "관세"],
    },
    "public": {
        "name": "공정·행정·공공",
        "icon": "🏛️",
        "description": "국가·공공기관 상대 사건",
        "specialties": ["공정거래", "국가계약", "행정법"],
    },
    "ip": {
        "name": "지식재산(IP)",
        "icon": "💡",
        "description": "기술·콘텐츠 권리 보호",
        "specialties": ["특허", "저작권"],
    },
    "it-media": {
        "name": "IT·미디어·콘텐츠",
        "icon": "📱",
        "description": "플랫폼, 데이터, 콘텐츠 산업",
        "specialties": ["IT", "언론·방송통신", "엔터테인먼트", "스포츠"],
    },
    "medical": {
        "name": "의료·바이오·식품",
        "icon": "🏥",
        "description": "의료 분쟁 + 규제",
        "specialties": ["의료", "식품·의약"],
    },
    "international": {
        "name": "국제·해외",
        "icon": "🌐",
        "description": "국제 거래·분쟁·이동",
        "specialties": ["국제관계법", "국제중재", "중재", "해외투자", "해상", "이주 및 비자"],
    },
}


def get_specialties_by_category(category: str) -> Set[str]:
    """카테고리 ID로 해당 카테고리의 전문분야 목록 조회"""
    if category in SPECIALTY_CATEGORIES:
        return set(SPECIALTY_CATEGORIES[category]["specialties"])
    return set()


def get_categories() -> List[dict]:
    """12대분류 목록 반환 (프론트엔드 표시용)"""
    return [
        {
            "id": cat_id,
            "name": cat["name"],
            "icon": cat["icon"],
            "description": cat["description"],
        }
        for cat_id, cat in SPECIALTY_CATEGORIES.items()
    ]

# 데이터 파일 경로
# __file__ = backend/app/modules/lawyer_finder/service/__init__.py
# 6 parents up = law-3-team/ (프로젝트 루트)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LAWYERS_WITH_SPECIALTIES_FILE = DATA_DIR / "lawyers_with_specialties.json"
LAWYERS_FILE = DATA_DIR / "lawyers_with_coords.json"
FALLBACK_FILE = PROJECT_ROOT / "all_lawyers.json"


@lru_cache(maxsize=1)
def load_lawyers_data() -> dict:
    """변호사 데이터 로드 (캐싱)"""
    # 전문분야 포함 파일 우선
    if LAWYERS_WITH_SPECIALTIES_FILE.exists():
        with open(LAWYERS_WITH_SPECIALTIES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    # 지오코딩된 파일
    if LAWYERS_FILE.exists():
        with open(LAWYERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    # 폴백: 원본 파일
    if FALLBACK_FILE.exists():
        with open(FALLBACK_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    return {"lawyers": [], "metadata": {}}


def get_available_specialties() -> List[str]:
    """사용 가능한 전문분야 목록 조회"""
    data = load_lawyers_data()
    lawyers = data.get("lawyers", [])

    specialties_set: set[str] = set()
    for lawyer in lawyers:
        specs = lawyer.get("specialties", [])
        if isinstance(specs, list):
            specialties_set.update(specs)

    return sorted(specialties_set)


def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    두 좌표 간 거리 계산 (Haversine 공식)
    반환: 거리 (km)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return 6371 * c  # 지구 반지름 (km)


def get_bounding_box(
    lat: float, lng: float, radius_km: float
) -> Tuple[float, float, float, float]:
    """
    반경 기준 바운딩 박스 계산 (성능 최적화용)
    반환: (min_lat, max_lat, min_lng, max_lng)
    """
    lat_delta = radius_km / 111.0  # 위도 1도 ≈ 111km
    lng_delta = radius_km / (111.0 * cos(radians(lat)))
    return (lat - lat_delta, lat + lat_delta, lng - lng_delta, lng + lng_delta)


def find_nearby_lawyers(
    latitude: float,
    longitude: float,
    radius_m: int = 5000,
    limit: Optional[int] = None,  # None이면 제한 없음
    category: Optional[str] = None,
    specialty: Optional[str] = None
) -> List[dict]:
    """
    Find lawyers within a radius of a geographic point.
    
    Performs an initial bounding-box filter, precise distance filtering using the Haversine formula, and specialty/category filtering (exact `specialty` match takes priority over `category`). Results are sorted by distance.
    
    Parameters:
        latitude (float): Query point latitude.
        longitude (float): Query point longitude.
        radius_m (int): Search radius in meters.
        limit (Optional[int]): Maximum number of results to return; if None, return all matches.
        category (Optional[str]): Category ID whose specialties are used for filtering when `specialty` is not provided.
        specialty (Optional[str]): Exact specialty name to filter lawyers by.
    
    Returns:
        List[dict]: Matching lawyer records with an added `id` (index in the loaded dataset) and `distance` (kilometers, rounded to 2 decimals), sorted by ascending distance.
    """
    data = load_lawyers_data()
    lawyers = data.get("lawyers", [])

    radius_km = radius_m / 1000
    min_lat, max_lat, min_lng, max_lng = get_bounding_box(latitude, longitude, radius_km)

    # 카테고리에 해당하는 전문분야 목록 (specialty가 없을 때만 사용)
    category_specs = get_specialties_by_category(category) if category and not specialty else set()

    results = []

    for idx, lawyer in enumerate(lawyers):
        lat = lawyer.get("latitude")
        lng = lawyer.get("longitude")

        # 좌표 없으면 스킵
        if lat is None or lng is None:
            continue

        # 1차 필터: 바운딩 박스
        if not (min_lat <= lat <= max_lat and min_lng <= lng <= max_lng):
            continue

        # 2차 필터: 정확한 거리 계산
        dist = haversine(longitude, latitude, lng, lat)
        if dist > radius_km:
            continue

        # 3차 필터: 전문분야 (specialty 우선, 없으면 category)
        lawyer_specs = lawyer.get("specialties", [])
        if not isinstance(lawyer_specs, list):
            lawyer_specs = []

        if specialty:
            # 정확한 전문분야 매칭 (예: "이혼"이 lawyer_specs에 있는지)
            if specialty not in lawyer_specs:
                continue
        elif category_specs:
            # 카테고리 내 전문분야 중 하나라도 있으면 통과
            if not category_specs.intersection(lawyer_specs):
                continue

        lawyer_copy = {**lawyer, "id": idx, "distance": round(dist, 2)}
        results.append(lawyer_copy)

    # 거리순 정렬
    results.sort(key=lambda x: x["distance"])

    return results[:limit] if limit else results


def get_lawyer_by_id(lawyer_id: int) -> Optional[dict]:
    """
    Retrieve a lawyer record by its index ID.
    
    Returns:
        dict: The lawyer's data with an added `id` field set to `lawyer_id`, or `None` if no lawyer exists with that ID.
    """
    data = load_lawyers_data()
    lawyers = data.get("lawyers", [])

    if 0 <= lawyer_id < len(lawyers):
        lawyer = lawyers[lawyer_id]
        return {**lawyer, "id": lawyer_id}

    return None


def search_lawyers(
    name: Optional[str] = None,
    office: Optional[str] = None,
    district: Optional[str] = None,
    category: Optional[str] = None,
    specialty: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    radius_m: int = 5000,
    limit: Optional[int] = None  # None이면 제한 없음
) -> List[dict]:
    """
    Search for lawyers by name, office, district, category or specialty, with optional location filtering.
    
    Parameters:
        name (Optional[str]): Substring to match against lawyer name (OR with `office`).
        office (Optional[str]): Substring to match against office name (OR with `name`).
        district (Optional[str]): Substring to require inside the lawyer's address (AND with other filters).
        category (Optional[str]): Category ID whose specialties are used as a fallback filter when `specialty` is not provided.
        specialty (Optional[str]): Exact specialty keyword to require (takes precedence over `category`).
        latitude (Optional[float]): Latitude for distance-based filtering; must be provided together with `longitude`.
        longitude (Optional[float]): Longitude for distance-based filtering; must be provided together with `latitude`.
        radius_m (int): Search radius in meters when location filtering is used.
        limit (Optional[int]): Maximum number of results to return; `None` means no limit.
    
    Returns:
        List[dict]: List of matching lawyer records. Each returned dict is the original lawyer record with an added `id` key containing the lawyer's index in the loaded data.
    
    Raises:
        ValueError: If exactly one of `latitude` or `longitude` is provided.
    """
    data = load_lawyers_data()
    lawyers = data.get("lawyers", [])

    # 위치 필터링 입력 검증: 둘 다 제공되거나 둘 다 없어야 함
    has_latitude = latitude is not None
    has_longitude = longitude is not None
    if has_latitude != has_longitude:
        missing = "longitude" if has_latitude else "latitude"
        provided = "latitude" if has_latitude else "longitude"
        raise ValueError(
            f"위치 필터링을 사용하려면 latitude와 longitude가 모두 필요합니다. "
            f"{provided}만 제공되었고 {missing}가 누락되었습니다."
        )

    # 위치 필터링용 바운딩 박스
    bbox = None
    if has_latitude and has_longitude:
        radius_km = radius_m / 1000
        bbox = get_bounding_box(latitude, longitude, radius_km)

    # 카테고리에 해당하는 전문분야 목록 (specialty가 없을 때만 사용)
    category_specs = get_specialties_by_category(category) if category and not specialty else set()

    results = []

    for idx, lawyer in enumerate(lawyers):
        # 이름 또는 사무소 검색 (OR 조건)
        if name or office:
            name_match = name and name in lawyer.get("name", "")
            office_match = office and office in (lawyer.get("office_name") or "")

            # 둘 다 제공된 경우 OR 조건, 하나만 제공된 경우 해당 조건만
            if not (name_match or office_match):
                continue

        # 지역(구/군) 검색 (AND 조건)
        if district:
            address = lawyer.get("address") or ""
            if district not in address:
                continue

        # 전문분야 필터링 (specialty 우선, 없으면 category)
        lawyer_specs = lawyer.get("specialties", [])
        if not isinstance(lawyer_specs, list):
            lawyer_specs = []

        if specialty:
            # 정확한 전문분야 매칭
            if specialty not in lawyer_specs:
                continue
        elif category_specs:
            # 카테고리 내 전문분야 중 하나라도 있으면 통과
            if not category_specs.intersection(lawyer_specs):
                continue

        # 위치 필터링 (AND 조건)
        if bbox:
            lat = lawyer.get("latitude")
            lng = lawyer.get("longitude")
            if lat is None or lng is None:
                continue
            min_lat, max_lat, min_lng, max_lng = bbox
            if not (min_lat <= lat <= max_lat and min_lng <= lng <= max_lng):
                continue
            # 정확한 거리 계산
            dist = haversine(longitude, latitude, lng, lat)
            if dist > (radius_m / 1000):
                continue

        results.append({**lawyer, "id": idx})

        if limit and len(results) >= limit:
            break

    return results


def get_clusters(
    min_lat: float,
    max_lat: float,
    min_lng: float,
    max_lng: float,
    grid_size: float = 0.01  # 약 1km 그리드
) -> List[dict]:
    """
    뷰포트 내 변호사를 그리드로 클러스터링
    """
    data = load_lawyers_data()
    lawyers = data.get("lawyers", [])

    # 그리드 집계
    grid = {}

    for lawyer in lawyers:
        lat = lawyer.get("latitude")
        lng = lawyer.get("longitude")

        if lat is None or lng is None:
            continue

        # 뷰포트 필터
        if not (min_lat <= lat <= max_lat and min_lng <= lng <= max_lng):
            continue

        # 그리드 셀 계산
        grid_lat = round(lat / grid_size) * grid_size
        grid_lng = round(lng / grid_size) * grid_size
        key = (grid_lat, grid_lng)

        if key not in grid:
            grid[key] = {"latitude": grid_lat, "longitude": grid_lng, "count": 0}
        grid[key]["count"] += 1

    return list(grid.values())


def get_zoom_grid_size(zoom: int) -> float:
    """줌 레벨에 따른 그리드 크기 결정"""
    # 줌 레벨이 높을수록 (확대) 그리드 크기 작게
    grid_sizes = {
        5: 0.1,    # 약 10km
        6: 0.08,
        7: 0.05,
        8: 0.03,
        9: 0.02,
        10: 0.01,  # 약 1km
        11: 0.005,
        12: 0.003,
    }
    return grid_sizes.get(zoom, 0.01)