"""변호사 찾기 모듈 - 서비스 레이어"""
import json
from math import radians, cos, sin, asin, sqrt
from pathlib import Path
from typing import List, Optional, Tuple
from functools import lru_cache

# 데이터 파일 경로
# __file__ = backend/app/modules/lawyer_finder/service/__init__.py
# 6 parents up = law-3-team/ (프로젝트 루트)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LAWYERS_FILE = DATA_DIR / "lawyers_with_coords.json"
FALLBACK_FILE = PROJECT_ROOT / "all_lawyers.json"


@lru_cache(maxsize=1)
def load_lawyers_data() -> dict:
    """변호사 데이터 로드 (캐싱)"""
    # 지오코딩된 파일 우선
    if LAWYERS_FILE.exists():
        with open(LAWYERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    # 폴백: 원본 파일
    if FALLBACK_FILE.exists():
        with open(FALLBACK_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    return {"lawyers": [], "metadata": {}}


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
    limit: int = 50
) -> List[dict]:
    """
    반경 내 변호사 검색

    1단계: 바운딩 박스로 1차 필터링 (빠름)
    2단계: Haversine 공식으로 정확한 거리 계산
    """
    data = load_lawyers_data()
    lawyers = data.get("lawyers", [])

    radius_km = radius_m / 1000
    min_lat, max_lat, min_lng, max_lng = get_bounding_box(latitude, longitude, radius_km)

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
        if dist <= radius_km:
            lawyer_copy = {**lawyer, "id": idx, "distance": round(dist, 2)}
            results.append(lawyer_copy)

    # 거리순 정렬
    results.sort(key=lambda x: x["distance"])

    return results[:limit]


def get_lawyer_by_id(lawyer_id: int) -> Optional[dict]:
    """ID로 변호사 조회"""
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
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    radius_m: int = 5000,
    limit: int = 50
) -> List[dict]:
    """
    이름/사무소/지역으로 검색 (이름/사무소는 OR 조건, 지역은 AND 조건)
    위치 필터가 제공되면 해당 반경 내 결과만 반환
    """
    data = load_lawyers_data()
    lawyers = data.get("lawyers", [])

    # 위치 필터링용 바운딩 박스
    bbox = None
    if latitude is not None and longitude is not None:
        radius_km = radius_m / 1000
        bbox = get_bounding_box(latitude, longitude, radius_km)

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

        if len(results) >= limit:
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
