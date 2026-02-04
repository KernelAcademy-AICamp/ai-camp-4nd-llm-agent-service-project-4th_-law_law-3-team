"""
변호사 서비스

위치 및 전문분야 추출, 변호사 검색 지원
"""

import json
import logging
import re
from functools import lru_cache
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# =============================================================================
# 데이터 파일 경로
# =============================================================================
# __file__ = backend/app/services/service_function/lawyer_service.py
# 5 parents up = law-3-team/ (프로젝트 루트)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LAWYERS_FILE = DATA_DIR / "lawyers_with_coords.json"
FALLBACK_FILE = PROJECT_ROOT / "all_lawyers.json"

# =============================================================================
# 전문분야 12대분류 (사용자에게는 이것만 표시)
# =============================================================================
SPECIALTY_CATEGORIES: Dict[str, Dict[str, Any]] = {
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

# =============================================================================
# 에이전트용 메시지 파싱 상수
# =============================================================================
SPECIALTY_KEYWORDS: Dict[str, List[str]] = {
    "민사": ["민사", "계약", "채권", "채무", "손해배상", "임대차", "전세", "월세"],
    "형사": ["형사", "범죄", "고소", "고발", "구속", "기소", "재판"],
    "가사": ["이혼", "양육권", "상속", "유언", "재산분할", "가사"],
    "부동산": ["부동산", "토지", "건물", "등기", "분양", "재개발"],
    "기업": ["회사", "법인", "기업", "M&A", "합병", "인수"],
    "노동": ["노동", "근로", "해고", "임금", "퇴직금", "산재"],
    "행정": ["행정", "허가", "인허가", "소송", "취소"],
    "지적재산권": ["특허", "상표", "저작권", "지식재산", "IP"],
    "세무": ["세금", "세무", "조세", "탈세", "국세"],
    "의료": ["의료", "병원", "의사", "의료사고", "의료분쟁"],
}

REGION_PATTERNS = [
    r"(서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|전북|전남|경북|경남|제주)",
    r"(강남|서초|송파|마포|영등포|종로|중구|용산|성동|광진|동대문|중랑|성북|강북|도봉|노원|"
    r"은평|서대문|양천|구로|금천|동작|관악|강서|강동|잠실|판교|분당|일산|수원|성남)",
]


# =============================================================================
# 카테고리/전문분야 유틸리티 함수
# =============================================================================
def get_specialties_by_category(category: str) -> Set[str]:
    """카테고리 ID로 해당 카테고리의 전문분야 목록 조회"""
    if category in SPECIALTY_CATEGORIES:
        return set(SPECIALTY_CATEGORIES[category]["specialties"])
    return set()


def get_categories() -> List[Dict[str, Any]]:
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


# =============================================================================
# 데이터 로드
# =============================================================================
@lru_cache(maxsize=1)
def load_lawyers_data() -> Dict[str, Any]:
    """변호사 데이터 로드 (캐싱)"""
    files_to_try = [
        LAWYERS_FILE,       # data/lawyers_with_coords.json (좌표 + 전문분야 포함)
        FALLBACK_FILE,      # all_lawyers.json (원본 데이터, 좌표 없음)
    ]

    for file_path in files_to_try:
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    result: Dict[str, Any] = json.load(f)
                    logger.info(f"Loaded lawyers data from: {file_path}")
                    return result
            except json.JSONDecodeError as e:
                logger.error(f"JSON 파싱 오류 ({file_path}): {e}")
                continue
            except UnicodeDecodeError as e:
                logger.error(f"인코딩 오류 ({file_path}): {e}")
                continue

    logger.warning("변호사 데이터 파일을 찾을 수 없습니다")
    return {"lawyers": [], "metadata": {}}


def get_available_specialties() -> List[str]:
    """사용 가능한 전문분야 목록 조회"""
    data = load_lawyers_data()
    lawyers = data.get("lawyers", [])

    specialties_set: Set[str] = set()
    for lawyer in lawyers:
        specs = lawyer.get("specialties", [])
        if isinstance(specs, list):
            specialties_set.update(specs)

    return sorted(specialties_set)


# =============================================================================
# 거리 계산 유틸리티
# =============================================================================
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


# =============================================================================
# 변호사 검색 함수
# =============================================================================
def find_nearby_lawyers(
    latitude: float,
    longitude: float,
    radius_m: int = 5000,
    limit: Optional[int] = None,
    category: Optional[str] = None,
    specialty: Optional[str] = None
) -> Dict[str, Any]:
    """
    반경 내 변호사 검색

    1단계: 바운딩 박스로 1차 필터링 (빠름)
    2단계: Haversine 공식으로 정확한 거리 계산
    3단계: 전문분야 필터링 (specialty > category 우선순위)

    Args:
        latitude: 위도
        longitude: 경도
        radius_m: 검색 반경 (미터)
        limit: 최대 결과 수 (None이면 제한 없음)
        specialty: 특정 전문분야 키워드 (예: "이혼", "형사법") - 정확히 일치하는 전문분야 필터
        category: 전문분야 카테고리 ID (예: "civil-family") - 카테고리 내 모든 전문분야 필터

    Returns:
        {"lawyers": [...], "total_count": int} - total_count는 limit 적용 전 전체 건수
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

    total_count = len(results)
    limited = results[:limit] if limit else results
    return {"lawyers": limited, "total_count": total_count}


def get_lawyer_by_id(lawyer_id: int) -> Optional[Dict[str, Any]]:
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
    category: Optional[str] = None,
    specialty: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    radius_m: int = 5000,
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """
    이름/사무소/지역/전문분야로 검색

    Args:
        name: 이름 검색 (OR 조건)
        office: 사무소명 검색 (OR 조건)
        district: 지역(구/군) 검색 (AND 조건)
        category: 전문분야 카테고리 ID (AND 조건)
        specialty: 특정 전문분야 키워드 (AND 조건, category보다 우선)
        latitude: 위치 필터링 위도
        longitude: 위치 필터링 경도
        radius_m: 반경 (미터)
        limit: 최대 결과 수

    Returns:
        {"lawyers": [...], "total_count": int} - total_count는 limit 적용 전 전체 건수

    Raises:
        ValueError: latitude와 longitude 중 하나만 제공된 경우
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
    if has_latitude and has_longitude and latitude is not None and longitude is not None:
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
        dist: Optional[float] = None
        if bbox:
            lat = lawyer.get("latitude")
            lng = lawyer.get("longitude")
            if lat is None or lng is None:
                continue
            min_lat, max_lat, min_lng, max_lng = bbox
            if not (min_lat <= lat <= max_lat and min_lng <= lng <= max_lng):
                continue
            # 정확한 거리 계산 (bbox가 있으면 latitude, longitude는 None이 아님)
            if latitude is not None and longitude is not None:
                dist = haversine(longitude, latitude, lng, lat)
                if dist > (radius_m / 1000):
                    continue

        result_item: Dict[str, Any] = {**lawyer, "id": idx}
        if dist is not None:
            result_item["distance"] = round(dist, 2)
        results.append(result_item)

    # 위치 검색 시 거리순 정렬
    if bbox:
        results.sort(key=lambda x: x.get("distance", float("inf")))

    total_count = len(results)
    limited = results[:limit] if limit else results
    return {"lawyers": limited, "total_count": total_count}


# =============================================================================
# 클러스터링 함수
# =============================================================================
def get_clusters(
    min_lat: float,
    max_lat: float,
    min_lng: float,
    max_lng: float,
    grid_size: float = 0.01,
    category: Optional[str] = None,
    specialty: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    뷰포트 내 변호사를 그리드로 클러스터링

    Args:
        category: 전문분야 카테고리 ID (예: "criminal")
        specialty: 특정 전문분야 (예: "이혼") - category보다 우선
    """
    data = load_lawyers_data()
    lawyers = data.get("lawyers", [])

    # 카테고리에 해당하는 전문분야 목록
    category_specs = get_specialties_by_category(category) if category and not specialty else set()

    # 그리드 집계
    grid: Dict[Tuple[float, float], Dict[str, Any]] = {}

    for lawyer in lawyers:
        lat = lawyer.get("latitude")
        lng = lawyer.get("longitude")

        if lat is None or lng is None:
            continue

        # 뷰포트 필터
        if not (min_lat <= lat <= max_lat and min_lng <= lng <= max_lng):
            continue

        # 전문분야 필터
        if specialty or category_specs:
            lawyer_specs = lawyer.get("specialties", [])
            if not isinstance(lawyer_specs, list):
                lawyer_specs = []
            if specialty:
                if specialty not in lawyer_specs:
                    continue
            elif category_specs:
                if not category_specs.intersection(lawyer_specs):
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


# =============================================================================
# 에이전트용 메시지 파싱 클래스
# =============================================================================
class LawyerService:
    """변호사 서비스 클래스 (에이전트용 메시지 파싱)"""

    def extract_location(self, message: str) -> Optional[Dict[str, Any]]:
        """
        메시지에서 위치 정보 추출

        Args:
            message: 사용자 메시지

        Returns:
            {"region": "지역명", "sub_region": "세부지역"} 또는 None
        """
        location: Dict[str, Any] = {}

        # 시/도 추출
        for pattern in REGION_PATTERNS:
            match = re.search(pattern, message)
            if match:
                region = match.group(1)
                if region in ["서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
                              "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"]:
                    location["region"] = region
                else:
                    location["sub_region"] = region

        if location:
            return location
        return None

    def extract_specialty(self, message: str) -> Optional[str]:
        """
        메시지에서 전문분야 추출

        Args:
            message: 사용자 메시지

        Returns:
            전문분야명 또는 None
        """
        message_lower = message.lower()

        for specialty, keywords in SPECIALTY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in message_lower:
                    return specialty

        return None

    def extract_requirements(self, message: str) -> Dict[str, Any]:
        """
        메시지에서 변호사 검색 요구사항 추출

        Args:
            message: 사용자 메시지

        Returns:
            {"location": {...}, "specialty": "...", "keywords": [...]}
        """
        return {
            "location": self.extract_location(message),
            "specialty": self.extract_specialty(message),
            "keywords": self._extract_keywords(message),
        }

    def _extract_keywords(self, message: str) -> List[str]:
        """메시지에서 검색 키워드 추출"""
        keywords = []

        # 법률 용어 키워드 추출
        legal_terms = [
            "손해배상", "계약위반", "사기", "횡령", "배임",
            "이혼", "상속", "유언", "임대차", "전세",
            "해고", "퇴직금", "산재", "의료사고",
        ]

        for term in legal_terms:
            if term in message:
                keywords.append(term)

        return keywords


# =============================================================================
# 싱글톤 인스턴스
# =============================================================================
_lawyer_service: Optional[LawyerService] = None


def get_lawyer_service() -> LawyerService:
    """LawyerService 싱글톤 인스턴스 반환"""
    global _lawyer_service
    if _lawyer_service is None:
        _lawyer_service = LawyerService()
    return _lawyer_service
