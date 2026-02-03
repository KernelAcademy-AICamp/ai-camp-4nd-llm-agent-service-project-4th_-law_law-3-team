"""
통합 서비스 함수 모듈
"""

from app.services.service_function.precedent_service import (
    fetch_precedent_details,
    PrecedentService,
    get_precedent_service,
)

from app.services.service_function.law_service import (
    extract_law_names,
    fetch_laws_by_names,
    fetch_reference_articles_from_docs,
    LawService,
    get_law_service,
)

from app.services.service_function.lawyer_service import (
    # 에이전트용 메시지 파싱
    SPECIALTY_KEYWORDS,
    REGION_PATTERNS,
    LawyerService,
    get_lawyer_service,
    # 전문분야 12대분류
    SPECIALTY_CATEGORIES,
    get_specialties_by_category,
    get_categories,
    # 데이터 로드
    load_lawyers_data,
    get_available_specialties,
    # 거리 계산
    haversine,
    get_bounding_box,
    # 변호사 검색
    find_nearby_lawyers,
    get_lawyer_by_id,
    search_lawyers,
    # 클러스터링
    get_clusters,
    get_zoom_grid_size,
)

from app.services.service_function.lawyer_stats_service import (
    # 인구 데이터
    get_population_data,
    get_population_meta,
    # 지역 정규화
    PROVINCE_NORMALIZE_MAP,
    normalize_province,
    extract_region,
    get_category_for_specialty,
    # 통계 계산
    calculate_overview,
    calculate_by_region,
    calculate_density_by_region,
    calculate_by_specialty,
    calculate_specialty_by_region,
    # 교차 분석
    calculate_cross_analysis,
    calculate_cross_analysis_by_regions,
    calculate_cross_analysis_by_province,
)

from app.services.service_function.small_claims_service import (
    SMALL_CLAIMS_STEPS,
    DISPUTE_TYPE_KEYWORDS,
    SmallClaimsService,
    get_small_claims_service,
)

__all__ = [
    # 판례
    "fetch_precedent_details",
    "PrecedentService",
    "get_precedent_service",
    # 법령
    "extract_law_names",
    "fetch_laws_by_names",
    "fetch_reference_articles_from_docs",
    "LawService",
    "get_law_service",
    # 변호사 - 에이전트용 메시지 파싱
    "SPECIALTY_KEYWORDS",
    "REGION_PATTERNS",
    "LawyerService",
    "get_lawyer_service",
    # 변호사 - 전문분야 12대분류
    "SPECIALTY_CATEGORIES",
    "get_specialties_by_category",
    "get_categories",
    # 변호사 - 데이터 로드
    "load_lawyers_data",
    "get_available_specialties",
    # 변호사 - 거리 계산
    "haversine",
    "get_bounding_box",
    # 변호사 - 검색
    "find_nearby_lawyers",
    "get_lawyer_by_id",
    "search_lawyers",
    # 변호사 - 클러스터링
    "get_clusters",
    "get_zoom_grid_size",
    # 변호사 통계 - 인구 데이터
    "get_population_data",
    "get_population_meta",
    # 변호사 통계 - 지역 정규화
    "PROVINCE_NORMALIZE_MAP",
    "normalize_province",
    "extract_region",
    "get_category_for_specialty",
    # 변호사 통계 - 통계 계산
    "calculate_overview",
    "calculate_by_region",
    "calculate_density_by_region",
    "calculate_by_specialty",
    "calculate_specialty_by_region",
    # 변호사 통계 - 교차 분석
    "calculate_cross_analysis",
    "calculate_cross_analysis_by_regions",
    "calculate_cross_analysis_by_province",
    # 소액소송
    "SMALL_CLAIMS_STEPS",
    "DISPUTE_TYPE_KEYWORDS",
    "SmallClaimsService",
    "get_small_claims_service",
]
