"""
관할법원 매핑 서비스

scourt_region_courts.json을 로드하여 region(시군구) → court_name 매핑 제공.
trial_statistics 테이블의 court_name 형식에 맞게 변환.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.services.service_function.lawyer_stats_service import (
    DISTRICT_NORMALIZE_MAP,
    PROVINCE_NORMALIZE_MAP,
)

# =============================================================================
# 상수 정의
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
SCOURT_JSON_PATH = PROJECT_ROOT / "data" / "scourt_region_courts.json"

# 일반 방위명 지원 (parent city prefix로 정규화 필요)
GENERIC_BRANCH_NAMES = frozenset({"서부지원", "동부지원"})

# 법원명 접미사 (도시 접두사 추출용)
COURT_SUFFIXES = ("지방법원", "가정법원", "행정법원", "회생법원")


# =============================================================================
# 데이터 구조
# =============================================================================
@dataclass(frozen=True)
class CourtHierarchy:
    """법원 계층 정보

    Attributes:
        main_court: 본원 (지방법원/가정법원/행정법원)
        branch_court: 지원 (None이면 본원에 직접 속함)
    """

    main_court: str
    branch_court: str | None


# =============================================================================
# 내부 함수
# =============================================================================
def _extract_city_prefix(court_name: str) -> str:
    """법원명에서 도시 접두사 추출.

    '대구지방법원' → '대구', '수원가정법원' → '수원'
    """
    for suffix in COURT_SUFFIXES:
        if court_name.endswith(suffix):
            return court_name[: -len(suffix)]
    return court_name


def _qualify_branch_name(parent_court: str, branch: str) -> str:
    """일반 방위명 지원을 parent_court 도시 접두사로 정규화.

    '서부지원' + '대구지방법원' → '대구서부지원'
    '강릉지원' + '춘천지방법원' → '강릉지원' (이미 지리적 식별자 포함)
    """
    if branch in GENERIC_BRANCH_NAMES:
        city = _extract_city_prefix(parent_court)
        return f"{city}{branch}"
    return branch


def _parse_scourt_court_name(court_name: str) -> CourtHierarchy:
    """scourt court_name을 CourtHierarchy로 변환.

    scourt 형식 → trial_statistics 호환 형식:
      "서울중앙지방법원"               → main="서울중앙지방법원", branch=None
      "의정부지방법원 고양지원"         → main="의정부지방법원", branch="고양지원"
      "대구지방법원 서부지원"           → main="대구지방법원", branch="대구서부지원"
      "춘천지방법원 강릉지원 동해시법원" → main="춘천지방법원", branch="강릉지원"
      "울산지방법원 양산시법원"         → main="울산지방법원", branch=None
      "대전지방법원 세종특별자치시법원"  → main="대전지방법원", branch=None
    """
    parts = court_name.split()

    if len(parts) == 1:
        # 본원: "서울중앙지방법원"
        return CourtHierarchy(main_court=parts[0], branch_court=None)

    main_court = parts[0]

    if len(parts) >= 3:
        # 3부분: "지방법원 지원 시법원/군법원"
        # 시법원/군법원은 무시, 지원 부분을 branch로 사용
        branch = _qualify_branch_name(main_court, parts[1])
        return CourtHierarchy(main_court=main_court, branch_court=branch)

    # 2부분: "지방법원 지원" 또는 "지방법원 시법원/군법원"
    second = parts[1]
    if second.endswith("지원"):
        branch = _qualify_branch_name(main_court, second)
        return CourtHierarchy(main_court=main_court, branch_court=branch)

    # 시법원/군법원이 본원 바로 아래에 있는 경우 → branch 없음
    return CourtHierarchy(main_court=main_court, branch_court=None)


def _normalize_region(address1: str, address2: str) -> str | None:
    """주소를 정규화된 region 문자열로 변환.

    Returns:
        "서울 강남구" 형태의 region, 또는 address2가 비어있으면 None
    """
    if not address2.strip():
        return None
    province = PROVINCE_NORMALIZE_MAP.get(address1, address1)
    region = f"{province} {address2}"
    return DISTRICT_NORMALIZE_MAP.get(region, region)


# =============================================================================
# 공개 함수
# =============================================================================
@lru_cache(maxsize=1)
def _load_scourt_data() -> list[dict[str, str]]:
    """scourt_region_courts.json 로드 및 캐싱."""
    with open(SCOURT_JSON_PATH, encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    return data["records"]  # type: ignore[no-any-return]


@lru_cache(maxsize=16)
def get_region_to_court_map(
    court_category: str,
) -> dict[str, CourtHierarchy]:
    """region → CourtHierarchy 매핑 반환.

    Args:
        court_category: scourt 카테고리 ("general", "family", "administrative")

    Returns:
        {"서울 강남구": CourtHierarchy(main="서울중앙지방법원", branch=None), ...}

    Notes:
        하나의 시군구에 여러 법원이 매핑될 수 있음 (시법원/군법원 + 지원).
        branch_court가 있는 매핑을 우선하여 최종 하나만 반환.
    """
    records = _load_scourt_data()
    result: dict[str, CourtHierarchy] = {}

    for record in records:
        if record["category"] != court_category:
            continue

        region = _normalize_region(record["address1"], record["address2"])
        if region is None:
            continue

        hierarchy = _parse_scourt_court_name(record["court_name"])

        existing = result.get(region)
        if existing is None:
            result[region] = hierarchy
        elif hierarchy.branch_court and not existing.branch_court:
            # branch가 있는 매핑으로 교체 (시법원/군법원보다 지원 우선)
            result[region] = hierarchy

    return result


def get_court_to_regions_map(
    court_category: str,
) -> dict[str, list[str]]:
    """court_name → [region list] 역방향 매핑.

    trial_statistics court_name 기준: branch_court 우선, 없으면 main_court.

    Args:
        court_category: scourt 카테고리

    Returns:
        {"서울중앙지방법원": ["서울 강남구", "서울 서초구", ...], ...}
    """
    region_to_court = get_region_to_court_map(court_category)
    result: dict[str, list[str]] = {}

    for region, hierarchy in region_to_court.items():
        court = hierarchy.branch_court or hierarchy.main_court
        result.setdefault(court, []).append(region)

    return result


def resolve_court_name(
    hierarchy: CourtHierarchy,
    available_courts: set[str],
) -> str | None:
    """CourtHierarchy에서 trial_statistics에 존재하는 court_name 결정.

    branch_court → main_court 순서로 매칭을 시도.

    Args:
        hierarchy: 법원 계층 정보
        available_courts: trial_statistics에 존재하는 court_name 집합

    Returns:
        매칭된 court_name, 없으면 None
    """
    if hierarchy.branch_court and hierarchy.branch_court in available_courts:
        return hierarchy.branch_court
    if hierarchy.main_court in available_courts:
        return hierarchy.main_court
    return None
