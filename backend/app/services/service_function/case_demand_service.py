"""
사건 수요 통계 서비스

trial_statistics DB + lawyers_2010_2025.csv를 결합하여
지역별 사건 수요·부담지수를 계산.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.trial_statistics import TrialStatistics
from app.services.service_function.court_mapping_service import (
    get_region_to_court_map,
    resolve_court_name,
)

# =============================================================================
# 상수 정의
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
LAWYER_CSV_PATH = PROJECT_ROOT / "data" / "lawyers_2010_2025.csv"

# UI 분야 → (scourt_category, trial_statistics categories) 매핑
CATEGORY_MAP: dict[str, tuple[str, list[str]]] = {
    "민사": ("general", ["민사_본안_단독", "민사_본안_합의"]),
    "형사": ("general", ["형사_공판", "형사_약식"]),
    "가사": ("family", ["가사"]),
    "행정": ("administrative", ["행정"]),
    "소년보호": ("family", ["소년보호"]),
    "가정보호": ("family", ["가정보호"]),
}

# CSV (province_group, court_short_name) → trial_statistics court_name 매핑
LAWYER_CSV_COURT_MAP: dict[tuple[str, str], str] = {
    ("서울", "중앙"): "서울중앙지방법원",
    ("서울", "동부"): "서울동부지방법원",
    ("서울", "서부"): "서울서부지방법원",
    ("서울", "남부"): "서울남부지방법원",
    ("서울", "북부"): "서울북부지방법원",
    ("경기북부", "의정부"): "의정부지방법원",
    ("경기북부", "고양"): "고양지원",
    ("경기북부", "남양주"): "남양주지원",
    ("경기중앙", "수원"): "수원지방법원",
    ("경기중앙", "안산"): "안산지원",
    ("경기중앙", "성남"): "성남지원",
    ("경기중앙", "여주"): "여주지원",
    ("경기중앙", "평택"): "평택지원",
    ("경기중앙", "안양"): "안양지원",
    ("인천", "인천"): "인천지방법원",
    ("인천", "부천"): "부천지원",
    ("강원", "춘천"): "춘천지방법원",
    ("강원", "강릉"): "강릉지원",
    ("강원", "원주"): "원주지원",
    ("강원", "속초"): "속초지원",
    ("강원", "영월"): "영월지원",
    ("충북", "청주"): "청주지방법원",
    ("충북", "충주"): "충주지원",
    ("충북", "제천"): "제천지원",
    ("충북", "영동"): "영동지원",
    ("대전", "대전"): "대전지방법원",
    ("대전", "홍성"): "홍성지원",
    ("대전", "공주"): "공주지원",
    ("대전", "논산"): "논산지원",
    ("대전", "서산"): "서산지원",
    ("대전", "천안"): "천안지원",
    ("대구", "대구"): "대구지방법원",
    ("대구", "서부"): "대구서부지원",
    ("대구", "안동"): "안동지원",
    ("대구", "김천"): "김천지원",
    ("대구", "상주"): "상주지원",
    ("대구", "경주"): "경주지원",
    ("대구", "영덕"): "영덕지원",
    ("대구", "포항"): "포항지원",
    ("대구", "의성"): "의성지원",
    ("부산", "부산"): "부산지방법원",
    ("부산", "동부"): "부산동부지원",
    ("부산", "서부"): "부산서부지원",
    ("울산", "울산"): "울산지방법원",
    ("울산", "양산"): "울산지방법원",  # 양산시법원은 울산지방법원 소속
    ("경남", "창원"): "창원지방법원",
    ("경남", "진주"): "진주지원",
    ("경남", "통영"): "통영지원",
    ("경남", "마산"): "마산지원",
    ("경남", "밀양"): "밀양지원",
    ("경남", "거창"): "거창지원",
    ("광주", "광주"): "광주지방법원",
    ("광주", "목포"): "목포지원",
    ("광주", "장흥"): "장흥지원",
    ("광주", "순천"): "순천지원",
    ("광주", "해남"): "해남지원",
    ("전북", "전주"): "전주지방법원",
    ("전북", "군산"): "군산지원",
    ("전북", "정읍"): "정읍지원",
    ("전북", "남원"): "남원지원",
    ("제주", "제주"): "제주지방법원",
    ("제주", "서귀포"): "제주지방법원",  # 서귀포시법원은 제주지방법원 소속
}

# trial_statistics court_name → CSV group 매핑
# 가정법원/행정법원 등 per-court 매핑이 불가능한 법원의 group-level 집계용
COURT_TO_CSV_GROUP: dict[str, str] = {
    # General courts + branches (per-court 매핑과 group-level 양쪽 모두 지원)
    "서울중앙지방법원": "서울",
    "서울동부지방법원": "서울",
    "서울서부지방법원": "서울",
    "서울남부지방법원": "서울",
    "서울북부지방법원": "서울",
    "의정부지방법원": "경기북부",
    "고양지원": "경기북부",
    "남양주지원": "경기북부",
    "수원지방법원": "경기중앙",
    "안산지원": "경기중앙",
    "성남지원": "경기중앙",
    "여주지원": "경기중앙",
    "평택지원": "경기중앙",
    "안양지원": "경기중앙",
    "인천지방법원": "인천",
    "부천지원": "인천",
    "춘천지방법원": "강원",
    "강릉지원": "강원",
    "원주지원": "강원",
    "속초지원": "강원",
    "영월지원": "강원",
    "청주지방법원": "충북",
    "충주지원": "충북",
    "제천지원": "충북",
    "영동지원": "충북",
    "대전지방법원": "대전",
    "홍성지원": "대전",
    "공주지원": "대전",
    "논산지원": "대전",
    "서산지원": "대전",
    "천안지원": "대전",
    "대구지방법원": "대구",
    "대구서부지원": "대구",
    "안동지원": "대구",
    "김천지원": "대구",
    "상주지원": "대구",
    "경주지원": "대구",
    "영덕지원": "대구",
    "포항지원": "대구",
    "의성지원": "대구",
    "부산지방법원": "부산",
    "부산동부지원": "부산",
    "부산서부지원": "부산",
    "울산지방법원": "울산",
    "창원지방법원": "경남",
    "진주지원": "경남",
    "통영지원": "경남",
    "마산지원": "경남",
    "밀양지원": "경남",
    "거창지원": "경남",
    "광주지방법원": "광주",
    "목포지원": "광주",
    "장흥지원": "광주",
    "순천지원": "광주",
    "해남지원": "광주",
    "전주지방법원": "전북",
    "군산지원": "전북",
    "정읍지원": "전북",
    "남원지원": "전북",
    "제주지방법원": "제주",
    # 가정법원 (본원) → CSV group
    "서울가정법원": "서울",
    "수원가정법원": "경기중앙",
    "인천가정법원": "인천",
    "대전가정법원": "대전",
    "대구가정법원": "대구",
    "부산가정법원": "부산",
    "울산가정법원": "울산",
    "광주가정법원": "광주",
    # 행정법원
    "서울행정법원": "서울",
}


# =============================================================================
# CSV 리더
# =============================================================================
@lru_cache(maxsize=1)
def _load_lawyer_csv_raw() -> dict[str, dict[int, int]]:
    """CSV 파싱 → {court_name: {year: count}}.

    CSV에서 총계 행 제외, LAWYER_CSV_COURT_MAP으로 법원명 변환.
    동일 법원에 매핑되는 행은 합산 (예: 울산+양산 → 울산지방법원).
    """
    result: dict[str, dict[int, int]] = {}

    with open(LAWYER_CSV_PATH, encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    # 헤더: ['', '', '2010', '2011', ...]
    header = rows[0]
    year_indices: list[tuple[int, int]] = []
    for col_idx in range(2, len(header)):
        val = header[col_idx].strip()
        if val.isdigit():
            year_indices.append((col_idx, int(val)))

    current_group = ""

    for row in rows[1:]:
        group = row[0].strip()
        court_short = row[1].strip() if len(row) > 1 else ""

        if group:
            current_group = group

        # 총계 행 또는 빈 court 스킵
        if current_group == "총계" or not court_short:
            continue

        court_name = LAWYER_CSV_COURT_MAP.get((current_group, court_short))
        if court_name is None:
            continue

        if court_name not in result:
            result[court_name] = {}

        for col_idx, year in year_indices:
            if col_idx >= len(row):
                continue
            val = row[col_idx].strip().replace(",", "")
            if not val:
                continue
            try:
                count = int(val)
            except ValueError:
                continue
            result[court_name][year] = result[court_name].get(year, 0) + count

    return result


def get_lawyer_count_by_court(year: int = 2024) -> dict[str, int]:
    """특정 연도의 per-court 변호사 수 반환.

    Returns:
        {"서울중앙지방법원": 19722, "고양지원": 212, ...}
    """
    raw = _load_lawyer_csv_raw()
    return {court: years.get(year, 0) for court, years in raw.items()}


def get_lawyer_count_by_group(year: int = 2024) -> dict[str, int]:
    """특정 연도의 CSV group별 변호사 수 합산.

    Returns:
        {"서울": 22923, "경기북부": 586, "경기중앙": 1234, ...}
    """
    per_court = get_lawyer_count_by_court(year)
    group_totals: dict[str, int] = {}

    for court_name, count in per_court.items():
        group = COURT_TO_CSV_GROUP.get(court_name)
        if group:
            group_totals[group] = group_totals.get(group, 0) + count

    return group_totals


def get_lawyer_count_for_court(court_name: str, year: int = 2024) -> int:
    """trial_statistics court_name에 대응하는 변호사 수 반환.

    1) per-court 정확 매칭 시도 (general courts + branches)
    2) 실패 시 group-level 합산 (가정법원, 행정법원 등)
    """
    per_court = get_lawyer_count_by_court(year)

    # 1) per-court 정확 매칭
    if court_name in per_court:
        return per_court[court_name]

    # 2) group-level 합산
    csv_group = COURT_TO_CSV_GROUP.get(court_name)
    if csv_group:
        group_totals = get_lawyer_count_by_group(year)
        return group_totals.get(csv_group, 0)

    return 0


def get_available_years() -> list[int]:
    """CSV에서 사용 가능한 연도 목록 반환."""
    raw = _load_lawyer_csv_raw()
    years: set[int] = set()
    for year_dict in raw.values():
        years.update(year_dict.keys())
    return sorted(years)


# =============================================================================
# 수요 통계 계산
# =============================================================================
async def get_court_case_counts(
    db: AsyncSession,
    trial_categories: list[str],
    year: int,
) -> dict[str, int]:
    """trial_statistics에서 법원별 사건 수 집계.

    Args:
        db: DB 세션
        trial_categories: trial_statistics category 목록
        year: 조회 연도

    Returns:
        {"서울중앙지방법원": 150000, "고양지원": 8000, ...}
    """
    stmt = (
        select(
            TrialStatistics.court_name,
            func.sum(TrialStatistics.case_count).label("total"),
        )
        .where(
            TrialStatistics.category.in_(trial_categories),
            TrialStatistics.year == year,
        )
        .group_by(TrialStatistics.court_name)
    )
    result = await db.execute(stmt)
    return {row.court_name: row.total for row in result.all()}


async def get_available_trial_years(
    db: AsyncSession,
) -> list[int]:
    """trial_statistics에 존재하는 연도 목록."""
    stmt = (
        select(TrialStatistics.year)
        .distinct()
        .order_by(TrialStatistics.year)
    )
    result = await db.execute(stmt)
    return [row[0] for row in result.all()]


async def calculate_demand_by_region(
    db: AsyncSession,
    category: str = "민사",
    year: int = 2024,
) -> dict[str, Any]:
    """지역별 사건 수요 통계 계산.

    Args:
        db: DB 세션
        category: UI 사건 분야 ("민사", "형사", "가사", "행정", "소년보호", "가정보호")
        year: 조회 연도 (2015~2024)

    Returns:
        {
            "data": [{"region": ..., "case_count": ..., ...}, ...],
            "year": 2024,
            "category": "민사",
            "available_years": [2015, ..., 2024],
            "available_categories": ["민사", "형사", ...],
        }
    """
    cat_info = CATEGORY_MAP.get(category)
    if cat_info is None:
        return {
            "data": [],
            "year": year,
            "category": category,
            "available_years": [],
            "available_categories": list(CATEGORY_MAP.keys()),
        }

    scourt_category, trial_categories = cat_info

    # 1) region → court hierarchy 매핑
    region_court_map = get_region_to_court_map(scourt_category)

    # 2) trial_statistics에서 법원별 사건 수 집계
    court_cases = await get_court_case_counts(db, trial_categories, year)
    available_courts = set(court_cases.keys())

    # 3) 사용 가능 연도
    available_years = await get_available_trial_years(db)

    # 4) 각 region에 대해 사건 수·변호사 수·부담지수 계산
    data: list[dict[str, Any]] = []

    for region, hierarchy in region_court_map.items():
        court = resolve_court_name(hierarchy, available_courts)
        if court is None:
            continue

        case_count = court_cases.get(court, 0)
        if case_count == 0:
            continue

        lawyer_count = get_lawyer_count_for_court(court, year)
        burden_index = round(case_count / max(lawyer_count, 1), 2)

        data.append({
            "region": region,
            "case_count": case_count,
            "lawyer_count": lawyer_count,
            "burden_index": burden_index,
            "court_name": court,
        })

    # case_count 내림차순 정렬
    data.sort(key=lambda x: x["case_count"], reverse=True)

    return {
        "data": data,
        "year": year,
        "category": category,
        "available_years": available_years,
        "available_categories": list(CATEGORY_MAP.keys()),
    }
