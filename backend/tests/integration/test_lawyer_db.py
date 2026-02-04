"""
변호사 DB 서비스 통합 테스트

JSON과 DB 결과를 비교하여 데이터 무결성을 검증합니다.

Usage:
    uv run pytest tests/integration/test_lawyer_db.py -v
    uv run python tests/integration/test_lawyer_db.py  # 직접 실행
"""

import asyncio
import sys
from pathlib import Path

import pytest

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings


# DB 모드가 아니면 테스트 건너뛰기
pytestmark = pytest.mark.skipif(
    not settings.USE_DB_LAWYERS,
    reason="USE_DB_LAWYERS=false (DB 모드 비활성화)",
)


@pytest.fixture
async def db_session():
    """테스트용 DB 세션."""
    from app.core.database import async_session_factory
    async with async_session_factory() as session:
        yield session


@pytest.mark.asyncio
async def test_db_total_count(db_session):
    """DB 총 건수가 JSON과 일치하는지 확인."""
    from sqlalchemy import func, select

    from app.models.lawyer import Lawyer
    from app.services.service_function.lawyer_service import load_lawyers_data

    json_data = load_lawyers_data()
    json_count = len(json_data.get("lawyers", []))

    db_count = await db_session.scalar(select(func.count(Lawyer.id)))

    assert db_count == json_count, (
        f"건수 불일치: DB={db_count}, JSON={json_count}"
    )


@pytest.mark.asyncio
async def test_db_coord_ratio(db_session):
    """좌표 보유 비율이 JSON과 유사한지 확인."""
    from sqlalchemy import func, select

    from app.models.lawyer import Lawyer
    from app.services.service_function.lawyer_service import load_lawyers_data

    json_data = load_lawyers_data()
    json_lawyers = json_data.get("lawyers", [])
    json_with_coords = sum(
        1 for l in json_lawyers
        if l.get("latitude") is not None and l.get("longitude") is not None
    )
    json_ratio = json_with_coords / len(json_lawyers) if json_lawyers else 0

    db_total = await db_session.scalar(select(func.count(Lawyer.id))) or 0
    db_with_coords = await db_session.scalar(
        select(func.count(Lawyer.id)).where(
            Lawyer.latitude.isnot(None),
            Lawyer.longitude.isnot(None),
        )
    ) or 0
    db_ratio = db_with_coords / db_total if db_total else 0

    assert abs(db_ratio - json_ratio) < 0.01, (
        f"좌표 비율 불일치: DB={db_ratio:.3f}, JSON={json_ratio:.3f}"
    )


@pytest.mark.asyncio
async def test_db_specialty_ratio(db_session):
    """전문분야 보유 비율이 JSON과 유사한지 확인."""
    from sqlalchemy import func, select

    from app.models.lawyer import Lawyer
    from app.services.service_function.lawyer_service import load_lawyers_data

    json_data = load_lawyers_data()
    json_lawyers = json_data.get("lawyers", [])
    json_with_specs = sum(
        1 for l in json_lawyers
        if isinstance(l.get("specialties"), list) and len(l["specialties"]) > 0
    )
    json_ratio = json_with_specs / len(json_lawyers) if json_lawyers else 0

    db_total = await db_session.scalar(select(func.count(Lawyer.id))) or 0
    db_with_specs = await db_session.scalar(
        select(func.count(Lawyer.id)).where(
            func.array_length(Lawyer.specialties, 1) > 0,
        )
    ) or 0
    db_ratio = db_with_specs / db_total if db_total else 0

    assert abs(db_ratio - json_ratio) < 0.01, (
        f"전문분야 비율 불일치: DB={db_ratio:.3f}, JSON={json_ratio:.3f}"
    )


@pytest.mark.asyncio
async def test_find_nearby_db(db_session):
    """DB 기반 근접 검색이 동작하는지 확인."""
    from app.services.service_function.lawyer_db_service import (
        find_nearby_lawyers_db,
    )

    # 서울 강남구 좌표
    results = await find_nearby_lawyers_db(
        db=db_session,
        latitude=37.4979,
        longitude=127.0276,
        radius_m=5000,
        limit=10,
    )

    assert isinstance(results, list)
    assert len(results) > 0, "강남구 5km 반경 내 변호사가 없음"
    assert "distance" in results[0]
    assert results[0]["distance"] <= 5.0  # 5km 이내


@pytest.mark.asyncio
async def test_search_lawyers_db(db_session):
    """DB 기반 이름 검색이 동작하는지 확인."""
    from app.services.service_function.lawyer_db_service import search_lawyers_db

    results = await search_lawyers_db(
        db=db_session,
        district="강남구",
        limit=5,
    )

    assert isinstance(results, list)
    assert len(results) > 0, "강남구 변호사 검색 결과 없음"


@pytest.mark.asyncio
async def test_clusters_db(db_session):
    """DB 기반 클러스터링이 동작하는지 확인."""
    from app.services.service_function.lawyer_db_service import get_clusters_db

    clusters = await get_clusters_db(
        db=db_session,
        min_lat=37.4,
        max_lat=37.6,
        min_lng=126.9,
        max_lng=127.1,
        grid_size=0.01,
    )

    assert isinstance(clusters, list)
    assert len(clusters) > 0, "서울 중심부 클러스터 없음"
    assert "count" in clusters[0]


@pytest.mark.asyncio
async def test_overview_db(db_session):
    """DB 기반 overview 통계가 동작하는지 확인."""
    from app.services.service_function.lawyer_stats_db_service import (
        calculate_overview_db,
    )

    overview = await calculate_overview_db(db_session)

    assert overview["total_lawyers"] > 0
    assert isinstance(overview["status_counts"], list)
    assert overview["coord_rate"] > 0
    assert overview["specialty_rate"] > 0


@pytest.mark.asyncio
async def test_by_region_db(db_session):
    """DB 기반 지역별 통계가 동작하는지 확인."""
    from app.services.service_function.lawyer_stats_db_service import (
        calculate_by_region_db,
    )

    regions = await calculate_by_region_db(db_session)

    assert isinstance(regions, list)
    assert len(regions) > 0
    # 서울 강남구가 상위에 있어야 함
    top_regions = [r["region"] for r in regions[:5]]
    assert any("서울" in r for r in top_regions), (
        f"서울 지역이 상위 5개에 없음: {top_regions}"
    )


# =============================================================================
# 직접 실행용
# =============================================================================
async def _run_all_tests():
    """직접 실행 시 모든 테스트 수행."""
    from app.core.database import async_session_factory

    if not settings.USE_DB_LAWYERS:
        print("USE_DB_LAWYERS=false 이므로 테스트를 건너뜁니다.")
        print("DB 테스트를 실행하려면 .env에 USE_DB_LAWYERS=true 설정 후 재시도하세요.")
        return

    async with async_session_factory() as session:
        tests = [
            ("총 건수 비교", test_db_total_count),
            ("좌표 비율 비교", test_db_coord_ratio),
            ("전문분야 비율 비교", test_db_specialty_ratio),
            ("근접 검색", test_find_nearby_db),
            ("이름 검색", test_search_lawyers_db),
            ("클러스터링", test_clusters_db),
            ("overview 통계", test_overview_db),
            ("지역별 통계", test_by_region_db),
        ]

        passed = 0
        failed = 0

        for name, test_func in tests:
            try:
                await test_func(session)
                print(f"  ✓ {name}")
                passed += 1
            except Exception as e:
                print(f"  ✗ {name}: {e}")
                failed += 1

        print(f"\n결과: {passed} passed, {failed} failed")


if __name__ == "__main__":
    asyncio.run(_run_all_tests())
