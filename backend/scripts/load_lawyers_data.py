"""
변호사 데이터 PostgreSQL 로드 스크립트

data/lawyers_with_coords.json → lawyers 테이블

Usage:
    uv run python scripts/load_lawyers_data.py           # 로드
    uv run python scripts/load_lawyers_data.py --reset    # 삭제 후 재로드
    uv run python scripts/load_lawyers_data.py --verify   # 검증만
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import create_engine, func, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.core.database import Base
from app.models.lawyer import Lawyer
from app.services.service_function.lawyer_stats_service import (
    DISTRICT_NORMALIZE_MAP,
    PROVINCE_NORMALIZE_MAP,
    REGION_PATTERN,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT.parent / "data"
LAWYERS_FILE = DATA_DIR / "lawyers_with_coords.json"

BATCH_SIZE = 1000


def normalize_province(province: str) -> str:
    """시/도명을 표준 형식으로 정규화."""
    return PROVINCE_NORMALIZE_MAP.get(province, province)


def extract_region_parts(address: str | None) -> tuple[str | None, str | None, str | None]:
    """
    주소에서 province, district, region 추출.

    Returns:
        (province, district, region) 또는 (None, None, None)
    """
    if not address:
        return None, None, None

    # 세종시 특별 처리
    first_token = address.split()[0] if address.split() else ""
    normalized_first = normalize_province(first_token)
    if normalized_first == "세종":
        return "세종", None, "세종"

    match = REGION_PATTERN.match(address)
    if match:
        province = normalize_province(match.group(1))
        district = match.group(2)
        region = f"{province} {district}"
        # 행정구역 통합/개칭 반영
        region = DISTRICT_NORMALIZE_MAP.get(region, region)
        # region에서 district 다시 추출 (정규화 후 달라질 수 있음)
        parts = region.split(" ", 1)
        if len(parts) == 2:
            province, district = parts
        return province, district, region

    return None, None, None


def load_json_data() -> dict:
    """JSON 파일 로드."""
    if not LAWYERS_FILE.exists():
        logger.error(f"파일이 없습니다: {LAWYERS_FILE}")
        sys.exit(1)

    with open(LAWYERS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    lawyers = data.get("lawyers", [])
    logger.info(f"JSON 파일 로드 완료: {len(lawyers)}건")
    return data


def prepare_record(lawyer: dict) -> dict:
    """JSON 레코드를 DB 레코드로 변환."""
    province, district, region = extract_region_parts(lawyer.get("address"))

    specialties = lawyer.get("specialties", [])
    if not isinstance(specialties, list):
        specialties = []

    return {
        "detail_id": lawyer.get("detail_id"),
        "name": lawyer.get("name", ""),
        "status": lawyer.get("status"),
        "birth_year": lawyer.get("birth_year"),
        "photo_url": lawyer.get("photo_url"),
        "office_name": lawyer.get("office_name"),
        "address": lawyer.get("address"),
        "phone": lawyer.get("phone"),
        "fax": lawyer.get("fax"),
        "email": lawyer.get("email"),
        "birthdate": lawyer.get("birthdate"),
        "local_bar": lawyer.get("local_bar"),
        "qualification": lawyer.get("qualification"),
        "klaw_url": lawyer.get("klaw_url"),
        "latitude": lawyer.get("latitude"),
        "longitude": lawyer.get("longitude"),
        "specialties": specialties,
        "province": province,
        "district": district,
        "region": region,
    }


def load_to_db(session: sessionmaker, lawyers: list[dict], reset: bool = False) -> int:
    """변호사 데이터를 DB에 로드."""
    with session() as db:
        if reset:
            count = db.query(Lawyer).count()
            db.execute(text("TRUNCATE TABLE lawyers RESTART IDENTITY CASCADE"))
            db.commit()
            logger.info(f"기존 데이터 {count}건 삭제 완료")

        total_loaded = 0
        start_time = time.time()

        for i in range(0, len(lawyers), BATCH_SIZE):
            batch = lawyers[i:i + BATCH_SIZE]
            records = [prepare_record(lawyer) for lawyer in batch]

            # ON CONFLICT (detail_id) DO UPDATE 로 멱등성 보장
            stmt = insert(Lawyer).values(records)
            update_cols = {
                col.name: col
                for col in stmt.excluded
                if col.name not in ("id", "detail_id", "created_at")
            }
            stmt = stmt.on_conflict_do_update(
                index_elements=["detail_id"],
                set_=update_cols,
            )
            db.execute(stmt)
            db.commit()

            total_loaded += len(batch)
            elapsed = time.time() - start_time
            logger.info(
                f"  진행: {total_loaded}/{len(lawyers)} "
                f"({total_loaded / len(lawyers) * 100:.1f}%) "
                f"[{elapsed:.1f}s]"
            )

        return total_loaded


def verify_data(session: sessionmaker, expected_count: int) -> None:
    """로드된 데이터 검증."""
    with session() as db:
        total = db.query(func.count(Lawyer.id)).scalar() or 0
        with_coords = db.query(func.count(Lawyer.id)).filter(
            Lawyer.latitude.isnot(None),
            Lawyer.longitude.isnot(None),
        ).scalar() or 0
        with_specialties = db.query(func.count(Lawyer.id)).filter(
            func.array_length(Lawyer.specialties, 1) > 0,
        ).scalar() or 0
        with_region = db.query(func.count(Lawyer.id)).filter(
            Lawyer.region.isnot(None),
        ).scalar() or 0

        # 지역별 상위 5개
        top_regions = (
            db.query(Lawyer.region, func.count(Lawyer.id).label("cnt"))
            .filter(Lawyer.region.isnot(None))
            .group_by(Lawyer.region)
            .order_by(func.count(Lawyer.id).desc())
            .limit(5)
            .all()
        )

    logger.info("=" * 60)
    logger.info("데이터 검증 결과")
    logger.info("=" * 60)
    logger.info(f"  총 건수:       {total:,}건 (예상: {expected_count:,}건)")
    logger.info(f"  좌표 보유:     {with_coords:,}건 ({with_coords / total * 100:.1f}%)")
    logger.info(f"  전문분야 보유: {with_specialties:,}건 ({with_specialties / total * 100:.1f}%)")
    logger.info(f"  지역 보유:     {with_region:,}건 ({with_region / total * 100:.1f}%)")
    logger.info("")
    logger.info("  지역별 상위 5:")
    for region, cnt in top_regions:
        logger.info(f"    {region}: {cnt:,}건")
    logger.info("=" * 60)

    if total != expected_count:
        logger.warning(f"건수 불일치: DB {total}건 != JSON {expected_count}건")
    else:
        logger.info("건수 일치 확인 완료")


def main() -> None:
    parser = argparse.ArgumentParser(description="변호사 데이터 PostgreSQL 로드")
    parser.add_argument("--reset", action="store_true", help="기존 데이터 삭제 후 재로드")
    parser.add_argument("--verify", action="store_true", help="검증만 실행")
    args = parser.parse_args()

    # Sync engine 사용 (스크립트용)
    engine = create_engine(
        settings.DATABASE_URL,
        echo=False,
        pool_size=5,
        pool_pre_ping=True,
    )
    Session = sessionmaker(engine)

    # JSON 데이터 로드
    data = load_json_data()
    lawyers = data.get("lawyers", [])

    if args.verify:
        verify_data(Session, len(lawyers))
        return

    # DB 로드
    logger.info(f"변호사 데이터 로드 시작 (reset={args.reset})")
    total_loaded = load_to_db(Session, lawyers, reset=args.reset)
    logger.info(f"로드 완료: {total_loaded:,}건")

    # 검증
    verify_data(Session, len(lawyers))


if __name__ == "__main__":
    main()
