"""
재판 통계 CSV → PostgreSQL 로드 스크립트

data/trial_statistics_data/*.csv → trial_statistics 테이블

Usage:
    uv run python scripts/load_trial_statistics_data.py           # 로드
    uv run python scripts/load_trial_statistics_data.py --reset    # 삭제 후 재로드
    uv run python scripts/load_trial_statistics_data.py --verify   # 검증만
"""

import argparse
import csv
import logging
import re
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
from app.models.trial_statistics import TrialStatistics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT.parent / "data" / "trial_statistics_data"

BATCH_SIZE = 1000

# CSV 파일명 → category 매핑
FILE_CATEGORY_MAP: dict[str, str] = {
    "제2항_민사_민사본안_단독_제1심.csv": "민사_본안_단독",
    "제2항_민사_민사본안_합의_제1심.csv": "민사_본안_합의",
    "제3항_가사_가사소송_제1심.csv": "가사",
    "제4항_행정_행정소송_제1심.csv": "행정",
    "제6항_형사_형사공판_제1심.csv": "형사_공판",
    "제6항_형사_약식명령.csv": "형사_약식",
    "제7항_소년보호_소년보호.csv": "소년보호",
    "제8항_가정보호_가정보호.csv": "가정보호",
}

# 소계/합계/법원 제외 키워드 (공백 제거 후 비교)
SKIP_KEYWORDS = frozenset({"합계", "소계", "법원"})

# 지리적 식별자가 없는 일반 방위명 지원 (parent_court 접두사로 정규화 필요)
GENERIC_BRANCH_NAMES = frozenset({"서부지원", "동부지원"})

# 법원명 접미사 (도시 접두사 추출용)
COURT_SUFFIXES = ("지방법원", "가정법원", "행정법원", "회생법원")

# 연도 패턴: 4자리 숫자
YEAR_PATTERN = re.compile(r"^\d{4}$")


def parse_number(value: str) -> int | None:
    """
    셀 값을 정수로 변환.

    Returns:
        정수 값, 또는 파싱 불가 시 None (빈 값, "-", 보정값 "[...]" 등)
    """
    stripped = value.strip()
    if not stripped or stripped == "-":
        return None
    # 보정값: "[218,497]" 같은 대괄호 값
    if stripped.startswith("["):
        return None
    # 쉼표 제거 후 정수 변환
    cleaned = stripped.replace(",", "")
    try:
        return int(cleaned)
    except ValueError:
        logger.warning(f"숫자 변환 실패: '{value}'")
        return None


def is_skip_row(court_name: str) -> bool:
    """
    제외해야 하는 행인지 판단.

    빈 문자열이거나, 공백 제거 후 '합계', '소계', '법원'이면 True.
    """
    if not court_name.strip():
        return True
    normalized = court_name.replace(" ", "")
    return normalized in SKIP_KEYWORDS


def determine_court_type(court_name: str) -> str:
    """
    법원 유형 판별.

    '지원'으로 끝나면 'branch', 그 외(법원으로 끝남)는 'main'.
    """
    name = court_name.strip()
    if name.endswith("지원"):
        return "branch"
    if name.endswith("법원"):
        return "main"
    # 예외적 이름 (안전장치)
    logger.warning(f"법원 유형 판별 불가, 'main'으로 처리: '{court_name}'")
    return "main"


def extract_city_prefix(court_name: str) -> str:
    """
    법원명에서 도시 접두사 추출.

    '대구지방법원' → '대구', '부산지방법원' → '부산'
    """
    for suffix in COURT_SUFFIXES:
        if court_name.endswith(suffix):
            return court_name[: -len(suffix)]
    return court_name


def qualify_court_name(court_name: str, parent_court: str | None) -> str:
    """
    일반 방위명 지원을 parent_court 도시 접두사로 정규화.

    '서부지원' + '대구지방법원' → '대구서부지원'
    '서부지원' + '부산지방법원' → '부산서부지원'
    '고양지원' → '고양지원' (이미 지리적 식별자 포함)
    """
    if court_name in GENERIC_BRANCH_NAMES and parent_court:
        city = extract_city_prefix(parent_court)
        return f"{city}{court_name}"
    return court_name


def parse_csv_file(filepath: Path, category: str) -> list[dict]:
    """
    CSV 파일을 파싱하여 레코드 리스트 반환.

    Args:
        filepath: CSV 파일 경로
        category: 사건 카테고리 (예: '민사_본안_단독')

    Returns:
        DB insert용 dict 리스트
    """
    records: list[dict] = []

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) < 2:
        logger.warning(f"데이터 부족: {filepath.name}")
        return records

    # 헤더에서 연도 컬럼 인덱스 추출 (4자리 숫자만)
    header = rows[0]
    year_columns: list[tuple[int, int]] = []  # (column_index, year)
    for col_idx, col_name in enumerate(header):
        col_stripped = col_name.strip()
        if YEAR_PATTERN.match(col_stripped):
            year_columns.append((col_idx, int(col_stripped)))

    if not year_columns:
        logger.error(f"연도 컬럼을 찾을 수 없음: {filepath.name}")
        return records

    logger.info(
        f"  {filepath.name}: 연도 {year_columns[0][1]}~{year_columns[-1][1]} "
        f"({len(year_columns)}개)"
    )

    # 본원 추적 (branch 법원의 parent_court 결정용)
    current_parent_court: str | None = None

    for row_idx, row in enumerate(rows[1:], start=2):
        if not row:
            continue

        court_name = row[0].strip()

        # 제외 행 필터링
        if is_skip_row(court_name):
            continue

        court_type = determine_court_type(court_name)

        if court_type == "main":
            current_parent_court = court_name
            parent_court = None
        else:
            parent_court = current_parent_court

        # 일반 방위명 지원 정규화 (서부지원 → 대구서부지원/부산서부지원)
        qualified_name = qualify_court_name(court_name, parent_court)

        # 연도별 레코드 생성
        for col_idx, year in year_columns:
            if col_idx >= len(row):
                continue
            case_count = parse_number(row[col_idx])
            if case_count is None:
                # "-" 또는 빈 값: 해당 연도에 법원이 존재하지 않음
                continue

            records.append({
                "category": category,
                "court_name": qualified_name,
                "court_type": court_type,
                "parent_court": parent_court,
                "year": year,
                "case_count": case_count,
            })

    return records


def load_all_csv_files() -> list[dict]:
    """모든 CSV 파일을 파싱하여 전체 레코드 리스트 반환."""
    if not DATA_DIR.exists():
        logger.error(f"데이터 디렉토리가 없습니다: {DATA_DIR}")
        sys.exit(1)

    all_records: list[dict] = []

    for filename, category in FILE_CATEGORY_MAP.items():
        filepath = DATA_DIR / filename
        if not filepath.exists():
            logger.warning(f"파일 없음, 건너뜀: {filepath}")
            continue

        records = parse_csv_file(filepath, category)
        all_records.extend(records)
        logger.info(f"  → {category}: {len(records):,}건")

    logger.info(f"전체 CSV 파싱 완료: {len(all_records):,}건")
    return all_records


def load_to_db(
    session_factory: sessionmaker,
    records: list[dict],
    reset: bool = False,
) -> int:
    """재판 통계 데이터를 DB에 로드."""
    with session_factory() as db:
        if reset:
            count = db.query(TrialStatistics).count()
            db.execute(
                text("TRUNCATE TABLE trial_statistics RESTART IDENTITY CASCADE")
            )
            db.commit()
            logger.info(f"기존 데이터 {count:,}건 삭제 완료")

        total_loaded = 0
        start_time = time.time()

        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i : i + BATCH_SIZE]

            stmt = insert(TrialStatistics).values(batch)
            # ON CONFLICT (category, court_name, year) DO UPDATE
            update_cols = {
                col.name: col
                for col in stmt.excluded
                if col.name not in ("id", "category", "court_name", "year", "created_at")
            }
            stmt = stmt.on_conflict_do_update(
                constraint="uq_trial_stats_cat_court_year",
                set_=update_cols,
            )
            db.execute(stmt)
            db.commit()

            total_loaded += len(batch)
            elapsed = time.time() - start_time
            logger.info(
                f"  진행: {total_loaded:,}/{len(records):,} "
                f"({total_loaded / len(records) * 100:.1f}%) "
                f"[{elapsed:.1f}s]"
            )

        return total_loaded


def verify_data(
    session_factory: sessionmaker,
    expected_count: int,
) -> None:
    """로드된 데이터 검증."""
    with session_factory() as db:
        total = db.query(func.count(TrialStatistics.id)).scalar() or 0

        # 카테고리별 건수
        category_counts = (
            db.query(
                TrialStatistics.category,
                func.count(TrialStatistics.id).label("cnt"),
            )
            .group_by(TrialStatistics.category)
            .order_by(func.count(TrialStatistics.id).desc())
            .all()
        )

        # court_type별 건수
        type_counts = (
            db.query(
                TrialStatistics.court_type,
                func.count(TrialStatistics.id).label("cnt"),
            )
            .group_by(TrialStatistics.court_type)
            .all()
        )

        # 연도 범위
        min_year = db.query(func.min(TrialStatistics.year)).scalar()
        max_year = db.query(func.max(TrialStatistics.year)).scalar()

        # 총 사건 수 합계
        total_cases = db.query(func.sum(TrialStatistics.case_count)).scalar() or 0

        # 상위 5개 법원 (사건 수 기준)
        top_courts = (
            db.query(
                TrialStatistics.court_name,
                func.sum(TrialStatistics.case_count).label("total"),
            )
            .group_by(TrialStatistics.court_name)
            .order_by(func.sum(TrialStatistics.case_count).desc())
            .limit(5)
            .all()
        )

        # 소계/합계 레코드 존재 확인
        skip_check = (
            db.query(func.count(TrialStatistics.id))
            .filter(
                TrialStatistics.court_name.in_(["합계", "소계", "법원"])
            )
            .scalar()
            or 0
        )

    logger.info("=" * 60)
    logger.info("데이터 검증 결과")
    logger.info("=" * 60)
    logger.info(f"  총 건수:       {total:,}건 (예상: {expected_count:,}건)")
    logger.info(f"  연도 범위:     {min_year}~{max_year}")
    logger.info(f"  총 사건 수:    {total_cases:,}건")
    logger.info("")

    logger.info("  카테고리별 건수:")
    for category, cnt in category_counts:
        logger.info(f"    {category}: {cnt:,}건")
    logger.info("")

    logger.info("  법원 유형별 건수:")
    for court_type, cnt in type_counts:
        label = "본원" if court_type == "main" else "지원"
        logger.info(f"    {label} ({court_type}): {cnt:,}건")
    logger.info("")

    logger.info("  사건 수 상위 5개 법원:")
    for court_name, total_count in top_courts:
        logger.info(f"    {court_name}: {total_count:,}건")
    logger.info("")

    logger.info(f"  소계/합계/법원 레코드: {skip_check}건 (0이어야 정상)")
    logger.info("=" * 60)

    if total != expected_count:
        logger.warning(
            f"건수 불일치: DB {total:,}건 != CSV {expected_count:,}건"
        )
    else:
        logger.info("건수 일치 확인 완료")

    if skip_check > 0:
        logger.error("소계/합계/법원 레코드가 존재합니다!")


def main() -> None:
    parser = argparse.ArgumentParser(description="재판 통계 CSV → PostgreSQL 로드")
    parser.add_argument(
        "--reset", action="store_true", help="기존 데이터 삭제 후 재로드"
    )
    parser.add_argument(
        "--verify", action="store_true", help="검증만 실행"
    )
    args = parser.parse_args()

    # Sync engine 사용 (스크립트용)
    engine = create_engine(
        settings.DATABASE_URL,
        echo=False,
        pool_size=5,
        pool_pre_ping=True,
    )
    Session = sessionmaker(engine)

    # CSV 데이터 파싱
    all_records = load_all_csv_files()

    if args.verify:
        verify_data(Session, len(all_records))
        return

    # DB 로드
    logger.info(f"재판 통계 데이터 로드 시작 (reset={args.reset})")
    total_loaded = load_to_db(Session, all_records, reset=args.reset)
    logger.info(f"로드 완료: {total_loaded:,}건")

    # 검증
    verify_data(Session, len(all_records))


if __name__ == "__main__":
    main()
