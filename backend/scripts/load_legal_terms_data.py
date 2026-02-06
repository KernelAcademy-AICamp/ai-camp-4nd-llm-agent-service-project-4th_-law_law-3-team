"""
법률 용어 데이터 PostgreSQL 로드 스크립트

data/law_data/lawterms_full.json → legal_terms 테이블

Usage:
    uv run python scripts/load_legal_terms_data.py           # 로드
    uv run python scripts/load_legal_terms_data.py --reset    # 삭제 후 재로드
    uv run python scripts/load_legal_terms_data.py --verify   # 검증만
    uv run python scripts/load_legal_terms_data.py --stats    # 통계만
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

from sqlalchemy import create_engine, func, text  # noqa: E402
from sqlalchemy.dialects.postgresql import insert  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.models.legal_term import LegalTerm  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT.parent / "data" / "law_data"
LAWTERMS_FILE = DATA_DIR / "lawterms_full.json"

BATCH_SIZE = 1000


def load_json_data() -> list[dict]:
    """JSON 파일 로드."""
    if not LAWTERMS_FILE.exists():
        logger.error("파일이 없습니다: %s", LAWTERMS_FILE)
        sys.exit(1)

    with open(LAWTERMS_FILE, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        logger.error("JSON 형식 오류: 리스트가 아닙니다")
        sys.exit(1)

    logger.info("JSON 파일 로드 완료: %d건", len(data))
    return data


def prepare_record(item: dict) -> dict:
    """JSON 레코드를 DB 레코드로 변환."""
    term = item.get("법령용어명_한글", "").strip()
    term_hanja = item.get("법령용어명_한자", "").strip() or None
    definition = item.get("법령용어정의", "").strip() or None
    source = item.get("출처", "").strip() or None
    source_code = item.get("법령용어코드명", "").strip() or None
    serial_number = str(item.get("법령용어 일련번호", "")).strip() or None

    term_length = len(term)
    is_korean_only = LegalTerm.compute_is_korean_only(term)
    priority = LegalTerm.compute_priority(
        term, source_code or "", term_length, is_korean_only,
    )

    return {
        "term": term,
        "term_hanja": term_hanja,
        "definition": definition,
        "source": source,
        "source_code": source_code,
        "serial_number": serial_number,
        "term_length": term_length,
        "is_korean_only": is_korean_only,
        "priority": priority,
    }


def load_to_db(
    session_factory: sessionmaker,
    items: list[dict],
    reset: bool = False,
) -> int:
    """법률 용어 데이터를 DB에 로드."""
    with session_factory() as db:
        if reset:
            count = db.query(LegalTerm).count()
            db.execute(text("TRUNCATE TABLE legal_terms RESTART IDENTITY CASCADE"))
            db.commit()
            logger.info("기존 데이터 %d건 삭제 완료", count)

        # 중복 제거 (term 기준, 먼저 나온 것 우선)
        seen_terms: set[str] = set()
        unique_records: list[dict] = []
        skipped = 0

        for item in items:
            record = prepare_record(item)
            term = record["term"]
            if not term:
                skipped += 1
                continue
            if term in seen_terms:
                skipped += 1
                continue
            seen_terms.add(term)
            unique_records.append(record)

        logger.info(
            "고유 용어: %d건 (중복/빈값 스킵: %d건)",
            len(unique_records), skipped,
        )

        total_loaded = 0
        start_time = time.time()

        for i in range(0, len(unique_records), BATCH_SIZE):
            batch = unique_records[i:i + BATCH_SIZE]

            # ON CONFLICT (term) DO UPDATE 로 멱등성 보장
            stmt = insert(LegalTerm).values(batch)
            update_cols = {
                col.name: col
                for col in stmt.excluded
                if col.name not in ("id", "term", "created_at")
            }
            stmt = stmt.on_conflict_do_update(
                constraint="uq_legal_terms_term",
                set_=update_cols,
            )
            db.execute(stmt)
            db.commit()

            total_loaded += len(batch)
            elapsed = time.time() - start_time
            logger.info(
                "  진행: %d/%d (%.1f%%) [%.1fs]",
                total_loaded, len(unique_records),
                total_loaded / len(unique_records) * 100,
                elapsed,
            )

        return total_loaded


def verify_data(session_factory: sessionmaker, expected_unique: int) -> None:
    """로드된 데이터 검증."""
    with session_factory() as db:
        total = db.query(func.count(LegalTerm.id)).scalar() or 0
        korean_only = db.query(func.count(LegalTerm.id)).filter(
            LegalTerm.is_korean_only.is_(True),
        ).scalar() or 0
        by_source = (
            db.query(LegalTerm.source_code, func.count(LegalTerm.id).label("cnt"))
            .group_by(LegalTerm.source_code)
            .order_by(func.count(LegalTerm.id).desc())
            .all()
        )
        tokenizer_candidates = db.query(func.count(LegalTerm.id)).filter(
            LegalTerm.is_korean_only.is_(True),
            LegalTerm.term_length >= 2,
            LegalTerm.term_length <= 10,
            LegalTerm.source_code == "법령정의사전",
        ).scalar() or 0

        # 길이 분포
        length_dist = (
            db.query(LegalTerm.term_length, func.count(LegalTerm.id).label("cnt"))
            .filter(LegalTerm.is_korean_only.is_(True))
            .group_by(LegalTerm.term_length)
            .order_by(LegalTerm.term_length)
            .all()
        )

    logger.info("=" * 60)
    logger.info("데이터 검증 결과")
    logger.info("=" * 60)
    logger.info("  총 건수:         %s건 (예상: %s건)", f"{total:,}", f"{expected_unique:,}")
    logger.info("  한글 전용:       %s건 (%.1f%%)", f"{korean_only:,}", korean_only / max(total, 1) * 100)
    logger.info("  토크나이저 후보: %s건 (법령정의사전+한글+2~10자)", f"{tokenizer_candidates:,}")
    logger.info("")
    logger.info("  사전유형별:")
    for source, cnt in by_source:
        logger.info("    %s: %s건", source or "(없음)", f"{cnt:,}")
    logger.info("")
    logger.info("  길이 분포 (한글 전용):")
    for length, cnt in length_dist:
        if 1 <= length <= 12:
            logger.info("    %d글자: %s건", length, f"{cnt:,}")
    logger.info("=" * 60)

    if total != expected_unique:
        logger.warning("건수 불일치: DB %d건 != 예상 %d건", total, expected_unique)
    else:
        logger.info("건수 일치 확인 완료")


def show_stats(session_factory: sessionmaker) -> None:
    """DB 통계만 출력."""
    with session_factory() as db:
        total = db.query(func.count(LegalTerm.id)).scalar() or 0
        if total == 0:
            logger.info("legal_terms 테이블이 비어 있습니다")
            return
    verify_data(session_factory, total)


def main() -> None:
    parser = argparse.ArgumentParser(description="법률 용어 데이터 PostgreSQL 로드")
    parser.add_argument("--reset", action="store_true", help="기존 데이터 삭제 후 재로드")
    parser.add_argument("--verify", action="store_true", help="검증만 실행")
    parser.add_argument("--stats", action="store_true", help="통계만 출력")
    args = parser.parse_args()

    # Sync engine 사용 (스크립트용)
    engine = create_engine(
        settings.DATABASE_URL,
        echo=False,
        pool_size=5,
        pool_pre_ping=True,
    )
    session_factory = sessionmaker(engine)

    if args.stats:
        show_stats(session_factory)
        return

    # JSON 데이터 로드
    raw_data = load_json_data()

    if args.verify:
        # 예상 고유 건수 계산
        seen: set[str] = set()
        for item in raw_data:
            term = item.get("법령용어명_한글", "").strip()
            if term:
                seen.add(term)
        verify_data(session_factory, len(seen))
        return

    # DB 로드
    logger.info("법률 용어 데이터 로드 시작 (reset=%s)", args.reset)
    total_loaded = load_to_db(session_factory, raw_data, reset=args.reset)
    logger.info("로드 완료: %s건", f"{total_loaded:,}")

    # 예상 고유 건수 계산
    seen_terms: set[str] = set()
    for item in raw_data:
        term = item.get("법령용어명_한글", "").strip()
        if term:
            seen_terms.add(term)

    # 검증
    verify_data(session_factory, len(seen_terms))


if __name__ == "__main__":
    main()
