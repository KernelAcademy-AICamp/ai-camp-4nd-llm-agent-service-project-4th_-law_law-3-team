"""
법률 용어 데이터 PostgreSQL 로드 스크립트

lawterms_v1.json (81,488건) → legal_terms 테이블
fallback: lawterms_full.json (37,169건)

기능:
- 리스트 타입 레코드 평탄화 (flatten)
- 법령한영사전 역방향 한글 용어 추출
- 우선순위 기반 중복 제거 (source_count 집계)
- 노이즈 필터링 통계

Usage:
    uv run python scripts/load_legal_terms_data.py           # 로드
    uv run python scripts/load_legal_terms_data.py --reset    # 삭제 후 재로드
    uv run python scripts/load_legal_terms_data.py --verify   # 검증만
    uv run python scripts/load_legal_terms_data.py --stats    # 통계만
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import func, text  # noqa: E402
from sqlalchemy.dialects.postgresql import insert  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.models.legal_term import LegalTerm  # noqa: E402
from scripts.common.db import create_sync_session_factory  # noqa: E402
from scripts.common.logging_config import setup_logging  # noqa: E402

logger = setup_logging(__name__)

DATA_DIR = PROJECT_ROOT.parent / "data"
LAWTERMS_FILE = DATA_DIR / "lawterms_v1.json"
LAWTERMS_FALLBACK = DATA_DIR / "law_data" / "lawterms_full.json"

BATCH_SIZE = 1000

# 사전유형 우선순위 (높을수록 dedup 시 채택)
SOURCE_PRIORITY: dict[str, int] = {
    "법령정의사전": 100,
    "생활용어사전": 80,
    "법령한영사전": 60,
    "법령용어사전": 40,
    "한영역추출": 20,
}

# 역추출 필터용 한글 전용 패턴
_KOREAN_ONLY = re.compile(r"^[가-힣]+$")


# ============================================================================
# 데이터 전처리 함수
# ============================================================================


def load_json_data() -> tuple[list[dict], str]:
    """JSON 파일 로드. lawterms_v1 우선, 없으면 fallback."""
    if LAWTERMS_FILE.exists():
        path = LAWTERMS_FILE
    elif LAWTERMS_FALLBACK.exists():
        path = LAWTERMS_FALLBACK
        logger.warning(
            "lawterms_v1.json 없음, fallback 사용: %s", LAWTERMS_FALLBACK,
        )
    else:
        logger.error("데이터 파일이 없습니다: %s 또는 %s", LAWTERMS_FILE, LAWTERMS_FALLBACK)
        sys.exit(1)

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        logger.error("JSON 형식 오류: 리스트가 아닙니다")
        sys.exit(1)

    logger.info("JSON 파일 로드 완료: %d건 (%s)", len(data), path.name)
    return data, path.name


def flatten_records(data: list[dict]) -> list[dict]:
    """리스트 타입 레코드를 개별 엔트리로 분리.

    lawterms_v1.json의 약 15%가 리스트 타입 필드를 가짐.
    각 인덱스별로 분리하여 플랫 레코드로 변환.
    """
    flat: list[dict] = []
    list_count = 0

    for item in data:
        # 리스트 타입 필드 확인
        sample_val = item.get("법령용어명_한글", "")
        if isinstance(sample_val, list):
            list_count += 1
            n = len(sample_val)
            for idx in range(n):
                entry: dict[str, str] = {}
                for key, val in item.items():
                    if isinstance(val, list):
                        entry[key] = val[idx] if idx < len(val) else ""
                    else:
                        entry[key] = val
                flat.append(entry)
        else:
            flat.append(item)

    logger.info(
        "평탄화 완료: %d → %d건 (리스트 레코드: %d건)",
        len(data), len(flat), list_count,
    )
    return flat


def extract_reverse_terms(entries: list[dict]) -> dict[str, dict]:
    """법령한영사전 역방향 정의에서 한글 법률 용어 추출.

    예: term="compensation for damage", definition="손해배상"
        → 추출: {"손해배상": {...}}

    세미콜론/쉼표 분리, 괄호 제거, 한글 전용 2-10자 필터.
    """
    reverse_terms: dict[str, dict] = {}

    for item in entries:
        source_code = item.get("법령용어코드명", "")
        if source_code != "법령한영사전":
            continue

        definition = str(item.get("법령용어정의", "")).strip()
        english_term = str(item.get("법령용어명_한글", "")).strip()

        if not definition:
            continue

        # 세미콜론/쉼표로 분리
        parts = re.split(r"[;；,，]", definition)
        for part in parts:
            part = part.strip()
            # 괄호 제거: "보증인(인적담보)" → "보증인"
            part = re.sub(r"[（(][^）)]*[）)]", "", part).strip()
            # 숫자/점/공백 prefix 제거: "1. 손해배상" → "손해배상"
            part = re.sub(r"^[\d.\s]+", "", part).strip()

            if not part:
                continue
            if not _KOREAN_ONLY.match(part):
                continue
            if not (2 <= len(part) <= 10):
                continue

            # 이미 추출된 용어보다 더 좋은 출처는 없으므로 첫 번째 채택
            if part not in reverse_terms:
                reverse_terms[part] = {
                    "법령용어명_한글": part,
                    "법령용어명_한자": None,
                    "법령용어정의": f"[한영역추출] {english_term}",
                    "출처": None,
                    "법령용어코드명": "한영역추출",
                    "법령용어ID": None,
                }

    logger.info("한영사전 역추출 용어: %d개", len(reverse_terms))
    return reverse_terms


def dedup_with_priority(
    flat_entries: list[dict],
    reverse_terms: dict[str, dict],
) -> tuple[list[dict], int, int]:
    """우선순위 기반 중복 제거.

    동일 용어가 여러 사전에 등재된 경우:
    1. source_code 우선순위가 높은 정의를 채택
    2. source_count에 총 출현 횟수 기록

    Returns:
        (unique_records, total_seen, duplicates_merged)
    """
    # term → (best_record, source_count)
    term_map: dict[str, tuple[dict, int]] = {}
    empty_count = 0

    for item in flat_entries:
        term = str(item.get("법령용어명_한글", "")).strip()
        if not term:
            empty_count += 1
            continue

        source_code = str(item.get("법령용어코드명", "")).strip()
        priority = SOURCE_PRIORITY.get(source_code, 0)

        if term in term_map:
            existing_record, count = term_map[term]
            existing_priority = SOURCE_PRIORITY.get(
                str(existing_record.get("법령용어코드명", "")), 0,
            )
            # 더 높은 우선순위면 교체
            if priority > existing_priority:
                term_map[term] = (item, count + 1)
            else:
                term_map[term] = (existing_record, count + 1)
        else:
            term_map[term] = (item, 1)

    # 역추출 용어 추가 (기존에 없는 것만)
    reverse_added = 0
    for term, record in reverse_terms.items():
        if term not in term_map:
            term_map[term] = (record, 1)
            reverse_added += 1
        else:
            # 이미 있으면 source_count만 증가
            existing_record, count = term_map[term]
            term_map[term] = (existing_record, count + 1)

    logger.info(
        "중복 제거: %d → %d 고유 용어 (역추출 신규: %d, 빈값 스킵: %d)",
        len(flat_entries), len(term_map), reverse_added, empty_count,
    )

    unique_records: list[dict] = []
    for term, (record, count) in term_map.items():
        unique_records.append({
            "_raw": record,
            "_term": term,
            "_source_count": count,
        })

    total_seen = len(flat_entries) + len(reverse_terms) - empty_count
    duplicates_merged = total_seen - len(term_map)
    return unique_records, total_seen, duplicates_merged


def prepare_record(item: dict) -> dict:
    """전처리된 레코드를 DB 레코드로 변환."""
    raw = item["_raw"]
    term = item["_term"]
    source_count = item["_source_count"]

    term_hanja = str(raw.get("법령용어명_한자", "") or "").strip() or None
    definition = str(raw.get("법령용어정의", "") or "").strip() or None
    source = str(raw.get("출처", "") or "").strip() or None
    source_code = str(raw.get("법령용어코드명", "") or "").strip() or None
    serial_number = str(raw.get("법령용어ID", "") or "").strip() or None

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
        "source_count": source_count,
    }


# ============================================================================
# DB 로드
# ============================================================================


def load_to_db(
    session_factory: sessionmaker,
    unique_items: list[dict],
    reset: bool = False,
) -> int:
    """법률 용어 데이터를 DB에 로드."""
    with session_factory() as db:
        if reset:
            count = db.query(LegalTerm).count()
            db.execute(text("TRUNCATE TABLE legal_terms RESTART IDENTITY CASCADE"))
            db.commit()
            logger.info("기존 데이터 %d건 삭제 완료", count)

        records = [prepare_record(item) for item in unique_items]

        total_loaded = 0
        start_time = time.time()

        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i:i + BATCH_SIZE]

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
                total_loaded, len(records),
                total_loaded / len(records) * 100,
                elapsed,
            )

        return total_loaded


# ============================================================================
# 검증 / 통계
# ============================================================================


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
        # 토크나이저 로드 대상: 한글 전용 + 2~10자 (모든 사전유형)
        tokenizer_candidates = db.query(func.count(LegalTerm.id)).filter(
            LegalTerm.is_korean_only.is_(True),
            LegalTerm.term_length >= 2,
            LegalTerm.term_length <= 10,
        ).scalar() or 0

        # 길이 분포
        length_dist = (
            db.query(LegalTerm.term_length, func.count(LegalTerm.id).label("cnt"))
            .filter(LegalTerm.is_korean_only.is_(True))
            .group_by(LegalTerm.term_length)
            .order_by(LegalTerm.term_length)
            .all()
        )

        # source_count > 1 (다중 출처)
        multi_source = db.query(func.count(LegalTerm.id)).filter(
            LegalTerm.source_count > 1,
        ).scalar() or 0

        # 제외 통계
        space_terms = db.query(func.count(LegalTerm.id)).filter(
            LegalTerm.is_korean_only.is_(False),
            LegalTerm.term.op("~")(r"^[가-힣\s]+$"),
        ).scalar() or 0
        long_terms = db.query(func.count(LegalTerm.id)).filter(
            LegalTerm.term_length > 10,
            LegalTerm.is_korean_only.is_(True),
        ).scalar() or 0

    logger.info("=" * 60)
    logger.info("데이터 검증 결과")
    logger.info("=" * 60)
    logger.info("  총 건수:         %s건 (예상: %s건)", f"{total:,}", f"{expected_unique:,}")
    logger.info("  한글 전용:       %s건 (%.1f%%)", f"{korean_only:,}", korean_only / max(total, 1) * 100)
    logger.info("  토크나이저 후보: %s건 (한글+2~10자)", f"{tokenizer_candidates:,}")
    logger.info("  다중 출처 용어:  %s건 (source_count > 1)", f"{multi_source:,}")
    logger.info("")
    logger.info("  사전유형별:")
    for source, cnt in by_source:
        logger.info("    %s: %s건", source or "(없음)", f"{cnt:,}")
    logger.info("")
    logger.info("  === 제외 통계 ===")
    logger.info("  공백 포함 한글 용어 (토크나이저 미로드): %s건", f"{space_terms:,}")
    logger.info("  11자 이상 한글 용어 (토크나이저 미로드): %s건", f"{long_terms:,}")
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


# ============================================================================
# 전처리 파이프라인
# ============================================================================


def preprocess_data(raw_data: list[dict]) -> tuple[list[dict], int]:
    """전체 전처리 파이프라인: flatten → reverse extract → dedup.

    Returns:
        (unique_records, expected_unique_count)
    """
    # 1. 평탄화
    flat = flatten_records(raw_data)

    # 2. 역추출
    reverse_terms = extract_reverse_terms(flat)

    # 3. 우선순위 기반 중복 제거
    unique_records, _, _ = dedup_with_priority(flat, reverse_terms)

    # 통계 로깅
    source_counter: Counter[str] = Counter()
    for item in unique_records:
        sc = str(item["_raw"].get("법령용어코드명", "") or "")
        source_counter[sc] += 1
    logger.info("사전유형별 고유 용어:")
    for sc, cnt in source_counter.most_common():
        logger.info("  %s: %d건", sc or "(없음)", cnt)

    return unique_records, len(unique_records)


# ============================================================================
# 메인
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="법률 용어 데이터 PostgreSQL 로드")
    parser.add_argument("--reset", action="store_true", help="기존 데이터 삭제 후 재로드")
    parser.add_argument("--verify", action="store_true", help="검증만 실행")
    parser.add_argument("--stats", action="store_true", help="통계만 출력")
    args = parser.parse_args()

    # Sync engine 사용 (스크립트용)
    session_factory = create_sync_session_factory()

    if args.stats:
        show_stats(session_factory)
        return

    # JSON 데이터 로드
    raw_data, _filename = load_json_data()

    if args.verify:
        unique_records, expected = preprocess_data(raw_data)
        verify_data(session_factory, expected)
        return

    # 전처리
    unique_records, expected = preprocess_data(raw_data)

    # DB 로드
    logger.info("법률 용어 데이터 로드 시작 (reset=%s)", args.reset)
    total_loaded = load_to_db(session_factory, unique_records, reset=args.reset)
    logger.info("로드 완료: %s건", f"{total_loaded:,}")

    # 검증
    verify_data(session_factory, expected)


if __name__ == "__main__":
    main()
