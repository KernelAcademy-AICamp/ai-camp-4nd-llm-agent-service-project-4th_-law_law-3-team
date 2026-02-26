"""
Neo4j 그래프 데이터를 PostgreSQL로 로드하는 스크립트

build_graph.py와 동일한 JSON 소스 파일을 사용하여
PostgreSQL 관계 테이블에 데이터를 적재합니다.

사용법:
    cd backend
    uv run python scripts/load_graph_data.py
    uv run python scripts/load_graph_data.py --verify
    uv run python scripts/load_graph_data.py --reset
"""

import argparse
import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import func, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from tqdm import tqdm

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

env_path = PROJECT_ROOT / ".env"
load_dotenv(env_path)

from app.core.database import sync_session_factory  # noqa: E402
from app.models.case_case_citation import CaseCaseCitation  # noqa: E402
from app.models.case_statute_citation import CaseStatuteCitation  # noqa: E402
from app.models.law_document import LawDocument  # noqa: E402
from app.models.precedent_document import PrecedentDocument  # noqa: E402
from app.models.statute_alias import StatuteAlias  # noqa: E402
from app.models.statute_hierarchy import StatuteHierarchy  # noqa: E402
from app.models.statute_relation import StatuteRelation  # noqa: E402

# 데이터 파일 경로
DATA_DIR = PROJECT_ROOT.parent / "data"
HIERARCHY_FILE = DATA_DIR / "law_hierarchy.json"
ABBREVIATION_FILE = DATA_DIR / "law_abbreviations.json"
CASE_FILE = DATA_DIR / "ingest_source" / "precedents_v2.json"
INFORMAL_ABBR_FILE = Path(__file__).parent / "informal_abbreviations.json"

BATCH_SIZE = 1000


def build_law_id_lookup() -> dict[str, int]:
    """law_id → law_documents.id 룩업 딕셔너리 빌드"""
    print("Building law_id → id lookup...")
    with sync_session_factory() as session:
        result = session.execute(
            select(LawDocument.id, LawDocument.law_id)
        )
        lookup = {row.law_id: row.id for row in result}
    print(f"  {len(lookup)} law documents indexed.")
    return lookup


def build_law_name_lookup() -> dict[str, int]:
    """law_name → law_documents.id 룩업 딕셔너리 빌드"""
    print("Building law_name → id lookup...")
    with sync_session_factory() as session:
        result = session.execute(
            select(LawDocument.id, LawDocument.law_name)
        )
        lookup = {row.law_name: row.id for row in result}
    print(f"  {len(lookup)} law names indexed.")
    return lookup


def build_case_serial_lookup() -> dict[str, int]:
    """serial_number → precedent_documents.id 룩업 딕셔너리 빌드"""
    print("Building serial_number → id lookup...")
    with sync_session_factory() as session:
        result = session.execute(
            select(PrecedentDocument.id, PrecedentDocument.serial_number)
        )
        lookup = {row.serial_number: row.id for row in result}
    print(f"  {len(lookup)} precedent documents indexed.")
    return lookup


def build_case_number_lookup() -> dict[str, int]:
    """case_number → precedent_documents.id 룩업 딕셔너리 빌드"""
    print("Building case_number → id lookup...")
    with sync_session_factory() as session:
        result = session.execute(
            select(PrecedentDocument.id, PrecedentDocument.case_number)
        )
        lookup: dict[str, int] = {}
        for row in result:
            if row.case_number:
                lookup[row.case_number] = row.id
    print(f"  {len(lookup)} case numbers indexed.")
    return lookup


def load_abbreviations(law_id_lookup: dict[str, int]) -> int:
    """법령 약칭 로드 → law_documents.abbreviation UPDATE"""
    print(f"\n=== Loading Abbreviations from {ABBREVIATION_FILE} ===")

    if not ABBREVIATION_FILE.exists():
        print(f"File not found: {ABBREVIATION_FILE}")
        return 0

    with open(ABBREVIATION_FILE, encoding="utf-8") as f:
        data = json.load(f)

    updated = 0
    batch: list[dict] = []

    with sync_session_factory() as session:
        for item in tqdm(data, desc="Abbreviations"):
            statute_id = item.get("법령ID")
            abbreviation = item.get("법령약칭명")

            if not statute_id or not abbreviation:
                continue

            doc_id = law_id_lookup.get(statute_id)
            if not doc_id:
                continue

            batch.append({"doc_id": doc_id, "abbreviation": abbreviation})

            if len(batch) >= BATCH_SIZE:
                for b in batch:
                    session.execute(
                        update(LawDocument)
                        .where(LawDocument.id == b["doc_id"])
                        .values(abbreviation=b["abbreviation"])
                    )
                session.commit()
                updated += len(batch)
                batch = []

        if batch:
            for b in batch:
                session.execute(
                    update(LawDocument)
                    .where(LawDocument.id == b["doc_id"])
                    .values(abbreviation=b["abbreviation"])
                )
            session.commit()
            updated += len(batch)

    print(f"Updated {updated} law documents with abbreviations.")
    return updated


def load_informal_abbreviations(law_name_lookup: dict[str, int]) -> int:
    """비공식 약칭 로드 → statute_aliases INSERT"""
    print(f"\n=== Loading Informal Abbreviations from {INFORMAL_ABBR_FILE} ===")

    if not INFORMAL_ABBR_FILE.exists():
        print(f"File not found: {INFORMAL_ABBR_FILE}")
        return 0

    with open(INFORMAL_ABBR_FILE, encoding="utf-8") as f:
        data = json.load(f)

    mappings = data.get("mappings", [])
    rows: list[dict] = []

    for category_data in mappings:
        category = category_data.get("category", "")
        items = category_data.get("items", [])

        for item in items:
            abbr = item.get("abbreviation")
            full_name = item.get("full_name")

            if not abbr or not full_name:
                continue

            doc_id = law_name_lookup.get(full_name)
            if not doc_id:
                continue

            rows.append({
                "law_doc_id": doc_id,
                "alias_name": abbr,
                "category": category,
            })

    if not rows:
        print("No informal abbreviations to load.")
        return 0

    inserted = 0
    with sync_session_factory() as session:
        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i:i + BATCH_SIZE]
            stmt = pg_insert(StatuteAlias).values(batch)
            stmt = stmt.on_conflict_do_update(
                index_elements=["alias_name"],
                set_={
                    "law_doc_id": stmt.excluded.law_doc_id,
                    "category": stmt.excluded.category,
                },
            )
            session.execute(stmt)
            session.commit()
            inserted += len(batch)

    print(f"Inserted {inserted} statute aliases.")
    return inserted


def load_hierarchy(law_id_lookup: dict[str, int]) -> int:
    """법령 계급 관계 로드 → statute_hierarchy INSERT"""
    print(f"\n=== Loading Hierarchy from {HIERARCHY_FILE} ===")

    if not HIERARCHY_FILE.exists():
        print(f"File not found: {HIERARCHY_FILE}")
        return 0

    with open(HIERARCHY_FILE, encoding="utf-8") as f:
        data = json.load(f)

    real_relations: list[dict[str, str]] = []

    def parse_node(node_dict: dict, parent_id: str | None = None) -> None:
        """재귀적으로 계급 관계 파싱"""
        current_id = None
        if "기본정보" in node_dict:
            current_id = node_dict["기본정보"].get("법령ID")
            if current_id and parent_id and current_id != parent_id:
                real_relations.append({"lower": current_id, "upper": parent_id})

        next_parent = current_id if current_id else parent_id

        for key, val in node_dict.items():
            if key in ("기본정보", "관련법령", "제개정구분"):
                continue
            targets: list[dict] = []
            if isinstance(val, dict):
                targets.append(val)
            elif isinstance(val, list):
                targets.extend(val)
            for t in targets:
                if isinstance(t, dict):
                    parse_node(t, next_parent)

    for item in tqdm(data, desc="Parsing Hierarchy"):
        parse_node(item)

    print(f"Found {len(real_relations)} hierarchy relationships.")

    # 매핑 및 INSERT
    rows: list[dict] = []
    skipped = 0
    for rel in real_relations:
        child_id = law_id_lookup.get(rel["lower"])
        parent_id = law_id_lookup.get(rel["upper"])
        if child_id and parent_id:
            rows.append({"child_id": child_id, "parent_id": parent_id})
        else:
            skipped += 1

    if skipped:
        print(f"  Skipped {skipped} (law_id not found in DB).")

    inserted = 0
    with sync_session_factory() as session:
        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i:i + BATCH_SIZE]
            stmt = pg_insert(StatuteHierarchy).values(batch)
            stmt = stmt.on_conflict_do_nothing(
                constraint="uq_statute_hierarchy",
            )
            session.execute(stmt)
            session.commit()
            inserted += len(batch)

    print(f"Inserted {inserted} hierarchy relationships.")
    return inserted


def load_related_statutes(law_id_lookup: dict[str, int]) -> int:
    """법령 관련 관계 로드 → statute_relations INSERT"""
    print(f"\n=== Loading Related Statutes from {HIERARCHY_FILE} ===")

    if not HIERARCHY_FILE.exists():
        print(f"File not found: {HIERARCHY_FILE}")
        return 0

    with open(HIERARCHY_FILE, encoding="utf-8") as f:
        data = json.load(f)

    relations: list[dict[str, str]] = []

    for item in tqdm(data, desc="Parsing Related Statutes"):
        if "기본정보" not in item:
            continue
        source_id = item["기본정보"].get("법령ID")
        if not source_id:
            continue

        related = item.get("관련법령", {})
        conlaw_list = related.get("conlaw", [])
        if not isinstance(conlaw_list, list):
            continue

        for conlaw in conlaw_list:
            target_id = conlaw.get("법령ID")
            if target_id and target_id != source_id:
                relations.append({"source_id": source_id, "target_id": target_id})

    print(f"Found {len(relations)} related statute relationships.")

    # 매핑 및 INSERT (id_1 < id_2 보장)
    rows: list[dict] = []
    seen: set[tuple[int, int]] = set()
    skipped = 0

    for rel in relations:
        doc_id_1 = law_id_lookup.get(rel["source_id"])
        doc_id_2 = law_id_lookup.get(rel["target_id"])
        if not doc_id_1 or not doc_id_2:
            skipped += 1
            continue
        # id_1 < id_2 보장
        pair = (min(doc_id_1, doc_id_2), max(doc_id_1, doc_id_2))
        if pair in seen:
            continue
        seen.add(pair)
        rows.append({"law_doc_id_1": pair[0], "law_doc_id_2": pair[1]})

    if skipped:
        print(f"  Skipped {skipped} (law_id not found in DB).")

    inserted = 0
    with sync_session_factory() as session:
        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i:i + BATCH_SIZE]
            stmt = pg_insert(StatuteRelation).values(batch)
            stmt = stmt.on_conflict_do_nothing(
                constraint="uq_statute_relations",
            )
            session.execute(stmt)
            session.commit()
            inserted += len(batch)

    print(f"Inserted {inserted} related statute relationships.")
    return inserted


def load_case_statute_citations(
    law_name_lookup: dict[str, int],
    case_serial_lookup: dict[str, int],
) -> int:
    """판례→법령 인용 로드 → case_statute_citations INSERT"""
    print(f"\n=== Loading Case→Statute Citations from {CASE_FILE} ===")

    if not CASE_FILE.exists():
        print(f"File not found: {CASE_FILE}")
        return 0

    with open(CASE_FILE, encoding="utf-8") as f:
        data = json.load(f)

    regex_statute = re.compile(r"([가-힣]+법(?:시행령|시행규칙)?)")

    rows: list[dict] = []
    seen: set[tuple[int, int]] = set()
    skipped_case = 0
    skipped_law = 0

    for item in tqdm(data, desc="Parsing Case→Statute"):
        c_id = item.get("판례정보일련번호")
        refs = item.get("참조조문", "")

        if not c_id or not refs:
            continue

        case_doc_id = case_serial_lookup.get(str(c_id))
        if not case_doc_id:
            skipped_case += 1
            continue

        found_laws = set(regex_statute.findall(refs))
        for law_name in found_laws:
            if len(law_name) < 2:
                continue
            law_doc_id = law_name_lookup.get(law_name)
            if not law_doc_id:
                skipped_law += 1
                continue
            pair = (case_doc_id, law_doc_id)
            if pair in seen:
                continue
            seen.add(pair)
            rows.append({"case_doc_id": case_doc_id, "law_doc_id": law_doc_id})

    print(f"Found {len(rows)} case→statute citation pairs.")
    if skipped_case:
        print(f"  Skipped {skipped_case} cases (serial not found).")
    if skipped_law:
        print(f"  Skipped {skipped_law} laws (name not found).")

    inserted = 0
    with sync_session_factory() as session:
        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i:i + BATCH_SIZE]
            stmt = pg_insert(CaseStatuteCitation).values(batch)
            stmt = stmt.on_conflict_do_nothing(
                constraint="uq_case_statute_citation",
            )
            session.execute(stmt)
            session.commit()
            inserted += len(batch)

    print(f"Inserted {inserted} case→statute citations.")
    return inserted


def load_case_case_citations(
    case_serial_lookup: dict[str, int],
    case_number_lookup: dict[str, int],
) -> int:
    """판례→판례 인용 로드 → case_case_citations INSERT"""
    print(f"\n=== Loading Case→Case Citations from {CASE_FILE} ===")

    if not CASE_FILE.exists():
        print(f"File not found: {CASE_FILE}")
        return 0

    with open(CASE_FILE, encoding="utf-8") as f:
        data = json.load(f)

    regex_case_number = re.compile(
        r"(\d{2,4})"
        r"([가-힣]{1,3})"
        r"(\d+)"
    )

    rows: list[dict] = []
    seen: set[tuple[int, int]] = set()
    skipped_citing = 0
    skipped_cited = 0

    for item in tqdm(data, desc="Parsing Case→Case"):
        c_id = item.get("판례정보일련번호")
        refs = item.get("참조판례", "")

        if not c_id or not refs:
            continue

        citing_id = case_serial_lookup.get(str(c_id))
        if not citing_id:
            skipped_citing += 1
            continue

        matches = regex_case_number.findall(refs)
        for match in matches:
            year, case_type, number = match
            ref_case_number = f"{year}{case_type}{number}"
            cited_id = case_number_lookup.get(ref_case_number)
            if not cited_id:
                skipped_cited += 1
                continue
            if citing_id == cited_id:
                continue
            pair = (citing_id, cited_id)
            if pair in seen:
                continue
            seen.add(pair)
            rows.append({"citing_case_id": citing_id, "cited_case_id": cited_id})

    print(f"Found {len(rows)} case→case citation pairs.")
    if skipped_citing:
        print(f"  Skipped {skipped_citing} citing cases (serial not found).")
    if skipped_cited:
        print(f"  Skipped {skipped_cited} cited cases (case_number not found).")

    inserted = 0
    with sync_session_factory() as session:
        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i:i + BATCH_SIZE]
            stmt = pg_insert(CaseCaseCitation).values(batch)
            stmt = stmt.on_conflict_do_nothing(
                constraint="uq_case_case_citation",
            )
            session.execute(stmt)
            session.commit()
            inserted += len(batch)

    print(f"Inserted {inserted} case→case citations.")
    return inserted


def compute_citation_counts() -> int:
    """법령별 인용 수 계산 → law_documents.citation_count UPDATE"""
    print("\n=== Computing Citation Counts ===")

    with sync_session_factory() as session:
        # 서브쿼리: 각 law_doc_id별 인용 수
        subq = (
            select(
                CaseStatuteCitation.law_doc_id,
                func.count(CaseStatuteCitation.id).label("cnt"),
            )
            .group_by(CaseStatuteCitation.law_doc_id)
            .subquery()
        )

        # UPDATE with subquery
        session.execute(
            update(LawDocument)
            .where(LawDocument.id == subq.c.law_doc_id)
            .values(citation_count=subq.c.cnt)
        )
        session.commit()

        # 결과 확인
        result = session.execute(
            select(func.count()).select_from(LawDocument).where(
                LawDocument.citation_count > 0
            )
        )
        updated = result.scalar() or 0

    print(f"Updated {updated} law documents with citation_count > 0.")
    return updated


def reset_tables() -> None:
    """관계 테이블 데이터 초기화"""
    print("\n=== Resetting graph tables ===")
    with sync_session_factory() as session:
        session.execute(text("DELETE FROM case_case_citations"))
        session.execute(text("DELETE FROM case_statute_citations"))
        session.execute(text("DELETE FROM statute_relations"))
        session.execute(text("DELETE FROM statute_hierarchy"))
        session.execute(text("DELETE FROM statute_aliases"))
        session.execute(
            update(LawDocument).values(abbreviation=None, citation_count=0)
        )
        session.commit()
    print("All graph tables cleared.")


def verify() -> None:
    """데이터 건수 검증"""
    print("\n=== Verification ===")
    with sync_session_factory() as session:
        tables = [
            (
                "law_documents (with abbreviation)",
                select(func.count())
                .select_from(LawDocument)
                .where(LawDocument.abbreviation.isnot(None)),
            ),
            (
                "law_documents (citation_count > 0)",
                select(func.count())
                .select_from(LawDocument)
                .where(LawDocument.citation_count > 0),
            ),
            (
                "statute_aliases",
                select(func.count()).select_from(StatuteAlias),
            ),
            (
                "statute_hierarchy",
                select(func.count()).select_from(StatuteHierarchy),
            ),
            (
                "statute_relations",
                select(func.count()).select_from(StatuteRelation),
            ),
            (
                "case_statute_citations",
                select(func.count()).select_from(CaseStatuteCitation),
            ),
            (
                "case_case_citations",
                select(func.count()).select_from(CaseCaseCitation),
            ),
        ]

        for name, query in tables:
            count = session.execute(query).scalar() or 0
            print(f"  {name}: {count:,}")

        # TOP 5 인용 법령
        print("\nTop 5 cited statutes:")
        result = session.execute(
            select(LawDocument.law_name, LawDocument.citation_count)
            .where(LawDocument.citation_count > 0)
            .order_by(LawDocument.citation_count.desc())
            .limit(5)
        )
        for row in result:
            print(f"  {row.law_name}: {row.citation_count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Load graph data to PostgreSQL")
    parser.add_argument("--verify", action="store_true", help="검증만 수행")
    parser.add_argument("--reset", action="store_true", help="기존 데이터 삭제 후 재로드")
    args = parser.parse_args()

    if args.verify:
        verify()
        return

    if args.reset:
        reset_tables()

    # 룩업 딕셔너리 빌드
    law_id_lookup = build_law_id_lookup()
    law_name_lookup = build_law_name_lookup()
    case_serial_lookup = build_case_serial_lookup()
    case_number_lookup = build_case_number_lookup()

    # 데이터 로드
    load_abbreviations(law_id_lookup)
    load_informal_abbreviations(law_name_lookup)
    load_hierarchy(law_id_lookup)
    load_related_statutes(law_id_lookup)
    load_case_statute_citations(law_name_lookup, case_serial_lookup)
    load_case_case_citations(case_serial_lookup, case_number_lookup)
    compute_citation_counts()

    # 검증
    verify()

    print("\n=== Graph data loading complete ===")


if __name__ == "__main__":
    main()
