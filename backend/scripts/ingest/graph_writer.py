"""
그래프 데이터 적재 모듈 (인제스트 파이프라인 통합)

법령 계급, 약칭, 인용 관계 JSON 소스 파일을 사용하여
PostgreSQL 관계 테이블에 데이터를 적재합니다.

사용법:
    uv run python -m scripts.ingest.cli --step graph
    uv run python -m scripts.ingest.cli --step graph --reset
    uv run python -m scripts.ingest.cli --step graph --verify
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path

_backend_root = Path(__file__).parent.parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))

from sqlalchemy import func, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.core.database import sync_session_factory
from app.models.case_case_citation import CaseCaseCitation
from app.models.case_statute_citation import CaseStatuteCitation
from app.models.law_document import LawDocument
from app.models.precedent_document import PrecedentDocument
from app.models.statute_alias import StatuteAlias
from app.models.statute_hierarchy import StatuteHierarchy
from app.models.statute_relation import StatuteRelation
from scripts.ingest.config import DATA_DIR, get_source_path

logger = logging.getLogger(__name__)

# 소스 파일 경로
_HIERARCHY_FILE = DATA_DIR / "law_hierarchy.json"
_ABBREVIATION_FILE = DATA_DIR / "law_abbreviations.json"
_INFORMAL_ABBR_FILE = Path(__file__).parent.parent / "informal_abbreviations.json"


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------


def run_graph_ingest(
    reset: bool = False,
    batch_size: int = 1000,
) -> dict[str, int]:
    """그래프 테이블 전체 적재 (단일 진입점)

    Args:
        reset: True이면 기존 데이터 삭제 후 재적재
        batch_size: 배치 INSERT 크기

    Returns:
        각 단계별 적재/업데이트 건수 dict

    Raises:
        RuntimeError: law_documents 또는 precedent_documents가 비어있을 때
    """
    overall_start = time.time()
    stats: dict[str, int] = {}

    _verify_prerequisites()

    if reset:
        _reset_graph_tables()

    # 룩업 딕셔너리 빌드
    law_id_lookup = _build_law_id_lookup()
    law_name_lookup = _build_law_name_lookup()
    case_serial_lookup = _build_case_serial_lookup()
    case_number_lookup = _build_case_number_lookup()

    # 7단계 순차 적재
    stats["abbreviations_updated"] = _load_abbreviations(
        law_id_lookup, batch_size
    )
    stats["informal_aliases_inserted"] = _load_informal_abbreviations(
        law_name_lookup, batch_size
    )
    stats["hierarchy_inserted"] = _load_hierarchy(
        law_id_lookup, batch_size
    )
    stats["relations_inserted"] = _load_related_statutes(
        law_id_lookup, batch_size
    )
    stats["case_statute_inserted"] = _load_case_statute_citations(
        law_name_lookup, case_serial_lookup, batch_size
    )
    stats["case_case_inserted"] = _load_case_case_citations(
        case_serial_lookup, case_number_lookup, batch_size
    )
    stats["citation_counts_updated"] = _compute_citation_counts()

    elapsed = time.time() - overall_start
    logger.info("그래프 적재 완료 (%.1f초)", elapsed)
    return stats


def verify_graph() -> dict[str, int]:
    """그래프 테이블 건수 검증

    Returns:
        테이블별 건수 dict
    """
    results: dict[str, int] = {}

    with sync_session_factory() as session:
        results["law_with_abbreviation"] = (
            session.execute(
                select(func.count())
                .select_from(LawDocument)
                .where(LawDocument.abbreviation.isnot(None))
            ).scalar()
            or 0
        )
        results["law_with_citation_count"] = (
            session.execute(
                select(func.count())
                .select_from(LawDocument)
                .where(LawDocument.citation_count > 0)
            ).scalar()
            or 0
        )
        results["statute_aliases"] = (
            session.execute(
                select(func.count()).select_from(StatuteAlias)
            ).scalar()
            or 0
        )
        results["statute_hierarchy"] = (
            session.execute(
                select(func.count()).select_from(StatuteHierarchy)
            ).scalar()
            or 0
        )
        results["statute_relations"] = (
            session.execute(
                select(func.count()).select_from(StatuteRelation)
            ).scalar()
            or 0
        )
        results["case_statute_citations"] = (
            session.execute(
                select(func.count()).select_from(CaseStatuteCitation)
            ).scalar()
            or 0
        )
        results["case_case_citations"] = (
            session.execute(
                select(func.count()).select_from(CaseCaseCitation)
            ).scalar()
            or 0
        )

        # TOP 5 인용 법령
        top5 = session.execute(
            select(LawDocument.law_name, LawDocument.citation_count)
            .where(LawDocument.citation_count > 0)
            .order_by(LawDocument.citation_count.desc())
            .limit(5)
        ).all()

    logger.info("=== 그래프 테이블 검증 ===")
    for key, count in results.items():
        logger.info("  %s: %s", key, f"{count:,}")

    if top5:
        logger.info("Top 5 인용 법령:")
        for row in top5:
            logger.info("  %s: %s", row.law_name, row.citation_count)

    return results


# ---------------------------------------------------------------------------
# 내부 함수
# ---------------------------------------------------------------------------


def _verify_prerequisites() -> None:
    """law_documents, precedent_documents 행 존재 확인"""
    with sync_session_factory() as session:
        law_count = (
            session.execute(
                select(func.count()).select_from(LawDocument)
            ).scalar()
            or 0
        )
        prec_count = (
            session.execute(
                select(func.count()).select_from(PrecedentDocument)
            ).scalar()
            or 0
        )

    if law_count == 0:
        raise RuntimeError(
            "law_documents 테이블이 비어있습니다. "
            "먼저 --type law --step db를 실행하세요."
        )
    if prec_count == 0:
        raise RuntimeError(
            "precedent_documents 테이블이 비어있습니다. "
            "먼저 --type precedent --step db를 실행하세요."
        )

    logger.info(
        "사전 조건 확인: law_documents=%s, precedent_documents=%s",
        f"{law_count:,}",
        f"{prec_count:,}",
    )


def _reset_graph_tables() -> None:
    """관계 테이블 데이터 초기화"""
    logger.info("그래프 테이블 초기화 중...")
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
    logger.info("그래프 테이블 초기화 완료")


def _build_law_id_lookup() -> dict[str, int]:
    """law_id → law_documents.id 룩업 딕셔너리 빌드"""
    logger.info("law_id → id 룩업 빌드 중...")
    with sync_session_factory() as session:
        result = session.execute(
            select(LawDocument.id, LawDocument.law_id)
        )
        lookup = {row.law_id: row.id for row in result}
    logger.info("  %s법령 인덱싱 완료", f"{len(lookup):,}")
    return lookup


def _build_law_name_lookup() -> dict[str, int]:
    """law_name → law_documents.id 룩업 딕셔너리 빌드"""
    logger.info("law_name → id 룩업 빌드 중...")
    with sync_session_factory() as session:
        result = session.execute(
            select(LawDocument.id, LawDocument.law_name)
        )
        lookup = {row.law_name: row.id for row in result}
    logger.info("  %s법령명 인덱싱 완료", f"{len(lookup):,}")
    return lookup


def _build_case_serial_lookup() -> dict[str, int]:
    """serial_number → precedent_documents.id 룩업 딕셔너리 빌드"""
    logger.info("serial_number → id 룩업 빌드 중...")
    with sync_session_factory() as session:
        result = session.execute(
            select(PrecedentDocument.id, PrecedentDocument.serial_number)
        )
        lookup = {row.serial_number: row.id for row in result}
    logger.info("  %s판례 인덱싱 완료", f"{len(lookup):,}")
    return lookup


def _build_case_number_lookup() -> dict[str, int]:
    """case_number → precedent_documents.id 룩업 딕셔너리 빌드"""
    logger.info("case_number → id 룩업 빌드 중...")
    with sync_session_factory() as session:
        result = session.execute(
            select(PrecedentDocument.id, PrecedentDocument.case_number)
        )
        lookup: dict[str, int] = {}
        for row in result:
            if row.case_number:
                lookup[row.case_number] = row.id
    logger.info("  %s사건번호 인덱싱 완료", f"{len(lookup):,}")
    return lookup


def _load_abbreviations(
    law_id_lookup: dict[str, int],
    batch_size: int,
) -> int:
    """법령 약칭 로드 → law_documents.abbreviation UPDATE"""
    logger.info("=== 법령 약칭 적재 (%s) ===", _ABBREVIATION_FILE.name)

    if not _ABBREVIATION_FILE.exists():
        logger.warning("파일 미존재: %s", _ABBREVIATION_FILE)
        return 0

    with open(_ABBREVIATION_FILE, encoding="utf-8") as f:
        data = json.load(f)

    updated = 0
    batch: list[dict[str, object]] = []

    with sync_session_factory() as session:
        for item in data:
            statute_id = item.get("법령ID")
            abbreviation = item.get("법령약칭명")

            if not statute_id or not abbreviation:
                continue

            doc_id = law_id_lookup.get(statute_id)
            if not doc_id:
                continue

            batch.append({"doc_id": doc_id, "abbreviation": abbreviation})

            if len(batch) >= batch_size:
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

    logger.info("  약칭 업데이트: %s건", f"{updated:,}")
    return updated


def _load_informal_abbreviations(
    law_name_lookup: dict[str, int],
    batch_size: int,
) -> int:
    """비공식 약칭 로드 → statute_aliases INSERT"""
    logger.info("=== 비공식 약칭 적재 (%s) ===", _INFORMAL_ABBR_FILE.name)

    if not _INFORMAL_ABBR_FILE.exists():
        logger.warning("파일 미존재: %s", _INFORMAL_ABBR_FILE)
        return 0

    with open(_INFORMAL_ABBR_FILE, encoding="utf-8") as f:
        data = json.load(f)

    mappings = data.get("mappings", [])
    rows: list[dict[str, object]] = []

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
        logger.info("  적재 대상 없음")
        return 0

    inserted = 0
    with sync_session_factory() as session:
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
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

    logger.info("  비공식 약칭: %s건", f"{inserted:,}")
    return inserted


def _load_hierarchy(
    law_id_lookup: dict[str, int],
    batch_size: int,
) -> int:
    """법령 계급 관계 로드 → statute_hierarchy INSERT"""
    logger.info("=== 법령 계급 적재 (%s) ===", _HIERARCHY_FILE.name)

    if not _HIERARCHY_FILE.exists():
        logger.warning("파일 미존재: %s", _HIERARCHY_FILE)
        return 0

    with open(_HIERARCHY_FILE, encoding="utf-8") as f:
        data = json.load(f)

    real_relations: list[dict[str, str]] = []

    def parse_node(node_dict: dict[str, object], parent_id: str | None = None) -> None:
        current_id = None
        if "기본정보" in node_dict:
            info = node_dict["기본정보"]
            if isinstance(info, dict):
                current_id = info.get("법령ID")
                if current_id and parent_id and current_id != parent_id:
                    real_relations.append(
                        {"lower": str(current_id), "upper": str(parent_id)}
                    )

        next_parent = str(current_id) if current_id else parent_id

        for key, val in node_dict.items():
            if key in ("기본정보", "관련법령", "제개정구분"):
                continue
            targets: list[dict[str, object]] = []
            if isinstance(val, dict):
                targets.append(val)
            elif isinstance(val, list):
                for v in val:
                    if isinstance(v, dict):
                        targets.append(v)
            for t in targets:
                parse_node(t, next_parent)

    for item in data:
        parse_node(item)

    logger.info("  파싱된 계급 관계: %s건", f"{len(real_relations):,}")

    rows: list[dict[str, int]] = []
    skipped = 0
    for rel in real_relations:
        child_id = law_id_lookup.get(rel["lower"])
        parent_id = law_id_lookup.get(rel["upper"])
        if child_id and parent_id:
            rows.append({"child_id": child_id, "parent_id": parent_id})
        else:
            skipped += 1

    if skipped:
        logger.info("  스킵 (law_id 미매칭): %s건", f"{skipped:,}")

    inserted = 0
    with sync_session_factory() as session:
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            stmt = pg_insert(StatuteHierarchy).values(batch)
            stmt = stmt.on_conflict_do_nothing(
                constraint="uq_statute_hierarchy",
            )
            session.execute(stmt)
            session.commit()
            inserted += len(batch)

    logger.info("  계급 관계: %s건", f"{inserted:,}")
    return inserted


def _load_related_statutes(
    law_id_lookup: dict[str, int],
    batch_size: int,
) -> int:
    """법령 관련 관계 로드 → statute_relations INSERT"""
    logger.info("=== 관련 법령 적재 (%s) ===", _HIERARCHY_FILE.name)

    if not _HIERARCHY_FILE.exists():
        logger.warning("파일 미존재: %s", _HIERARCHY_FILE)
        return 0

    with open(_HIERARCHY_FILE, encoding="utf-8") as f:
        data = json.load(f)

    relations: list[dict[str, str]] = []

    for item in data:
        if "기본정보" not in item:
            continue
        source_id = item["기본정보"].get("법령ID")
        if not source_id:
            continue

        related = item.get("관련법령", {})
        if not isinstance(related, dict):
            continue
        conlaw_list = related.get("conlaw", [])
        if not isinstance(conlaw_list, list):
            continue

        for conlaw in conlaw_list:
            if not isinstance(conlaw, dict):
                continue
            target_id = conlaw.get("법령ID")
            if target_id and target_id != source_id:
                relations.append({
                    "source_id": str(source_id),
                    "target_id": str(target_id),
                })

    logger.info("  파싱된 관련 관계: %s건", f"{len(relations):,}")

    rows: list[dict[str, int]] = []
    seen: set[tuple[int, int]] = set()
    skipped = 0

    for rel in relations:
        doc_id_1 = law_id_lookup.get(rel["source_id"])
        doc_id_2 = law_id_lookup.get(rel["target_id"])
        if not doc_id_1 or not doc_id_2:
            skipped += 1
            continue
        pair = (min(doc_id_1, doc_id_2), max(doc_id_1, doc_id_2))
        if pair in seen:
            continue
        seen.add(pair)
        rows.append({"law_doc_id_1": pair[0], "law_doc_id_2": pair[1]})

    if skipped:
        logger.info("  스킵 (law_id 미매칭): %s건", f"{skipped:,}")

    inserted = 0
    with sync_session_factory() as session:
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            stmt = pg_insert(StatuteRelation).values(batch)
            stmt = stmt.on_conflict_do_nothing(
                constraint="uq_statute_relations",
            )
            session.execute(stmt)
            session.commit()
            inserted += len(batch)

    logger.info("  관련 법령: %s건", f"{inserted:,}")
    return inserted


def _load_case_statute_citations(
    law_name_lookup: dict[str, int],
    case_serial_lookup: dict[str, int],
    batch_size: int,
) -> int:
    """판례→법령 인용 로드 → case_statute_citations INSERT"""
    case_file = get_source_path("precedent")
    logger.info("=== 판례→법령 인용 적재 (%s) ===", case_file.name)

    if not case_file.exists():
        logger.warning("파일 미존재: %s", case_file)
        return 0

    with open(case_file, encoding="utf-8") as f:
        data = json.load(f)

    regex_statute = re.compile(r"([가-힣]+법(?:시행령|시행규칙)?)")

    rows: list[dict[str, int]] = []
    seen: set[tuple[int, int]] = set()
    skipped_case = 0
    skipped_law = 0

    for item in data:
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

    logger.info("  파싱된 판례→법령 인용: %s건", f"{len(rows):,}")
    if skipped_case:
        logger.info("  스킵 (판례 serial 미매칭): %s건", f"{skipped_case:,}")
    if skipped_law:
        logger.info("  스킵 (법령명 미매칭): %s건", f"{skipped_law:,}")

    inserted = 0
    with sync_session_factory() as session:
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            stmt = pg_insert(CaseStatuteCitation).values(batch)
            stmt = stmt.on_conflict_do_nothing(
                constraint="uq_case_statute_citation",
            )
            session.execute(stmt)
            session.commit()
            inserted += len(batch)

    logger.info("  판례→법령 인용: %s건", f"{inserted:,}")
    return inserted


def _load_case_case_citations(
    case_serial_lookup: dict[str, int],
    case_number_lookup: dict[str, int],
    batch_size: int,
) -> int:
    """판례→판례 인용 로드 → case_case_citations INSERT"""
    case_file = get_source_path("precedent")
    logger.info("=== 판례→판례 인용 적재 (%s) ===", case_file.name)

    if not case_file.exists():
        logger.warning("파일 미존재: %s", case_file)
        return 0

    with open(case_file, encoding="utf-8") as f:
        data = json.load(f)

    regex_case_number = re.compile(
        r"(\d{2,4})"
        r"([가-힣]{1,3})"
        r"(\d+)"
    )

    rows: list[dict[str, int]] = []
    seen: set[tuple[int, int]] = set()
    skipped_citing = 0
    skipped_cited = 0

    for item in data:
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

    logger.info("  파싱된 판례→판례 인용: %s건", f"{len(rows):,}")
    if skipped_citing:
        logger.info("  스킵 (인용 판례 serial 미매칭): %s건", f"{skipped_citing:,}")
    if skipped_cited:
        logger.info("  스킵 (피인용 사건번호 미매칭): %s건", f"{skipped_cited:,}")

    inserted = 0
    with sync_session_factory() as session:
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            stmt = pg_insert(CaseCaseCitation).values(batch)
            stmt = stmt.on_conflict_do_nothing(
                constraint="uq_case_case_citation",
            )
            session.execute(stmt)
            session.commit()
            inserted += len(batch)

    logger.info("  판례→판례 인용: %s건", f"{inserted:,}")
    return inserted


def _compute_citation_counts() -> int:
    """법령별 인용 수 계산 → law_documents.citation_count UPDATE"""
    logger.info("=== 인용 수 계산 ===")

    with sync_session_factory() as session:
        subq = (
            select(
                CaseStatuteCitation.law_doc_id,
                func.count(CaseStatuteCitation.id).label("cnt"),
            )
            .group_by(CaseStatuteCitation.law_doc_id)
            .subquery()
        )

        session.execute(
            update(LawDocument)
            .where(LawDocument.id == subq.c.law_doc_id)
            .values(citation_count=subq.c.cnt)
        )
        session.commit()

        result = session.execute(
            select(func.count()).select_from(LawDocument).where(
                LawDocument.citation_count > 0
            )
        )
        updated = result.scalar() or 0

    logger.info("  citation_count > 0: %s건", f"{updated:,}")
    return updated
