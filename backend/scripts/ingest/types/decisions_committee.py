"""
위원회 결정례 인제스트 설정

data/ingest_source/decisions_committee/ 디렉토리 (11파일, 10개 위원회)을 대상으로:
- 벡터 DB: 결정문요약 1문서=1벡터
- PostgreSQL: 원문 전체 + FTS 인덱스

10개 위원회별로 JSON 스키마가 상이하므로 다중 필드명 매핑 적용:
- 제목: 사건명 / 안건명 / 제목 → case_name
- 날짜: 의결일자 / 의결일 / 결정일자 / 등록일 → decision_date
- 번호: 결정번호 / 의결번호 / 의안번호 → decision_number
- 요지: 결정요지 / 판단요지 / 판정요지 / 판정사항 → decision_summary
- 전문: 의결문 / 결정례전문 / 내용 → full_text
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

_backend_root = Path(__file__).parent.parent.parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))

from app.models.ingest.decisions_committee_document import (  # noqa: E402
    DecisionsCommitteeDocument,
)
from scripts.embedding_common.schema import create_chunk  # noqa: E402
from scripts.ingest.config import DATA_DIR, IngestConfig, register_config  # noqa: E402

_DEFAULT_SOURCE = DATA_DIR / "ingest_source" / "decisions_committee"


# ---------------------------------------------------------------------------
# 헬퍼: 다중 필드명 매핑
# ---------------------------------------------------------------------------


def _first(*values: Optional[str]) -> Optional[str]:
    """첫 번째 truthy 값 반환 (None/빈 문자열 스킵)"""
    for v in values:
        if v:
            return v
    return None


def _get_title(item: dict[str, Any]) -> Optional[str]:
    """사건명 / 안건명 / 제목 → 통합 제목"""
    return _first(item.get("사건명"), item.get("안건명"), item.get("제목"))


def _get_date(item: dict[str, Any]) -> Optional[str]:
    """의결일자 / 의결일 / 결정일자 / 등록일 → 통합 날짜"""
    return _first(
        item.get("의결일자"),
        item.get("의결일"),
        item.get("결정일자"),
        item.get("등록일"),
    )


def _get_decision_number(item: dict[str, Any]) -> Optional[str]:
    """결정번호 / 의결번호 / 의안번호 → 통합 결정번호"""
    return _first(
        item.get("결정번호"),
        item.get("의결번호"),
        item.get("의안번호"),
    )


def _get_decision_summary(item: dict[str, Any]) -> Optional[str]:
    """결정요지 / 판단요지 / 판정요지 / 판정사항 → 통합 결정요지"""
    return _first(
        item.get("결정요지"),
        item.get("판단요지"),
        item.get("판정요지"),
        item.get("판정사항"),
    )


def _get_full_text(item: dict[str, Any]) -> Optional[str]:
    """의결문 / 결정례전문 / 내용 → 통합 전문"""
    return _first(
        item.get("의결문"),
        item.get("결정례전문"),
        item.get("내용"),
    )


# ---------------------------------------------------------------------------
# ORM 팩토리
# ---------------------------------------------------------------------------


def _orm_factory(item: dict[str, Any]) -> DecisionsCommitteeDocument:
    """JSON item → DecisionsCommitteeDocument 인스턴스"""
    return DecisionsCommitteeDocument(
        serial_number=item.get("결정문일련번호", ""),
        case_name=_get_title(item),
        decision_date=_get_date(item),
        case_number=item.get("사건번호"),
        decision_number=_get_decision_number(item),
        ruling=item.get("주문"),
        reason=item.get("이유"),
        decision_summary=_get_decision_summary(item),
        claim=item.get("청구취지"),
        appendix=item.get("별지"),
        action_reason=item.get("조치이유"),
        action_content=item.get("조치내용"),
        full_text=_get_full_text(item),
        committee_name=item.get("__source_group__"),
        ai_summary=item.get("결정문요약"),
    )


# ---------------------------------------------------------------------------
# 벡터 DB 함수
# ---------------------------------------------------------------------------


def _vector_metadata_fn(
    item: dict[str, Any],
    vector: list[float],
) -> dict[str, Any]:
    """JSON item + embedding vector → LanceDB record dict"""
    committee = item.get("__source_group__", "")
    date_str = _get_date(item) or ""
    return create_chunk(
        data_type="위원회결정례",
        source_id=str(item.get("결정문일련번호", "")),
        title=_get_title(item) or "",
        content=item.get("결정문요약", "") or "",
        vector=vector,
        source_name=committee or "",
        date=date_str if date_str else None,
        chunk_index=0,
        total_chunks=1,
    )


# ---------------------------------------------------------------------------
# FTS 함수 (JSON 소스)
# ---------------------------------------------------------------------------


def _fulltext_fn(item: dict[str, Any]) -> str:
    """JSON item → FTS용 원문 텍스트 concat"""
    parts: list[str] = []

    title = _get_title(item)
    if title:
        parts.append(f"[{title}]")

    committee = item.get("__source_group__")
    if committee:
        parts.append(f"위원회: {committee}")

    # 주요 텍스트 필드 (위원회별로 다른 필드명)
    for field in (
        "주문", "이유", "결정요지", "판단요지", "판정요지", "판정사항",
        "청구취지", "조치이유", "조치내용",
    ):
        value = item.get(field)
        if value:
            parts.append(value)

    return "\n".join(parts)


def _fts_metadata_fn(item: dict[str, Any]) -> dict[str, Any]:
    """JSON item → fts_index 메타데이터 dict"""
    return {
        "source_id": str(item.get("결정문일련번호", "")),
        "data_type": "위원회결정례",
        "title": _get_title(item) or "",
        "date": _get_date(item),
        "source_name": item.get("__source_group__"),
        "case_number": item.get("사건번호"),
    }


# ---------------------------------------------------------------------------
# FTS 함수 (ORM 소스 — DB 재빌드용)
# ---------------------------------------------------------------------------


def _orm_fulltext_fn(row: Any) -> str:
    """DecisionsCommitteeDocument ORM 인스턴스 → FTS용 원문 텍스트 concat"""
    parts: list[str] = []

    if row.case_name:
        parts.append(f"[{row.case_name}]")
    if row.committee_name:
        parts.append(f"위원회: {row.committee_name}")
    if row.ruling:
        parts.append(row.ruling)
    if row.reason:
        parts.append(row.reason)
    if row.decision_summary:
        parts.append(row.decision_summary)
    if row.claim:
        parts.append(row.claim)
    if row.action_reason:
        parts.append(row.action_reason)
    if row.action_content:
        parts.append(row.action_content)

    return "\n".join(parts)


def _orm_fts_metadata_fn(row: Any) -> dict[str, Any]:
    """DecisionsCommitteeDocument ORM 인스턴스 → fts_index 메타데이터 dict"""
    return {
        "source_id": row.serial_number,
        "data_type": "위원회결정례",
        "title": row.case_name or "",
        "date": row.decision_date,
        "source_name": row.committee_name,
        "case_number": row.case_number,
    }


# ---------------------------------------------------------------------------
# 설정 등록
# ---------------------------------------------------------------------------

DECISIONS_COMMITTEE_CONFIG = IngestConfig(
    name="decisions_committee",
    data_type_label="위원회결정례",
    source_path=_DEFAULT_SOURCE,
    id_field="결정문일련번호",
    summary_field="결정문요약",
    title_field="사건명",
    orm_class=DecisionsCommitteeDocument,
    orm_id_attr="serial_number",
    orm_factory_fn=_orm_factory,
    vector_metadata_fn=_vector_metadata_fn,
    fulltext_fn=_fulltext_fn,
    fts_metadata_fn=_fts_metadata_fn,
    orm_fulltext_fn=_orm_fulltext_fn,
    orm_fts_metadata_fn=_orm_fts_metadata_fn,
)

register_config(DECISIONS_COMMITTEE_CONFIG)
