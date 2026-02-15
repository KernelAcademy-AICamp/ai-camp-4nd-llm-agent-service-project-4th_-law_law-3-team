"""
행정규칙 인제스트 설정

data/admin_rule_v2.json (5,258건)을 대상으로:
- 벡터 DB: 행정규칙요약 1문서=1벡터
- PostgreSQL: 원문 전체 + FTS 인덱스
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_backend_root = Path(__file__).parent.parent.parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))

from app.models.ingest.admin_rule_document import AdminRuleDocument  # noqa: E402
from scripts.embedding_common.schema import create_chunk  # noqa: E402
from scripts.ingest.config import (  # noqa: E402
    IngestConfig,
    get_source_path,
    register_config,
)

_DEFAULT_SOURCE = get_source_path("admin_rule")


# ---------------------------------------------------------------------------
# ORM 팩토리
# ---------------------------------------------------------------------------


def _orm_factory(item: dict[str, Any]) -> AdminRuleDocument:
    """JSON item → AdminRuleDocument 인스턴스"""
    # 조문내용: list[str] → 텍스트 concat
    content_list = item.get("조문내용")
    content: str | None = None
    if isinstance(content_list, list):
        content = "\n".join(str(s) for s in content_list if s)
    elif isinstance(content_list, str):
        content = content_list

    return AdminRuleDocument(
        admin_rule_id=item.get("행정규칙ID", ""),
        serial_number=item.get("행정규칙일련번호"),
        admin_rule_name=item.get("행정규칙명", ""),
        admin_rule_type=item.get("행정규칙종류"),
        ministry=item.get("소관부처명"),
        content=content,
        supplementary=item.get("부칙내용"),
        ai_summary=item.get("행정규칙요약"),
    )


# ---------------------------------------------------------------------------
# 벡터 DB 함수
# ---------------------------------------------------------------------------


def _vector_metadata_fn(
    item: dict[str, Any],
    vector: list[float],
) -> dict[str, Any]:
    """JSON item + embedding vector → LanceDB record dict"""
    return create_chunk(
        data_type="행정규칙",
        source_id=str(item.get("행정규칙ID", "")),
        title=item.get("행정규칙명", "") or "",
        content=item.get("행정규칙요약", "") or "",
        vector=vector,
        source_name=item.get("소관부처명", "") or "",
        chunk_index=0,
        total_chunks=1,
    )


# ---------------------------------------------------------------------------
# FTS 함수 (JSON 소스)
# ---------------------------------------------------------------------------


def _fulltext_fn(item: dict[str, Any]) -> str:
    """JSON item → FTS용 원문 텍스트 concat"""
    parts: list[str] = []

    name = item.get("행정규칙명")
    if name:
        parts.append(f"[{name}]")

    content_list = item.get("조문내용")
    if isinstance(content_list, list):
        for text in content_list:
            if text:
                parts.append(str(text))
    elif isinstance(content_list, str) and content_list:
        parts.append(content_list)

    supplementary = item.get("부칙내용")
    if supplementary:
        parts.append(supplementary)

    return "\n".join(parts)


def _fts_metadata_fn(item: dict[str, Any]) -> dict[str, Any]:
    """JSON item → fts_index 메타데이터 dict"""
    return {
        "source_id": str(item.get("행정규칙ID", "")),
        "data_type": "행정규칙",
        "title": item.get("행정규칙명", "") or "",
        "date": None,
        "source_name": item.get("소관부처명"),
        "case_number": None,
    }


# ---------------------------------------------------------------------------
# FTS 함수 (ORM 소스 — DB 재빌드용)
# ---------------------------------------------------------------------------


def _orm_fulltext_fn(row: Any) -> str:
    """AdminRuleDocument ORM 인스턴스 → FTS용 원문 텍스트 concat"""
    parts: list[str] = []

    if row.admin_rule_name:
        parts.append(f"[{row.admin_rule_name}]")
    if row.content:
        parts.append(row.content)
    if row.supplementary:
        parts.append(row.supplementary)

    return "\n".join(parts)


def _orm_fts_metadata_fn(row: Any) -> dict[str, Any]:
    """AdminRuleDocument ORM 인스턴스 → fts_index 메타데이터 dict"""
    return {
        "source_id": row.admin_rule_id,
        "data_type": "행정규칙",
        "title": row.admin_rule_name or "",
        "date": None,
        "source_name": row.ministry,
        "case_number": None,
    }


# ---------------------------------------------------------------------------
# 설정 등록
# ---------------------------------------------------------------------------

ADMIN_RULE_CONFIG = IngestConfig(
    name="admin_rule",
    data_type_label="행정규칙",
    source_path=_DEFAULT_SOURCE,
    id_field="행정규칙ID",
    summary_field="행정규칙요약",
    title_field="행정규칙명",
    orm_class=AdminRuleDocument,
    orm_id_attr="admin_rule_id",
    orm_factory_fn=_orm_factory,
    vector_metadata_fn=_vector_metadata_fn,
    fulltext_fn=_fulltext_fn,
    fts_metadata_fn=_fts_metadata_fn,
    orm_fulltext_fn=_orm_fulltext_fn,
    orm_fts_metadata_fn=_orm_fts_metadata_fn,
)

register_config(ADMIN_RULE_CONFIG)
