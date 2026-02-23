"""
자치법규 인제스트 설정

data/local_rules_v1.json (160,276건)을 대상으로:
- PostgreSQL: 원문 전체 + FTS 인덱스
- 벡터 DB: 전용 라이터 사용 (1문서 = 1전체요약 + N조문요약)
  → vector_metadata_fn은 Basic(전체요약) 1:1 fallback용
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_backend_root = Path(__file__).parent.parent.parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))

from app.models.ingest.local_ordinance_document import (  # noqa: E402
    LocalOrdinanceDocument,
)
from app.tools.vectorstore.local_ordinance_schema import (  # noqa: E402
    create_overall_summary_chunk,
)
from scripts.ingest.config import (  # noqa: E402
    IngestConfig,
    get_source_path,
    register_config,
)

_DEFAULT_SOURCE = get_source_path("local_ordinance")


# ---------------------------------------------------------------------------
# ORM 팩토리
# ---------------------------------------------------------------------------


def _concat_articles(articles: Any) -> str | None:
    """조문 리스트 → 텍스트 concat"""
    if not articles or not isinstance(articles, list):
        return None
    parts: list[str] = []
    for art in articles:
        if isinstance(art, dict):
            no = str(art.get("조문번호", ""))
            text = str(art.get("조내용", "") or "")
            if text:
                parts.append(f"{no} {text}" if no else text)
    return "\n\n".join(parts) if parts else None


def _orm_factory(item: dict[str, Any]) -> LocalOrdinanceDocument:
    """JSON item → LocalOrdinanceDocument 인스턴스"""
    content = _concat_articles(item.get("조"))
    overall_summary = item.get("전체요약")

    return LocalOrdinanceDocument(
        ordinance_id=str(item.get("자치법규ID", "")),
        ordinance_serial=str(item.get("자치법규일련번호", "") or ""),
        ordinance_name=str(item.get("자치법규명", "") or ""),
        local_government=str(item.get("지자체기관명", "") or ""),
        overall_summary=overall_summary,
        content=content,
        supplementary=item.get("부칙내용"),
        ai_summary=overall_summary,
    )


# ---------------------------------------------------------------------------
# 벡터 DB 함수 (Basic 전체요약 1:1 fallback)
# ---------------------------------------------------------------------------


def _vector_metadata_fn(
    item: dict[str, Any],
    vector: list[float],
) -> dict[str, Any]:
    """JSON item + vector → LanceDB record (전체요약 1:1 fallback)

    실제 다중 벡터 저장은 local_ordinance_vector_writer.py에서 처리.
    이 함수는 기존 파이프라인 호환용 stub입니다.
    """
    source_id = str(item.get("자치법규ID", ""))
    articles = item.get("조", [])
    total_chunks = 1 + (len(articles) if isinstance(articles, list) else 0)

    return create_overall_summary_chunk(
        source_id=source_id,
        title=str(item.get("자치법규명", "") or ""),
        content=str(item.get("전체요약", "") or ""),
        vector=vector,
        source_name=str(item.get("지자체기관명", "") or ""),
        total_chunks=total_chunks,
    )


# ---------------------------------------------------------------------------
# FTS 함수 (JSON 소스)
# ---------------------------------------------------------------------------


def _fulltext_fn(item: dict[str, Any]) -> str:
    """JSON item → FTS용 원문 텍스트 concat"""
    parts: list[str] = []

    title = item.get("자치법규명")
    if title:
        parts.append(f"[{title}]")

    articles = item.get("조")
    if articles and isinstance(articles, list):
        for art in articles:
            if isinstance(art, dict):
                text = art.get("조내용", "")
                if text:
                    parts.append(str(text))

    supplementary = item.get("부칙내용")
    if supplementary:
        parts.append(str(supplementary))

    return "\n".join(parts)


def _fts_metadata_fn(item: dict[str, Any]) -> dict[str, Any]:
    """JSON item → fts_index 메타데이터 dict"""
    return {
        "source_id": str(item.get("자치법규ID", "")),
        "data_type": "자치법규",
        "title": str(item.get("자치법규명", "") or ""),
        "date": None,
        "source_name": str(item.get("지자체기관명", "") or ""),
        "case_number": None,
    }


# ---------------------------------------------------------------------------
# FTS 함수 (ORM 소스 — DB 재빌드용)
# ---------------------------------------------------------------------------


def _orm_fulltext_fn(row: Any) -> str:
    """LocalOrdinanceDocument ORM → FTS용 원문 텍스트 concat"""
    parts: list[str] = []

    if row.ordinance_name:
        parts.append(f"[{row.ordinance_name}]")
    if row.content:
        parts.append(row.content)
    if row.supplementary:
        parts.append(row.supplementary)

    return "\n".join(parts)


def _orm_fts_metadata_fn(row: Any) -> dict[str, Any]:
    """LocalOrdinanceDocument ORM → fts_index 메타데이터 dict"""
    return {
        "source_id": row.ordinance_id,
        "data_type": "자치법규",
        "title": row.ordinance_name or "",
        "date": None,
        "source_name": row.local_government,
        "case_number": None,
    }


# ---------------------------------------------------------------------------
# 설정 등록
# ---------------------------------------------------------------------------

LOCAL_ORDINANCE_CONFIG = IngestConfig(
    name="local_ordinance",
    data_type_label="자치법규",
    source_path=_DEFAULT_SOURCE,
    id_field="자치법규ID",
    summary_field="전체요약",
    title_field="자치법규명",
    orm_class=LocalOrdinanceDocument,
    orm_id_attr="ordinance_id",
    orm_factory_fn=_orm_factory,
    vector_metadata_fn=_vector_metadata_fn,
    fulltext_fn=_fulltext_fn,
    fts_metadata_fn=_fts_metadata_fn,
    orm_fulltext_fn=_orm_fulltext_fn,
    orm_fts_metadata_fn=_orm_fts_metadata_fn,
)

register_config(LOCAL_ORDINANCE_CONFIG)
