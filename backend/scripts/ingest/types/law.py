"""
법령 인제스트 설정

data/raw/law.json (한국어 키)을 대상으로:
- 벡터 DB: 법령 요약 1문서=1벡터
- PostgreSQL: 원문 전체 + FTS 인덱스
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# 백엔드 app 모듈 import를 위한 경로 추가
_backend_root = Path(__file__).parent.parent.parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))

from app.models.law_document import LawDocument  # noqa: E402
from scripts.embedding_common.schema import create_law_chunk  # noqa: E402
from scripts.ingest.config import DATA_DIR, IngestConfig, register_config  # noqa: E402

# 기본 데이터 소스 경로
_DEFAULT_SOURCE = DATA_DIR / "raw" / "law.json"


# ---------------------------------------------------------------------------
# ORM 팩토리: JSON item → LawDocument 인스턴스 (적재용 SSOT)
# ---------------------------------------------------------------------------


def _orm_factory(item: dict[str, Any]) -> LawDocument:
    """JSON item → LawDocument 인스턴스"""
    # 조문 리스트 → 텍스트 concat
    content = _concat_articles(item.get("조문"))

    # 부칙 리스트 → 텍스트 concat
    supplementary = _concat_supplementary(item.get("부칙"))

    return LawDocument(
        law_id=item.get("법령ID", ""),
        law_name=item.get("법령명_한글", ""),
        content=content,
        supplementary=supplementary,
        ai_summary=item.get("법령 요약"),
    )


def _concat_articles(articles: Any) -> str | None:
    """조문 리스트 → 텍스트 concat"""
    if not articles or not isinstance(articles, list):
        return None

    parts: list[str] = []
    for article in articles:
        if isinstance(article, dict):
            no = article.get("조문번호", "")
            text = article.get("조문내용", "")
            if text:
                parts.append(f"{no} {text}" if no else text)
        elif isinstance(article, str):
            parts.append(article)

    return "\n".join(parts) if parts else None


def _flatten_list(data: Any) -> list[Any]:
    """중첩 리스트 [[{...}]] → [{...}] 평탄화 (1단계)"""
    if not isinstance(data, list):
        return []
    result: list[Any] = []
    for item in data:
        if isinstance(item, list):
            result.extend(item)
        else:
            result.append(item)
    return result


def _concat_supplementary(supplementary: Any) -> str | None:
    """부칙 리스트 → 텍스트 concat (중첩 리스트 [[{...}]] 대응)"""
    if not supplementary or not isinstance(supplementary, list):
        return None

    parts: list[str] = []
    for item in _flatten_list(supplementary):
        if isinstance(item, dict):
            text = item.get("부칙내용", "")
            if text:
                parts.append(text)
        elif isinstance(item, str):
            parts.append(item)

    return "\n".join(parts) if parts else None


# ---------------------------------------------------------------------------
# 벡터 DB 함수
# ---------------------------------------------------------------------------


def _vector_metadata_fn(
    item: dict[str, Any],
    vector: list[float],
) -> dict[str, Any]:
    """JSON item + embedding vector → LanceDB record dict"""
    source_id = str(item.get("법령ID", ""))
    title = item.get("법령명_한글", "") or ""
    content = item.get("법령 요약", "") or ""

    return create_law_chunk(
        source_id=source_id,
        chunk_index=0,
        title=title,
        content=content,
        vector=vector,
        enforcement_date="",
        department="",
        total_chunks=1,
    )


# ---------------------------------------------------------------------------
# FTS 함수 (JSON 소스)
# ---------------------------------------------------------------------------


def _fulltext_fn(item: dict[str, Any]) -> str:
    """JSON item → FTS용 원문 텍스트 concat"""
    parts: list[str] = []

    law_name = item.get("법령명_한글")
    if law_name:
        parts.append(f"[{law_name}]")

    # 조문 텍스트
    articles = item.get("조문")
    if articles and isinstance(articles, list):
        for article in articles:
            if isinstance(article, dict):
                text = article.get("조문내용", "")
                if text:
                    parts.append(text)

    # 부칙 텍스트 (중첩 리스트 [[{...}]] 대응)
    supplementary = item.get("부칙")
    if supplementary and isinstance(supplementary, list):
        for supp in _flatten_list(supplementary):
            if isinstance(supp, dict):
                text = supp.get("부칙내용", "")
                if text:
                    parts.append(text)

    return "\n".join(parts)


def _fts_metadata_fn(item: dict[str, Any]) -> dict[str, Any]:
    """JSON item → fts_index 메타데이터 dict"""
    return {
        "source_id": str(item.get("법령ID", "")),
        "data_type": "법령",
        "title": item.get("법령명_한글", "") or "",
        "date": None,
        "source_name": None,
        "case_number": None,
    }


# ---------------------------------------------------------------------------
# FTS 함수 (ORM 소스 — DB 재빌드용)
# ---------------------------------------------------------------------------


def _orm_fulltext_fn(row: Any) -> str:
    """LawDocument ORM 인스턴스 → FTS용 원문 텍스트 concat"""
    parts: list[str] = []

    if row.law_name:
        parts.append(f"[{row.law_name}]")
    if row.content:
        parts.append(row.content)
    if row.supplementary:
        parts.append(row.supplementary)

    return "\n".join(parts)


def _orm_fts_metadata_fn(row: Any) -> dict[str, Any]:
    """LawDocument ORM 인스턴스 → fts_index 메타데이터 dict"""
    return {
        "source_id": row.law_id,
        "data_type": "법령",
        "title": row.law_name or "",
        "date": None,
        "source_name": None,
        "case_number": None,
    }


# ---------------------------------------------------------------------------
# 설정 등록
# ---------------------------------------------------------------------------

LAW_CONFIG = IngestConfig(
    name="law",
    data_type_label="법령",
    source_path=_DEFAULT_SOURCE,
    id_field="법령ID",
    summary_field="법령 요약",
    title_field="법령명_한글",
    orm_class=LawDocument,
    orm_id_attr="law_id",
    orm_factory_fn=_orm_factory,
    vector_metadata_fn=_vector_metadata_fn,
    fulltext_fn=_fulltext_fn,
    fts_metadata_fn=_fts_metadata_fn,
    orm_fulltext_fn=_orm_fulltext_fn,
    orm_fts_metadata_fn=_orm_fts_metadata_fn,
)

register_config(LAW_CONFIG)
