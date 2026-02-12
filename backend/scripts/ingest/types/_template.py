"""
[타입명] 인제스트 설정 — 복사하여 새 타입 생성

사용법:
    1. 이 파일을 types/new_type.py로 복사
    2. TODO 주석을 따라 수정
    3. 자동 등록됨 (__init__.py 수정 불필요)

필수 사전 작업:
    - app/models/new_type_document.py 생성 (순수 테이블 정의, ai_summary 포함)
    - alembic 마이그레이션 작성
    - embedding_common/schema.py에 create_xxx_chunk 함수 추가 (벡터 사용 시)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_backend_root = Path(__file__).parent.parent.parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))

# TODO: ORM 모델 import 변경
# from app.models.new_type_document import NewTypeDocument  # noqa: E402
# TODO: 벡터 스키마 함수 import 변경
# from scripts.embedding_common.schema import create_xxx_chunk  # noqa: E402
from scripts.ingest.config import (  # noqa: E402, F401
    DATA_DIR,
    IngestConfig,
    register_config,
)

# TODO: 데이터 소스 경로 변경
_DEFAULT_SOURCE = DATA_DIR / "raw" / "new_type.json"


# ---------------------------------------------------------------------------
# ORM 팩토리: JSON item → ORM 인스턴스 (적재용 SSOT)
# ---------------------------------------------------------------------------


def _orm_factory(item: dict[str, Any]) -> Any:
    """JSON item → ORM 인스턴스

    TODO: ORM 클래스와 JSON 키 매핑을 수정하세요.
    """
    raise NotImplementedError("_orm_factory를 구현하세요")
    # return NewTypeDocument(
    #     source_id=item.get("ID필드", ""),
    #     title=item.get("제목필드", ""),
    #     content=item.get("내용필드"),
    #     ai_summary=item.get("요약필드"),
    # )


# ---------------------------------------------------------------------------
# 벡터 DB 함수
# ---------------------------------------------------------------------------


def _vector_metadata_fn(
    item: dict[str, Any],
    vector: list[float],
) -> dict[str, Any]:
    """JSON item + embedding vector → LanceDB record dict

    TODO: create_xxx_chunk 호출로 변경하세요.
    """
    raise NotImplementedError("_vector_metadata_fn를 구현하세요")
    # return create_xxx_chunk(
    #     source_id=str(item.get("ID필드", "")),
    #     chunk_index=0,
    #     title=item.get("제목필드", "") or "",
    #     content=item.get("요약필드", "") or "",
    #     vector=vector,
    #     total_chunks=1,
    # )


# ---------------------------------------------------------------------------
# FTS 함수 (JSON 소스)
# ---------------------------------------------------------------------------


def _fulltext_fn(item: dict[str, Any]) -> str:
    """JSON item → FTS용 원문 텍스트 concat

    TODO: FTS 검색에 포함할 텍스트 필드를 조합하세요.
    """
    parts: list[str] = []

    title = item.get("제목필드")
    if title:
        parts.append(f"[{title}]")

    content = item.get("내용필드")
    if content:
        parts.append(content)

    return "\n".join(parts)


def _fts_metadata_fn(item: dict[str, Any]) -> dict[str, Any]:
    """JSON item → fts_index 메타데이터 dict

    TODO: data_type, title 등을 수정하세요.
    """
    return {
        "source_id": str(item.get("ID필드", "")),
        "data_type": "타입라벨",  # TODO: 한글 라벨 (예: "법령", "판례")
        "title": item.get("제목필드", "") or "",
        "date": None,
        "source_name": None,
        "case_number": None,
    }


# ---------------------------------------------------------------------------
# FTS 함수 (ORM 소스 — DB 재빌드용)
# ---------------------------------------------------------------------------


def _orm_fulltext_fn(row: Any) -> str:
    """ORM 인스턴스 → FTS용 원문 텍스트 concat

    TODO: ORM 속성명으로 변경하세요.
    """
    parts: list[str] = []

    if row.title:
        parts.append(f"[{row.title}]")
    if row.content:
        parts.append(row.content)

    return "\n".join(parts)


def _orm_fts_metadata_fn(row: Any) -> dict[str, Any]:
    """ORM 인스턴스 → fts_index 메타데이터 dict

    TODO: ORM 속성명으로 변경하세요.
    """
    return {
        "source_id": row.source_id,
        "data_type": "타입라벨",  # TODO: _fts_metadata_fn과 동일
        "title": row.title or "",
        "date": None,
        "source_name": None,
        "case_number": None,
    }


# ---------------------------------------------------------------------------
# 설정 등록 — TODO: 전체 수정 필요
# ---------------------------------------------------------------------------

# NEW_TYPE_CONFIG = IngestConfig(
#     name="new_type",                    # CLI에서 사용할 이름
#     data_type_label="타입라벨",           # 한글 라벨
#     source_path=_DEFAULT_SOURCE,
#     id_field="ID필드",                   # JSON 내 고유 ID 키
#     summary_field="요약필드",             # 벡터 임베딩 대상 텍스트 키
#     title_field="제목필드",               # 제목 키
#     orm_class=NewTypeDocument,
#     orm_id_attr="source_id",            # ORM의 고유 ID 속성명
#     orm_factory_fn=_orm_factory,
#     vector_metadata_fn=_vector_metadata_fn,
#     fulltext_fn=_fulltext_fn,
#     fts_metadata_fn=_fts_metadata_fn,
#     orm_fulltext_fn=_orm_fulltext_fn,
#     orm_fts_metadata_fn=_orm_fts_metadata_fn,
# )
#
# register_config(NEW_TYPE_CONFIG)
