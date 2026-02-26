"""
위원회 결정례 공통 헬퍼 (11개 위원회 타입 설정에서 공유)

파일 접두사 '_'로 자동 등록(auto-scan) 대상에서 제외됨.
벡터 DB·FTS 함수 팩토리 및 IngestConfig 등록 유틸.

각 타입 모듈은 ORM 팩토리, fulltext_fn, orm_fulltext_fn만 정의하고
register_dec_comm()을 호출하면 벡터/FTS/메타데이터 함수를 자동 생성합니다.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Optional, Type

_backend_root = Path(__file__).parent.parent.parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))

from scripts.embedding_common.schema import create_chunk  # noqa: E402
from scripts.ingest.config import (  # noqa: E402
    IngestConfig,
    get_source_path,
    register_config,
)


def register_dec_comm(
    *,
    name: str,
    committee_name: str,
    orm_class: Type[Any],
    orm_factory_fn: Callable[[dict[str, Any]], Any],
    fulltext_fn: Callable[[dict[str, Any]], str],
    orm_fulltext_fn: Callable[[Any], str],
    title_field: Optional[str] = "사건명",
    date_field: Optional[str] = "의결일자",
    case_number_field: Optional[str] = "사건번호",
) -> IngestConfig:
    """
    위원회 결정례 개별 타입 IngestConfig 생성 + 등록

    Args:
        name: 설정 식별자 (예: "dec_employment")
        committee_name: 위원회명 (예: "고용보험심사위원회")
        orm_class: SQLAlchemy ORM 클래스
        orm_factory_fn: JSON item -> ORM 인스턴스
        fulltext_fn: JSON item -> FTS 원문 텍스트
        orm_fulltext_fn: ORM row -> FTS 원문 텍스트
        title_field: JSON 제목 필드명 (None이면 사건번호 등 대체)
        date_field: JSON 날짜 필드명 (None이면 날짜 없음)
        case_number_field: JSON 사건번호 필드명 (None이면 사건번호 없음)

    Returns:
        등록된 IngestConfig 인스턴스
    """

    source_path = get_source_path(name)

    def _vector_metadata_fn(
        item: dict[str, Any],
        vector: list[float],
    ) -> dict[str, Any]:
        date_str = ""
        if date_field:
            date_str = str(item.get(date_field, "") or "")
        title = ""
        if title_field:
            title = item.get(title_field, "") or ""
        elif case_number_field:
            title = item.get(case_number_field, "") or ""
        return create_chunk(
            data_type="위원회결정례",
            source_id=str(item.get("결정문일련번호", "")),
            title=title,
            content=item.get("결정문요약", "") or "",
            vector=vector,
            source_name=committee_name,
            date=date_str if date_str else None,
            chunk_index=0,
            total_chunks=1,
        )

    def _fts_metadata_fn(item: dict[str, Any]) -> dict[str, Any]:
        date_str = ""
        if date_field:
            date_str = str(item.get(date_field, "") or "")
        title = ""
        if title_field:
            title = item.get(title_field, "") or ""
        elif case_number_field:
            title = item.get(case_number_field, "") or ""
        return {
            "source_id": f"{name}:{item.get('결정문일련번호', '')}",
            "data_type": "위원회결정례",
            "title": title,
            "date": date_str if date_str else None,
            "source_name": committee_name,
            "case_number": (
                item.get(case_number_field) if case_number_field else None
            ),
        }

    def _orm_fts_metadata_fn(row: Any) -> dict[str, Any]:
        return {
            "source_id": f"{name}:{row.serial_number}",
            "data_type": "위원회결정례",
            "title": getattr(row, "case_name", None)
            or getattr(row, "case_label", None)
            or getattr(row, "case_number", None)
            or "",
            "date": getattr(row, "decision_date", None),
            "source_name": committee_name,
            "case_number": getattr(row, "case_number", None),
        }

    config = IngestConfig(
        name=name,
        data_type_label="위원회결정례",
        source_path=source_path,
        id_field="결정문일련번호",
        summary_field="결정문요약",
        title_field=title_field or case_number_field or "결정문일련번호",
        orm_class=orm_class,
        orm_id_attr="serial_number",
        orm_factory_fn=orm_factory_fn,
        vector_metadata_fn=_vector_metadata_fn,
        fulltext_fn=fulltext_fn,
        fts_metadata_fn=_fts_metadata_fn,
        orm_fulltext_fn=orm_fulltext_fn,
        orm_fts_metadata_fn=_orm_fts_metadata_fn,
    )

    register_config(config)
    return config
