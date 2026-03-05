"""
RAG 테이블 레지스트리

PostgreSQL 원문 테이블 매핑 및 SQL 식별자 검증 유틸리티.
data_type(한국어) ↔ PostgreSQL 테이블 이름 + 컬럼 설정 관리.
"""

from __future__ import annotations

import re
from typing import NamedTuple

__all__ = [
    "TableConfig",
    "DOCUMENT_TABLE_REGISTRY",
    "validate_identifier",
    "resolve_data_type",
    "to_doc_type",
    "group_dec_source_ids",
]

_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def validate_identifier(name: str) -> str:
    """SQL 식별자(테이블명/컬럼명)가 안전한 형식인지 검증."""
    if not _IDENTIFIER_PATTERN.match(name):
        raise ValueError(f"안전하지 않은 SQL 식별자: {name!r}")
    return name


# 하위 호환용 내부 별칭 (retrieval.py에서 _validate_identifier로 참조)
_validate_identifier = validate_identifier


class TableConfig(NamedTuple):
    """PostgreSQL 원문 테이블 조회 설정."""

    table_name: str
    id_column: str
    content_columns: tuple[str, ...]


# fts_index.data_type / LanceDB data_type (한국어) → PostgreSQL 테이블 매핑
DOCUMENT_TABLE_REGISTRY: dict[str, list[TableConfig]] = {
    "판례": [
        TableConfig("precedent_documents", "serial_number", ("ruling", "reasoning")),
    ],
    "법령": [
        TableConfig("law_documents", "law_id", ("content",)),
    ],
    "행정규칙": [
        TableConfig("admin_rule_documents", "serial_number", ("content",)),
    ],
    "부처유권해석": [
        TableConfig(
            "interpretation_ministry_documents",
            "serial_number",
            ("answer", "reason"),
        ),
    ],
    "헌재결정례": [
        TableConfig(
            "constitutional_documents", "serial_number", ("ruling", "reasoning")
        ),
    ],
    "행정심판례": [
        TableConfig("administration_documents", "serial_number", ("ruling", "reason")),
    ],
    "법령해석례": [
        TableConfig("legislation_documents", "serial_number", ("answer", "reason")),
    ],
    "조약": [
        TableConfig("treaty_documents", "serial_number", ("content",)),
    ],
    "특별행정심판": [
        TableConfig(
            "special_admin_appeal_documents", "serial_number", ("ruling", "reason")
        ),
    ],
    "위원회결정례": [
        TableConfig(
            "dec_labor_documents",
            "serial_number",
            ("judgment_summary", "judgment_result"),
        ),
        TableConfig(
            "dec_human_rights_documents",
            "serial_number",
            ("ruling", "judgment_summary"),
        ),
        TableConfig("dec_privacy_documents", "serial_number", ("reason",)),
        TableConfig(
            "dec_employment_documents", "serial_number", ("ruling", "reason")
        ),
        TableConfig(
            "dec_financial_documents",
            "serial_number",
            ("action_reason", "action_content"),
        ),
        TableConfig(
            "dec_industrial_documents", "serial_number", ("ruling", "reason")
        ),
        TableConfig(
            "dec_environment_documents",
            "serial_number",
            ("ruling", "evaluation_opinion"),
        ),
        TableConfig(
            "dec_securities_documents",
            "serial_number",
            ("action_reason", "action_content"),
        ),
        TableConfig(
            "dec_civil_rights_documents", "serial_number", ("ruling", "reason")
        ),
        TableConfig(
            "dec_fair_trade_documents", "serial_number", ("ruling", "reason")
        ),
        TableConfig("dec_media_documents", "serial_number", ("ruling",)),
    ],
    "자치법규": [
        TableConfig(
            "local_ordinance_documents", "ordinance_id", ("content", "overall_summary")
        ),
    ],
}

# doc_type (API, 영어) ↔ data_type (DB, 한국어) 변환
_DOC_TYPE_TO_DATA_TYPE: dict[str, str] = {"precedent": "판례", "law": "법령"}
_DATA_TYPE_TO_DOC_TYPE: dict[str, str] = {"판례": "precedent", "법령": "law"}

# 위원회결정례 접두사 → TableConfig 직접 라우팅 (11테이블 순차 스캔 회피)
# FTS source_id "dec_labor:12345" → prefix "dec_labor" → dec_labor_documents 테이블
_DEC_TABLE_BY_PREFIX: dict[str, TableConfig] = {
    tc.table_name.removesuffix("_documents"): tc
    for tc in DOCUMENT_TABLE_REGISTRY.get("위원회결정례", [])
}


def resolve_data_type(doc_type_or_data_type: str) -> str:
    """doc_type(영어) 또는 data_type(한국어) → 항상 한국어 data_type."""
    return _DOC_TYPE_TO_DATA_TYPE.get(doc_type_or_data_type, doc_type_or_data_type)


def to_doc_type(data_type: str) -> str:
    """data_type(한국어) → doc_type. 매핑 없으면 원본 반환."""
    return _DATA_TYPE_TO_DOC_TYPE.get(data_type, data_type)


def group_dec_source_ids(
    source_ids: list[str],
) -> tuple[dict[TableConfig, dict[str, str]], list[str]]:
    """위원회결정례 source_id를 접두사 기반으로 테이블별 그룹핑.

    FTS source_id: "dec_labor:12345" → dec_labor_documents, serial "12345"
    벡터 source_id: "12345" → 접두사 없음, 기존 순차 스캔 필요

    Returns:
        (routed: {TableConfig: {original_sid: serial_number}}, unrouted: [sid])
    """
    routed: dict[TableConfig, dict[str, str]] = {}
    unrouted: list[str] = []

    for sid in source_ids:
        if ":" in sid:
            prefix, serial = sid.split(":", 1)
            tc = _DEC_TABLE_BY_PREFIX.get(prefix)
            if tc:
                routed.setdefault(tc, {})[sid] = serial
            else:
                unrouted.append(sid)
        else:
            unrouted.append(sid)

    return routed, unrouted
