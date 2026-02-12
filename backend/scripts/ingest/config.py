"""
인제스트 설정 (config-driven)

각 데이터 타입(판례, 법령 등)의 인제스트 설정을 정의하는 dataclass.
types/ 하위에 데이터 타입별 설정을 등록하여 사용.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Type

from sqlalchemy.orm import DeclarativeBase

# backend/ 디렉토리 (scripts/ingest/config.py → scripts/ → backend/)
BACKEND_ROOT = Path(__file__).parent.parent.parent

# 프로젝트 루트 (backend/ → project root)
PROJECT_ROOT = BACKEND_ROOT.parent

# 기본 데이터 디렉토리
DATA_DIR = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class IngestConfig:
    """
    데이터 타입별 인제스트 설정

    Attributes:
        name: 설정 식별자 (예: "precedent")
        data_type_label: 데이터 유형 라벨 (예: "판례")
        source_path: 기본 JSON 소스 경로
        id_field: JSON 문서 ID 필드명
        summary_field: 벡터 DB용 요약 필드명
        title_field: 제목 필드명
        orm_class: SQLAlchemy ORM 클래스
        orm_id_attr: ORM 클래스의 고유 ID 속성명 (예: "serial_number")
        vector_metadata_fn: JSON item + vector → LanceDB record dict
        fulltext_fn: JSON item → FTS용 원문 텍스트 concat
        fts_metadata_fn: JSON item → fts_index 메타데이터 dict
        orm_fulltext_fn: ORM 인스턴스 → FTS용 원문 텍스트 concat (FTS 재빌드용)
        orm_fts_metadata_fn: ORM 인스턴스 → fts_index 메타데이터 dict (FTS 재빌드용)
    """

    name: str
    data_type_label: str
    source_path: Path
    id_field: str
    summary_field: str
    title_field: str
    orm_class: Type[DeclarativeBase]
    orm_id_attr: str
    vector_metadata_fn: Callable[[dict[str, Any], list[float]], dict[str, Any]]
    fulltext_fn: Callable[[dict[str, Any]], str]
    fts_metadata_fn: Callable[[dict[str, Any]], dict[str, Any]]
    orm_fulltext_fn: Callable[[Any], str]
    orm_fts_metadata_fn: Callable[[Any], dict[str, Any]]


# 등록된 설정을 관리하는 레지스트리
_registry: dict[str, IngestConfig] = {}


def register_config(config: IngestConfig) -> None:
    """인제스트 설정 등록"""
    _registry[config.name] = config


def get_config(name: str) -> IngestConfig:
    """등록된 인제스트 설정 조회"""
    if name not in _registry:
        available = ", ".join(_registry.keys()) or "(없음)"
        raise KeyError(
            f"등록되지 않은 인제스트 타입: '{name}'. 사용 가능: {available}"
        )
    return _registry[name]


def list_configs() -> list[str]:
    """등록된 인제스트 타입 목록"""
    return list(_registry.keys())
