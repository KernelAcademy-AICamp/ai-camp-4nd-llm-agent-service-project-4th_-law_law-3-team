"""scripts/common — 스크립트 공통 유틸리티 모듈.

경로, 로깅, JSON 로딩, DB 세션, 배치 처리, 인용 추출 등
전처리/EDA/로드 스크립트에서 공통으로 사용하는 기능을 제공합니다.
"""

from scripts.common.batch import batch_iterate, process_in_batches
from scripts.common.citation import (
    extract_case_numbers,
    extract_citations,
    extract_law_names,
    extract_statute_names_plain,
)
from scripts.common.db import create_sync_engine, create_sync_session_factory
from scripts.common.json_loader import (
    get_file_size_mb,
    load_items,
    load_json_directory,
    load_json_file,
    resolve_source_path,
    should_stream,
    smart_load,
    stream_json,
)
from scripts.common.logging_config import setup_logging
from scripts.common.paths import (
    BACKEND_DIR,
    DATA_DIR,
    OUTPUT_DIR,
    PROJECT_ROOT,
    SCRIPTS_DIR,
    setup_sys_path,
)

__all__ = [
    # paths
    "BACKEND_DIR",
    "DATA_DIR",
    "OUTPUT_DIR",
    "PROJECT_ROOT",
    "SCRIPTS_DIR",
    "setup_sys_path",
    # logging
    "setup_logging",
    # json_loader
    "get_file_size_mb",
    "load_items",
    "load_json_directory",
    "load_json_file",
    "resolve_source_path",
    "should_stream",
    "smart_load",
    "stream_json",
    # db
    "create_sync_engine",
    "create_sync_session_factory",
    # batch
    "batch_iterate",
    "process_in_batches",
    # citation
    "extract_case_numbers",
    "extract_citations",
    "extract_law_names",
    "extract_statute_names_plain",
]
