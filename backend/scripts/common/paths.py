"""경로 상수 및 sys.path 설정 유틸리티.

모든 스크립트에서 사용하는 경로 상수와 sys.path 설정을 통합합니다.

Usage:
    from scripts.common.paths import BACKEND_DIR, PROJECT_ROOT, DATA_DIR, setup_sys_path
    setup_sys_path()  # backend/ 루트를 sys.path에 추가
"""

from __future__ import annotations

import sys
from pathlib import Path

# 이 파일 기준: backend/scripts/common/paths.py
_THIS_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = _THIS_DIR.parent  # backend/scripts/
BACKEND_DIR = SCRIPTS_DIR.parent  # backend/
PROJECT_ROOT = BACKEND_DIR.parent  # law-3-team/
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = BACKEND_DIR / "eda_output"


def setup_sys_path() -> None:
    """backend/ 루트를 sys.path에 추가 (app 모듈 import 가능하게)."""
    backend_str = str(BACKEND_DIR)
    if backend_str not in sys.path:
        sys.path.insert(0, backend_str)
