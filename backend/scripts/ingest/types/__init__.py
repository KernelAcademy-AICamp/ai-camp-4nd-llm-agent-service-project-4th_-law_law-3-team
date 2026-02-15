"""
인제스트 타입 자동 등록

types/ 폴더의 .py 파일을 자동 스캔하여 import합니다.
각 모듈은 모듈-레벨에서 register_config()를 호출해야 합니다.

새 타입 추가 시 types/new_type.py 파일만 생성하면 자동 등록됩니다.
(_template.py, __로 시작하는 파일은 제외)
"""

import importlib
import pkgutil
from pathlib import Path

_pkg_dir = str(Path(__file__).parent)

for _finder, _name, _ispkg in pkgutil.iter_modules([_pkg_dir]):
    if _name.startswith("_"):
        continue
    importlib.import_module(f"{__name__}.{_name}")
