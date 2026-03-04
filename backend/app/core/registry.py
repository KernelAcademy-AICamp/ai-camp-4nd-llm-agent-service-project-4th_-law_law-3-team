import importlib
import logging
from pathlib import Path
from typing import List

from fastapi import FastAPI

from app.core.config import settings

logger = logging.getLogger(__name__)


class ModuleRegistry:
    """모듈을 동적으로 등록/해제하는 레지스트리"""

    MODULES_PATH = "app.modules"
    MODULES_DIR = Path(__file__).resolve().parent.parent / "modules"

    def __init__(self, app: FastAPI):
        self.app = app
        self._registered_modules: List[str] = []

    def _get_available_modules(self) -> List[str]:
        """modules 폴더에서 사용 가능한 모듈 목록 반환

        Note: router/__init__.py가 존재하는 모듈만 반환
              (__pycache__만 남은 빈 폴더 무시)
        """
        modules = []
        for item in self.MODULES_DIR.iterdir():
            router_init = item / "router" / "__init__.py"
            if (
                item.is_dir()
                and not item.name.startswith("_")
                and router_init.is_file()
            ):
                modules.append(item.name)
        return modules

    def _is_module_enabled(self, module_name: str) -> bool:
        """모듈이 활성화되어 있는지 확인"""
        if not settings.ENABLED_MODULES:
            return True  # 빈 리스트면 모든 모듈 활성화
        return module_name in settings.ENABLED_MODULES

    def register_module(self, module_name: str) -> bool:
        """단일 모듈 등록"""
        if not self._is_module_enabled(module_name):
            return False

        try:
            # 모듈의 router 패키지에서 router 가져오기
            router_module = importlib.import_module(
                f"{self.MODULES_PATH}.{module_name}.router"
            )

            if hasattr(router_module, "router"):
                self.app.include_router(
                    router_module.router,
                    prefix=f"/api/{module_name.replace('_', '-')}",
                    tags=[module_name],
                )
                self._registered_modules.append(module_name)
                logger.info("Module '%s' registered successfully", module_name)
                return True
        except ImportError as e:
            logger.error("Failed to import module '%s': %s", module_name, e)
            if settings.ENVIRONMENT == "production":
                raise RuntimeError(f"프로덕션 환경에서 모듈 '{module_name}' 임포트 실패: {e}")
        except Exception as e:
            logger.error("Error registering module '%s': %s", module_name, e)
            if settings.ENVIRONMENT == "production":
                raise RuntimeError(f"프로덕션 환경에서 모듈 '{module_name}' 등록 실패: {e}")

        return False

    def register_all_modules(self) -> None:
        """모든 사용 가능한 모듈 등록"""
        available_modules = self._get_available_modules()
        for module_name in available_modules:
            self.register_module(module_name)

    def get_registered_modules(self) -> List[str]:
        """등록된 모듈 목록 반환"""
        return self._registered_modules
