from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    APP_NAME: str = "Law Platform API"
    DEBUG: bool = True

    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/lawdb"

    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]

    # API Keys
    KAKAO_MAP_API_KEY: str = ""
    OPENAI_API_KEY: str = ""

    # 활성화할 모듈 목록 (빈 리스트면 모든 모듈 활성화)
    ENABLED_MODULES: List[str] = []

    class Config:
        env_file = ".env"


settings = Settings()
