from urllib.parse import urlparse, urlunparse

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    APP_NAME: str = "Law Platform API"
    DEBUG: bool = True

    # Environment: development, docker, production
    ENVIRONMENT: str = "development"

    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/lawdb"

    # Async Database URL (for SQLAlchemy async)
    @property
    def DATABASE_URL_ASYNC(self) -> str:  # noqa: N802
        """Convert to async driver URL, handling various PostgreSQL scheme variants."""
        parsed = urlparse(self.DATABASE_URL)
        # Normalize scheme: postgres, postgresql, postgresql+psycopg2 → postgresql+asyncpg
        if parsed.scheme in ("postgres", "postgresql", "postgresql+psycopg2"):
            parsed = parsed._replace(scheme="postgresql+asyncpg")
        return urlunparse(parsed)

    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]

    # API Keys
    KAKAO_MAP_API_KEY: str = ""
    KAKAO_REST_API_KEY: str = ""
    OPENAI_API_KEY: str = ""

    # Vector DB 선택 (chroma, qdrant)
    VECTOR_DB: str = "chroma"

    # ChromaDB 설정
    CHROMA_PERSIST_DIR: str = "./data/chroma"
    CHROMA_COLLECTION_NAME: str = "legal_documents"

    # Qdrant 설정 (VECTOR_DB=qdrant 일 때 사용)
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""  # Qdrant Cloud 사용 시
    QDRANT_COLLECTION_NAME: str = "legal_documents"

    # Embedding
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_BATCH_SIZE: int = 100

    # Local Embedding (sentence-transformers)
    USE_LOCAL_EMBEDDING: bool = True
    LOCAL_EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # 활성화할 모듈 목록 (빈 리스트면 모든 모듈 활성화)
    ENABLED_MODULES: List[str] = []

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
