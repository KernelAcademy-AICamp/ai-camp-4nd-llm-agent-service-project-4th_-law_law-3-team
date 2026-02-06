from typing import List
from urllib.parse import urlparse, urlunparse

from pydantic_settings import BaseSettings


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

    # LLM 설정
    LLM_PROVIDER: str = "openai"  # openai, anthropic, google

    # OpenAI
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"

    # Anthropic (Claude)
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL: str = "claude-3-5-sonnet-20241022"

    # Google (Gemini)
    GOOGLE_API_KEY: str = ""
    GOOGLE_MODEL: str = "gemini-3-flash-preview"

    # Vector DB 선택 (chroma, qdrant, lancedb)
    VECTOR_DB: str = "lancedb"

    # ChromaDB 설정
    CHROMA_PERSIST_DIR: str = "./data/chroma"
    CHROMA_COLLECTION_NAME: str = "legal_documents"

    # Qdrant 설정 (VECTOR_DB=qdrant 일 때 사용)
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""  # Qdrant Cloud 사용 시
    QDRANT_COLLECTION_NAME: str = "legal_documents"

    # LanceDB 설정 (VECTOR_DB=lancedb 일 때 사용)
    LANCEDB_URI: str = "./lancedb_data"
    LANCEDB_TABLE_NAME: str = "legal_chunks"

    # Embedding
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_BATCH_SIZE: int = 100

    # Local Embedding (sentence-transformers)
    USE_LOCAL_EMBEDDING: bool = True
    LOCAL_EMBEDDING_MODEL: str = "nlpai-lab/KURE-v1"  # 1024차원, LanceDB 데이터와 일치

    # Neo4j Graph DB 설정
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"

    # 변호사 데이터 소스 (True: PostgreSQL, False: JSON 파일)
    USE_DB_LAWYERS: bool = False

    # 법률 용어 사전 (MeCab 토크나이저 법률 복합명사 보강)
    USE_LEGAL_TERM_DICT: bool = False

    # 활성화할 모듈 목록 (빈 리스트면 모든 모듈 활성화)
    ENABLED_MODULES: List[str] = []

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
