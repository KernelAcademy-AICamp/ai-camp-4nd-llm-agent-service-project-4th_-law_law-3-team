import os
from typing import List
from urllib.parse import urlparse, urlunparse

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "Law Platform API"
    DEBUG: bool = False

    # Environment: development, docker, production
    ENVIRONMENT: str = "development"

    # Database
    DATABASE_URL: str = ""

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
    LANCEDB_INDEX_TYPE: str = "IVF_FLAT"  # 빈 문자열이면 brute-force, "IVF_FLAT" 등 설정 가능
    LANCEDB_NPROBES: int = 30  # IVF 인덱스 검색 시 탐색할 파티션 수 (높을수록 정확, 느림)
    LANCEDB_MODE: str = "local"  # "local" | "remote"
    LANCEDB_SERVICE_URL: str = "http://localhost:8100"  # remote 모드 시 마이크로서비스 URL
    LANCEDB_SERVICE_TIMEOUT: float = 30.0  # HTTP timeout (초)

    # Embedding
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_BATCH_SIZE: int = 100

    # Local Embedding (sentence-transformers)
    USE_LOCAL_EMBEDDING: bool = True
    LOCAL_EMBEDDING_MODEL: str = "nlpai-lab/KURE-v1"  # 1024차원, LanceDB 데이터와 일치

    # Neo4j Graph DB 설정
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = ""

    # 변호사 데이터 소스 (True: PostgreSQL, False: JSON 파일)
    USE_DB_LAWYERS: bool = False

    # 영속 체크포인터 사용 여부 (True: PostgreSQL, False: InMemory)
    USE_PERSISTENT_CHECKPOINTER: bool = True

    # LLM 타임아웃 (초)
    LLM_TIMEOUT_SECONDS: int = 60

    # 에이전트 노드 전체 타임아웃 (초)
    AGENT_TIMEOUT_SECONDS: int = 120

    # 법률 용어 사전 (MeCab 토크나이저 법률 복합명사 보강)
    USE_LEGAL_TERM_DICT: bool = False

    # 하이브리드 검색 (벡터 + 키워드)
    USE_HYBRID_SEARCH: bool = True

    # MeCab 사용자 사전 경로 (법률 복합명사 인식)
    MECAB_USERDIC_PATH: str = "data/mecab_userdic/legal_terms.dic"

    # LangSmith 트레이싱
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_PROJECT: str = "law-platform"
    LANGCHAIN_API_KEY: str = ""

    # ONNX 임베딩 최적화
    USE_ONNX_EMBEDDING: bool = False
    ONNX_EMBEDDING_VARIANT: str = "ort-opt"  # ort-opt (FP32 무손실) | ort-opt-qdq (INT8, cosine 0.999) | onnx-fp16 (FP16, cosine 1.0)
    USE_ONNX_RERANKER: bool = False
    ONNX_RERANKER_VARIANT: str = "ort-opt"  # ort-opt | ort-opt-qdq
    ONNX_QUALITY_GATE_ENABLED: bool = True
    ONNX_QUALITY_GATE_FALLBACK: bool = True
    ONNX_INTRA_OP_THREADS: int = 0  # 0 = 자동 (P코어 감지)
    ONNX_ENABLE_IO_BINDING: bool = False  # CPU EP에서 무효 (GPU EP에서만 유의미)
    ONNX_INFERENCE_TIMEOUT_SECONDS: float = 30.0  # 추론 타임아웃 (초)
    ONNX_ENABLE_BF16_FASTMATH: bool = False  # Graviton3+ 전용 (Mac ARM 미지원)
    ONNX_QDQ_SENSITIVE_LAYERS: str = ""  # 쉼표 구분 민감 레이어 인덱스 (예: "0,1,22,23"), 빈 문자열이면 기본값 사용

    # Context 압축 (LLMLingua-2)
    ENABLE_CONTEXT_COMPRESSION: bool = False
    COMPRESSION_MODEL: str = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
    COMPRESSION_DEFAULT_RATE: float = 0.5
    COMPRESSION_MIN_LENGTH: int = 500
    COMPRESSION_DEVICE: str = "cpu"  # "cpu" | "cuda"

    # 활성화할 모듈 목록 (빈 리스트면 모든 모듈 활성화)
    ENABLED_MODULES: List[str] = []

    # API 인증 (빈 문자열이면 인증 비활성화)
    API_KEY: str = ""

    # Rate Limiting (분당 요청 수)
    RATE_LIMIT_PER_MINUTE: int = 30
    RATE_LIMIT_AI_PER_MINUTE: int = 10
    RATE_LIMIT_STORAGE_URI: str = "memory://"  # 프로덕션: "redis://localhost:6379"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

# LangSmith/LangChain SDK는 os.environ에서 직접 읽으므로 설정값을 내보냄
if settings.LANGCHAIN_TRACING_V2:
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", settings.LANGCHAIN_PROJECT)
    if settings.LANGCHAIN_API_KEY:
        os.environ.setdefault("LANGCHAIN_API_KEY", settings.LANGCHAIN_API_KEY)
