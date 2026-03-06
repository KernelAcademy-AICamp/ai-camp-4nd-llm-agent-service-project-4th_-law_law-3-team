import logging
import os
from pathlib import Path
from typing import List
from urllib.parse import urlparse, urlunparse

from pydantic import model_validator
from pydantic_settings import BaseSettings

_config_logger = logging.getLogger(__name__)

# backend/ 디렉토리 (이 파일 기준: backend/app/core/config.py)
_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
_PROJECT_ROOT = _BACKEND_DIR.parent

# Docker 환경 감지: WORKDIR=/app → PROJECT_ROOT=/ (루트)
# 이 경우 data/는 볼륨 마운트로 /app/data/에 있음
if _PROJECT_ROOT == Path("/"):
    RUNTIME_DATA_DIR = _BACKEND_DIR / "data"
else:
    RUNTIME_DATA_DIR = _PROJECT_ROOT / "data"


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

    # 쿼리 리라이팅 전용 LLM (빈 문자열이면 LLM_PROVIDER/모델 사용)
    QUERY_REWRITE_PROVIDER: str = ""  # openai | google (빈 문자열 → LLM_PROVIDER)
    QUERY_REWRITE_MODEL: str = ""  # 빈 문자열 → 프로바이더 기본 모델

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

    # 변호사 데이터 소스 (True: PostgreSQL, False: JSON 파일)
    USE_DB_LAWYERS: bool = False

    # 영속 체크포인터 사용 여부 (True: PostgreSQL, False: InMemory)
    USE_PERSISTENT_CHECKPOINTER: bool = True

    # LLM 타임아웃 (초)
    LLM_TIMEOUT_SECONDS: int = 60

    # 에이전트 노드 전체 타임아웃 (초)
    AGENT_TIMEOUT_SECONDS: int = 120

    # 법률 용어 사전 (MeCab 토크나이저 법률 복합명사 보강)
    USE_LEGAL_TERM_DICT: bool = True

    # 라우터 선택 (True: LLM 의도 분류, False: 키워드 기반)
    USE_LLM_ROUTER: bool = True

    # 하이브리드 검색 (벡터 + 키워드)
    USE_HYBRID_SEARCH: bool = True

    # BM25 검색 (pg_textsearch). True: BM25 인덱스 사용, False: FTS 비활성화
    USE_BM25_SEARCH: bool = True

    # ML 모델 캐시 디렉토리 (임베딩/리랭커, 상대경로는 backend/ 기준)
    MODEL_CACHE_DIR: str = "data/models"

    # MeCab 사용자 사전 경로 (법률 복합명사 인식)
    MECAB_USERDIC_PATH: str = "data/mecab_userdic/legal_terms.dic"

    # Upstage (Solar)
    UPSTAGE_API_KEY: str = ""
    UPSTAGE_MODEL: str = "solar-pro"

    # Content Marketing (콘텐츠 마케팅 자동화)
    TAVILY_API_KEY: str = ""
    NAVER_CLIENT_ID: str = ""
    NAVER_CLIENT_SECRET: str = ""
    PERPLEXITY_API_KEY: str = ""  # Phase 2
    YOUTUBE_API_KEY: str = ""  # Phase 2
    GOOGLE_CSE_API_KEY: str = ""  # Google Custom Search JSON API 키
    GOOGLE_CSE_ID: str = ""  # Programmable Search Engine ID
    NEWSDATA_API_KEY: str = ""  # NewsData.io API 키
    NEWSAPI_API_KEY: str = ""  # NewsAPI.org API 키
    CONTENT_MARKETING_CACHE_TTL: int = 86400  # 24시간

    # Content Marketing v2.1 — Keyword Flow
    KEYWORD_COLLECT_CACHE_TTL: int = 3600  # 1시간
    KEYWORD_NEWS_CACHE_TTL: int = 1800  # 30분
    KEYWORD_MAX_RESULTS: int = 10
    KEYWORD_NEWS_MAX_RESULTS: int = 10
    KEYWORD_COMMUNITY_DOMAINS: List[str] = [
        "dcinside.com",
        "fmkorea.com",
        "theqoo.net",
        "bobaedream.co.kr",
        "inven.co.kr",
    ]
    KEYWORD_COLLECT_RATE_LIMIT: int = 5  # 시간당
    KEYWORD_NEWS_RATE_LIMIT: int = 30  # 시간당

    # Content Marketing v2.1 — Legal Gate + 5차원 가중합 스코어링
    TREND_LEGAL_THRESHOLD: float = 0.3  # Legal Gate 임계값 (설계 기준 복원)
    TREND_MENTION_WEIGHT: float = 0.25  # 언급 빈도
    TREND_LEGAL_WEIGHT: float = 0.6  # v1.0 하위호환 유지 (v2에서는 _LEGAL_ADDITIVE_WEIGHT 사용)
    TREND_CONTROVERSY_WEIGHT: float = 0.20  # 논란/찬반 대립
    TREND_SPREAD_WEIGHT: float = 0.10  # 확산도 (소스 다양성)
    TREND_FITNESS_WEIGHT: float = 0.25  # 채널 적합도

    # Content Marketing v2.0 — 페르소나
    PERSONA_MIN_HISTORY: int = 30
    PERSONA_CONFIDENCE_THRESHOLD: float = 0.6

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

    # News Pipeline
    NEWS_PIPELINE_ENABLED: bool = True
    NEWS_PIPELINE_SUMMARY_PROVIDER: str = "upstage"
    NEWS_PIPELINE_SUMMARY_MODEL: str = "solar-pro2"
    NEWS_PIPELINE_MIN_ARTICLE_LENGTH: int = 500
    NEWS_PIPELINE_CHUNK_SIZE: int = 1500
    NEWS_PIPELINE_CHUNK_OVERLAP: int = 200
    NEWS_PIPELINE_MAX_ARTICLES_PER_RUN: int = 200
    NEWS_PIPELINE_RATE_LIMIT_CRAWL: float = 2.0
    NEWS_PIPELINE_LAWTIMES_ENABLED: bool = True
    NEWS_PIPELINE_NAVER_ENABLED: bool = True
    NEWS_PIPELINE_LANCEDB_TABLE: str = "news_chunks"
    NEWS_PIPELINE_RETENTION_DAYS: int = 90
    NEWS_PIPELINE_LLM_FALLBACK_PROVIDER: str = "openai"
    NEWS_PIPELINE_SLACK_WEBHOOK_URL: str = ""
    NEWS_PIPELINE_OTEL_ENDPOINT: str = ""
    NEWS_PIPELINE_LINEAGE_URL: str = ""
    NEWS_PIPELINE_REDIS_URL: str = ""
    NEWS_PIPELINE_SIMHASH_THRESHOLD: int = 3

    # Webtoon Storyboard (콘텐츠 마케팅)
    STORYBOARD_IMAGE_MODEL: str = "gemini-3-pro-image-preview"
    STORYBOARD_MAX_PANELS: int = 14
    STORYBOARD_IMAGE_FORMAT: str = "webp"
    STORYBOARD_CACHE_TTL: int = 604800  # 7일 (초)
    STORYBOARD_MAX_CONCURRENT: int = 3

    # Rate Limiting (분당 요청 수)
    RATE_LIMIT_PER_MINUTE: int = 30
    RATE_LIMIT_AI_PER_MINUTE: int = 10
    RATE_LIMIT_STORAGE_URI: str = "memory://"  # 프로덕션: "redis://localhost:6379"

    @model_validator(mode="after")
    def _validate_production_settings(self) -> "Settings":
        """프로덕션 환경에서 필수 설정이 누락되면 시작 시 에러 발생"""
        if self.ENVIRONMENT == "production":
            missing: list[str] = []
            if not self.DATABASE_URL:
                missing.append("DATABASE_URL")
            if not self.API_KEY:
                missing.append("API_KEY")
            if missing:
                raise ValueError(
                    f"프로덕션 환경에서 필수 설정이 누락되었습니다: {', '.join(missing)}"
                )
            if self.DEBUG:
                _config_logger.warning(
                    "프로덕션 환경에서 DEBUG=True가 설정되어 있습니다. 보안상 권장하지 않습니다."
                )
            if self.RATE_LIMIT_STORAGE_URI == "memory://":
                _config_logger.warning(
                    "프로덕션 환경에서 RATE_LIMIT_STORAGE_URI=memory:// 사용 중입니다. "
                    "다중 인스턴스 배포 시 Redis 등 외부 저장소를 권장합니다."
                )
        return self

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
