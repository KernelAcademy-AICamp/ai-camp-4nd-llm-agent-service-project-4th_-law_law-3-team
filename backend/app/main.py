from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.router import chat_router
from app.core.config import settings
from app.core.registry import ModuleRegistry

# 미디어 디렉토리 경로
MEDIA_DIR = Path(__file__).parent.parent / "data" / "media"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    import logging

    from app.services.rag import check_embedding_model_availability, get_local_model

    logger = logging.getLogger(__name__)

    # 시작 시: 임베딩 모델 캐시 상태 확인
    model_available = check_embedding_model_availability()

    # 로컬 임베딩 사용 시 미리 로드 (Eager Loading)
    if model_available and settings.USE_LOCAL_EMBEDDING:
        logger.info("임베딩 모델을 미리 로드합니다...")
        try:
            get_local_model()
            logger.info("임베딩 모델 로드 완료: %s", settings.LOCAL_EMBEDDING_MODEL)
        except Exception as e:
            logger.error("임베딩 모델 로드 실패: %s", e)

    # 법률 용어 사전 초기화 (USE_LEGAL_TERM_DICT=true 시)
    if settings.USE_LEGAL_TERM_DICT:
        from app.tools.vectorstore.legal_term_dict import init_legal_term_dict_from_db

        try:
            count = await init_legal_term_dict_from_db()
            logger.info("법률 용어 사전 로드 완료: %d개", count)
        except Exception as e:
            logger.error("법률 용어 사전 로드 실패: %s", e)

    yield
    # 종료 시: 정리 작업 (필요 시)


app = FastAPI(
    title=settings.APP_NAME,
    description="법률 서비스 플랫폼 API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모듈 자동 등록
registry = ModuleRegistry(app)
registry.register_all_modules()

# API 라우터 수동 등록 (모듈 시스템과 별도)
app.include_router(chat_router, prefix="/api")

# 미디어 정적 파일 마운트
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")


@app.get("/health")
async def health_check() -> dict[str, object]:
    return {"status": "healthy", "modules": registry.get_registered_modules()}
