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

    # 시작 시: LangGraph 체크포인터 초기화
    if settings.USE_PERSISTENT_CHECKPOINTER:
        from app.multi_agent.graph import init_checkpointer

        logger.info("PostgreSQL 체크포인터를 초기화합니다...")
        await init_checkpointer(settings.DATABASE_URL)

    # MeCab userdic 필수 검증 (실제 로드는 각 토크나이저 초기화 시)
    userdic_path = Path(settings.MECAB_USERDIC_PATH)
    decomp_path = userdic_path.parent / "decomposition_map.json"
    if userdic_path.exists() and decomp_path.exists():
        import json

        with open(decomp_path, encoding="utf-8") as f:
            decomp_count = len(json.load(f))
        logger.info(
            "MeCab userdic 확인: %s (%d 분해맵)",
            userdic_path, decomp_count,
        )
    else:
        missing = []
        if not userdic_path.exists():
            missing.append(f"dic: {userdic_path}")
        if not decomp_path.exists():
            missing.append(f"map: {decomp_path}")
        logger.error(
            "MeCab userdic 필수 파일 없음: %s\n"
            "  빌드 명령: uv run python scripts/build_mecab_userdic.py --from-json",
            ", ".join(missing),
        )

    yield

    # 종료 시: 체크포인터 정리
    if settings.USE_PERSISTENT_CHECKPOINTER:
        from app.multi_agent.graph import shutdown_checkpointer

        await shutdown_checkpointer()


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
