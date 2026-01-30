from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.registry import ModuleRegistry

# 미디어 디렉토리 경로
MEDIA_DIR = Path(__file__).parent.parent / "data" / "media"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    # 시작 시: 임베딩 모델 캐시 상태 확인
    from app.common.chat_service import check_embedding_model_availability

    check_embedding_model_availability()

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

# 미디어 정적 파일 마운트
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")


@app.get("/health")
async def health_check() -> dict[str, object]:
    return {"status": "healthy", "modules": registry.get_registered_modules()}
