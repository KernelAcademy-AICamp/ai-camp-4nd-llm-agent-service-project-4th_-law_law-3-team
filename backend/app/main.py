from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.registry import ModuleRegistry

# 미디어 디렉토리 경로
MEDIA_DIR = Path(__file__).parent.parent / "data" / "media"

app = FastAPI(
    title=settings.APP_NAME,
    description="법률 서비스 플랫폼 API",
    version="1.0.0",
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
if MEDIA_DIR.exists():
    app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "modules": registry.get_registered_modules()}
