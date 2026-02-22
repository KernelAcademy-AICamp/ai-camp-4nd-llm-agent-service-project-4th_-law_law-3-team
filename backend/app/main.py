from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi.errors import RateLimitExceeded

from app.api.router import chat_router
from app.core.auth import verify_api_key
from app.core.config import settings
from app.core.rate_limit import limiter
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

    # 로컬 임베딩 사용 시 미리 로드 + JIT warm-up
    if model_available and settings.USE_LOCAL_EMBEDDING:
        logger.info("임베딩 모델을 미리 로드합니다...")
        try:
            model = get_local_model()
            model.encode("warm-up", normalize_embeddings=True)
            logger.info("임베딩 모델 로드 + warm-up 완료: %s", settings.LOCAL_EMBEDDING_MODEL)
        except Exception as e:
            logger.error("임베딩 모델 로드 실패: %s", e)

    # 리랭커 모델 미리 로드 + JIT warm-up
    from app.services.rag.rerank import _load_reranker_model

    logger.info("리랭커 모델을 미리 로드합니다...")
    try:
        reranker = _load_reranker_model()
        if reranker is not None:
            reranker.predict([("warm-up", "warm-up")])
            logger.info("리랭커 모델 로드 + warm-up 완료")
        else:
            logger.warning("리랭커 모델 로드 실패 → 리랭킹 비활성화")
    except Exception as e:
        logger.error("리랭커 모델 로드 실패: %s", e)

    # ONNX 세션 로드 + 품질 게이트 + warmup
    if settings.USE_ONNX_EMBEDDING or settings.USE_ONNX_RERANKER:
        from app.services.rag.onnx_session import (
            load_embedding_session,
            load_reranker_session,
            warmup_embedding,
            warmup_reranker,
        )

        if settings.USE_ONNX_EMBEDDING:
            logger.info("ONNX 임베딩 세션을 로드합니다...")
            if load_embedding_session():
                warmup_embedding()
            else:
                logger.warning("ONNX 임베딩 로드 실패 → PyTorch 유지")
                settings.USE_ONNX_EMBEDDING = False

        if settings.USE_ONNX_RERANKER:
            logger.info("ONNX 리랭커 세션을 로드합니다...")
            if load_reranker_session():
                warmup_reranker()
            else:
                logger.warning("ONNX 리랭커 로드 실패 → PyTorch 유지")
                settings.USE_ONNX_RERANKER = False

        # 품질 게이트 (ONNX가 로드된 경우에만)
        if settings.USE_ONNX_EMBEDDING or settings.USE_ONNX_RERANKER:
            from app.services.rag.onnx_quality_gate import run_quality_gate

            logger.info("ONNX 품질 게이트를 실행합니다...")
            gate_results = run_quality_gate()
            for name, result in gate_results.items():
                status = "PASS" if result.passed else "FAIL"
                logger.info(
                    "품질 게이트 [%s] %s: %s=%.6f (기준 %.4f) [%.0fms]",
                    name, status, result.metric_name,
                    result.metric_value, result.threshold, result.elapsed_ms,
                )

    # 벡터 인덱스 생성 (LANCEDB_INDEX_TYPE이 설정된 경우에만)
    if settings.LANCEDB_INDEX_TYPE:
        try:
            if settings.LANCEDB_MODE == "remote":
                from app.tools.vectorstore import get_vector_store

                store = get_vector_store()
                created = store.create_vector_index(settings.LANCEDB_INDEX_TYPE)  # type: ignore[attr-defined]
            else:
                from app.tools.vectorstore.lancedb import LanceDBStore

                lance_store = LanceDBStore()
                created = lance_store.create_vector_index(settings.LANCEDB_INDEX_TYPE)
            if created:
                logger.info("LanceDB 벡터 인덱스 생성 완료: %s", settings.LANCEDB_INDEX_TYPE)
        except Exception as e:
            logger.error("LanceDB 벡터 인덱스 생성 실패 (검색은 brute-force로 동작): %s", e)

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
    dependencies=[Depends(verify_api_key)],
)

# Rate Limiting
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"detail": "요청이 너무 많습니다. 잠시 후 다시 시도해주세요."},
    )


# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
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
