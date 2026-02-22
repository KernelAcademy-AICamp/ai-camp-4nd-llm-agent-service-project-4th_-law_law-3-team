"""
임베딩 서비스

쿼리/문서 텍스트를 벡터로 변환
"""

import asyncio
import logging
import threading
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from openai import OpenAI

from app.core.config import settings
from app.core.errors import EmbeddingModelNotFoundError

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# 모델 캐시 디렉토리
MODEL_CACHE_DIR = Path(__file__).parent.parent.parent.parent / "data" / "models"

# 모델 가용성 상태 (모듈 레벨 캐싱)
_embedding_model_available: Optional[bool] = None
_embedding_model_warning_shown = False

# 쿼리 임베딩 LRU 캐시 (thread-safe)
# 1024차원 float * 1024개 ≈ 4MB (무시할 수 있는 수준)
QUERY_CACHE_MAX_SIZE = 1024
_query_cache: OrderedDict[str, List[float]] = OrderedDict()
_query_cache_lock = threading.Lock()
_query_cache_hits = 0
_query_cache_misses = 0


def _get_model_cache_path(model_name: str) -> Path:
    """모델 캐시 경로 반환 (HuggingFace 캐시 구조)"""
    sanitized = model_name.replace("/", "--")
    return MODEL_CACHE_DIR / f"models--{sanitized}"


def is_embedding_model_cached(model_name: Optional[str] = None) -> bool:
    """
    임베딩 모델이 로컬에 캐시되어 있는지 확인

    Args:
        model_name: 모델명 (기본값: settings.LOCAL_EMBEDDING_MODEL)

    Returns:
        캐시 존재 여부
    """
    model_name = model_name or settings.LOCAL_EMBEDDING_MODEL
    cache_path = _get_model_cache_path(model_name)

    if not cache_path.exists():
        return False

    # blobs 디렉토리에 .incomplete 파일이 있으면 다운로드 미완료
    blobs_dir = cache_path / "blobs"
    if blobs_dir.exists():
        for file in blobs_dir.iterdir():
            if file.name.endswith(".incomplete"):
                return False

    # snapshots 디렉토리에 실제 모델 파일이 있어야 함
    snapshots_dir = cache_path / "snapshots"
    if not snapshots_dir.exists():
        return False

    snapshots = list(snapshots_dir.iterdir())
    return len(snapshots) > 0


def check_embedding_model_availability() -> bool:
    """
    임베딩 모델 사용 가능 여부 확인 (서버 시작 시 호출)

    Returns:
        모델 사용 가능 여부
    """
    global _embedding_model_available, _embedding_model_warning_shown

    if _embedding_model_available is not None:
        return _embedding_model_available

    if not settings.USE_LOCAL_EMBEDDING:
        _embedding_model_available = True
        return True

    _embedding_model_available = is_embedding_model_cached()

    if not _embedding_model_available and not _embedding_model_warning_shown:
        _embedding_model_warning_shown = True
        warning_msg = (
            "\n" + "=" * 60 + "\n"
            "[WARNING] 임베딩 모델이 캐시되지 않았습니다.\n"
            f"모델명: {settings.LOCAL_EMBEDDING_MODEL}\n"
            "검색 API 사용 전 먼저 모델을 다운로드해주세요:\n"
            "  uv run python scripts/download_models.py\n"
            "=" * 60
        )
        print(warning_msg)
        logger.warning("임베딩 모델 미캐시: %s", settings.LOCAL_EMBEDDING_MODEL)

    return _embedding_model_available


@lru_cache(maxsize=1)
def get_local_model() -> "SentenceTransformer":
    """
    sentence-transformers 모델 로드 (캐싱)

    Returns:
        SentenceTransformer 모델 인스턴스

    Raises:
        EmbeddingModelNotFoundError: 모델이 캐시되지 않은 경우
    """
    global _embedding_model_available

    if _embedding_model_available is None:
        _embedding_model_available = is_embedding_model_cached()

    if not _embedding_model_available:
        raise EmbeddingModelNotFoundError(settings.LOCAL_EMBEDDING_MODEL)

    from sentence_transformers import SentenceTransformer

    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    return SentenceTransformer(
        settings.LOCAL_EMBEDDING_MODEL,
        cache_folder=str(MODEL_CACHE_DIR),
        trust_remote_code=True,
        local_files_only=True,
    )


def create_query_embedding(query: str) -> List[float]:
    """
    쿼리 텍스트를 임베딩 벡터로 변환 (LRU 캐시 적용)

    동일 쿼리의 반복 호출 시 캐시된 결과를 즉시 반환한다.
    임베딩 모델이 변경되지 않는 한 동일 쿼리의 결과는 불변이므로 TTL 없음.

    Args:
        query: 검색 쿼리 텍스트

    Returns:
        임베딩 벡터 (float 리스트)

    Raises:
        EmbeddingModelNotFoundError: 로컬 모델 미캐시 시
    """
    global _query_cache_hits, _query_cache_misses

    # 캐시 히트 확인
    with _query_cache_lock:
        if query in _query_cache:
            _query_cache_hits += 1
            _query_cache.move_to_end(query)
            total = _query_cache_hits + _query_cache_misses
            if total % 100 == 0:
                hit_rate = _query_cache_hits / total * 100
                logger.info(
                    "쿼리 캐시 히트율: %.1f%% (%d/%d)",
                    hit_rate, _query_cache_hits, total,
                )
            return list(_query_cache[query])

    # 캐시 미스 — 임베딩 계산
    if settings.USE_LOCAL_EMBEDDING:
        model = get_local_model()
        embedding = model.encode(
            query,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        result = embedding.tolist()
    else:
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        response = client.embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input=query,
        )
        result = response.data[0].embedding

    # 캐시에 저장
    with _query_cache_lock:
        _query_cache_misses += 1
        _query_cache[query] = list(result)
        if len(_query_cache) > QUERY_CACHE_MAX_SIZE:
            _query_cache.popitem(last=False)

    return result


async def create_query_embedding_async(query: str) -> List[float]:
    """
    쿼리 텍스트를 비동기로 임베딩 벡터로 변환

    sync 함수를 별도 스레드에서 실행하여 FastAPI 이벤트 루프 블로킹 방지.

    Args:
        query: 검색 쿼리 텍스트

    Returns:
        임베딩 벡터 (float 리스트)

    Raises:
        EmbeddingModelNotFoundError: 로컬 모델 미캐시 시
    """
    return await asyncio.to_thread(create_query_embedding, query)
