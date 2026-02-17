"""
MD5 기반 임베딩 캐시

동일 텍스트의 재임베딩을 방지합니다.
"""

import hashlib
import json
from pathlib import Path
from typing import Callable, Optional

# 기대 임베딩 차원 (KURE-v1: 1024)
EXPECTED_EMBEDDING_DIM = 1024


class EmbeddingCache:
    """MD5 해시 기반 디스크 + 메모리 임베딩 캐시"""

    def __init__(
        self,
        cache_dir: str = "./embedding_cache",
        expected_dim: int = EXPECTED_EMBEDDING_DIM,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: dict[str, list[float]] = {}
        self._expected_dim = expected_dim
        self._hits = 0
        self._misses = 0
        self._dim_mismatch = 0

    def _hash(self, text: str) -> str:
        """텍스트의 MD5 해시"""
        return hashlib.md5(text.encode()).hexdigest()

    def _get_path(self, text_hash: str) -> Path:
        """캐시 파일 경로 (2-level 디렉토리 구조)"""
        subdir = self.cache_dir / text_hash[:2]
        subdir.mkdir(exist_ok=True)
        return subdir / f"{text_hash}.json"

    def get(self, text: str) -> Optional[list[float]]:
        """캐시에서 임베딩 조회 (차원 검증 포함)"""
        text_hash = self._hash(text)

        # 메모리 캐시 확인
        if text_hash in self._memory_cache:
            cached = self._memory_cache[text_hash]
            if len(cached) == self._expected_dim:
                self._hits += 1
                return cached
            else:
                # 차원 불일치 - 메모리에서 제거
                del self._memory_cache[text_hash]
                self._dim_mismatch += 1

        # 디스크 캐시 확인
        cache_path = self._get_path(text_hash)
        if cache_path.exists():
            try:
                embedding = json.loads(cache_path.read_text())
                # 차원 검증
                if len(embedding) != self._expected_dim:
                    # 차원 불일치 - 디스크에서 삭제
                    cache_path.unlink(missing_ok=True)
                    self._dim_mismatch += 1
                else:
                    self._memory_cache[text_hash] = embedding
                    self._hits += 1
                    return embedding
            except (json.JSONDecodeError, OSError):
                pass

        self._misses += 1
        return None

    def set(self, text: str, embedding: list[float]) -> None:
        """임베딩을 캐시에 저장"""
        text_hash = self._hash(text)
        self._memory_cache[text_hash] = embedding

        cache_path = self._get_path(text_hash)
        try:
            cache_path.write_text(json.dumps(embedding))
        except OSError:
            pass

    def get_or_compute(
        self,
        text: str,
        compute_fn: Callable[[list[str]], list[list[float]]],
    ) -> list[float]:
        """캐시에서 조회 후, 없으면 계산하여 저장"""
        cached = self.get(text)
        if cached is not None:
            return cached

        embeddings = compute_fn([text])
        embedding = embeddings[0]
        self.set(text, embedding)
        return embedding

    def get_stats(self) -> dict[str, str | int]:
        """캐시 통계"""
        total = self._hits + self._misses
        hit_rate = f"{self._hits / total * 100:.1f}%" if total > 0 else "0.0%"
        stats: dict[str, str | int] = {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "memory_cache_size": len(self._memory_cache),
        }
        if self._dim_mismatch > 0:
            stats["dim_mismatch_purged"] = self._dim_mismatch
        return stats

    def clear_memory_cache(self) -> None:
        """메모리 캐시만 정리"""
        self._memory_cache.clear()

    def clear_all(self) -> None:
        """전체 캐시 정리 (디스크 포함)"""
        self._memory_cache.clear()
        import shutil

        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
