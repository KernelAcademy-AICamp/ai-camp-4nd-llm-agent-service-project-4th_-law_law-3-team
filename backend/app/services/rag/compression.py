"""
RAG Context 압축 모듈 (LLMLingua-2 기반)

리랭킹 후 원문 조회된 content_fields를 컬럼별 정책에 따라 압축.
DOCUMENT_TABLE_REGISTRY의 컬럼 구조와 1:1 매핑하여
결론 컬럼은 보존, 근거 컬럼은 압축하는 차등 전략을 적용한다.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MeCab 동적 force_tokens 추출
# ---------------------------------------------------------------------------


def _extract_legal_nouns(text: str) -> list[str]:
    """MeCab userdic으로 텍스트에서 법률 복합명사를 추출.

    압축 시 force_tokens에 동적으로 추가하여
    SentencePiece 서브워드 분할로 인한 법률 용어 탈락을 방지한다.

    Args:
        text: 분석할 원문

    Returns:
        2자 이상 명사 리스트.

    Raises:
        RuntimeError: MeCab 미설치 또는 userdic 미빌드 시.
    """
    from app.tools.vectorstore.lancedb import _get_thread_tokenizer

    tokenizer = _get_thread_tokenizer()
    return tokenizer.morphs(text)


# ---------------------------------------------------------------------------
# 압축 시 보존할 토큰
# ---------------------------------------------------------------------------

FORCE_TOKENS: list[str] = [
    "\n", ".", ",", "?",
    "제", "조", "항", "호",
    "원고", "피고", "법원", "대법원",
    "기각", "인용", "각하", "취소",
    "손해배상", "위법", "위반",
    "민법", "형법", "상법", "민사소송법", "형사소송법",
]


# ---------------------------------------------------------------------------
# 컬럼별 압축 정책
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnCompressionPolicy:
    """컬럼별 압축 정책.

    Attributes:
        rate: 압축률 (0.0~1.0). 1.0이면 압축 안 함.
        min_length: 이 길이 미만이면 압축 스킵.
        force_tokens: 컬럼 전용 추가 보존 토큰.
    """

    rate: float = 0.4
    min_length: int = 500
    force_tokens: tuple[str, ...] = ()


# DOCUMENT_TABLE_REGISTRY 컬럼 → 압축 정책 매핑
COLUMN_COMPRESSION_POLICIES: dict[str, ColumnCompressionPolicy] = {
    # 결론/결과 → 압축 안 함
    "ruling": ColumnCompressionPolicy(rate=1.0),
    "answer": ColumnCompressionPolicy(rate=1.0),
    "judgment_summary": ColumnCompressionPolicy(rate=1.0),
    "judgment_result": ColumnCompressionPolicy(rate=1.0),
    "action_content": ColumnCompressionPolicy(rate=1.0),
    "overall_summary": ColumnCompressionPolicy(rate=1.0),
    # 근거/이유 → 적극 압축
    "reasoning": ColumnCompressionPolicy(rate=0.4),
    "reason": ColumnCompressionPolicy(rate=0.4),
    "evaluation_opinion": ColumnCompressionPolicy(rate=0.5),
    # 혼합 → 중간 압축
    "content": ColumnCompressionPolicy(rate=0.5),
    "action_reason": ColumnCompressionPolicy(rate=0.5),
}


def _default_policy() -> ColumnCompressionPolicy:
    """미등록 컬럼의 기본 압축 정책."""
    return ColumnCompressionPolicy(
        rate=settings.COMPRESSION_DEFAULT_RATE,
        min_length=settings.COMPRESSION_MIN_LENGTH,
    )


# ---------------------------------------------------------------------------
# ContextCompressor
# ---------------------------------------------------------------------------


class ContextCompressor:
    """LLMLingua-2 기반 컨텍스트 압축기.

    lazy loading으로 모델을 초기화하며,
    content_fields dict의 각 value를 컬럼별 정책에 따라 독립 압축한다.
    """

    def __init__(self) -> None:
        self._compressor: Any = None

    def _get_compressor(self) -> Any:
        """PromptCompressor lazy initialization."""
        if self._compressor is None:
            from llmlingua import PromptCompressor

            logger.info(
                "LLMLingua-2 모델 로딩: %s (device=%s)",
                settings.COMPRESSION_MODEL,
                settings.COMPRESSION_DEVICE,
            )
            load_start = time.monotonic()
            self._compressor = PromptCompressor(
                model_name=settings.COMPRESSION_MODEL,
                use_llmlingua2=True,
                device_map=settings.COMPRESSION_DEVICE,
            )
            logger.info(
                "LLMLingua-2 모델 로딩 완료 (%.1fs)",
                time.monotonic() - load_start,
            )
        return self._compressor

    def compress_field(
        self,
        text: str,
        policy: ColumnCompressionPolicy,
    ) -> str:
        """단일 텍스트를 정책에 따라 압축.

        Args:
            text: 압축할 원문
            policy: 컬럼별 압축 정책

        Returns:
            압축된 텍스트. 압축 스킵 시 원문 반환.
        """
        if policy.rate >= 1.0 or len(text) < policy.min_length:
            return text

        compressor = self._get_compressor()
        dynamic_tokens = _extract_legal_nouns(text)

        # 중복 제거 (순서 유지)
        seen: set[str] = set()
        all_force_tokens: list[str] = []
        for token in FORCE_TOKENS + list(policy.force_tokens) + dynamic_tokens:
            if token not in seen:
                seen.add(token)
                all_force_tokens.append(token)

        # LLMLingua-2 내부 상한 (added_tokens 100개)
        all_force_tokens = all_force_tokens[:100]

        result = compressor.compress_prompt(
            [text],
            rate=policy.rate,
            force_tokens=all_force_tokens,
        )
        return str(result.get("compressed_prompt", text))

    def compress_document_fields(
        self,
        content_fields: dict[str, str],
    ) -> dict[str, str]:
        """content_fields를 컬럼별 정책에 따라 압축.

        Args:
            content_fields: {column_name: text} 매핑

        Returns:
            {column_name: compressed_text} 매핑
        """
        default = _default_policy()
        compressed: dict[str, str] = {}

        for col_name, text in content_fields.items():
            policy = COLUMN_COMPRESSION_POLICIES.get(col_name, default)
            compressed[col_name] = self.compress_field(text, policy)

        return compressed

    def compress_documents(
        self,
        documents: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """문서 리스트의 content_fields를 in-place 압축.

        Args:
            documents: 파이프라인 검색 결과 문서 리스트

        Returns:
            압축 메트릭 dict (before_chars, after_chars, compression_ratio, time_ms)
        """
        start = time.monotonic()
        total_before = 0
        total_after = 0

        for doc in documents:
            fields = doc.get("content_fields")
            if not fields:
                continue

            before_len = sum(len(v) for v in fields.values())
            total_before += before_len

            compressed = self.compress_document_fields(fields)
            doc["content_fields"] = compressed
            doc["content"] = "\n\n".join(compressed.values())

            after_len = sum(len(v) for v in compressed.values())
            total_after += after_len

        elapsed_ms = (time.monotonic() - start) * 1000
        ratio = total_after / total_before if total_before > 0 else 1.0

        logger.info(
            "Context 압축 완료: %d자 → %d자 (%.0f%%, %.0fms, %d건)",
            total_before,
            total_after,
            ratio * 100,
            elapsed_ms,
            len(documents),
        )

        return {
            "before_chars": total_before,
            "after_chars": total_after,
            "compression_ratio": round(ratio, 3),
            "time_ms": round(elapsed_ms, 1),
        }


# ---------------------------------------------------------------------------
# 싱글턴
# ---------------------------------------------------------------------------

_compressor_instance: ContextCompressor | None = None


def get_context_compressor() -> ContextCompressor:
    """ContextCompressor 싱글턴 반환."""
    global _compressor_instance  # noqa: PLW0603
    if _compressor_instance is None:
        _compressor_instance = ContextCompressor()
    return _compressor_instance
