"""RAG용 청킹 + 임베딩 + LanceDB 저장"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.services.rag.embedding import create_query_embedding
from app.tools.news_pipeline.config import NewsPipelineConfig
from app.tools.news_pipeline.models import ProcessedArticle

logger = logging.getLogger(__name__)


@dataclass
class NewsChunk:
    """LanceDB 저장용 청크"""

    chunk_id: str
    doc_id: str
    chunk_text: str
    chunk_type: str  # "summary" | "body"
    published_at: str
    source: str
    publisher: str
    title: str
    url: str
    is_secondary: bool = True  # 항상 True
    data_type: str = "news_article"


class Chunker:
    """기사를 RAG용 청크로 분할 + 임베딩 생성

    청킹 전략:
    1. 요약 → 별도 chunk (항상 포함, chunk_type="summary")
    2. 본문 → 고정 크기 분절 + overlap (chunk_type="body")
    3. 모든 청크에 is_secondary=true 메타데이터 필수
    """

    def __init__(self, config: NewsPipelineConfig) -> None:
        self._chunk_size = config.chunk_size
        self._chunk_overlap = config.chunk_overlap

    def create_chunks(self, article: ProcessedArticle) -> list[NewsChunk]:
        """기사를 청크 목록으로 변환"""
        chunks: list[NewsChunk] = []
        pub_at = article.published_at.isoformat() if article.published_at else ""

        # 1. 요약 청크
        summary_text = self._build_summary_text(article)
        chunks.append(NewsChunk(
            chunk_id=f"{article.doc_id}_summary",
            doc_id=article.doc_id,
            chunk_text=summary_text,
            chunk_type="summary",
            published_at=pub_at,
            source=article.source.value,
            publisher=article.publisher,
            title=article.title,
            url=article.url,
        ))

        # 2. 본문 청크
        body_chunks = self._split_text(article.cleaned_text)
        for idx, chunk_text in enumerate(body_chunks):
            chunks.append(NewsChunk(
                chunk_id=f"{article.doc_id}_body_{idx}",
                doc_id=article.doc_id,
                chunk_text=chunk_text,
                chunk_type="body",
                published_at=pub_at,
                source=article.source.value,
                publisher=article.publisher,
                title=article.title,
                url=article.url,
            ))

        return chunks

    async def embed_and_store(
        self, chunks: list[NewsChunk], lancedb_table_name: str,
    ) -> int:
        """청크를 임베딩하고 LanceDB에 저장. 저장 건수 반환.

        v0.3.0: FTS 인덱스 자동 생성 → 하이브리드 검색 지원
        """
        import lancedb  # type: ignore[import-untyped]  # noqa: PLC0415

        from app.core.config import settings

        if not chunks:
            return 0

        # 임베딩 생성
        records: list[dict[str, Any]] = []
        for chunk in chunks:
            vector = create_query_embedding(chunk.chunk_text)
            records.append({
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "chunk_text": chunk.chunk_text,
                "chunk_type": chunk.chunk_type,
                "published_at": chunk.published_at,
                "source": chunk.source,
                "publisher": chunk.publisher,
                "title": chunk.title,
                "url": chunk.url,
                "is_secondary": chunk.is_secondary,
                "data_type": chunk.data_type,
                "vector": vector,
            })

        # LanceDB 저장
        db = lancedb.connect(settings.LANCEDB_URI)
        try:
            table = db.open_table(lancedb_table_name)
            table.add(records)
        except Exception:
            # 테이블 미존재 시 생성
            table = db.create_table(lancedb_table_name, data=records)

        # v0.3.0: FTS 인덱스 (하이브리드 검색용, 최초 1회만 생성)
        self._ensure_fts_index(table)

        logger.info("LanceDB 저장 완료: %d 청크 → %s", len(records), lancedb_table_name)
        return len(records)

    @staticmethod
    def _ensure_fts_index(table: Any) -> None:
        """FTS 인덱스 존재 확인 및 생성 (v0.3.0)"""
        try:
            table.create_fts_index("chunk_text", replace=False)
        except Exception:
            pass  # 이미 존재하면 무시

    @staticmethod
    async def hybrid_search(
        query: str,
        lancedb_table_name: str,
        *,
        limit: int = 20,
        rerank_top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """v0.3.0: 하이브리드 검색 (Vector + FTS) + 리랭커

        Consultant 피드백: LanceDB 저장만으로는 검색 품질 부족.
        Vector + FTS 결합 후 리랭커로 정밀 정렬.

        Args:
            query: 검색 쿼리
            lancedb_table_name: LanceDB 테이블명
            limit: 1차 검색 결과 수
            rerank_top_k: 리랭킹 후 반환할 최종 결과 수

        Returns:
            정렬된 검색 결과 목록
        """
        import lancedb  # noqa: PLC0415

        from app.core.config import settings

        db = lancedb.connect(settings.LANCEDB_URI)
        table = db.open_table(lancedb_table_name)

        # 1. 벡터 검색
        query_vector = create_query_embedding(query)
        vector_results = (
            table.search(query_vector)
            .limit(limit)
            .to_list()
        )

        # 2. FTS 검색
        fts_results = (
            table.search(query, query_type="fts")
            .limit(limit)
            .to_list()
        )

        # 3. RRF (Reciprocal Rank Fusion) 결합
        merged = _reciprocal_rank_fusion(vector_results, fts_results, k=60)

        # 4. 리랭커 적용 (기존 인프라 재활용)
        reranked = await _apply_reranker(query, merged[:rerank_top_k * 2])

        return reranked[:rerank_top_k]

    @staticmethod
    def _build_summary_text(article: ProcessedArticle) -> str:
        """요약 정보를 검색용 텍스트로 조합"""
        s = article.summary
        parts = [
            f"[요약] {s.one_liner}",
            f"[쟁점] {', '.join(s.issues)}" if s.issues else "",
            f"[법령] {', '.join(s.laws)}" if s.laws else "",
            f"[판례] {', '.join(s.cases)}" if s.cases else "",
            f"[기관] {', '.join(s.institutions)}" if s.institutions else "",
            f"[시사점] {', '.join(s.implications)}" if s.implications else "",
        ]
        return "\n".join(p for p in parts if p)

    def _split_text(self, text: str) -> list[str]:
        """본문을 고정 크기 청크로 분할 (overlap 적용)"""
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + self._chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk.strip())
            start += self._chunk_size - self._chunk_overlap
        return chunks


async def _apply_reranker(
    query: str, candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """기존 리랭커 인프라로 후보 재정렬

    app/services/rag/ 하위 리랭커를 재사용.
    리랭커 미설정 시 원래 순서 유지.
    """
    try:
        from app.services.rag.rerank import rerank_documents
        texts = [c.get("chunk_text", "") for c in candidates]
        scores = rerank_documents(query, texts)
        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = score
        candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
    except ImportError:
        logger.debug("리랭커 미설치, RRF 순서 유지")
    return candidates


def _reciprocal_rank_fusion(
    results_a: list[dict[str, Any]],
    results_b: list[dict[str, Any]],
    k: int = 60,
) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion — 두 검색 결과를 통합 정렬"""
    scores: dict[str, float] = {}
    items: dict[str, dict[str, Any]] = {}

    for rank, item in enumerate(results_a):
        doc_id = item.get("chunk_id", str(rank))
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
        items[doc_id] = item

    for rank, item in enumerate(results_b):
        doc_id = item.get("chunk_id", str(rank))
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
        items[doc_id] = item

    sorted_ids = sorted(scores, key=scores.get, reverse=True)  # type: ignore[arg-type]
    return [items[doc_id] for doc_id in sorted_ids if doc_id in items]
