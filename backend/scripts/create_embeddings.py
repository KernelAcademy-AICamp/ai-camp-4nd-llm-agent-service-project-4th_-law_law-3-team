#!/usr/bin/env python3
"""
법률 문서 임베딩 생성 스크립트 (청킹 지원)

PostgreSQL에서 법률 문서를 읽어 청크 단위로 임베딩을 생성하고 ChromaDB에 저장합니다.
로컬 모델(sentence-transformers) 또는 OpenAI API를 사용할 수 있습니다.

사용법:
    # 모든 문서 임베딩 생성 (로컬 모델 사용)
    uv run python scripts/create_embeddings.py

    # 특정 유형만 임베딩
    uv run python scripts/create_embeddings.py --type precedent
    uv run python scripts/create_embeddings.py --type constitutional
    uv run python scripts/create_embeddings.py --type administration
    uv run python scripts/create_embeddings.py --type legislation
    uv run python scripts/create_embeddings.py --type committee

    # OpenAI API 사용 (USE_LOCAL_EMBEDDING=False 또는 --use-openai)
    uv run python scripts/create_embeddings.py --use-openai

    # 배치 크기 조정
    uv run python scripts/create_embeddings.py --batch-size 50

    # 청킹 설정 조정
    uv run python scripts/create_embeddings.py --chunk-size 500 --chunk-overlap 50

    # 기존 임베딩 삭제 후 재생성
    uv run python scripts/create_embeddings.py --reset

    # 현재 통계만 확인
    uv run python scripts/create_embeddings.py --stats
"""

import argparse
import asyncio
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import func, select

from app.core.database import async_session_factory
from app.tools.vectorstore import VectorStore
from app.core.config import settings
from app.models.legal_document import DocType, LegalDocument

# ============================================================================
# 청킹 설정
# ============================================================================

@dataclass
class ChunkConfig:
    """청크 설정"""
    chunk_size: int = 500      # 문자 수
    chunk_overlap: int = 50    # 오버랩 문자 수
    min_chunk_size: int = 100  # 최소 청크 크기


@dataclass
class Chunk:
    """문서 청크"""
    chunk_id: str           # {source}_{doc_id}_chunk_{index}
    doc_id: int             # PostgreSQL PK (포인터)
    source: str             # 데이터 출처
    doc_type: str           # 문서 유형
    chunk_index: int        # 청크 순서
    chunk_start: int        # 원문 내 시작 위치
    chunk_end: int          # 원문 내 종료 위치
    chunk_text: str         # 청크 텍스트
    case_number: str        # 사건번호 (필터용)
    court_name: str         # 기관명 (필터용)
    decision_date: str      # 날짜 (필터용)


def create_chunks(doc: LegalDocument, config: ChunkConfig) -> List[Chunk]:
    """
    문서를 청크로 분할

    Args:
        doc: 법률 문서
        config: 청크 설정

    Returns:
        청크 목록
    """
    text = doc.embedding_text
    if not text or len(text) < config.min_chunk_size:
        # 텍스트가 너무 짧으면 청크 하나만 생성
        if text:
            return [Chunk(
                chunk_id=f"{doc.source}_{doc.id}_chunk_0",
                doc_id=doc.id,
                source=doc.source,
                doc_type=doc.doc_type,
                chunk_index=0,
                chunk_start=0,
                chunk_end=len(text),
                chunk_text=text,
                case_number=doc.case_number or "",
                court_name=doc.court_name or "",
                decision_date=doc.decision_date.isoformat() if doc.decision_date else "",
            )]
        return []

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        # 청크 끝 위치 계산
        end = min(start + config.chunk_size, len(text))

        # 문장 경계에서 자르기 시도 (마지막 청크가 아닌 경우)
        if end < len(text):
            # 마침표, 느낌표, 물음표, 줄바꿈 등에서 자르기
            for sep in ['. ', '.\n', '! ', '!\n', '? ', '?\n', '\n\n', '\n']:
                sep_pos = text.rfind(sep, start + config.min_chunk_size, end)
                if sep_pos > start:
                    end = sep_pos + len(sep)
                    break

        chunk_text = text[start:end].strip()

        if chunk_text and len(chunk_text) >= config.min_chunk_size:
            chunks.append(Chunk(
                chunk_id=f"{doc.source}_{doc.id}_chunk_{chunk_index}",
                doc_id=doc.id,
                source=doc.source,
                doc_type=doc.doc_type,
                chunk_index=chunk_index,
                chunk_start=start,
                chunk_end=end,
                chunk_text=chunk_text,
                case_number=doc.case_number or "",
                court_name=doc.court_name or "",
                decision_date=doc.decision_date.isoformat() if doc.decision_date else "",
            ))
            chunk_index += 1

        # 다음 청크 시작 위치 (오버랩 적용)
        start = end - config.chunk_overlap
        if start >= len(text) - config.min_chunk_size:
            break

    return chunks


# ============================================================================
# 임베딩 모델
# ============================================================================

_local_model = None


def get_local_model():
    """sentence-transformers 모델 로드 (lazy loading)"""
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer

        # 모델 캐시 디렉토리를 워크스페이스 내로 설정
        cache_dir = Path(__file__).parent.parent.parent / "data" / "models"
        cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Loading local model: {settings.LOCAL_EMBEDDING_MODEL}")
        print(f"[INFO] Model cache directory: {cache_dir}")
        _local_model = SentenceTransformer(
            settings.LOCAL_EMBEDDING_MODEL,
            cache_folder=str(cache_dir)
        )
        print(f"[INFO] Model loaded. Embedding dimension: {_local_model.get_sentence_embedding_dimension()}")
    return _local_model


def create_embeddings_batch_local(texts: List[str]) -> List[List[float]]:
    """
    로컬 모델을 사용한 임베딩 생성

    Args:
        texts: 임베딩할 텍스트 목록

    Returns:
        임베딩 벡터 목록
    """
    model = get_local_model()

    # 빈 텍스트 처리
    processed_texts = []
    for text in texts:
        if not text or not text.strip():
            processed_texts.append("(내용 없음)")
        else:
            # 로컬 모델은 보통 512 토큰 제한이므로 텍스트 자르기
            processed_texts.append(text.strip()[:2000])

    embeddings = model.encode(processed_texts, show_progress_bar=False)
    return [emb.tolist() for emb in embeddings]


def create_embeddings_batch_openai(texts: List[str]) -> List[List[float]]:
    """
    OpenAI API를 사용한 임베딩 생성

    Args:
        texts: 임베딩할 텍스트 목록

    Returns:
        임베딩 벡터 목록
    """
    import tiktoken
    from openai import OpenAI

    openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
    tokenizer = tiktoken.get_encoding("cl100k_base")
    max_tokens = 8191

    def truncate_text(text: str) -> str:
        tokens = tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return tokenizer.decode(tokens[:max_tokens])

    # 빈 텍스트 처리
    processed_texts = []
    for text in texts:
        if not text or not text.strip():
            processed_texts.append("(내용 없음)")
        else:
            processed_texts.append(truncate_text(text.strip()))

    response = openai_client.embeddings.create(
        model=settings.EMBEDDING_MODEL,
        input=processed_texts,
    )

    return [item.embedding for item in response.data]


def create_embeddings_batch(texts: List[str], use_local: bool = True) -> List[List[float]]:
    """
    임베딩 생성 (로컬 또는 OpenAI)

    Args:
        texts: 임베딩할 텍스트 목록
        use_local: 로컬 모델 사용 여부

    Returns:
        임베딩 벡터 목록
    """
    if use_local:
        return create_embeddings_batch_local(texts)
    else:
        return create_embeddings_batch_openai(texts)


# ============================================================================
# DB 조회
# ============================================================================

async def get_documents_for_embedding(
    doc_type: Optional[DocType] = None,
    offset: int = 0,
    limit: int = 1000,
) -> List[LegalDocument]:
    """
    임베딩할 문서 조회

    Args:
        doc_type: 특정 문서 유형만 조회 (None이면 전체)
        offset: 시작 위치
        limit: 조회 개수

    Returns:
        LegalDocument 목록
    """
    async with async_session_factory() as session:
        query = select(LegalDocument).order_by(LegalDocument.id)

        if doc_type:
            query = query.where(LegalDocument.doc_type == doc_type.value)

        query = query.offset(offset).limit(limit)

        result = await session.execute(query)
        return list(result.scalars().all())


async def get_document_count(doc_type: Optional[DocType] = None) -> int:
    """문서 수 조회"""
    async with async_session_factory() as session:
        query = select(func.count(LegalDocument.id))

        if doc_type:
            query = query.where(LegalDocument.doc_type == doc_type.value)

        result = await session.execute(query)
        return result.scalar() or 0


def get_existing_chunk_ids(store: VectorStore, doc_type: Optional[DocType] = None) -> set:
    """ChromaDB에 이미 존재하는 청크 ID 조회"""
    try:
        # 메타데이터 필터링으로 조회
        if doc_type:
            results = store.collection.get(
                where={"doc_type": doc_type.value},
                include=[]
            )
        else:
            results = store.collection.get(include=[])

        return set(results["ids"]) if results["ids"] else set()
    except Exception:
        return set()


def get_existing_doc_ids_from_chunks(store: VectorStore, doc_type: Optional[DocType] = None) -> set:
    """이미 임베딩된 문서의 doc_id 조회"""
    try:
        if doc_type:
            results = store.collection.get(
                where={"doc_type": doc_type.value},
                include=["metadatas"]
            )
        else:
            results = store.collection.get(include=["metadatas"])

        if results["metadatas"]:
            return set(m.get("doc_id") for m in results["metadatas"] if m.get("doc_id"))
        return set()
    except Exception:
        return set()


# ============================================================================
# 임베딩 생성
# ============================================================================

async def create_embeddings_for_type(
    doc_type: Optional[DocType] = None,
    batch_size: int = 100,
    reset: bool = False,
    use_local: bool = True,
    chunk_config: ChunkConfig = None,
) -> dict:
    """
    특정 유형의 문서 임베딩 생성 (청킹 적용)

    Args:
        doc_type: 문서 유형 (None이면 전체)
        batch_size: 배치 크기
        reset: True면 기존 임베딩 삭제 후 재생성
        use_local: 로컬 임베딩 모델 사용 여부
        chunk_config: 청킹 설정

    Returns:
        처리 결과 통계
    """
    if chunk_config is None:
        chunk_config = ChunkConfig()

    stats = {
        "doc_type": doc_type.value if doc_type else "all",
        "total_documents": 0,
        "documents_processed": 0,
        "chunks_created": 0,
        "chunks_stored": 0,
        "skipped": 0,
        "errors": 0,
    }

    # 벡터 저장소 초기화
    store = VectorStore()

    if reset:
        if doc_type:
            # 특정 타입만 삭제
            print(f"[INFO] Deleting existing embeddings for {doc_type.value}...")
            existing_ids = get_existing_chunk_ids(store, doc_type)
            if existing_ids:
                store.delete_by_ids(list(existing_ids))
                print(f"[INFO] Deleted {len(existing_ids)} existing chunks")
            existing_doc_ids = set()
        else:
            # 전체 삭제
            print("[INFO] Resetting entire vector store...")
            store.reset()
            existing_doc_ids = set()
    else:
        existing_doc_ids = get_existing_doc_ids_from_chunks(store, doc_type)
        print(f"[INFO] Found {len(existing_doc_ids)} documents already embedded")

    # 문서 수 조회
    total_count = await get_document_count(doc_type)
    stats["total_documents"] = total_count
    print(f"[INFO] Total documents to process: {total_count:,}")
    print(f"[INFO] Chunk config: size={chunk_config.chunk_size}, overlap={chunk_config.chunk_overlap}")

    if total_count == 0:
        print("[WARN] No documents found")
        return stats

    # 로컬 모델 사전 로드
    if use_local:
        get_local_model()

    # 배치 처리
    offset = 0
    db_batch_size = 500  # DB 조회 배치

    while offset < total_count:
        # DB에서 문서 조회
        documents = await get_documents_for_embedding(doc_type, offset, db_batch_size)

        if not documents:
            break

        # 청크 배치 준비
        batch_chunks: List[Chunk] = []

        for doc in documents:
            # 이미 처리된 문서 스킵
            if doc.id in existing_doc_ids:
                stats["skipped"] += 1
                continue

            # 문서를 청크로 분할
            try:
                chunks = create_chunks(doc, chunk_config)
                if not chunks:
                    stats["skipped"] += 1
                    continue

                batch_chunks.extend(chunks)
                stats["documents_processed"] += 1
                stats["chunks_created"] += len(chunks)

            except Exception as e:
                stats["errors"] += 1
                if stats["errors"] <= 5:
                    print(f"  [ERROR] Chunking error for doc {doc.id}: {e}")
                continue

            # 배치 크기 도달 시 처리
            if len(batch_chunks) >= batch_size:
                try:
                    stored = _store_chunk_batch(store, batch_chunks, use_local)
                    stats["chunks_stored"] += stored
                    existing_doc_ids.update(c.doc_id for c in batch_chunks)
                except Exception as e:
                    stats["errors"] += len(batch_chunks)
                    print(f"  [ERROR] Batch store error: {e}")

                batch_chunks = []

        # 남은 배치 처리
        if batch_chunks:
            try:
                stored = _store_chunk_batch(store, batch_chunks, use_local)
                stats["chunks_stored"] += stored
            except Exception as e:
                stats["errors"] += len(batch_chunks)
                print(f"  [ERROR] Final batch store error: {e}")

        # 진행률 출력
        progress = offset + len(documents)
        pct = progress / total_count * 100
        print(
            f"  [PROGRESS] {progress:,}/{total_count:,} ({pct:.1f}%) - "
            f"Docs: {stats['documents_processed']:,}, Chunks: {stats['chunks_stored']:,}"
        )

        offset += db_batch_size

    return stats


def _store_chunk_batch(store: VectorStore, chunks: List[Chunk], use_local: bool) -> int:
    """청크 배치를 벡터 저장소에 저장"""
    if not chunks:
        return 0

    # 임베딩 생성
    texts = [c.chunk_text for c in chunks]
    embeddings = create_embeddings_batch(texts, use_local)

    # 벡터 저장소에 저장
    # LanceDB는 디스크 기반이라 텍스트 저장에 부담 없음
    # ChromaDB/Qdrant는 용량 최적화를 위해 텍스트 저장 안 함
    ids = [c.chunk_id for c in chunks]
    metadatas = [
        {
            "doc_id": c.doc_id,
            "source": c.source,
            "doc_type": c.doc_type,
            "chunk_index": c.chunk_index,
            "chunk_start": c.chunk_start,
            "chunk_end": c.chunk_end,
            "case_number": c.case_number,
            "court_name": c.court_name,
            "decision_date": c.decision_date,
        }
        for c in chunks
    ]

    # LanceDB는 텍스트를 저장해도 효율적이므로 texts를 넘김
    documents_to_save = texts if settings.VECTOR_DB == "lancedb" else None

    store.add_documents(
        ids=ids,
        documents=documents_to_save,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return len(chunks)


async def create_all_embeddings(
    batch_size: int = 100,
    reset: bool = False,
    use_local: bool = True,
    chunk_config: ChunkConfig = None,
) -> dict:
    """모든 문서 타입의 임베딩 생성"""
    all_stats = {}
    start_time = datetime.now()

    for doc_type in DocType:
        print(f"\n{'='*60}")
        print(f"Creating embeddings for {doc_type.value}...")
        print('='*60)

        stats = await create_embeddings_for_type(
            doc_type, batch_size, reset, use_local, chunk_config
        )
        all_stats[doc_type.value] = stats

    elapsed = datetime.now() - start_time
    all_stats["elapsed_time"] = str(elapsed)

    return all_stats


# ============================================================================
# 통계
# ============================================================================

def show_stats():
    """현재 임베딩 통계 출력"""
    store = VectorStore()

    print("\n" + "="*60)
    print("Vector Store Statistics (Chunks)")
    print("="*60)

    total = store.count()
    print(f"Total chunks: {total:,}")

    # 타입별 카운트
    print("\nBy doc_type:")
    for doc_type in DocType:
        try:
            results = store.collection.get(
                where={"doc_type": doc_type.value},
                include=["metadatas"]
            )
            chunk_count = len(results["ids"]) if results["ids"] else 0
            doc_ids = set(m.get("doc_id") for m in results["metadatas"] if m.get("doc_id")) if results["metadatas"] else set()
            doc_count = len(doc_ids)
            print(f"  - {doc_type.value}: {chunk_count:,} chunks from {doc_count:,} documents")
        except Exception:
            print(f"  - {doc_type.value}: (error)")

    # 소스별 카운트
    print("\nBy source (for committee):")
    try:
        results = store.collection.get(
            where={"doc_type": "committee"},
            include=["metadatas"]
        )
        if results["metadatas"]:
            source_counts = {}
            for m in results["metadatas"]:
                source = m.get("source", "unknown")
                source_counts[source] = source_counts.get(source, 0) + 1
            for source, count in sorted(source_counts.items()):
                print(f"  - {source}: {count:,} chunks")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="법률 문서 임베딩 생성 (청킹 지원)"
    )
    parser.add_argument(
        "--type",
        choices=["precedent", "constitutional", "administration", "legislation", "committee", "all"],
        default="all",
        help="임베딩할 문서 유형 (기본: all)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="배치 크기 (기본: 100)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="청크 크기 (문자 수, 기본: 500)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="청크 오버랩 (문자 수, 기본: 50)"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="기존 임베딩 삭제 후 재생성"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="현재 임베딩 통계만 출력"
    )
    parser.add_argument(
        "--use-openai",
        action="store_true",
        help="OpenAI API 사용 (기본: 로컬 모델)"
    )

    args = parser.parse_args()

    use_local = settings.USE_LOCAL_EMBEDDING and not args.use_openai
    model_name = settings.LOCAL_EMBEDDING_MODEL if use_local else settings.EMBEDDING_MODEL

    chunk_config = ChunkConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    print("="*60)
    print("Legal Document Embedding Creator (with Chunking)")
    print("="*60)
    print(f"Embedding model: {model_name}")
    print(f"Use local model: {use_local}")
    print(f"Batch size: {args.batch_size}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Chunk overlap: {args.chunk_overlap}")
    print(f"Reset mode: {args.reset}")

    if args.stats:
        show_stats()
        return

    if args.type == "all":
        stats = asyncio.run(create_all_embeddings(
            args.batch_size, args.reset, use_local, chunk_config
        ))
    else:
        doc_type = DocType(args.type)
        stats = asyncio.run(create_embeddings_for_type(
            doc_type, args.batch_size, args.reset, use_local, chunk_config
        ))

    # 결과 출력
    print("\n" + "="*60)
    print("Embedding Results")
    print("="*60)

    import json
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    # 최종 통계
    show_stats()


if __name__ == "__main__":
    main()
