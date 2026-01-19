#!/usr/bin/env python3
"""
법률 문서 임베딩 생성 스크립트

PostgreSQL에서 법률 문서를 읽어 임베딩을 생성하고 ChromaDB에 저장합니다.
로컬 모델(sentence-transformers) 또는 OpenAI API를 사용할 수 있습니다.

사용법:
    # 모든 문서 임베딩 생성 (로컬 모델 사용)
    uv run python scripts/create_embeddings.py

    # 특정 유형만 임베딩
    uv run python scripts/create_embeddings.py --type precedent

    # OpenAI API 사용 (USE_LOCAL_EMBEDDING=False 또는 --use-openai)
    uv run python scripts/create_embeddings.py --use-openai

    # 배치 크기 조정
    uv run python scripts/create_embeddings.py --batch-size 50

    # 기존 임베딩 삭제 후 재생성
    uv run python scripts/create_embeddings.py --reset

    # 현재 통계만 확인
    uv run python scripts/create_embeddings.py --stats
"""

import argparse
import asyncio
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select, func

from app.core.config import settings
from app.common.database import async_session_factory
from app.common.vectorstore import VectorStore
from app.models.legal_document import LegalDocument, DocType


# 임베딩 모델 (로컬 또는 OpenAI)
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


def get_existing_ids(store: VectorStore, doc_type: Optional[DocType] = None) -> set:
    """ChromaDB에 이미 존재하는 문서 ID 조회"""
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


async def create_embeddings_for_type(
    doc_type: Optional[DocType] = None,
    batch_size: int = 100,
    reset: bool = False,
    use_local: bool = True,
) -> dict:
    """
    특정 유형의 문서 임베딩 생성

    Args:
        doc_type: 문서 유형 (None이면 전체)
        batch_size: 배치 크기
        reset: True면 기존 임베딩 삭제 후 재생성
        use_local: 로컬 임베딩 모델 사용 여부

    Returns:
        처리 결과 통계
    """
    stats = {
        "doc_type": doc_type.value if doc_type else "all",
        "total_documents": 0,
        "processed": 0,
        "skipped": 0,
        "errors": 0,
    }

    # 벡터 저장소 초기화
    store = VectorStore()

    if reset:
        if doc_type:
            # 특정 타입만 삭제
            print(f"[INFO] Deleting existing embeddings for {doc_type.value}...")
            existing_ids = get_existing_ids(store, doc_type)
            if existing_ids:
                store.delete_by_ids(list(existing_ids))
                print(f"[INFO] Deleted {len(existing_ids)} existing embeddings")
            existing_ids = set()
        else:
            # 전체 삭제
            print("[INFO] Resetting entire vector store...")
            store.reset()
            existing_ids = set()
    else:
        existing_ids = get_existing_ids(store, doc_type)
        print(f"[INFO] Found {len(existing_ids)} existing embeddings")

    # 문서 수 조회
    total_count = await get_document_count(doc_type)
    stats["total_documents"] = total_count
    print(f"[INFO] Total documents to process: {total_count:,}")

    if total_count == 0:
        print("[WARN] No documents found")
        return stats

    # 로컬 모델 사전 로드
    if use_local:
        get_local_model()

    # 배치 처리
    offset = 0
    db_batch_size = 1000  # DB 조회 배치

    while offset < total_count:
        # DB에서 문서 조회
        documents = await get_documents_for_embedding(doc_type, offset, db_batch_size)

        if not documents:
            break

        # 임베딩 배치 준비
        batch_ids = []
        batch_texts = []
        batch_metadatas = []
        batch_documents_text = []

        for doc in documents:
            doc_id = f"{doc.doc_type}_{doc.serial_number}"

            # 이미 존재하는 문서 스킵
            if doc_id in existing_ids:
                stats["skipped"] += 1
                continue

            embedding_text = doc.embedding_text
            if not embedding_text:
                stats["skipped"] += 1
                continue

            batch_ids.append(doc_id)
            batch_texts.append(embedding_text)
            batch_documents_text.append(embedding_text[:1000])  # 저장용 요약
            batch_metadatas.append({
                "doc_type": doc.doc_type,
                "case_number": doc.case_number or "",
                "case_name": doc.case_name or "",
                "decision_date": doc.decision_date.isoformat() if doc.decision_date else "",
                "court_name": doc.court_name or "",
                "db_id": doc.id,
            })

            # 배치 크기 도달 시 처리
            if len(batch_ids) >= batch_size:
                try:
                    # 임베딩 생성
                    embeddings = create_embeddings_batch(batch_texts, use_local)

                    # ChromaDB에 저장
                    store.add_documents(
                        ids=batch_ids,
                        documents=batch_documents_text,
                        metadatas=batch_metadatas,
                        embeddings=embeddings,
                    )

                    stats["processed"] += len(batch_ids)
                    existing_ids.update(batch_ids)

                except Exception as e:
                    stats["errors"] += len(batch_ids)
                    print(f"  [ERROR] Batch error: {e}")

                # 배치 초기화
                batch_ids = []
                batch_texts = []
                batch_metadatas = []
                batch_documents_text = []

        # 남은 배치 처리
        if batch_ids:
            try:
                embeddings = create_embeddings_batch(batch_texts, use_local)

                store.add_documents(
                    ids=batch_ids,
                    documents=batch_documents_text,
                    metadatas=batch_metadatas,
                    embeddings=embeddings,
                )

                stats["processed"] += len(batch_ids)

            except Exception as e:
                stats["errors"] += len(batch_ids)
                print(f"  [ERROR] Final batch error: {e}")

        # 진행률 출력
        progress = offset + len(documents)
        pct = progress / total_count * 100
        print(
            f"  [PROGRESS] {progress:,}/{total_count:,} ({pct:.1f}%) - "
            f"Processed: {stats['processed']:,}, Skipped: {stats['skipped']:,}"
        )

        offset += db_batch_size

    return stats


async def create_all_embeddings(
    batch_size: int = 100,
    reset: bool = False,
    use_local: bool = True,
) -> dict:
    """모든 문서 타입의 임베딩 생성"""
    all_stats = {}
    start_time = datetime.now()

    for doc_type in DocType:
        print(f"\n{'='*60}")
        print(f"Creating embeddings for {doc_type.value}...")
        print('='*60)

        stats = await create_embeddings_for_type(doc_type, batch_size, reset, use_local)
        all_stats[doc_type.value] = stats

    elapsed = datetime.now() - start_time
    all_stats["elapsed_time"] = str(elapsed)

    return all_stats


def show_stats():
    """현재 임베딩 통계 출력"""
    store = VectorStore()

    print("\n" + "="*50)
    print("Vector Store Statistics")
    print("="*50)

    total = store.count()
    print(f"Total embeddings: {total:,}")

    # 타입별 카운트
    print("\nBy type:")
    for doc_type in DocType:
        try:
            results = store.collection.get(
                where={"doc_type": doc_type.value},
                include=[]
            )
            count = len(results["ids"]) if results["ids"] else 0
            print(f"  - {doc_type.value}: {count:,}")
        except Exception:
            print(f"  - {doc_type.value}: (error)")


def main():
    parser = argparse.ArgumentParser(
        description="법률 문서 임베딩 생성"
    )
    parser.add_argument(
        "--type",
        choices=["precedent", "constitutional", "administration", "legislation", "all"],
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

    print("="*60)
    print("Legal Document Embedding Creator")
    print("="*60)
    print(f"Embedding model: {model_name}")
    print(f"Use local model: {use_local}")
    print(f"Batch size: {args.batch_size}")
    print(f"Reset mode: {args.reset}")

    if args.stats:
        show_stats()
        return

    if args.type == "all":
        stats = asyncio.run(create_all_embeddings(args.batch_size, args.reset, use_local))
    else:
        doc_type = DocType(args.type)
        stats = asyncio.run(
            create_embeddings_for_type(doc_type, args.batch_size, args.reset, use_local)
        )

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
