"""
RAG 기반 법률 챗봇 서비스

VectorStore에서 관련 문서를 검색하고 OpenAI로 응답 생성
"""

from typing import List, Optional
from pathlib import Path

from openai import OpenAI
from sqlalchemy import select

from app.core.config import settings
from app.common.vectorstore import VectorStore
from app.common.database import sync_session_factory
from app.models.legal_document import LegalDocument


# 로컬 임베딩 모델 (lazy loading)
_local_model = None


def get_local_model():
    """sentence-transformers 모델 로드"""
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer

        cache_dir = Path(__file__).parent.parent.parent / "data" / "models"
        cache_dir.mkdir(parents=True, exist_ok=True)

        _local_model = SentenceTransformer(
            settings.LOCAL_EMBEDDING_MODEL,
            cache_folder=str(cache_dir)
        )
    return _local_model


def create_query_embedding(query: str) -> List[float]:
    """쿼리 텍스트를 임베딩 벡터로 변환"""
    if settings.USE_LOCAL_EMBEDDING:
        model = get_local_model()
        embedding = model.encode(query, show_progress_bar=False)
        return embedding.tolist()
    else:
        # OpenAI 임베딩
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        response = client.embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input=query,
        )
        return response.data[0].embedding


def _fetch_document_texts(doc_ids: List[int], chunk_positions: dict) -> dict:
    """
    PostgreSQL에서 문서 원문을 가져와 청크 텍스트 추출

    Args:
        doc_ids: 문서 ID 목록
        chunk_positions: {doc_id: [(chunk_start, chunk_end), ...]}

    Returns:
        {chunk_key: chunk_text}
    """
    if not doc_ids:
        return {}

    with sync_session_factory() as session:
        result = session.execute(
            select(LegalDocument).where(LegalDocument.id.in_(doc_ids))
        )
        docs = {doc.id: doc for doc in result.scalars().all()}

    chunk_texts = {}
    for doc_id, positions in chunk_positions.items():
        doc = docs.get(doc_id)
        if doc:
            text = doc.embedding_text or ""
            for start, end in positions:
                key = f"{doc_id}_{start}_{end}"
                chunk_texts[key] = text[start:end] if text else ""

    return chunk_texts


def search_relevant_documents(
    query: str,
    n_results: int = 5,
    doc_type: Optional[str] = None,
) -> List[dict]:
    """
    관련 법률 문서 검색

    Args:
        query: 검색 쿼리
        n_results: 반환할 결과 수
        doc_type: 문서 유형 필터 (precedent, constitutional, etc.)

    Returns:
        관련 문서 목록
    """
    store = VectorStore()

    # 쿼리 임베딩 생성
    query_embedding = create_query_embedding(query)

    # 필터 조건
    where = {"doc_type": doc_type} if doc_type else None

    # 검색 (documents 없이 metadatas만)
    results = store.search(
        query_embedding=query_embedding,
        n_results=n_results,
        where=where,
        include=["metadatas", "distances"],
    )

    # 결과가 없으면 빈 목록 반환
    if not results or not results.get("ids") or not results["ids"][0]:
        return []

    # doc_id와 청크 위치 수집
    doc_ids = set()
    chunk_positions = {}  # {doc_id: [(start, end), ...]}

    for i, chunk_id in enumerate(results["ids"][0]):
        metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
        doc_id = metadata.get("doc_id")
        chunk_start = metadata.get("chunk_start", 0)
        chunk_end = metadata.get("chunk_end", 0)

        if doc_id:
            doc_ids.add(doc_id)
            if doc_id not in chunk_positions:
                chunk_positions[doc_id] = []
            chunk_positions[doc_id].append((chunk_start, chunk_end))

    # PostgreSQL에서 원문 가져오기
    chunk_texts = _fetch_document_texts(list(doc_ids), chunk_positions)

    # 결과 정리
    documents = []
    for i, chunk_id in enumerate(results["ids"][0]):
        metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
        doc_id = metadata.get("doc_id")
        chunk_start = metadata.get("chunk_start", 0)
        chunk_end = metadata.get("chunk_end", 0)

        # 청크 텍스트 가져오기
        chunk_key = f"{doc_id}_{chunk_start}_{chunk_end}"
        content = chunk_texts.get(chunk_key, "")

        doc = {
            "id": chunk_id,
            "content": content,
            "metadata": metadata,
            "similarity": 1 - results["distances"][0][i] if results.get("distances") else 0,
        }
        documents.append(doc)

    return documents


SYSTEM_PROMPT = """당신은 한국 법률 전문 AI 어시스턴트입니다.
사용자의 법률 질문에 대해 제공된 판례 및 법률 문서를 참고하여 정확하고 도움이 되는 답변을 제공합니다.

답변 시 주의사항:
1. 제공된 참고 자료를 기반으로 답변하세요
2. 관련 판례가 있다면 사건번호와 함께 인용하세요
3. 법률 용어는 쉽게 설명해주세요
4. 확실하지 않은 내용은 추측하지 말고 "법률 전문가와 상담이 필요합니다"라고 안내하세요
5. 답변은 친절하고 이해하기 쉽게 작성하세요

중요: 이 서비스는 법률 정보 제공 목적이며, 실제 법률 조언을 대체하지 않습니다."""


def generate_chat_response(
    user_message: str,
    chat_history: Optional[List[dict]] = None,
    n_context_docs: int = 5,
) -> dict:
    """
    RAG 기반 챗봇 응답 생성

    Args:
        user_message: 사용자 메시지
        chat_history: 이전 대화 기록 [{"role": "user/assistant", "content": "..."}]
        n_context_docs: 컨텍스트로 사용할 문서 수

    Returns:
        {
            "response": "AI 응답",
            "sources": [관련 문서 목록]
        }
    """
    # 1. 관련 문서 검색
    relevant_docs = search_relevant_documents(user_message, n_results=n_context_docs)

    # 2. 컨텍스트 구성
    context_parts = []
    for i, doc in enumerate(relevant_docs, 1):
        metadata = doc.get("metadata", {})
        case_name = metadata.get("case_name", "")
        case_number = metadata.get("case_number", "")
        doc_type = metadata.get("doc_type", "")

        context_parts.append(
            f"[참고자료 {i}]\n"
            f"유형: {doc_type}\n"
            f"사건명: {case_name}\n"
            f"사건번호: {case_number}\n"
            f"내용: {doc.get('content', '')[:500]}\n"
        )

    context = "\n".join(context_parts)

    # 3. 메시지 구성
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # 이전 대화 기록 추가
    if chat_history:
        for msg in chat_history[-10:]:  # 최근 10개만
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            })

    # 사용자 메시지 + 컨텍스트
    user_prompt = f"""사용자 질문: {user_message}

---
참고 자료:
{context if context else "(관련 자료 없음)"}
---

위 참고 자료를 바탕으로 사용자의 질문에 답변해주세요."""

    messages.append({"role": "user", "content": user_prompt})

    # 4. OpenAI API 호출
    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # 비용 효율적인 모델
        messages=messages,
        temperature=0.7,
        max_tokens=1000,
    )

    assistant_message = response.choices[0].message.content

    # 5. 결과 반환
    sources = [
        {
            "case_name": doc.get("metadata", {}).get("case_name", ""),
            "case_number": doc.get("metadata", {}).get("case_number", ""),
            "doc_type": doc.get("metadata", {}).get("doc_type", ""),
            "similarity": round(doc.get("similarity", 0), 3),
        }
        for doc in relevant_docs
    ]

    return {
        "response": assistant_message,
        "sources": sources,
    }
