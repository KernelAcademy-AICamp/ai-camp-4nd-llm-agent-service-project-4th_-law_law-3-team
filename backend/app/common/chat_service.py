"""
RAG 기반 법률 챗봇 서비스

VectorStore에서 관련 문서를 검색하고 LLM으로 응답 생성
LangChain/LangGraph 호환 구조
"""

import re
from typing import List, Optional, Set
from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from openai import OpenAI
from sqlalchemy import select, or_

from app.core.config import settings
from app.common.vectorstore import VectorStore
from app.common.database import sync_session_factory
from app.common.llm import get_chat_model
from app.models.legal_document import LegalDocument
from app.models.law import Law


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


def extract_law_names(reference_articles: str) -> Set[str]:
    """
    참조조문에서 법령명 추출

    Args:
        reference_articles: 참조조문 문자열
            예: "민사소송법 제704조 제2항", "근로기준법 제34조"

    Returns:
        법령명 집합 (예: {"민사소송법", "근로기준법"})
    """
    if not reference_articles:
        return set()

    law_names = set()

    # 패턴 1: "구 법령명(날짜...)" 형식에서 법령명 추출
    # 예: "구 지방세법(2002. 12. 30. 법률 제6838호로 개정되기 전의 것)"
    pattern_old = r'구\s+([가-힣]+(?:법|령|규칙|규정))'
    for match in re.finditer(pattern_old, reference_articles):
        law_names.add(match.group(1))

    # 패턴 2: 일반 법령명 추출
    # 예: "민법 제750조", "형법 제250조 제1항"
    # 법령명 패턴: 한글로 된 법/령/규칙/규정으로 끝나는 단어 (최소 2글자)
    # 형법, 민법 등 짧은 법령명도 추출하기 위해 {1,} 사용
    pattern_normal = r'([가-힣]+(?:법|령|규칙|규정))(?:\s|$|제|\(|,)'
    for match in re.finditer(pattern_normal, reference_articles):
        name = match.group(1)
        # "구"로 시작하는 것 제외 (이미 위에서 처리)
        # 최소 2글자 이상만 (예: "법" 단독은 제외)
        if not name.startswith("구") and len(name) >= 2:
            law_names.add(name)

    return law_names


def fetch_laws_by_names(law_names: Set[str], limit: int = 5) -> List[dict]:
    """
    법령명으로 laws 테이블에서 법령 조회

    Args:
        law_names: 법령명 집합
        limit: 최대 조회 건수

    Returns:
        법령 정보 목록
    """
    if not law_names:
        return []

    with sync_session_factory() as session:
        # 법령명으로 검색 (LIKE 검색으로 부분 매칭)
        conditions = []
        for name in law_names:
            conditions.append(Law.law_name.ilike(f"%{name}%"))

        if not conditions:
            return []

        result = session.execute(
            select(Law)
            .where(or_(*conditions))
            .limit(limit)
        )
        laws = result.scalars().all()

        return [
            {
                "law_id": law.law_id,
                "law_name": law.law_name,
                "law_type": law.law_type,
                "content": law.content[:1000] if law.content else "",  # 앞부분만
            }
            for law in laws
        ]


def fetch_reference_articles_from_docs(doc_ids: List[int]) -> str:
    """
    문서 ID 목록에서 reference_articles 수집

    Args:
        doc_ids: 문서 ID 목록

    Returns:
        모든 reference_articles를 합친 문자열
    """
    if not doc_ids:
        return ""

    with sync_session_factory() as session:
        result = session.execute(
            select(LegalDocument.reference_articles)
            .where(LegalDocument.id.in_(doc_ids))
            .where(LegalDocument.reference_articles.isnot(None))
        )
        articles = [row[0] for row in result.fetchall() if row[0]]

    return " ".join(articles)


def _fetch_document_texts(doc_ids: List[int], chunk_positions: dict) -> tuple[dict, dict]:
    """
    PostgreSQL에서 문서 원문을 가져와 청크 텍스트 추출

    Args:
        doc_ids: 문서 ID 목록
        chunk_positions: {doc_id: [(chunk_start, chunk_end), ...]}

    Returns:
        (chunk_texts, doc_info)
        - chunk_texts: {chunk_key: chunk_text}
        - doc_info: {doc_id: {"case_name": ..., "case_number": ..., ...}}
    """
    if not doc_ids:
        return {}, {}

    with sync_session_factory() as session:
        result = session.execute(
            select(LegalDocument).where(LegalDocument.id.in_(doc_ids))
        )
        docs = {doc.id: doc for doc in result.scalars().all()}

    chunk_texts = {}
    doc_info = {}

    for doc_id, positions in chunk_positions.items():
        doc = docs.get(doc_id)
        if doc:
            text = doc.embedding_text or ""
            for start, end in positions:
                key = f"{doc_id}_{start}_{end}"
                chunk_texts[key] = text[start:end] if text else ""

            # 문서 정보 저장
            doc_info[doc_id] = {
                "case_name": doc.case_name or "",
                "case_number": doc.case_number or "",
                "court_name": doc.court_name or "",
                "doc_type": doc.doc_type or "",
            }

    return chunk_texts, doc_info


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
    chunk_texts, doc_info = _fetch_document_texts(list(doc_ids), chunk_positions)

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

        # PostgreSQL에서 가져온 문서 정보로 메타데이터 보강
        if doc_id and doc_id in doc_info:
            info = doc_info[doc_id]
            metadata = {
                **metadata,
                "case_name": info.get("case_name", ""),
                "case_number": info.get("case_number", metadata.get("case_number", "")),
                "court_name": info.get("court_name", metadata.get("court_name", "")),
                "doc_type": info.get("doc_type", metadata.get("doc_type", "")),
            }

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

    # 2. 검색된 문서에서 참조 법령 추출
    doc_ids = [
        doc.get("metadata", {}).get("doc_id")
        for doc in relevant_docs
        if doc.get("metadata", {}).get("doc_id")
    ]
    reference_articles_text = fetch_reference_articles_from_docs(doc_ids)
    law_names = extract_law_names(reference_articles_text)

    # 3. 법령 조회
    related_laws = fetch_laws_by_names(law_names, limit=3) if law_names else []

    # 4. 컨텍스트 구성
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

    # 법령 정보 추가
    for i, law in enumerate(related_laws, len(relevant_docs) + 1):
        context_parts.append(
            f"[참고자료 {i} - 법령]\n"
            f"법령명: {law.get('law_name', '')}\n"
            f"유형: {law.get('law_type', '')}\n"
            f"내용: {law.get('content', '')[:500]}\n"
        )

    context = "\n".join(context_parts)

    # 3. LangChain 메시지 구성
    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    # 이전 대화 기록 추가
    if chat_history:
        for msg in chat_history[-10:]:  # 최근 10개만
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

    # 사용자 메시지 + 컨텍스트
    user_prompt = f"""사용자 질문: {user_message}

---
참고 자료:
{context if context else "(관련 자료 없음)"}
---

위 참고 자료를 바탕으로 사용자의 질문에 답변해주세요."""

    messages.append(HumanMessage(content=user_prompt))

    # 4. LLM 호출 (LangChain ChatModel)
    llm = get_chat_model(temperature=0.7)
    response = llm.invoke(messages)

    assistant_message = response.content

    # 5. 결과 반환
    sources = [
        {
            "case_name": doc.get("metadata", {}).get("case_name", ""),
            "case_number": doc.get("metadata", {}).get("case_number", ""),
            "doc_type": doc.get("metadata", {}).get("doc_type", ""),
            "similarity": round(doc.get("similarity", 0), 3),
            "content": doc.get("content", ""),  # 전문
        }
        for doc in relevant_docs
    ]

    # 법령 sources 추가
    for law in related_laws:
        sources.append({
            "law_name": law.get("law_name", ""),
            "law_type": law.get("law_type", ""),
            "doc_type": "law",
            "similarity": 1.0,  # 정확 매칭
            "content": law.get("content", ""),  # 법령 전문
        })

    return {
        "response": assistant_message,
        "sources": sources,
    }
