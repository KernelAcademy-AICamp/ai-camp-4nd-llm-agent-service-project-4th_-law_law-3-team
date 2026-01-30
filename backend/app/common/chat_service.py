"""
RAG 기반 법률 챗봇 서비스

VectorStore에서 관련 문서를 검색하고 LLM으로 응답 생성
LangChain/LangGraph 호환 구조
"""

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Set

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from openai import OpenAI
from sqlalchemy import or_, select

from app.common.database import sync_session_factory
from app.common.llm import get_chat_model
from app.common.vectorstore import get_vector_store
from app.core.config import settings
from app.models.law import Law
from app.models.legal_document import LegalDocument
from app.models.precedent_document import PrecedentDocument

# ============================================================================
# 상수 정의
# ============================================================================
MAX_CONTENT_PREVIEW_LENGTH = 500
MAX_CITED_STATUTES_DISPLAY = 3
MAX_SIMILAR_CASES_DISPLAY = 2
MAX_CHAT_HISTORY_LENGTH = 10
MAX_CONTENT_TRUNCATE_LENGTH = 1000

# 그래프 서비스 (lazy import로 Neo4j 연결 실패 시에도 동작)
_graph_service = None
_graph_service_init_attempted = False


def _get_graph_service():
    """
    GraphService lazy 로드

    한번 초기화 시도 후 실패하면 재시도하지 않음 (성능 최적화)
    """
    global _graph_service, _graph_service_init_attempted

    if _graph_service_init_attempted:
        return _graph_service

    _graph_service_init_attempted = True

    try:
        from app.common.graph_service import get_graph_service
        service = get_graph_service()
        if service.is_connected:
            _graph_service = service
        else:
            logger.warning("GraphService 연결 실패: Neo4j 서버에 연결할 수 없습니다")
    except Exception as e:
        logger.warning("GraphService 로드 실패: %s", e)

    return _graph_service


if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# 임베딩 모델 관련 상수
MODEL_CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "models"
_embedding_model_available = None  # None = 체크 안함, True/False = 체크 완료
_embedding_model_warning_shown = False  # 경고 메시지 출력 여부


class EmbeddingModelNotFoundError(Exception):
    """임베딩 모델이 캐시되지 않았을 때 발생하는 예외"""

    pass


def _get_model_cache_path(model_name: str) -> Path:
    """모델 캐시 경로 반환 (HuggingFace 캐시 구조)"""
    # 모델명 변환: "nlpai-lab/KURE-v1" -> "models--nlpai-lab--KURE-v1"
    sanitized = model_name.replace("/", "--")
    return MODEL_CACHE_DIR / f"models--{sanitized}"


def is_embedding_model_cached(model_name: Optional[str] = None) -> bool:
    """
    임베딩 모델이 로컬에 캐시되어 있는지 확인

    Args:
        model_name: 모델명 (기본값: settings.LOCAL_EMBEDDING_MODEL)

    Returns:
        True if 모델이 완전히 다운로드됨, False otherwise
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

    # 최소한 하나의 스냅샷이 있어야 함
    snapshots = list(snapshots_dir.iterdir())
    return len(snapshots) > 0


def check_embedding_model_availability() -> bool:
    """
    임베딩 모델 사용 가능 여부 확인 (서버 시작 시 호출)

    모델이 없으면 경고 로그 출력 (한 번만)
    """
    global _embedding_model_available, _embedding_model_warning_shown

    # 이미 체크한 경우 캐시된 결과 반환
    if _embedding_model_available is not None:
        return _embedding_model_available

    if not settings.USE_LOCAL_EMBEDDING:
        _embedding_model_available = True
        return True

    _embedding_model_available = is_embedding_model_cached()

    # 경고는 한 번만 출력 (print로 콘솔에 명확하게 표시)
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

    Raises:
        EmbeddingModelNotFoundError: 모델이 캐시되지 않았을 때
    """
    global _embedding_model_available

    # 첫 호출 시 캐시 상태 확인 (서버 시작 시 check 안했을 경우 대비)
    if _embedding_model_available is None:
        _embedding_model_available = is_embedding_model_cached()

    if not _embedding_model_available:
        raise EmbeddingModelNotFoundError(
            f"임베딩 모델 '{settings.LOCAL_EMBEDDING_MODEL}'이 캐시되지 않았습니다. "
            "먼저 'uv run python scripts/download_models.py'를 실행해주세요."
        )

    from sentence_transformers import SentenceTransformer

    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    return SentenceTransformer(
        settings.LOCAL_EMBEDDING_MODEL,
        cache_folder=str(MODEL_CACHE_DIR),
        trust_remote_code=True,  # KURE-v1 모델 요구사항
        local_files_only=True,  # 다운로드 방지, 로컬 캐시만 사용
    )


def create_query_embedding(query: str) -> List[float]:
    """쿼리 텍스트를 임베딩 벡터로 변환"""
    if settings.USE_LOCAL_EMBEDDING:
        model = get_local_model()
        embedding = model.encode(
            query,
            show_progress_bar=False,
            normalize_embeddings=True,  # 코사인 유사도를 위한 정규화
        )
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


def fetch_laws_by_names(law_names: Set[str], limit: int = 5) -> List[dict[str, Any]]:
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
                "content": (
                    law.content[:MAX_CONTENT_TRUNCATE_LENGTH] if law.content else ""
                ),
            }
            for law in laws
        ]


def fetch_precedent_details(source_ids: List[str]) -> dict:
    """
    source_id 목록으로 PostgreSQL에서 판례 상세 정보 조회

    Args:
        source_ids: 판례 serial_number 목록

    Returns:
        {serial_number: {ruling, claim, reasoning, full_reason}} 딕셔너리
    """
    if not source_ids:
        return {}

    with sync_session_factory() as session:
        result = session.execute(
            select(PrecedentDocument).where(
                PrecedentDocument.serial_number.in_(source_ids)
            )
        )
        precedents = result.scalars().all()

        return {
            p.serial_number: {
                "ruling": p.ruling or "",
                "claim": p.claim or "",
                "reasoning": p.reasoning or "",
                "full_reason": p.full_reason or "",
            }
            for p in precedents
        }


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


def _fetch_document_texts(doc_ids: List[int], chunk_positions: dict[int, list[tuple[int, int]]]) -> tuple[dict[str, str], dict[int, dict[str, str]]]:
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
        docs: dict[int, LegalDocument] = {int(doc.id): doc for doc in result.scalars().all()}

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
                "case_name": str(doc.case_name or ""),
                "case_number": str(doc.case_number or ""),
                "court_name": str(doc.court_name or ""),
                "doc_type": str(doc.doc_type or ""),
            }

    return chunk_texts, doc_info


def _map_data_type(data_type: str) -> str:
    """LanceDB data_type을 표준 doc_type으로 변환"""
    mapping = {
        "판례": "precedent",
        "법령": "law",
        "헌법재판소": "constitutional",
    }
    return mapping.get(data_type, data_type.lower() if data_type else "")


def _get_chunk_content(store, chunk_id: str, source_id: Optional[str] = None) -> str:
    """
    청크 ID로 content 조회

    1차: LanceDB에서 조회
    2차: LanceDB 비어있으면 PostgreSQL에서 fallback
    """
    # 1. LanceDB에서 조회
    try:
        result = store.get_by_id(chunk_id)
        if result:
            content = result.get("content", "")
            if content:
                return content
    except Exception as e:
        logger.debug(f"LanceDB 청크 조회 실패: {chunk_id}, {e}")

    # 2. PostgreSQL fallback (source_id가 있는 경우)
    if source_id:
        try:
            with sync_session_factory() as session:
                result = session.execute(
                    select(LegalDocument.embedding_text)
                    .where(LegalDocument.serial_number == source_id)
                )
                row = result.scalar_one_or_none()
                if row:
                    return row
        except Exception as e:
            logger.debug(f"PostgreSQL fallback 실패: {source_id}, {e}")

    return ""


def search_relevant_documents(
    query: str,
    n_results: int = 5,
    doc_type: Optional[str] = None,
) -> List[dict[str, Any]]:
    """
    관련 법률 문서 검색

    Args:
        query: 검색 쿼리
        n_results: 반환할 결과 수
        doc_type: 문서 유형 필터 (precedent, constitutional, etc.)

    Returns:
        관련 문서 목록
    """
    store = get_vector_store()

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

    # 결과 정리 - LanceDB 메타데이터 필드명 매핑
    documents = []
    for i, chunk_id in enumerate(results["ids"][0]):
        raw_metadata = results["metadatas"][0][i] if results.get("metadatas") else {}

        # LanceDB 필드명 → 표준 필드명 매핑
        # LanceDB: title, data_type, source_name, source_id
        # 표준: case_name, doc_type, court_name, doc_id
        metadata = {
            "case_name": raw_metadata.get("title", ""),
            "case_number": raw_metadata.get("case_number", ""),
            "doc_type": _map_data_type(raw_metadata.get("data_type", "")),
            "court_name": raw_metadata.get("source_name", ""),
            "doc_id": raw_metadata.get("source_id"),
            "date": raw_metadata.get("date", ""),
            # 원본 메타데이터 유지
            "chunk_index": raw_metadata.get("chunk_index", 0),
            "total_chunks": raw_metadata.get("total_chunks", 1),
        }

        # content 가져오기 (LanceDB → PostgreSQL fallback)
        source_id = raw_metadata.get("source_id")
        content = _get_chunk_content(store, chunk_id, source_id)

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
    chat_history: Optional[List[dict[str, Any]]] = None,
    n_context_docs: int = 5,
) -> dict[str, Any]:
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

    # 4. 그래프 컨텍스트 보강 (Neo4j)
    graph_service = _get_graph_service()
    graph_contexts = []  # 그래프에서 가져온 추가 정보

    if graph_service and graph_service.is_connected:
        for doc in relevant_docs:
            metadata = doc.get("metadata", {})
            case_number = metadata.get("case_number", "")
            if case_number:
                # 판례가 인용한 법령 정보 조회
                case_context = graph_service.enrich_case_context(case_number)
                if case_context.get("cited_statutes"):
                    graph_contexts.append({
                        "case_number": case_number,
                        "cited_statutes": case_context["cited_statutes"],
                        "similar_cases": case_context.get("similar_cases", []),
                    })

    # 5. 컨텍스트 구성
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
            f"내용: {doc.get('content', '')[:MAX_CONTENT_PREVIEW_LENGTH]}\n"
        )

    # 법령 정보 추가
    for i, law in enumerate(related_laws, len(relevant_docs) + 1):
        context_parts.append(
            f"[참고자료 {i} - 법령]\n"
            f"법령명: {law.get('law_name', '')}\n"
            f"유형: {law.get('law_type', '')}\n"
            f"내용: {law.get('content', '')[:MAX_CONTENT_PREVIEW_LENGTH]}\n"
        )

    # 그래프 컨텍스트 추가 (인용 법령, 유사 판례)
    if graph_contexts:
        context_parts.append("\n[그래프 분석 정보]")
        for gc in graph_contexts:
            case_num = gc["case_number"]
            cited = gc.get("cited_statutes", [])
            similar = gc.get("similar_cases", [])

            if cited:
                statute_names = ", ".join(
                    s.get("name", "") for s in cited[:MAX_CITED_STATUTES_DISPLAY]
                )
                context_parts.append(f"- {case_num} 판례가 인용한 법령: {statute_names}")

            if similar:
                similar_nums = ", ".join(
                    s.get("case_number", "") for s in similar[:MAX_SIMILAR_CASES_DISPLAY]
                )
                context_parts.append(f"- {case_num}와 유사한 판례: {similar_nums}")

    context = "\n".join(context_parts)

    # 6. LangChain 메시지 구성
    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    # 이전 대화 기록 추가
    if chat_history:
        for msg in chat_history[-MAX_CHAT_HISTORY_LENGTH:]:
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

    # 7. LLM 호출 (LangChain ChatModel)
    llm = get_chat_model(temperature=0.7)
    response = llm.invoke(messages)

    assistant_message = response.content

    # 8. 결과 반환
    # 판례 상세 정보 조회 (역할별 차등 표시용)
    source_ids = [
        doc.get("metadata", {}).get("doc_id")
        for doc in relevant_docs
        if doc.get("metadata", {}).get("doc_type") == "precedent"
        and doc.get("metadata", {}).get("doc_id")
    ]
    precedent_details = fetch_precedent_details(source_ids) if source_ids else {}

    sources = []
    for doc in relevant_docs:
        metadata = doc.get("metadata", {})
        case_number = metadata.get("case_number", "")
        doc_type = metadata.get("doc_type", "")
        source_id = metadata.get("doc_id", "")

        source_item = {
            "case_name": metadata.get("case_name", ""),
            "case_number": case_number,
            "doc_type": doc_type,
            "court_name": metadata.get("court_name", ""),
            "similarity": round(doc.get("similarity", 0), 3),
            "content": doc.get("content", ""),
        }

        # 판례 상세 정보 추가 (역할별 차등 표시용)
        if doc_type == "precedent" and source_id in precedent_details:
            details = precedent_details[source_id]
            source_item["ruling"] = details.get("ruling", "")
            source_item["claim"] = details.get("claim", "")
            source_item["reasoning"] = details.get("reasoning", "")
            source_item["full_reason"] = details.get("full_reason", "")

        # 그래프 정보 추가 (인용 법령)
        for gc in graph_contexts:
            if gc["case_number"] == case_number:
                source_item["cited_statutes"] = [
                    s.get("name", "")
                    for s in gc.get("cited_statutes", [])[:MAX_CITED_STATUTES_DISPLAY]
                ]
                source_item["similar_cases"] = [
                    s.get("case_number", "")
                    for s in gc.get("similar_cases", [])[:MAX_SIMILAR_CASES_DISPLAY]
                ]
                break

        sources.append(source_item)

    # 법령 sources 추가
    for law in related_laws:
        sources.append({
            "law_name": law.get("law_name", ""),
            "law_type": law.get("law_type", ""),
            "doc_type": "law",
            "similarity": 1.0,  # 정확 매칭
            "content": law.get("content", ""),
        })

    return {
        "response": assistant_message,
        "sources": sources,
    }
