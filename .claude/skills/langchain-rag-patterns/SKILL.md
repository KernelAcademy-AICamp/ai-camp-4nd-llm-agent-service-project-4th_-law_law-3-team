---
name: langchain-rag-patterns
description: LangChain/LangGraph 기반 RAG 시스템 구현 패턴. 메시지 타입, LLM 호출, 컨텍스트 구성, 프롬프트 템플릿. RAG 챗봇, LLM 연동 작업 시 사용.
---

# LangChain RAG Patterns

LangChain/LangGraph 기반 RAG 시스템 구현 패턴과 가이드라인.

## 1. 아키텍처 개요

```
사용자 쿼리
     │
     ▼
┌─────────────────┐
│ 임베딩 생성     │  ← create_query_embedding()
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 벡터 검색       │  ← LanceDB / VectorStore
│ (Top-K 문서)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 컨텍스트 구성   │  ← 검색 결과 + 그래프 보강
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ LLM 호출        │  ← get_chat_model()
│ (LangChain)     │
└────────┬────────┘
         │
         ▼
    응답 반환
```

## 2. 핵심 모듈

### 2.1 위치
```
backend/app/common/
├── chat_service.py    # RAG 메인 로직
├── vectorstore.py     # 벡터 DB 추상화
├── graph_service.py   # Neo4j 그래프 서비스
└── llm/
    └── __init__.py    # get_chat_model, get_llm_config
```

### 2.2 Import 패턴
```python
# LLM 사용
from app.common.llm import get_chat_model

# RAG 서비스 사용
from app.common.chat_service import (
    generate_chat_response,
    search_relevant_documents,
    create_query_embedding,
)

# 벡터 스토어 사용
from app.common.vectorstore import get_vector_store

# LangChain 메시지 타입
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
```

## 3. LLM 호출 패턴

### 3.1 get_chat_model 사용
```python
from app.common.llm import get_chat_model

# 기본 사용
llm = get_chat_model()

# 옵션 지정
llm = get_chat_model(
    provider="openai",      # openai | anthropic | google
    model="gpt-4o-mini",    # 모델명 (없으면 settings에서)
    temperature=0.7,        # 생성 온도
)

# 환경 변수로 프로바이더 전환
# .env: LLM_PROVIDER=anthropic
llm = get_chat_model()  # Claude 사용
```

### 3.2 메시지 구성
```python
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

messages: list[BaseMessage] = [
    SystemMessage(content=SYSTEM_PROMPT),
    HumanMessage(content="사용자 질문 1"),
    AIMessage(content="AI 응답 1"),
    HumanMessage(content="사용자 질문 2"),
]

# LLM 호출
response = llm.invoke(messages)
answer = response.content
```

### 3.3 대화 기록 변환
```python
def build_messages(
    system_prompt: str,
    chat_history: list[dict[str, str]] | None,
    user_message: str,
) -> list[BaseMessage]:
    """대화 기록을 LangChain 메시지로 변환"""
    messages = [SystemMessage(content=system_prompt)]

    if chat_history:
        for msg in chat_history[-10:]:  # 최근 10개만
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=user_message))
    return messages
```

## 4. RAG 컨텍스트 구성

### 4.1 기본 컨텍스트 구성
```python
def build_rag_context(documents: list[dict]) -> str:
    """검색된 문서들로 컨텍스트 구성"""
    context_parts = []

    for i, doc in enumerate(documents, 1):
        metadata = doc.get("metadata", {})
        content = doc.get("content", "")[:500]  # 길이 제한

        context_parts.append(
            f"[참고자료 {i}]\n"
            f"유형: {metadata.get('doc_type', '')}\n"
            f"제목: {metadata.get('case_name', '')}\n"
            f"내용: {content}\n"
        )

    return "\n".join(context_parts)
```

### 4.2 그래프 보강 컨텍스트
```python
def enrich_context_with_graph(
    documents: list[dict],
    graph_service,
) -> str:
    """Neo4j 그래프 정보로 컨텍스트 보강"""
    context_parts = []

    for doc in documents:
        case_number = doc.get("metadata", {}).get("case_number")
        if not case_number:
            continue

        # 그래프에서 관련 정보 조회
        graph_context = graph_service.enrich_case_context(case_number)

        if graph_context.get("cited_statutes"):
            statutes = ", ".join(
                s.get("name", "") for s in graph_context["cited_statutes"][:3]
            )
            context_parts.append(f"- {case_number}이 인용한 법령: {statutes}")

        if graph_context.get("similar_cases"):
            similar = ", ".join(
                s.get("case_number", "") for s in graph_context["similar_cases"][:2]
            )
            context_parts.append(f"- 유사 판례: {similar}")

    if context_parts:
        return "\n[그래프 분석 정보]\n" + "\n".join(context_parts)
    return ""
```

### 4.3 전체 프롬프트 구성
```python
def build_user_prompt(query: str, context: str) -> str:
    """사용자 프롬프트 구성"""
    return f"""사용자 질문: {query}

---
참고 자료:
{context if context else "(관련 자료 없음)"}
---

위 참고 자료를 바탕으로 사용자의 질문에 답변해주세요."""
```

## 5. 시스템 프롬프트 패턴

### 5.1 법률 RAG 시스템 프롬프트
```python
SYSTEM_PROMPT = """당신은 한국 법률 전문 AI 어시스턴트입니다.
사용자의 법률 질문에 대해 제공된 판례 및 법률 문서를 참고하여 정확하고 도움이 되는 답변을 제공합니다.

답변 시 주의사항:
1. 제공된 참고 자료를 기반으로 답변하세요
2. 관련 판례가 있다면 사건번호와 함께 인용하세요
3. 법률 용어는 쉽게 설명해주세요
4. 확실하지 않은 내용은 추측하지 말고 "법률 전문가와 상담이 필요합니다"라고 안내하세요
5. 답변은 친절하고 이해하기 쉽게 작성하세요

중요: 이 서비스는 법률 정보 제공 목적이며, 실제 법률 조언을 대체하지 않습니다."""
```

### 5.2 역할별 프롬프트 분리
```python
PROMPTS = {
    "user": """당신은 일반 사용자를 위한 법률 정보 제공 AI입니다.
법률 용어를 쉽게 풀어서 설명하고, 실생활 예시를 들어주세요.""",

    "lawyer": """당신은 변호사를 위한 법률 리서치 AI입니다.
판례 법리, 조문 해석, 판결 요지를 정확하게 제공하세요.
법률 전문 용어를 그대로 사용해도 됩니다.""",
}

def get_system_prompt(user_role: str) -> str:
    return PROMPTS.get(user_role, PROMPTS["user"])
```

## 6. 벡터 검색 패턴

### 6.1 기본 검색
```python
from app.common.vectorstore import get_vector_store
from app.common.chat_service import create_query_embedding

def search_documents(
    query: str,
    n_results: int = 5,
    doc_type: str | None = None,
) -> list[dict]:
    """관련 문서 검색"""
    store = get_vector_store()

    # 쿼리 임베딩 생성
    query_embedding = create_query_embedding(query)

    # 필터 조건
    where = {"doc_type": doc_type} if doc_type else None

    # 검색
    results = store.search(
        query_embedding=query_embedding,
        n_results=n_results,
        where=where,
        include=["metadatas", "distances"],
    )

    return results
```

### 6.2 하이브리드 검색 (벡터 + 키워드)
```python
def hybrid_search(
    query: str,
    n_results: int = 10,
) -> list[dict]:
    """벡터 검색 + 키워드 매칭"""
    # 1. 벡터 검색
    vector_results = search_documents(query, n_results=n_results * 2)

    # 2. 키워드 추출 및 필터링
    keywords = extract_keywords(query)

    # 3. 키워드 매칭 점수 추가
    scored_results = []
    for doc in vector_results:
        keyword_score = calculate_keyword_match(doc["content"], keywords)
        combined_score = doc["similarity"] * 0.7 + keyword_score * 0.3
        scored_results.append({**doc, "combined_score": combined_score})

    # 4. 재정렬
    scored_results.sort(key=lambda x: x["combined_score"], reverse=True)
    return scored_results[:n_results]
```

## 7. 전체 RAG 파이프라인

### 7.1 generate_chat_response 구조
```python
def generate_chat_response(
    user_message: str,
    chat_history: list[dict[str, Any]] | None = None,
    n_context_docs: int = 5,
) -> dict[str, Any]:
    """RAG 기반 챗봇 응답 생성"""

    # 1. 관련 문서 검색
    relevant_docs = search_relevant_documents(user_message, n_results=n_context_docs)

    # 2. 참조 법령 추출 (판례에서)
    law_names = extract_law_names_from_docs(relevant_docs)
    related_laws = fetch_laws_by_names(law_names)

    # 3. 그래프 컨텍스트 보강 (Neo4j)
    graph_contexts = enrich_with_graph(relevant_docs)

    # 4. 컨텍스트 구성
    context = build_context(relevant_docs, related_laws, graph_contexts)

    # 5. 메시지 구성
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *convert_history(chat_history),
        HumanMessage(content=build_user_prompt(user_message, context)),
    ]

    # 6. LLM 호출
    llm = get_chat_model(temperature=0.7)
    response = llm.invoke(messages)

    # 7. 결과 반환
    return {
        "response": response.content,
        "sources": format_sources(relevant_docs, related_laws),
    }
```

## 8. 상수 및 설정

### 8.1 RAG 관련 상수
```python
# 컨텍스트 길이 제한
MAX_CONTENT_PREVIEW_LENGTH = 500
MAX_CONTENT_TRUNCATE_LENGTH = 1000

# 검색 결과 제한
MAX_CITED_STATUTES_DISPLAY = 3
MAX_SIMILAR_CASES_DISPLAY = 2

# 대화 기록 제한
MAX_CHAT_HISTORY_LENGTH = 10
```

### 8.2 문서 타입 매핑
```python
DOC_TYPE_MAPPING = {
    "판례": "precedent",
    "법령": "law",
    "헌법재판소": "constitutional",
}

def map_data_type(data_type: str) -> str:
    """LanceDB data_type을 표준 doc_type으로 변환"""
    return DOC_TYPE_MAPPING.get(data_type, data_type.lower() if data_type else "")
```

## 9. 임베딩 모델 관리

### 9.1 로컬 모델 로드 (캐싱)
```python
from functools import lru_cache

@lru_cache(maxsize=1)
def get_local_model() -> SentenceTransformer:
    """sentence-transformers 모델 로드 (싱글톤)"""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        settings.LOCAL_EMBEDDING_MODEL,
        cache_folder=str(MODEL_CACHE_DIR),
        trust_remote_code=True,
        local_files_only=True,  # 다운로드 방지
    )
```

### 9.2 임베딩 생성
```python
def create_query_embedding(query: str) -> list[float]:
    """쿼리 텍스트를 임베딩 벡터로 변환"""
    if settings.USE_LOCAL_EMBEDDING:
        model = get_local_model()
        embedding = model.encode(
            query,
            show_progress_bar=False,
            normalize_embeddings=True,  # 코사인 유사도용
        )
        return embedding.tolist()
    else:
        # OpenAI 임베딩 fallback
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        response = client.embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input=query,
        )
        return response.data[0].embedding
```

## 10. Lazy 초기화 패턴

### 10.1 외부 서비스 Lazy 로드
```python
_graph_service = None
_graph_service_init_attempted = False

def _get_graph_service():
    """GraphService lazy 로드 (한 번 시도 후 캐싱)"""
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
            logger.warning("GraphService 연결 실패")
    except Exception as e:
        logger.warning("GraphService 로드 실패: %s", e)

    return _graph_service
```

### 10.2 사용 패턴
```python
def generate_response(...):
    # 그래프 서비스 (없어도 동작)
    graph_service = _get_graph_service()

    if graph_service and graph_service.is_connected:
        # 그래프 보강 사용
        graph_context = graph_service.enrich_case_context(case_number)
    else:
        # 그래프 없이 진행
        graph_context = {}
```

## 11. 에러 처리

### 11.1 임베딩 모델 없음
```python
class EmbeddingModelNotFoundError(Exception):
    """임베딩 모델이 캐시되지 않았을 때"""
    pass

def get_local_model():
    if not is_embedding_model_cached():
        raise EmbeddingModelNotFoundError(
            f"임베딩 모델 '{settings.LOCAL_EMBEDDING_MODEL}'이 캐시되지 않았습니다. "
            "먼저 'uv run python scripts/download_models.py'를 실행해주세요."
        )
    # ...
```

### 11.2 검색 결과 없음 처리
```python
def search_relevant_documents(query: str, n_results: int = 5) -> list[dict]:
    results = store.search(...)

    if not results or not results.get("ids") or not results["ids"][0]:
        return []  # 빈 목록 반환

    # 결과 처리
    return documents
```

## 12. LangGraph 연동 (확장)

### 12.1 StateGraph 정의
```python
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], "대화 메시지"]
    context: str
    sources: list[dict]

def create_rag_graph():
    graph = StateGraph(AgentState)

    # 노드 추가
    graph.add_node("retrieve", retrieve_documents)
    graph.add_node("generate", generate_response)

    # 엣지 연결
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()
```

### 12.2 노드 함수
```python
def retrieve_documents(state: AgentState) -> AgentState:
    """문서 검색 노드"""
    query = state["messages"][-1].content
    docs = search_relevant_documents(query)

    return {
        **state,
        "context": build_rag_context(docs),
        "sources": docs,
    }

def generate_response(state: AgentState) -> AgentState:
    """응답 생성 노드"""
    llm = get_chat_model()

    messages = state["messages"] + [
        HumanMessage(content=f"컨텍스트:\n{state['context']}")
    ]

    response = llm.invoke(messages)

    return {
        **state,
        "messages": state["messages"] + [response],
    }
```

## 13. 테스트 패턴

### 13.1 RAG 서비스 테스트
```python
def test_search_relevant_documents():
    results = search_relevant_documents("손해배상 요건", n_results=5)

    assert len(results) > 0
    assert all("content" in doc for doc in results)
    assert all("metadata" in doc for doc in results)

def test_generate_chat_response():
    result = generate_chat_response("민법 제750조 설명해줘")

    assert "response" in result
    assert "sources" in result
    assert len(result["response"]) > 0
```

### 13.2 LLM Mock 테스트
```python
from unittest.mock import MagicMock, patch

@patch("app.common.llm.get_chat_model")
def test_with_mock_llm(mock_get_model):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="테스트 응답")
    mock_get_model.return_value = mock_llm

    result = generate_chat_response("테스트 질문")

    assert result["response"] == "테스트 응답"
    mock_llm.invoke.assert_called_once()
```

## 14. 성능 최적화

### 14.1 배치 임베딩
```python
def create_batch_embeddings(texts: list[str]) -> list[list[float]]:
    """배치 임베딩 생성"""
    model = get_local_model()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return embeddings.tolist()
```

### 14.2 컨텍스트 길이 최적화
```python
def optimize_context(documents: list[dict], max_tokens: int = 3000) -> str:
    """토큰 제한 내에서 컨텍스트 최적화"""
    context_parts = []
    current_tokens = 0

    for doc in documents:
        content = doc.get("content", "")
        doc_tokens = len(content) // 4  # 대략적인 토큰 수

        if current_tokens + doc_tokens > max_tokens:
            # 남은 공간에 맞게 자르기
            remaining = max_tokens - current_tokens
            content = content[:remaining * 4]

        context_parts.append(format_document(doc, content))
        current_tokens += doc_tokens

        if current_tokens >= max_tokens:
            break

    return "\n\n".join(context_parts)
```
