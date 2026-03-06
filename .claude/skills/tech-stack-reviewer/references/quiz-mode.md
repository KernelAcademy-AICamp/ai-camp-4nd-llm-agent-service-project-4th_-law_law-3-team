# 퀴즈 모드 상세 가이드

사용자가 "퀴즈", "테스트", "문제"를 언급하면 이 가이드를 따른다.

## 퀴즈 유형 3가지

### 1. 코드 빈칸 채우기

실제 프로젝트 코드에서 핵심 부분을 `___`로 가리고 사용자에게 채우게 한다.

**진행 방식:**
1. Read로 실제 파일을 연다
2. 핵심 줄을 `___`로 대체하여 보여준다
3. 힌트를 제공한다 (선택적)
4. 사용자 답변 후 실제 코드와 비교하며 설명한다

**예시 (LangGraph):**
```python
# graph.py에서 발췌
def build_graph() -> StateGraph[Any]:
    builder = StateGraph(___)              # Q1: 상태 타입은?
    builder.add_edge(___, "router_node")   # Q2: 시작점 상수는?
    return builder
```
정답: `ChatState`, `START`

**예시 (FastAPI):**
```python
# main.py에서 발췌
app = ___(____)                            # Q1: 프레임워크와 lifespan 파라미터
@app.get("___")                            # Q2: 헬스체크 경로
async def health():
    return {"status": "ok"}
```

**예시 (SQLAlchemy):**
```python
# database.py에서 발췌
engine = create_async_engine(
    settings.DATABASE_URL,
    ___=True,                              # Q1: SQL 로그 출력 옵션
)
async_session = ___(engine)                # Q2: 세션 팩토리 함수
```

### 2. 에러 시나리오

"이 코드에서 X를 빼면/바꾸면 어떤 일이 일어날까?" 형태의 사고 실험.

**진행 방식:**
1. 실제 코드를 보여주고 변경 시나리오를 제시한다
2. 사용자가 예상한 결과를 말한다
3. 실제로 어떤 에러/동작 변경이 일어나는지 설명한다

**예시 문제 풀:**

| 시나리오 | 관련 기술 | 난이도 |
|---------|----------|--------|
| `builder.add_edge(START, "router_node")` 삭제하면? | LangGraph | 중 |
| `router_node`에서 Command 대신 dict 반환하면? | LangGraph | 상 |
| `ChatState`에서 `total=False`를 `total=True`로 바꾸면? | TypedDict | 중 |
| `async_session`에서 `expire_on_commit=False` 제거하면? | SQLAlchemy | 상 |
| `next.config.js`에서 rewrites 프록시 제거하면? | Next.js | 하 |
| `StreamWriter`를 안 쓰고 직접 return하면? | LangGraph | 중 |
| `AGENT_NODE_MAP`에 새 에이전트를 안 넣으면? | 라우팅 | 하 |
| MeCab userdic 없이 "손해배상청구권"을 토크나이징하면? | MeCab | 중 |
| ONNX 세션을 매 요청마다 새로 만들면? | ONNX | 중 |
| `checkpointer`를 None으로 compile하면? | LangGraph | 하 |

### 3. 설계 과제

"만약 ~를 추가해야 한다면?" 형태의 열린 질문.

**진행 방식:**
1. 과제를 제시한다
2. 사용자가 설계를 말한다
3. 프로젝트의 기존 패턴과 비교하며 피드백한다
4. 실제로 수정해야 할 파일 목록을 보여준다

**예시 과제 풀:**

| 과제 | 관련 기술 | 난이도 |
|------|----------|--------|
| 새 에이전트 `tax_agent` 추가 | LangGraph, 모듈 시스템 | 중 |
| 규칙 기반 라우터를 LLM 기반으로 교체 | LangGraph, LangChain | 상 |
| RAG에 새 데이터 소스 (뉴스 기사) 추가 | 인제스트, LanceDB | 중 |
| 실시간 알림 기능 추가 (WebSocket) | FastAPI, Next.js | 상 |
| 사용자 인증 (JWT) 도입 | FastAPI, Next.js | 상 |
| 검색 결과 캐싱 레이어 추가 | FastAPI, Redis | 중 |
| 새 프론트엔드 모듈 페이지 추가 | Next.js, 모듈 시스템 | 하 |
| BM25 인덱스를 Elasticsearch로 교체 | PostgreSQL, 검색 | 상 |

## 퀴즈 진행 규칙

1. 사용자가 기술을 지정하면 해당 기술 퀴즈를, 지정하지 않으면 랜덤으로 낸다
2. 한 번에 1문제씩 출제한다
3. 사용자 답변 후 정답 + 설명을 보여주고 "다음 문제?" 물어본다
4. 난이도를 점진적으로 올린다 (하 → 중 → 상)
5. 틀린 문제의 관련 코드를 Read로 열어서 함께 복습한다
