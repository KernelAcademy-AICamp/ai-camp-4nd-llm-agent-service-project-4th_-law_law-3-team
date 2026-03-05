# 아키텍처 모드 상세 가이드

사용자가 "전체 흐름", "아키텍처", "구조"를 언급하면 이 가이드를 따른다.

## End-to-End 요청 흐름

사용자 메시지가 입력되어 응답이 표시되기까지의 전체 과정:

```
[사용자 브라우저]
    | 메시지 입력 + 전송 버튼 클릭
    v
[Next.js Frontend]  <- ChatWidget.tsx
    | SSE 연결: POST /api/chat/stream
    | next.config.js rewrites -> localhost:8000
    v
[FastAPI Backend]  <- app/api/router/chat.py
    | 세션 검증 (session_token 쿠키)
    | ChatRequest -> ChatState 변환
    | graph.astream_events() 호출
    v
[LangGraph StateGraph]  <- multi_agent/graph.py
    |
    +- router_node  <- multi_agent/nodes.py
    |   | RulesRouter.route() -> 키워드 기반 에이전트 선택
    |   | Command(goto="legal_search_node") 반환
    |   v
    +- legal_search_node  <- agents/legal_search_agent.py
    |   |
    |   +- [쿼리 리라이팅]  <- services/rag/query_rewrite.py
    |   +- [벡터 검색]      <- services/rag/retrieval.py -> LanceDB
    |   +- [키워드 검색]    <- services/rag/keyword_search.py -> PostgreSQL BM25
    |   +- [리랭킹]        <- services/rag/rerank.py -> ONNX Runtime
    |   +- [컨텍스트 구성]  <- services/rag/format_utils.py
    |   +- [LLM 응답 생성] <- LangChain ChatOpenAI (스트리밍)
    |       | StreamWriter로 토큰 단위 전송
    |       v
    +- END
    v
[FastAPI]  SSE 이벤트 스트리밍
    v
[Next.js]  EventSource로 실시간 수신 + UI 렌더링
    v
[사용자 브라우저]  응답 표시 + 출처 링크
```

이 흐름을 설명할 때 각 단계의 실제 파일을 Read로 열어서 코드와 함께 보여준다.

## 데이터 흐름도

법률 데이터가 시스템에 들어와서 사용되기까지:

```
[원본 데이터]
    | 법령 XML, 판례 JSON (외부 API/크롤링)
    v
[인제스트 파이프라인]  <- scripts/ingest/cli.py
    | 파싱 -> 정규화 -> 청킹
    +-->  [PostgreSQL] 원본 저장 + search_text (BM25용)
    +-->  [LanceDB] 벡터 임베딩 (KURE-v1, 656K건)
    +-->  [MeCab userdic] 법률 용어 사전 보강 (37K+)
    v
[검색 시점]
    | 사용자 쿼리 입력
    +-->  [LanceDB] 시맨틱 검색 (의미 유사도)
    +-->  [PostgreSQL] BM25 키워드 검색 (정확 매칭)
    +-->  [ONNX Reranker] 결과 재정렬
    +-->  [LLM] 최종 답변 생성
```

## 모듈 시스템 구조

프론트엔드-백엔드 모듈이 어떻게 연결되는지:

```
[Frontend 모듈 정의]
    src/lib/modules.ts         <- 모듈 목록 (id, name, enabled)
    src/lib/api.ts             <- API endpoint 매핑
    next.config.js             <- rewrites 프록시 규칙
    src/features/<module>/     <- 모듈별 컴포넌트/훅/타입
         |
         | HTTP 요청 (프록시 경유)
         v
[Backend 모듈 구현]
    app/modules/<module>/
        router/__init__.py     <- FastAPI 라우터
        schema/                <- Pydantic 스키마
        service/               <- 비즈니스 로직
         |
         | ModuleRegistry 자동 등록
         v
    app/core/registry.py       <- modules/ 폴더 스캔 -> 라우터 등록
```

모듈 추가 시 4곳 동시 수정: `modules.ts`, `api.ts`, `next.config.js`, `router/__init__.py`

## 멀티에이전트 아키텍처

```
[통합 채팅 API]
    POST /api/chat/stream
         |
         v
[LangGraph StateGraph]
         |
    router_node (RulesRouter)
    |    키워드 매칭 -> AgentPlan(agent_type, confidence)
    |    Command(goto=target_node) 반환
    |
    +-- legal_search_node      (RAG: 판례/법령 검색)
    +-- lawyer_finder_node     (변호사 찾기)
    +-- small_claims_subgraph  (소액소송 - 다단계 서브그래프)
    +-- mock_trial_subgraph    (모의 법정 - 서브그래프)
    +-- storyboard_subgraph    (사건 타임라인 - 서브그래프)
    +-- lawyer_stats_node      (변호사 통계)
    +-- law_study_node         (로스쿨 학습)
    +-- content_marketing_node (콘텐츠 마케팅)
    +-- workspace_node         (워크스페이스)
    +-- simple_chat_node       (일반 대화)
         |
         v
       END -> SSE 스트리밍 응답
```

### 에이전트 상속 구조

```
BaseChatAgent (추상 클래스)
    |-- process_stream()   <- 스트리밍 에이전트
    |-- process()          <- 비스트리밍 에이전트
    |
    +-- LegalSearchAgent   (RAG + LLM 스트리밍)
    +-- LawyerFinderAgent  (DB 조회 + 비스트리밍)
    +-- SmallClaimsAgent   (서브그래프 내 다단계)
    +-- MockTrialAgent     (서브그래프, 형사/민사 분기)
    +-- StoryboardAgent    (태그 수집 + 타임라인)
    +-- ...
```

### 라우팅 의사결정 흐름

```
메시지 입력
    |
    +-- agent_override 있음?
    |       |-- Yes: ROLE_AGENTS 검증 -> 허용되면 직접 라우팅
    |       |        단, 강한 의도(confidence >= 0.9)면 override 해제
    |       |-- No: 아래로
    |
    +-- active_agent (세션) 있음?
    |       |-- Yes: 세션 유지 (탈출 키워드 체크)
    |       |        단, 강한 의도(>= 0.9)이면 세션 전환
    |       |-- No: 아래로
    |
    +-- 키워드 매칭 (INTENT_PATTERNS)
    |       |-- 매칭됨: 최고 점수 에이전트 선택
    |       |-- 매칭 없음: GENERAL (일반 채팅)
    |
    +-- AgentPlan(agent_type, confidence, reason) 반환
```

## 설계 결정 진화 기록

면접에서 "설계가 어떻게 진화했나요?"에 답하기 위한 소재.
기술 설명 시 관련 진화 과정을 프로젝트 코드/문서에서 찾아 보여준다:

| 영역 | Before | After | 이유 |
|------|--------|-------|------|
| 라우팅 | if/else 분기 | RulesRouter + Command 패턴 | 에이전트 9개 확장 시 관리 불가 |
| 검색 | 벡터 검색만 | 벡터 + BM25 하이브리드 + 리랭킹 | 키워드 정확 매칭 부족 |
| 모델 추론 | PyTorch 직접 | ONNX QDQ INT8 양자화 | 추론 속도 8.97x 개선 |
| 그래프 DB | Neo4j | PostgreSQL Recursive CTE | 인프라 단순화 (DB 통합) |
| 체크포인터 | InMemorySaver | AsyncPostgresSaver | 서버 재시작 시 대화 유실 방지 |
| 토크나이저 | 기본 MeCab | 법률 userdic 37K 보강 | 복합명사 처리 |

## 기술 레이어 맵

전체 기술 스택을 레이어로 정리:

```
┌─────────────────────────────────────────────┐
│              Presentation Layer              │
│  Next.js App Router + React + Tailwind CSS   │
│  ChatWidget (SSE), 모듈별 페이지             │
├─────────────────────────────────────────────┤
│              API Gateway Layer               │
│  FastAPI + Pydantic + 세션 관리              │
│  next.config.js rewrites (프록시)            │
├─────────────────────────────────────────────┤
│           Orchestration Layer                │
│  LangGraph StateGraph + RulesRouter          │
│  9 에이전트 + 3 서브그래프                   │
├─────────────────────────────────────────────┤
│             AI/ML Layer                      │
│  LangChain (LLM) + RAG Pipeline              │
│  임베딩(KURE-v1) + 리랭커(bge-reranker)     │
│  ONNX Runtime (최적화 추론)                  │
├─────────────────────────────────────────────┤
│              Data Layer                      │
│  PostgreSQL (RDBMS + BM25 FTS)               │
│  LanceDB (벡터 DB, 656K + 240만건)           │
│  MeCab userdic (37K 법률 용어)               │
├─────────────────────────────────────────────┤
│              Infra Layer                     │
│  Docker Compose + rclone (백업)              │
│  uv (패키지) + Ruff/mypy (품질)             │
└─────────────────────────────────────────────┘
```
