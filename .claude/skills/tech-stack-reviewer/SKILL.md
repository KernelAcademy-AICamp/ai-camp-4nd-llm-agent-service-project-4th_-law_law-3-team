---
name: tech-stack-reviewer
description: 프로젝트에 사용된 기술 스택을 코드 기반으로 학습할 수 있게 리뷰해주는 교육용 스킬. "이 기술이 뭐야?", "프로젝트에서 어떻게 쓰여?", "코드 보면서 공부하고 싶어", "기술 스택 설명해줘", "LangGraph 어떻게 동작해?", "RAG 파이프라인 알려줘", "면접 준비", "포트폴리오 정리", "기술 면접 질문", "퀴즈 내줘", "이 파일 설명해줘", "이 폴더 뭐하는 곳이야", "graph.py 알려줘", "services/rag/ 설명해줘" 등 프로젝트 기술을 배우거나 면접/포트폴리오를 준비할 때 사용. 특정 기술명, 파일 경로, 폴더 경로를 언급하며 학습/이해를 요청하거나, 면접 대비를 요청하면 반드시 이 스킬을 사용할 것.
---

# Tech Stack Reviewer

프로젝트에 사용된 기술 스택을 실제 코드 기반으로 설명하여 학습을 돕고, 포트폴리오/기술 면접까지 대비하는 교육용 스킬.

## 리뷰 모드

사용자의 키워드에 따라 모드를 자동 감지한다. 지정하지 않으면 **학습 모드**로 시작.

| 모드 | 트리거 키워드 | 상세 가이드 |
|------|-------------|-----------|
| **학습 모드** | "공부", "설명", "알려줘" | 이 파일 하단의 리뷰 템플릿 사용 |
| **면접 모드** | "면접", "인터뷰", "질문" | → `references/interview-mode.md` 읽기 |
| **포트폴리오 모드** | "포트폴리오", "정리", "발표" | → `references/portfolio-mode.md` 읽기 |
| **퀴즈 모드** | "퀴즈", "테스트", "문제" | → `references/quiz-mode.md` 읽기 |
| **아키텍처 모드** | "전체 흐름", "아키텍처", "구조" | → `references/architecture.md` 읽기 |
| **파일/폴더 탐색 모드** | 파일 경로 또는 폴더 경로 언급 | 아래 "파일/폴더 탐색 모드" 섹션 참조 |

## 리뷰 원칙

1. **코드 기반 설명**: 실제 파일/코드를 Read 도구로 열어서 보여주며 설명
2. **Why 중심**: 왜 선택했는지, 어떤 문제를 해결하는지, 대안과의 차이
3. **점진적 깊이**: 핵심 개념 → 프로젝트 적용 → 심화 패턴
4. **연결 관계**: 기술 간 연결고리 (예: FastAPI ↔ SQLAlchemy ↔ Alembic)

## 기술 스택 카탈로그

특정 기술을 지정하지 않으면 이 카탈로그를 보여주고 선택하게 한다.

### Backend (Python)

| # | 기술 | 역할 | 핵심 파일 |
|---|------|------|----------|
| 1 | **FastAPI** | 웹 프레임워크, API 서버 | `backend/app/main.py`, `backend/app/core/config.py` |
| 2 | **SQLAlchemy** | ORM, DB 모델 정의 | `backend/app/core/database.py`, `backend/app/models/` |
| 3 | **Alembic** | DB 마이그레이션 | `backend/alembic/`, `backend/alembic/env.py` |
| 4 | **Pydantic** | 데이터 검증, 스키마 | `backend/app/core/config.py`, `backend/app/modules/*/schema/` |
| 5 | **LangGraph** | 멀티에이전트 오케스트레이션 | `backend/app/multi_agent/graph.py`, `nodes.py` |
| 6 | **LangChain** | LLM 호출, 프롬프트 체이닝 | `backend/app/multi_agent/agents/base_chat.py` |
| 7 | **LanceDB** | 벡터 DB, 임베딩 저장/검색 | `backend/app/services/rag/retrieval.py` |
| 8 | **PostgreSQL** | RDBMS, BM25 FTS | `backend/app/core/database.py` |
| 9 | **MeCab** | 한국어 형태소 분석 | `backend/app/services/rag/keyword_search.py` |
| 10 | **ONNX Runtime** | 모델 추론 최적화 | `backend/app/services/rag/onnx_session.py` |
| 11 | **uv** | Python 패키지 매니저 | `backend/pyproject.toml` |
| 12 | **Ruff / mypy** | 린터 / 타입 체커 | `backend/pyproject.toml` |

### Frontend (TypeScript)

| # | 기술 | 역할 | 핵심 파일 |
|---|------|------|----------|
| 13 | **Next.js (App Router)** | React 프레임워크, SSR/SSG | `frontend/src/app/`, `frontend/next.config.js` |
| 14 | **React** | UI 컴포넌트 라이브러리 | `frontend/src/components/`, `frontend/src/features/` |
| 15 | **Tailwind CSS** | 유틸리티 CSS 프레임워크 | `frontend/tailwind.config.ts` |
| 16 | **TypeScript** | 정적 타입 시스템 | `frontend/tsconfig.json` |

### Infra / DevOps

| # | 기술 | 역할 | 핵심 파일 |
|---|------|------|----------|
| 17 | **Docker Compose** | 컨테이너 오케스트레이션 | `docker-compose.yml` |
| 18 | **rclone** | Google Drive 백업/복원 | `scripts/backup_to_gdrive.sh` |

### AI/ML 파이프라인 (복합)

| # | 주제 | 관련 기술 조합 | 핵심 파일 |
|---|------|--------------|----------|
| 19 | **RAG 파이프라인** | LanceDB + BM25 + Reranker + LLM | `backend/app/services/rag/pipeline.py` |
| 20 | **멀티에이전트 시스템** | LangGraph + BaseChatAgent + Router | `backend/app/multi_agent/` |
| 21 | **임베딩/리랭킹** | KURE-v1 + bge-reranker + ONNX | `backend/app/services/rag/embedding.py` |

---

## 학습 모드: 리뷰 템플릿

사용자가 기술을 선택하면 아래 구조로 리뷰한다.

```
## [기술명] 리뷰

### 1. 이 기술은 무엇인가?
- 한 줄 정의
- 어떤 문제를 해결하는 도구인지
- 비유를 통한 설명

### 2. 핵심 개념 (3-5개)

### 3. 이 프로젝트에서 어떻게 쓰이나?
- 아키텍처에서의 위치
- 실제 코드 읽기 (Read 도구로 열어서 주석과 함께)
- 코드 흐름 추적

### 4. 왜 이 기술을 선택했나? (Why This, Not That)
- 대안 비교 + 프로젝트 맥락에서의 선택 근거 + 한계점

### 5. 연결된 다른 기술

### 6. 더 깊이 공부하려면
- 공식 문서, 프로젝트 내 관련 파일, 관련 스킬
```

### 대안 비교 가이드

| 기술 | 대안들 | 비교 포인트 |
|------|--------|-----------|
| LangGraph | AutoGen, CrewAI, 직접 구현 | 상태 관리, 그래프 시각화, 체크포인터 |
| LanceDB | Pinecone, ChromaDB, Milvus | 비용, 스케일, 설치 복잡도 |
| FastAPI | Django, Flask | 비동기, 자동 문서, 타입 검증 |
| Next.js | Vite+React, Remix | SSR, 라우팅, 에코시스템 |
| PostgreSQL | MySQL, MongoDB | FTS, JSON, GIN 인덱스 |
| MeCab | Kiwi, Komoran, Okt | 속도, 사용자 사전, 법률 용어 |
| ONNX Runtime | TorchScript, TensorRT | 플랫폼 호환, ARM 지원 |
| Tailwind CSS | styled-components, Chakra UI | 번들 크기, DX |
| uv | pip, Poetry, PDM | 속도, lockfile |

## 코드 읽기 규칙

1. **Read 도구로 실제 파일을 열어서** 보여준다 (암기 아닌 실제 코드)
2. **핵심 부분만 발췌**, 파일 경로와 줄 번호 명시
3. **한국어 주석**으로 각 줄 설명
4. 긴 코드: **흐름 요약 → 핵심 발췌 → 전체 구조**

## 난이도 조절

| 수준 | 설명 방식 |
|------|----------|
| **입문** | 비유 중심, 코드 최소화, "이게 뭔지"에 집중 |
| **중급** | 코드와 개념 균형, "어떻게 동작하는지" (기본값) |
| **심화** | 코드 위주, 내부 구현, 성능, 트레이드오프 |

## 대화형 학습 지원

- 후속 질문 답변 ("async with가 왜 필요해?")
- 실습 제안 ("비슷한 코드를 작성해볼래?")
- 사고 실험 ("이 부분을 바꾸면 어떻게 될까?")

## 파일/폴더 탐색 모드

사용자가 특정 파일 경로(`graph.py`, `services/rag/pipeline.py` 등)나 폴더 경로(`multi_agent/`, `services/rag/` 등)를 언급하며 설명을 요청하면 이 모드로 진행한다.

### 파일 탐색 진행 방식

1. **Read로 해당 파일을 열고** 전체 구조를 파악한다
2. 아래 템플릿으로 설명한다:

```
## [파일명] 탐색

### 이 파일은 뭘 하는 파일인가?
- 한 줄 요약
- 아키텍처에서의 위치 (어느 레이어에 속하는지)

### 사용된 기술들
- 이 파일에서 import/사용하는 기술 목록
- 각 기술이 이 파일에서 어떤 역할을 하는지

### 핵심 코드 해설
- 주요 함수/클래스를 한국어 주석으로 설명
- 데이터가 들어와서 나가는 흐름

### 연결된 파일들
- 이 파일을 호출하는 곳 (upstream)
- 이 파일이 호출하는 곳 (downstream)
- import 관계 다이어그램

### 이 파일과 관련된 기술 더 알아보기
- 카탈로그의 해당 기술 번호 안내
```

### 폴더 탐색 진행 방식

1. **Glob/Bash로 폴더 구조를 파악**한다
2. 아래 템플릿으로 설명한다:

```
## [폴더명/] 탐색

### 이 폴더의 역할
- 한 줄 요약
- 아키텍처에서의 위치

### 폴더 구조
- 트리 형태로 파일 목록 + 각 파일의 한 줄 설명

### 사용된 기술 맵
- 이 폴더에서 사용하는 기술 목록 (파일별로 매핑)

### 파일 간 관계
- 파일들이 서로 어떻게 연결되는지 흐름도
- 진입점(entry point)과 의존 관계

### 추천 읽기 순서
- 이 폴더를 이해하려면 어떤 파일부터 읽어야 하는지 순서 제안
```

### 연결 파일 추적 규칙

파일/폴더를 설명할 때 반드시 **연결 관계**를 추적한다:
- `import` 문을 Grep으로 검색하여 upstream/downstream 파일을 찾는다
- 해당 파일을 호출하는 곳을 Grep으로 찾아 "누가 이걸 쓰는지" 보여준다
- 연결된 파일도 궁금하면 이어서 탐색할 수 있음을 안내한다

## 관련 스킬 매핑

| 기술 | 관련 스킬 |
|------|----------|
| LangGraph | `langgraph-debugging`, `multi-agent-patterns`, `agent-development-guide` |
| LangChain | `langchain-rag-patterns` |
| FastAPI | `error-handling-patterns`, `api-contract-sync` |
| SQLAlchemy/Alembic | `postgresql-migration`, `alembic-migration-safety` |
| Next.js/React | `react-nextjs-frontend`, `vercel-react-best-practices` |
| PostgreSQL | `postgresql-migration`, `spatial-query-patterns` |
| RAG | `rag-evaluation-workflow`, `legal-rag-experiment-tracking` |
| Docker | `docker-containerization` |
