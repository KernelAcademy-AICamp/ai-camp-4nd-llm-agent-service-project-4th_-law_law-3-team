# Skills & Agents Catalog

스킬 30개, 에이전트 5개, 규칙 6개의 분류 및 의존관계 인덱스.

> 최종 업데이트: 2026-02-20

---

## 카테고리 요약

| 카테고리 | 스킬 수 | 에이전트 수 | 설명 |
|---------|--------|-----------|------|
| [워크플로우](#1-워크플로우) | 4 | - | 커밋, 리뷰, 문서 동기화, 플랜 검토 |
| [코드 품질](#2-코드-품질) | 5 | 1 | 검증, 코딩 표준, TDD, 에러 처리 |
| [프론트엔드](#3-프론트엔드) | 6 | - | React/Next.js, 성능, UI/UX, 모의 법정 |
| [RAG/검색](#4-rag검색) | 4 | 1 | RAG 패턴, 평가, 실험 추적, 인제스트 |
| [데이터/DB](#5-데이터db) | 4 | - | PostgreSQL, Alembic, 위치검색, 요약감사 |
| [도메인 지식](#6-도메인-지식) | 2 | - | 한국 법률, Neo4j 그래프 |
| [멀티에이전트](#7-멀티에이전트) | 2 | - | LangGraph 패턴, 디버깅 |
| [외부 CLI](#8-외부-cli) | 3 | 1 | Gemini, Codex, CLI 조합 |
| [운영](#9-운영) | - | 2 | 의존성 감사, E2E 테스트 |

---

## 1. 워크플로우

일상적인 개발 사이클에서 반복 사용하는 스킬.

| 스킬 | 줄 수 | 트리거 | 설명 |
|------|------|--------|------|
| `project-commit` | 151 | `/commit` | 변경 분석 → 정적 검증 → 머지/충돌 해결 → 커밋 |
| `project-review` | 64 | `/review` | staged 변경 코드 리뷰 |
| `update-docs` | 292 | `/update-docs` | 코드 변경 후 문서 동기화 + 크기 가드레일 |
| `plan-review` | 203 | 플랜 작성 후 | Gemini/Codex CLI로 플랜 교차 검토 |

**의존관계**: `project-review` → `project-commit` → `update-docs` (순서)

---

## 2. 코드 품질

코드 작성/수정 시 항상 적용되는 표준.

| 스킬 | 줄 수 | 적용 시점 | 설명 |
|------|------|----------|------|
| `code-verification` | 463 | 코드 수정 후 | 빌드/린트/타입체크 + API 동기화 검증 |
| `python-coding-standards` | 298 | Python 작성 시 | AI 친화적 코딩 가이드, PEP 8 |
| `tdd-methodology` | 145 | 기능 구현 시 | Red-Green-Refactor, 테스트 전략 |
| `error-handling-patterns` | 626 | 에러 처리 구현 시 | FastAPI + Next.js 표준 에러 패턴 |
| `api-contract-sync` | 199 | API 변경 시 | Pydantic ↔ TypeScript 타입 동기화 |

| 에이전트 | 설명 |
|---------|------|
| `code-refactorer` | 코드 리팩토링 자동 수행 |

**규칙 연동**: `rules/code-verification.md`, `rules/coding-style.md`

> **참고**: `code-verification` 스킬은 동명 rule의 상세 가이드 버전. rule = 요약, skill = 실행 절차 포함.

---

## 3. 프론트엔드

React/Next.js 컴포넌트 개발, UI 디자인, 모의 법정 Phaser.js.

| 스킬 | 줄 수 | 적용 시점 | 설명 |
|------|------|----------|------|
| `react-nextjs-frontend` | 180 | 컴포넌트 작성 시 | 접근성, Tailwind, FastAPI 연동 |
| `vercel-react-best-practices` | 120 | 성능 최적화 시 | 45개 규칙, 8개 카테고리 |
| `ui-ux-pro-max` | 377 | UI 디자인 시 | 67개 스타일, 96개 팔레트 |
| `phaser-nextjs-integration` | 598 | Phaser 코드 수정 시 | Next.js + Phaser 통합 패턴 |
| `court-eventbus-patterns` | 665 | EventBus 수정 시 | Phaser ↔ React 통신 + 에러 복구 |
| `court-dialog-system` | 753 | 대화/캐릭터 수정 시 | 말풍선, 배심원, 단계 시스템 |

**관계**:
- `react-nextjs-frontend` (기본) → `vercel-react-best-practices` (성능) → `ui-ux-pro-max` (디자인)
- `react-nextjs-frontend` (기본) → `phaser-nextjs-integration` (Phaser 확장) → `court-eventbus-patterns` (통신) → `court-dialog-system` (게임 로직)

---

## 4. RAG/검색

RAG 파이프라인 구현, 평가, 데이터 적재.

| 스킬 | 줄 수 | 적용 시점 | 설명 |
|------|------|----------|------|
| `langchain-rag-patterns` | 601 | RAG 구현 시 | LangChain/LangGraph RAG 패턴 |
| `rag-evaluation-workflow` | 182 | RAG 변경 후 | Recall/MRR/NDCG 자동 평가 |
| `legal-rag-experiment-tracking` | 219 | 실험 기록 시 | 실험 메타데이터 템플릿 |
| `ingest-pipeline` | 135 | 데이터 적재 시 | 19개 타입 벡터 임베딩/FTS |

| 에이전트 | 설명 |
|---------|------|
| `rag-quality-monitor` | query_rewrite → retrieval → rerank 품질 측정 |

**흐름**: `ingest-pipeline` → `langchain-rag-patterns` → `rag-evaluation-workflow` → `legal-rag-experiment-tracking`

---

## 5. 데이터/DB

PostgreSQL 마이그레이션, 공간 쿼리, 데이터 품질.

| 스킬 | 줄 수 | 적용 시점 | 설명 |
|------|------|----------|------|
| `postgresql-migration` | 138 | JSON→DB 전환 시 | ORM, Alembic, 배치 로드, Feature Flag |
| `alembic-migration-safety` | 166 | 마이그레이션 작성 시 | 롤백/데이터 손실/인덱스 검증 |
| `spatial-query-patterns` | 133 | 위치 검색 시 | Bounding Box, Haversine (PostGIS 없이) |
| `summary-quality-audit` | 179 | 요약 검증 시 | LLM 요약 품질 감사 프로토콜 |

**규칙 연동**: `rules/database-operations.md`
**의존관계**: `postgresql-migration` → `alembic-migration-safety` (마이그레이션 작성 후 안전성 검증)

---

## 6. 도메인 지식

한국 법률 도메인 특화 지식.

| 스킬 | 줄 수 | 적용 시점 | 설명 |
|------|------|----------|------|
| `korean-legal-domain` | 443 | 법률 데이터 처리 시 | 법령 XML, 판례 구조, 청킹 전략 |
| `neo4j-graph-construction` | 357 | 그래프 DB 작업 시 | 법령 계급, 판례 인용, 에이전트 연동 |

**관계**: `korean-legal-domain` (데이터 구조) → `neo4j-graph-construction` (그래프 저장)

---

## 7. 멀티에이전트

LangGraph 기반 멀티 에이전트 시스템.

| 스킬 | 줄 수 | 적용 시점 | 설명 |
|------|------|----------|------|
| `multi-agent-patterns` | 797 | 에이전트 추가/수정 시 | BaseChatAgent, StateGraph, Command |
| `langgraph-debugging` | 225 | 에이전트 디버깅 시 | 라우팅/상태 전파 디버깅 |

**관계**: `multi-agent-patterns` (구현) → `langgraph-debugging` (디버깅)

---

## 8. 외부 CLI

외부 AI CLI 도구 활용 패턴.

| 스킬 | 줄 수 | 적용 시점 | 설명 |
|------|------|----------|------|
| `gemini-cli-delegation` | 249 | 대규모 분석 시 | 1M 토큰, 멀티모달, Deep Think |
| `codex-cli-delegation` | 381 | 코드 리뷰/실행 시 | /review, 샌드박스, 웹 검색 |
| `multi-cli-integration` | 496 | CLI 조합 시 | Claude+Gemini+Codex 복합 워크플로우 + 결정 흐름도 |

| 에이전트 | 설명 |
|---------|------|
| `ai-orchestrator` | CLI 설치 확인 → 최적 도구 선택 → 실행 |

**규칙 연동**: `rules/cli-tool-routing.md`
**관계**: `gemini-cli-delegation` + `codex-cli-delegation` → `multi-cli-integration` (상위 조합)

---

## 9. 운영

배포 전 검증, 의존성 관리.

| 에이전트 | 설명 |
|---------|------|
| `dependency-auditor` | CVE 스캔, 라이선스/버전 호환성 |
| `e2e-scenario-tester` | 판례검색→변호사찾기→소액소송 E2E 시나리오 |

---

## 규칙 (Rules) 목록

| 규칙 | 적용 범위 | 설명 |
|------|----------|------|
| `code-verification.md` | 코드 변경 후 | 정적 검증 프로토콜 |
| `coding-style.md` | 코드 작성 시 | 네이밍, 파일 구조, 금지 사항 |
| `database-operations.md` | DB 작업 시 | 마이그레이션 절차, Feature Flag |
| `git-convention.md` | 커밋/PR 시 | Conventional Commits, 브랜치 전략 |
| `cli-tool-routing.md` | CLI 사용 시 | 도구 선택 매트릭스, Fallback |
| `wsl2-docker.md` | Docker 실행 시 | WSL2 환경 docker.exe 규칙 |

---

## 스킬 크기 분포

```
700+ 줄 : multi-agent-patterns (797), court-dialog-system (753)
600+ 줄 : court-eventbus-patterns (665), error-handling-patterns (626)
           langchain-rag-patterns (601), phaser-nextjs-integration (598)
400+ 줄 : multi-cli-integration (496), code-verification (463), korean-legal-domain (443)
300+ 줄 : codex-cli-delegation (381), ui-ux-pro-max (377), neo4j-graph-construction (357)
200+ 줄 : python-coding-standards (298), update-docs (292), gemini-cli-delegation (249)
           langgraph-debugging (225), legal-rag-experiment-tracking (219), plan-review (203)
           api-contract-sync (199), rag-evaluation-workflow (182), react-nextjs-frontend (180)
           summary-quality-audit (179), alembic-migration-safety (166), project-commit (151)
100+ 줄 : tdd-methodology (145), postgresql-migration (138), ingest-pipeline (135)
           spatial-query-patterns (133), vercel-react-best-practices (120)
 ~64 줄 : project-review (64)
```

**총 줄 수**: ~9,836줄 (평균 328줄/스킬, 30개)
