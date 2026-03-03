# Agent Team Model Configuration

에이전트 팀 구성 시 모델 배정 규칙을 **항상(ALWAYS)** 따라야 합니다.

## 팀 구성 (7명)

| # | 파트 | 역할 | 모델 | 책임 |
|---|------|------|------|------|
| 1 | Product | **PM (리드)** | `opus` | 기획 조율, 요구사항, 우선순위, 최종 의사결정 |
| 2 | Design | UI/UX 디자이너 | `sonnet` | UX 설계, UI 컴포넌트, 접근성, 디자인 시스템 |
| 3 | Engineering | 백엔드 개발자 | `sonnet` | API, DB, 서버 로직, 성능 최적화 |
| 4 | Engineering | 프론트엔드 개발자 | `sonnet` | UI 구현, 상태 관리, API 연동 |
| 5 | Engineering | AI/ML 엔지니어 | `sonnet` | RAG, LangGraph, 임베딩/리랭커, 프롬프트 |
| 6 | Ops/QA | QA 엔지니어 | `sonnet` | 테스트 전략, 갭 분석, 코드 품질 |
| 7 | Ops/QA | 데브옵스 엔지니어 | `sonnet` | 인프라, CI/CD, Docker, 배포, 모니터링 |

## 모델 배정 규칙

| 역할 | 모델 | Task 도구 `model` 값 |
|------|------|---------------------|
| **PM (리드)** | Claude Opus 4.6 | `opus` |
| **나머지 팀원 (6명)** | Claude Sonnet 4.6 | `sonnet` |

**핵심 규칙: PM의 의사결정 Task는 반드시 Opus. 나머지 팀원은 모두 Sonnet.**
단, 팀원이 수행하는 단순 탐색성 서브태스크(파일 검색, 패턴 매칭)는 Haiku 허용.

## 적용 방법

### 1. PM (Product Part - 리드)

PM은 `model: "opus"`로 생성:

```
Task(subagent_type="bkit:product-manager", model="opus", ...)
```

### 2. Design Part

UI/UX 디자이너는 `model: "sonnet"`으로 생성:

```
Task(subagent_type="bkit:frontend-architect", model="sonnet", ...)
```

### 3. Engineering Part

백엔드, 프론트엔드, AI/ML 엔지니어는 `model: "sonnet"`으로 생성:

```
Task(subagent_type="bkit:bkend-expert", model="sonnet", ...)       # 백엔드
Task(subagent_type="bkit:frontend-architect", model="sonnet", ...)  # 프론트엔드
Task(subagent_type="bkit:enterprise-expert", model="sonnet", ...)   # AI/ML
```

### 4. Ops/QA Part

QA, 데브옵스 엔지니어는 `model: "sonnet"`으로 생성:

```
Task(subagent_type="bkit:qa-strategist", model="sonnet", ...)   # QA
Task(subagent_type="bkit:infra-architect", model="sonnet", ...)  # DevOps
```

### 5. 간단한 탐색/검색 작업

빠른 탐색이나 단순 검색은 `model: "haiku"`로 비용 절감 가능:

```
Task(subagent_type="Explore", model="haiku", ...)
```

## 비용 최적화 원칙

| 모델 | 사용 대상 | 적용 범위 |
|------|----------|----------|
| **Opus** | PM만 사용 | 의사결정, 기획 조율, 최종 확인, 팀 조율 |
| **Sonnet** | 나머지 6명 | 설계, 구현, 분석, 검증, 인프라 (핵심 작업) |
| **Haiku** | 모든 팀원 가능 | 단순 파일 검색, 패턴 매칭, 빠른 탐색 (서브태스크만) |

> Haiku는 팀원의 **탐색성 서브태스크**에만 사용. 핵심 구현/분석/결정 작업에는 사용 금지.

## subagent_type 선택 가이드

| 역할 | 기본 subagent_type | 대안 (작업 성격에 따라) |
|------|-------------------|----------------------|
| PM | `bkit:product-manager` | `bkit:cto-lead` |
| UI/UX 디자이너 | `bkit:frontend-architect` | `general-purpose` |
| 백엔드 개발자 | `bkit:bkend-expert` | `bkit:enterprise-expert` |
| 프론트엔드 개발자 | `bkit:frontend-architect` | `general-purpose` |
| AI/ML 엔지니어 | `bkit:enterprise-expert` | `general-purpose` |
| QA 엔지니어 | `bkit:qa-strategist` | `bkit:code-analyzer`, `bkit:gap-detector` |
| 데브옵스 엔지니어 | `bkit:infra-architect` | `bkit:security-architect` |
