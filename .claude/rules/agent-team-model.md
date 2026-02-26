# Agent Team Model Configuration

에이전트 팀 구성 시 모델 배정 규칙을 **항상(ALWAYS)** 따라야 합니다.

## 모델 배정 규칙

| 역할 | 모델 | Task 도구 `model` 값 |
|------|------|---------------------|
| **리드 (Team Lead)** | Claude Opus 4.6 | `opus` |
| **워커 (Teammate)** | Claude Sonnet 4.6 | `sonnet` |

## 적용 방법

### 1. 리드 에이전트 (orchestrator, CTO 등)

리드 역할의 에이전트는 `model: "opus"`로 생성:

```
Task(subagent_type="bkit:cto-lead", model="opus", ...)
```

### 2. 워커 에이전트 (실행, 분석, 검증 등)

워커 역할의 에이전트는 `model: "sonnet"`으로 생성:

```
Task(subagent_type="bkit:frontend-architect", model="sonnet", ...)
Task(subagent_type="bkit:bkend-expert", model="sonnet", ...)
Task(subagent_type="bkit:code-analyzer", model="sonnet", ...)
Task(subagent_type="bkit:gap-detector", model="sonnet", ...)
Task(subagent_type="bkit:qa-strategist", model="sonnet", ...)
Task(subagent_type="bkit:security-architect", model="sonnet", ...)
```

### 3. 간단한 탐색/검색 작업

빠른 탐색이나 단순 검색은 `model: "haiku"`로 비용 절감 가능:

```
Task(subagent_type="Explore", model="haiku", ...)
```

## 비용 최적화 원칙

- **Opus**: 의사결정, 아키텍처 설계, 복잡한 조율 작업
- **Sonnet**: 구현, 분석, 검증, 코드 리뷰
- **Haiku**: 단순 파일 검색, 패턴 매칭, 빠른 탐색
