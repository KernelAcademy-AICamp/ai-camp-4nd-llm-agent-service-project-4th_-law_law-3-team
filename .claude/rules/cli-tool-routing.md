# CLI Tool Routing Rules

Claude는 외부 AI CLI 도구(Gemini CLI, Codex CLI)를 활용할 때 이 규칙들을 **항상(ALWAYS)** 따라야 합니다.

> **관련 스킬**:
> - `.claude/skills/gemini-cli-delegation/SKILL.md` - Gemini CLI 단독 사용 패턴
> - `.claude/skills/codex-cli-delegation/SKILL.md` - Codex CLI 단독 사용 패턴
> - `.claude/skills/multi-cli-integration/SKILL.md` - CLI 조합 패턴

---

## 1. CLI 도구 선택 매트릭스

### 작업 유형별 도구 선택

| 작업 유형 | 1순위 도구 | 2순위 (Fallback) | 판단 기준 |
|----------|-----------|-----------------|----------|
| **대규모 코드 분석** (10+ 파일) | Gemini CLI | Claude (Task/Explore) | 파일 수, 컨텍스트 크기 |
| **프로젝트 아키텍처 파악** | Gemini CLI | Claude (Task/Explore) | 전체 구조 이해 필요 여부 |
| **보안 감사** (다수 파일) | Gemini CLI | Claude (순차 분석) | 분석 범위 |
| **마이그레이션 영향 분석** | Gemini CLI | Claude (Grep/Read) | 변경 영향 범위 |
| **코드 리뷰** (PR/커밋) | Codex CLI | Claude (git diff 분석) | 리뷰 대상 존재 여부 |
| **샌드박스 실행 테스트** | Codex CLI | Claude (코드 설명만) | 실행 필요 여부 |
| **웹 검색 + 코딩** | Codex CLI | Claude (WebSearch) | 외부 정보 필요 여부 |
| **정밀 코드 작성** | Claude | - | 항상 Claude |
| **단일 파일 분석** | Claude | - | 항상 Claude |
| **5개 이하 파일 분석** | Claude | - | 항상 Claude |

### 결정 흐름도

```
사용자 요청 수신
    │
    ├─ 파일 10개 이상 OR 컨텍스트 100K+ 토큰?
    │   ├─ YES → Gemini CLI 설치 확인
    │   │         ├─ 설치됨 → Gemini CLI 실행
    │   │         └─ 미설치 → Claude (Task/Explore + Glob/Grep)
    │   └─ NO → 다음 판단으로
    │
    ├─ 코드 리뷰 요청? (PR, 커밋 diff 분석)
    │   ├─ YES → Codex CLI 설치 확인
    │   │         ├─ 설치됨 → Codex CLI (codex review)
    │   │         └─ 미설치 → Claude (git diff 직접 분석)
    │   └─ NO → 다음 판단으로
    │
    ├─ 샌드박스 실행 필요?
    │   ├─ YES → Codex CLI 설치 확인
    │   │         ├─ 설치됨 → Codex CLI (-s read-only)
    │   │         └─ 미설치 → Claude (실행 불가 안내 + 코드 설명)
    │   └─ NO → 다음 판단으로
    │
    ├─ 웹 검색 + 코딩 필요?
    │   ├─ YES → Codex CLI 설치 확인
    │   │         ├─ 설치됨 → Codex CLI (--search)
    │   │         └─ 미설치 → Claude (WebSearch 도구 사용)
    │   └─ NO → Claude 직접 수행
    │
    └─ 그 외 모든 경우 → Claude 직접 수행
```

---

## 2. 설치 여부 확인 (필수)

외부 CLI를 사용하기 전 **반드시** 설치 여부를 확인합니다.

### 확인 명령어

```bash
# Gemini CLI 확인
which gemini 2>/dev/null && echo "GEMINI_AVAILABLE" || echo "GEMINI_NOT_AVAILABLE"

# Codex CLI 확인
which codex 2>/dev/null && echo "CODEX_AVAILABLE" || echo "CODEX_NOT_AVAILABLE"

# Windows 환경
where gemini 2>nul && echo "GEMINI_AVAILABLE" || echo "GEMINI_NOT_AVAILABLE"
where codex 2>nul && echo "CODEX_AVAILABLE" || echo "CODEX_NOT_AVAILABLE"
```

### 확인 규칙

1. **매 세션 첫 CLI 호출 전** 설치 여부 확인
2. 확인 결과를 세션 내에서 캐싱 (반복 확인 불필요)
3. 미설치 시 사용자에게 안내 후 Fallback 수행

### 미설치 시 사용자 안내 템플릿

```
[CLI명] CLI가 설치되어 있지 않아 Claude가 직접 분석을 수행합니다.
[CLI명] CLI 설치 시 더 빠르고 광범위한 분석이 가능합니다.

진행하시겠습니까?
```

---

## 3. Fallback 전략

### Gemini CLI 미설치 시

| 원래 작업 | Fallback 방법 | 품질 영향 |
|----------|--------------|----------|
| 대규모 파일 분석 | Task(Explore) 에이전트 + Glob/Grep | 중간 (순차 처리로 느림) |
| 아키텍처 파악 | Task(Explore) + 핵심 파일만 읽기 | 중간 (컨텍스트 제한) |
| 보안 감사 | 보안 관련 패턴 Grep + 순차 파일 읽기 | 낮음 (놓칠 수 있음) |
| 마이그레이션 분석 | Grep 패턴 검색 + 영향 파일 목록화 | 중간 |

### Codex CLI 미설치 시

| 원래 작업 | Fallback 방법 | 품질 영향 |
|----------|--------------|----------|
| 코드 리뷰 | `git diff` 출력을 직접 분석 | 낮음 (대부분 동등) |
| 샌드박스 실행 | 실행 불가, 코드 흐름 설명만 제공 | 높음 (실행 검증 불가) |
| 웹 검색 + 코딩 | WebSearch 도구 사용 | 낮음 (대부분 동등) |

### Fallback 품질 안내 의무

Fallback 수행 시 사용자에게 품질 영향을 반드시 안내합니다:

```
[참고] Gemini CLI 없이 분석을 수행하므로 다음 제한이 있습니다:
- 대규모 컨텍스트를 한번에 처리하지 못해 일부 패턴을 놓칠 수 있습니다
- 분석 시간이 더 소요될 수 있습니다

중요한 보안 감사의 경우 Gemini CLI 설치를 권장합니다.
```

---

## 4. 결과 검증 의무

외부 CLI 실행 후 Claude는 **반드시** 결과를 검증합니다.

### 검증 항목

| 항목 | 검증 방법 | 실패 시 조치 |
|------|----------|-------------|
| 파일 경로 존재 | Glob/Read로 확인 | 잘못된 경로 제거 후 보고 |
| 함수/클래스 존재 | Grep으로 확인 | 존재하지 않는 항목 제거 |
| 코드 문법 정확성 | 린트 실행 | 문법 오류 수정 |
| 보안 취약점 오탐 | 실제 코드 확인 | 오탐 제거 |

### 검증 워크플로우

```
1. CLI 실행 → 결과 수신
2. 결과에서 파일 경로 추출 → 존재 여부 확인
3. 결과에서 함수/클래스명 추출 → 존재 여부 확인
4. 오탐/오류 발견 시 결과에서 제거
5. 검증된 결과만 사용자에게 보고
```

### 검증 보고 형식

```
## CLI 실행 결과 검증

- **실행 도구**: [Gemini CLI / Codex CLI]
- **원본 결과 항목 수**: N개
- **검증 통과 항목 수**: M개
- **제거된 항목**: K개 (이유: 파일 미존재 / 함수 미존재 / 오탐)

### 검증된 결과
[검증 통과한 결과만 표시]
```

---

## 5. 금지 사항

### CLI 실행 시 금지

| 금지 사항 | 이유 |
|----------|------|
| 민감 파일 전달 (`.env`, `credentials.json`, `*.pem`) | 보안 위험 |
| 검증 없이 결과 그대로 사용 | 오탐/오류 가능성 |
| 사용자 동의 없이 CLI 실행 | 외부 서비스 호출이므로 |
| 파일 수정 모드로 CLI 실행 | Claude만 코드 수정 담당 |
| 미설치 CLI를 반복 호출 시도 | 불필요한 에러 |

### 결과 사용 시 금지

| 금지 사항 | 이유 |
|----------|------|
| CLI 결과를 검증 없이 코드에 반영 | 오류 코드 작성 위험 |
| 존재하지 않는 파일/함수 언급 | 사용자 혼란 |
| CLI 오류를 무시하고 진행 | 불완전한 분석 |

---

## 6. 보고 형식

### CLI 사용 보고 (필수)

모든 CLI 사용 후 다음 형식으로 보고합니다:

```
─────────────────────────────────────────────────
📊 CLI 도구 사용 보고
─────────────────────────────────────────────────
🔧 사용 도구: [Gemini CLI / Codex CLI / Fallback(Claude)]
📋 작업 유형: [대규모 분석 / 코드 리뷰 / 샌드박스 실행 / 웹 검색]
✅ 결과 검증: 통과 (N/M 항목)
⚠️ 제한 사항: [있으면 기술, 없으면 "없음"]
─────────────────────────────────────────────────
```

### Fallback 사용 시 추가 보고

```
─────────────────────────────────────────────────
⚠️ Fallback 사용 안내
─────────────────────────────────────────────────
📌 원래 도구: [Gemini CLI / Codex CLI]
📌 Fallback 이유: 미설치 / 네트워크 오류 / 실행 실패
📌 대체 방법: [Task(Explore) / git diff 분석 / WebSearch]
📌 품질 영향: [낮음 / 중간 / 높음]
─────────────────────────────────────────────────
```

---

## 7. 도구별 강점 요약

| 도구 | 강점 | 약점 | 최적 사용 시나리오 |
|------|------|------|------------------|
| **Claude** | 정밀 코딩, 맥락 이해, 오케스트레이션 | 대규모 컨텍스트 제한 | 코드 작성, 계획 수립, 최종 검토 |
| **Gemini CLI** | 1M+ 토큰 컨텍스트, 광범위 분석 | 정밀 코딩 미흡 | 10+ 파일 분석, 아키텍처 파악 |
| **Codex CLI** | 코드 리뷰, 샌드박스 실행, 웹 검색 | 대규모 분석 미흡 | PR 리뷰, 실행 테스트, 웹 연동 |

---

## 8. 에이전트/스킬 연동

이 규칙은 다음 에이전트/스킬과 연동됩니다:

- **ai-orchestrator 에이전트**: CLI 라우팅 결정 트리 실행
- **gemini-cli-delegation 스킬**: Gemini CLI 상세 사용법
- **codex-cli-delegation 스킬**: Codex CLI 상세 사용법
- **multi-cli-integration 스킬**: CLI 조합 워크플로우

---

**중요**: 이 규칙들은 외부 CLI 도구 사용 시 예외 없이 적용됩니다. CLI 도구는 보조 수단이며, 최종 코드 작성과 검증은 항상 Claude가 담당합니다.
