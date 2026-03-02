# Triple Review Workflow (3중 검증 워크플로우)

기획 및 구현 작업 시 이 워크플로우를 **항상(ALWAYS)** 따라야 합니다.

## 팀 구성 (7명 + 외부 2)

### 내부 Agent Team (7명)

| 파트 | 역할 | Agent # | 모델 | 책임 |
|------|------|---------|------|------|
| **Product** | PM (리드) | #1 | `opus` | 기획 조율, 요구사항 관리, 우선순위 결정, 팀 조율, 최종 의사결정 |
| **Design** | UI/UX 디자이너 | #2 | `sonnet` | 사용자 경험 설계, UI 컴포넌트 설계, 접근성, 디자인 시스템 |
| **Engineering** | 백엔드 개발자 | #3 | `sonnet` | API 설계/구현, DB 스키마, 서버 로직, 성능 최적화 |
| **Engineering** | 프론트엔드 개발자 | #4 | `sonnet` | UI 구현, 상태 관리, API 연동, 반응형 레이아웃 |
| **Engineering** | AI/ML 엔지니어 | #5 | `sonnet` | RAG 파이프라인, LangGraph 에이전트, 임베딩/리랭커, 프롬프트 엔지니어링 |
| **Ops/QA** | QA 엔지니어 | #6 | `sonnet` | 테스트 전략, 갭 분석, 코드 품질 검증, 버그 탐지 |
| **Ops/QA** | 데브옵스 엔지니어 | #7 | `sonnet` | 인프라 설계, CI/CD, Docker, 배포 전략, 모니터링 |

### 외부 검증 (변경 없음)

| 역할 | 담당 | 도구 | 책임 |
|------|------|------|------|
| **Red Team** | 보안/기술 전문가 | Gemini CLI | 보고서/코드 검증, 취약점 분석, 고급 기능 제안 |
| **External Consultant** | 외부 컨설턴트 | Codex CLI | 최종 검증, 컨설팅 보고서, 최고 수준 기능 제안 |

### 모델 배정 원칙

- **PM만 Opus 사용** — 의사결정, 조율, 최종 확인 등 리드 역할
- **나머지 6명은 Sonnet 사용** — 설계, 구현, 분석, 검증 등 실행 역할
- **단순 탐색 작업**은 Haiku 사용 가능 (비용 절감)

## 보안 사전 검증 (외부 CLI 호출 전 필수)

외부 CLI(Gemini CLI, Codex CLI)에 코드/보고서를 전달하기 전 **반드시** 다단계 보안 스캔 수행:

### 필수 스캔 절차 (QA 엔지니어 담당)

```bash
# 1단계: 파일 deny-list 필터링 (절대 전달 금지 파일)
# .env, *.pem, credentials.json, rclone.conf, secrets/ 디렉토리
# → 대상 파일 목록에서 자동 제외

# 2단계: 시크릿 패턴 스캔
grep -rn "API_KEY\|SECRET\|PASSWORD\|TOKEN\|PRIVATE_KEY\|Bearer " {대상_파일} || echo "SAFE"

# 3단계: 의존성 보안 점검 (새 라이브러리 도입 시)
# Backend: uv run pip-audit (Python)
# Frontend: npm audit (Node.js)
```

### 스캔 기준

| 검출 항목 | 조치 |
|----------|------|
| 시크릿/토큰 하드코딩 | 해당 파일 제외, 시크릿 제거 후 재포함 |
| PII (개인정보) | 마스킹 처리 후 전달 |
| 민감 설정 파일 | 절대 전달 금지 |
| 의존성 취약점 (Critical/High) | DevOps 엔지니어에게 보고, 채택 보류 |

## 파트별 협업 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                    PM (Product Part) - Opus                     │
│         기획 조율 · 요구사항 관리 · 최종 의사결정               │
├─────────────┬───────────────────────────┬───────────────────────┤
│ Design Part │    Engineering Part       │    Ops/QA Part        │
│             │                           │                       │
│ UI/UX       │ Backend   Frontend  AI/ML │ QA        DevOps     │
│ 디자이너    │ 개발자    개발자    엔지니어│ 엔지니어  엔지니어   │
│ (Sonnet)    │ (Sonnet)  (Sonnet) (Sonnet)│ (Sonnet)  (Sonnet)  │
├─────────────┴───────────────────────────┴───────────────────────┤
│               Red Team (Gemini CLI) - 외부 검증                 │
├─────────────────────────────────────────────────────────────────┤
│          External Consultant (Codex CLI) - 외부 검증            │
└─────────────────────────────────────────────────────────────────┘
```

## 파트별 담당 영역 상세

### Product Part (PM)

| 담당 | 설명 |
|------|------|
| 요구사항 분석 | 사용자 요구를 기능 명세로 변환 |
| 우선순위 결정 | 기능/작업 우선순위 판단 |
| 팀 조율 | 파트 간 의존성 관리, 작업 배분 |
| 이해관계자 소통 | 사용자 승인 요청, 진행 보고 |
| 최종 의사결정 | 기술적 쟁점 발생 시 최종 판단 |
| 외부 검증 관리 | Red Team/Consultant 결과 수용 여부 최종 결정 |

### Design Part (UI/UX 디자이너)

| 담당 | 설명 |
|------|------|
| 사용자 경험 설계 | 사용자 흐름, 인터랙션 패턴 |
| UI 컴포넌트 설계 | 컴포넌트 구조, 레이아웃, 디자인 시스템 |
| 접근성 검토 | WCAG 기준 접근성 확보 |
| 프로토타입 | 목업/와이어프레임 제작 |
| 프론트엔드 협업 | 프론트엔드 개발자와 긴밀히 협력하여 설계 구현 |

### Engineering Part (백엔드 + 프론트엔드 + AI/ML)

| 역할 | 담당 | 설명 |
|------|------|------|
| 백엔드 개발자 | API/서버 | FastAPI 엔드포인트, DB 스키마, 서비스 로직, 성능 최적화 |
| 프론트엔드 개발자 | UI 구현 | Next.js 컴포넌트, 상태 관리, API 연동, 반응형 |
| AI/ML 엔지니어 | AI 파이프라인 | LangGraph 에이전트, RAG 검색, 임베딩/리랭커, 프롬프트 |

Engineering 파트 내 협업 규칙:
- **API 계약**: 백엔드-프론트엔드 간 스키마 변경 시 양쪽 동시 확인
- **AI 연동**: AI/ML 엔지니어의 에이전트 출력 형식은 백엔드가 API로 노출, 프론트엔드가 표시
- **코드 리뷰**: Engineering 파트 내 상호 리뷰 권장

### Ops/QA Part (QA + DevOps)

| 역할 | 담당 | 설명 |
|------|------|------|
| QA 엔지니어 | 품질 보증 | 테스트 전략, 갭 분석, 코드 품질, 버그 탐지, 회귀 테스트, **보안 사전 검증**, **의존성 보안 점검** |
| 데브옵스 엔지니어 | 인프라/배포 | Docker 구성, CI/CD, 배포 전략, 모니터링, 성능 계측, **의존성 취약점 스캔** |

## Fast-track 워크플로우 (저위험 변경용)

다음 **고위험 영역에 해당하지 않는** 변경에 한해, 전체 3중 검증 대신 Fast-track을 사용할 수 있음:

### 고위험 영역 (Fast-track 불가 — 정규 워크플로우 필수)

| 영역 | 예시 |
|------|------|
| 인증/권한 | 로그인, 토큰, 접근 제어 변경 |
| 개인정보/데이터 접근 | 사용자 데이터 처리, DB 스키마 변경 |
| 프롬프트 체인 | LangGraph 에이전트, 시스템 프롬프트 변경 |
| 외부 연동 | 외부 API 호출, 결제, 메일 발송 |
| 새로운 아키텍처 패턴 | 기존 패턴에 없는 새로운 구조 도입 |

### Fast-track 대상 (위 고위험 영역에 해당하지 않는 경우)

- 기존 패턴 내 단순 변경 (버그 수정, UI 텍스트, 스타일)
- 문서/설정 변경
- 테스트 추가/수정

**Fast-track 절차:**
1. PM이 변경 영향도 확인 → 고위험 영역 해당 없음 확인 → Fast-track 승인
2. 해당 파트 담당자가 구현
3. QA 엔지니어가 코드 리뷰 + 보안 스캔
4. PM 최종 확인

> Fast-track에서도 **자동 테스트 + 보안 스캔 + 롤백 계획**은 필수.
> Fast-track 여부는 PM이 판단. 의심스러우면 정규 워크플로우 진행.

## Phase 1: 기획/계획 단계 워크플로우

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Agent Team → 단계별 보고서 작성                     │
│          PM(Opus): 요구사항 분석, 작업 배분, 조율             │
│          UI/UX(Sonnet): UX 관점 요구사항, 사용자 흐름         │
│          Backend(Sonnet): API/DB 설계 관점 분석               │
│          Frontend(Sonnet): UI 구현 관점 분석                  │
│          AI/ML(Sonnet): AI 파이프라인 관점 분석               │
│          QA(Sonnet): 테스트 전략, 품질 기준 수립              │
│          DevOps(Sonnet): 인프라/배포 관점 분석                │
├─────────────────────────────────────────────────────────────┤
│  Step 1.5: QA 엔지니어 → Pre-mortem 분석 (선택적)            │
│          - 계획의 잠재적 실패 지점 사전 식별                  │
│          - 기술적 리스크, 의존성 충돌, 일정 리스크 평가       │
│          - PM에게 리스크 보고서 제출                          │
├─────────────────────────────────────────────────────────────┤
│  Step 2: Gemini CLI (Red Team) → 보고서 검증                 │
│          - 취약점 분석                                       │
│          - 현업 수준 고급 기술/기능 추가 제안                 │
│          - Red Team 보고서 작성                               │
├─────────────────────────────────────────────────────────────┤
│  Step 3: Agent Team → Red Team 보고서 분석                   │
│          - 각 파트별 담당 영역의 피드백 반영                  │
│          - 계획 수정 및 보고서 업데이트                       │
│          - PM 최종 확인                                      │
├─────────────────────────────────────────────────────────────┤
│  Step 4: Codex CLI (External Consultant) → 보고서 검증       │
│          - 취약점 분석                                       │
│          - 최고 수준 서비스를 위한 고급 기술/기능 추가 제안   │
│          - 컨설팅 보고서 작성                                 │
├─────────────────────────────────────────────────────────────┤
│  Step 5: Agent Team → 컨설팅 보고서 분석                     │
│          - 각 파트별 담당 영역의 피드백 반영                  │
│          - 계획 수정 및 보고서 최종 업데이트                  │
│          - PM 최종 확인 및 결정                               │
├─────────────────────────────────────────────────────────────┤
│  Step 6: 사용자 승인 요청                                    │
└─────────────────────────────────────────────────────────────┘
```

## Phase 2: 기술 구현/코딩 단계 워크플로우

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Agent Team → 코드 구현                              │
│          PM: 작업 분배 및 진행 관리                           │
│          UI/UX: 컴포넌트 설계 명세 제공                      │
│          Backend: API/서버 코드 구현                          │
│          Frontend: UI 컴포넌트 구현                           │
│          AI/ML: 에이전트/RAG 파이프라인 구현                  │
│          QA: 구현 중 품질 기준 모니터링                       │
│          DevOps: 인프라/배포 구성                             │
├─────────────────────────────────────────────────────────────┤
│  Step 2: Gemini CLI (Red Team) → 코드 리뷰                   │
│          - 오류/취약점 발견 → Agent Team이 수정               │
├─────────────────────────────────────────────────────────────┤
│  Step 3: Codex CLI (External Consultant) → 코드 리뷰 (읽기 전용) │
│          - 리뷰 결과를 Agent Team에게 전달                     │
│          - `codex exec "..." --ephemeral -s read-only` (필수)  │
│          - ⚠ `--full-auto` 사용 금지 (#7852 좀비 프로세스)     │
│          - ⚠ 프롬프트 200줄 이하 필수 (#11122 출력 붕괴)       │
│          - ⚠ 순차 실행 필수 (#11435 세션 충돌)                 │
│          - 코드 수정은 Claude Agent Team이 직접 수행            │
├─────────────────────────────────────────────────────────────┤
│  Step 4: Agent Team → Codex 리뷰 결과 분석 및 코드 수정        │
│          - Codex 리뷰에서 지적한 이슈 분석                    │
│          - 각 파트별 담당 영역의 이슈를 해당 역할이 평가      │
│          - 유효한 지적만 채택하여 Agent Team이 직접 코드 수정  │
│          - 구현 계획 및 기획 보고서 업데이트                  │
│          - PM 최종 확인                                      │
└─────────────────────────────────────────────────────────────┘
```

## 팀 내 합의 불가 시 조율 프로토콜

에이전트 팀 내에서 합의가 되지 않는 토픽/문제 발생 시:

1. **PM**이 쟁점 정리 및 각 파트의 입장 수렴
2. **Gemini CLI**에 교차 질문 → 외부 의견 수렴
3. **Codex CLI**에 교차 질문 → 외부 의견 수렴
4. 3개 소스의 의견을 종합하여 **PM**이 최종 결정

```bash
# Gemini CLI 교차 질문 템플릿
gemini "다음 기술적 쟁점에 대해 분석하고 추천 방향을 제시해주세요.
쟁점: {논쟁_주제}
Option A: {옵션A_설명}
Option B: {옵션B_설명}
프로젝트 컨텍스트: FastAPI + Next.js + LangGraph 법률 서비스 플랫폼
평가 기준: 성능, 확장성, 유지보수성, 보안, 사용자 경험" -y -o text

# Codex CLI 교차 질문 템플릿 (비대화형 exec 서브커맨드 사용)
codex exec "다음 기술적 쟁점에 대해 분석하고 추천 방향을 제시해주세요.
쟁점: {논쟁_주제}
Option A: {옵션A_설명}
Option B: {옵션B_설명}
프로젝트 컨텍스트: FastAPI + Next.js + LangGraph 법률 서비스 플랫폼
평가 기준: 성능, 확장성, 유지보수성, 보안, 사용자 경험" --ephemeral -s read-only
```

## CLI 호출 규칙

### Gemini CLI (Red Team) 보고서 요청 템플릿

```bash
gemini "당신은 Red Team 보안/기술 전문가입니다.
다음 보고서를 검증하고 Red Team 보고서를 작성해주세요.

검토 관점:
1. 보안 취약점 분석 (OWASP Top 10 기준)
2. 아키텍처 약점 및 단일 장애점
3. 현업 수준 서비스를 위한 고급 기술/기능 추가 제안
4. 성능 병목 지점 및 확장성 이슈
5. 운영 관점의 모니터링/로깅/알림 부재 항목

출력 형식:
## Red Team 검증 보고서
### 1. 취약점 (Critical/High/Medium/Low)
### 2. 아키텍처 개선 제안
### 3. 고급 기능 추가 제안
### 4. 성능 최적화 제안
### 5. 운영 안정성 제안

---
$(cat {보고서_경로})
---" -y -o text
```

### Codex CLI (External Consultant) 보고서 요청 템플릿

```bash
# ⚠ Claude Code Bash 환경에서는 반드시 `codex exec` 서브커맨드 사용
codex exec "당신은 외부 기술 컨설턴트입니다.
다음 보고서를 검증하고 컨설팅 보고서를 작성해주세요.

검토 관점:
1. 업계 최고 수준 서비스 대비 격차 분석
2. 최신 기술 트렌드 반영 여부
3. 사용자 경험(UX) 최적화 방안
4. 데이터 파이프라인 및 AI/ML 고도화 방안
5. 비즈니스 확장성 및 수익화 전략

출력 형식:
## 외부 컨설팅 보고서
### 1. 격차 분석 (현재 vs 목표)
### 2. 기술 고도화 제안
### 3. UX 개선 제안
### 4. AI/ML 고도화 제안
### 5. 비즈니스 전략 제안

---
$(cat {보고서_경로})
---" --ephemeral -s read-only
```

### Gemini CLI (Red Team) 코드 리뷰 템플릿

```bash
gemini "당신은 시니어 코드 리뷰어이자 보안 전문가입니다.
다음 코드를 리뷰하고 상세한 리뷰 보고서를 작성해주세요.

리뷰 관점:
1. 보안 취약점 (인젝션, XSS, CSRF, 인증/인가 취약점)
2. 버그 및 논리적 오류
3. 성능 이슈 및 최적화 기회
4. 코드 품질 및 유지보수성
5. 에러 핸들링 및 엣지 케이스

$(for f in {파일_목록}; do echo '--- '$f' ---'; cat $f; done)
" -y -o text
```

### Codex CLI (External Consultant) 코드 리뷰 + 직접 수정 템플릿

**방법 A: uncommitted 변경 자동 리뷰 + 수정** (권장)
```bash
# Codex가 uncommitted 변경사항을 리뷰하고 직접 수정
codex review --uncommitted
```

**방법 B: 특정 파일 대상 리뷰** (비대화형, 읽기 전용)
```bash
# ⚠ --full-auto 사용 금지 (#7852 좀비 프로세스 위험)
# ⚠ --ephemeral 필수 (#11435 세션 충돌 방지)
# ⚠ 프롬프트 200줄 이하 (#11122 출력 형식 붕괴 방지)
codex exec "당신은 외부 기술 컨설턴트이자 코드 품질 전문가입니다.
다음 파일들을 리뷰하고 개선점을 제시해주세요. (코드 수정은 하지 말고 리뷰만 수행)

리뷰 관점:
1. 아키텍처 패턴 및 설계 원칙 준수
2. 테스트 가능성 및 테스트 커버리지
3. 타입 안전성 및 API 계약 정합성
4. 확장성 및 모듈화 수준
5. 업계 모범 사례 대비 개선점

대상 파일: {파일_목록}" --ephemeral -s read-only
# → 코드 수정은 Agent Team이 리뷰 결과를 분석한 후 직접 수행
```

**방법 C: 리뷰만 (수정 없이)**
```bash
# codex review: 비대화형 리뷰 전용 서브커맨드
codex review --uncommitted

# codex exec + 읽기 전용 샌드박스
codex exec "리뷰만 수행. 코드 수정하지 말 것. ..." --ephemeral -s read-only
```

### Codex 리뷰 결과 분석 및 Agent Team 수정 프로토콜

> **중요**: Codex CLI는 `--ephemeral -s read-only`로만 실행하므로 코드를 직접 수정하지 않습니다.
> 코드 수정은 Agent Team이 리뷰 결과를 분석한 후 직접 수행합니다.

```bash
# 1. Codex 리뷰 출력을 보고서로 저장
# → docs/03-analysis/{feature}.code-review-consulting.md

# 2. Agent Team이 리뷰 결과 분석
# 각 파트별 담당 영역의 지적 사항을 해당 역할이 평가:
#   - Backend 지적 → 백엔드 개발자 평가
#   - Frontend 지적 → 프론트엔드 개발자 + UI/UX 디자이너 평가
#   - AI/ML 지적 → AI/ML 엔지니어 평가
#   - 인프라/배포 지적 → 데브옵스 엔지니어 평가
#   - 전체 품질 → QA 엔지니어 평가
# 평가 기준:
#   - 지적이 기존 아키텍처 패턴과 일관성 있는가?
#   - 코딩 스타일 규칙(coding-style.md) 관점에서 유효한가?
#   - 프로젝트 컨텍스트에 적합한 제안인가?

# 3. 유효한 지적만 채택하여 Agent Team이 직접 코드 수정

# 4. 수정 후 검증
# Backend: uv run ruff check + mypy
# Frontend: npm run build
```

## 보고서 저장 경로

| 보고서 유형 | 저장 경로 |
|------------|----------|
| 기획 보고서 (Agent Team) | `docs/01-plan/features/{feature}.plan.md` |
| 설계 보고서 (Agent Team) | `docs/02-design/features/{feature}.design.md` |
| Red Team 보고서 | `docs/03-analysis/{feature}.redteam.md` |
| 컨설팅 보고서 | `docs/03-analysis/{feature}.consulting.md` |
| 최종 분석 보고서 | `docs/03-analysis/{feature}.analysis.md` |
| 완료 보고서 | `docs/04-report/{feature}.report.md` |
| 코드 리뷰 (Red Team) | `docs/03-analysis/{feature}.code-review-redteam.md` |
| 코드 리뷰 (Consultant) | `docs/03-analysis/{feature}.code-review-consulting.md` |

## Agent Team 구성 (7명)

| Agent # | 파트 | 역할 | 모델 | subagent_type |
|---------|------|------|------|---------------|
| #1 | Product | **PM** (리드/조율/결정) | `opus` | `bkit:product-manager` |
| #2 | Design | UI/UX 디자이너 | `sonnet` | `bkit:frontend-architect` |
| #3 | Engineering | 백엔드 개발자 | `sonnet` | `bkit:bkend-expert` |
| #4 | Engineering | 프론트엔드 개발자 | `sonnet` | `bkit:frontend-architect` |
| #5 | Engineering | AI/ML 엔지니어 | `sonnet` | `bkit:enterprise-expert` |
| #6 | Ops/QA | QA 엔지니어 | `sonnet` | `bkit:qa-strategist` |
| #7 | Ops/QA | 데브옵스 엔지니어 | `sonnet` | `bkit:infra-architect` |

### subagent_type 선택 가이드

작업 성격에 따라 subagent_type을 유연하게 조정 가능:

| 역할 | 기본 subagent_type | 대안 |
|------|-------------------|------|
| PM | `bkit:product-manager` | `bkit:cto-lead` (기술 의사결정 비중 높을 때) |
| UI/UX 디자이너 | `bkit:frontend-architect` | `general-purpose` (목업/프로토타입 작업) |
| 백엔드 개발자 | `bkit:bkend-expert` | `bkit:enterprise-expert` (마이크로서비스) |
| 프론트엔드 개발자 | `bkit:frontend-architect` | `general-purpose` (복합 작업) |
| AI/ML 엔지니어 | `bkit:enterprise-expert` | `general-purpose` (RAG/프롬프트 작업) |
| QA 엔지니어 | `bkit:qa-strategist` | `bkit:code-analyzer`, `bkit:gap-detector` |
| 데브옵스 엔지니어 | `bkit:infra-architect` | `bkit:security-architect` (보안 중심) |

## 승인 체계 (다중 게이트)

PM 단독 결정의 병목을 방지하기 위해, 영역별 승인 게이트를 운영:

| 승인 유형 | 승인자 | 적용 시점 |
|----------|--------|----------|
| 기획/요구사항 승인 | PM | Phase 1 완료 후 |
| 기술 설계 승인 | PM + 해당 Engineering 담당자 | 설계 확정 시 |
| 품질 게이트 승인 | QA 엔지니어 | 구현 완료 후, 외부 리뷰 전 |
| 배포 승인 | DevOps 엔지니어 | 배포 전 인프라 준비 확인 |
| 최종 릴리스 승인 | PM (위 3개 게이트 통과 후) | 사용자 전달 전 |

> PM이 최종 결정권을 가지되, 각 영역 전문가의 게이트 승인을 거쳐야 함.

## 외부 CLI 실행 실패 시 처리

Red Team(Gemini CLI) 또는 External Consultant(Codex CLI) 실행이 실패할 경우:

1. **자동 건너뛰기/Fallback 금지** — 기본 동작은 **중단 및 재시도**
2. **사용자에게 즉시 보고** — 실패한 CLI, 원인, 에러 메시지를 명확히 설명
3. **사용자 지침에 따라 진행** — `AskUserQuestion`으로 다음 선택지 제시:
   - CLI 문제 해결 후 재시도 **(기본 권장)**
   - Claude Agent Team이 해당 역할을 대체 수행
   - 해당 단계 건너뛰기 **(사용자 명시 승인 필수 + 리스크 기록)**
   - 작업 중단

> 건너뛰기를 선택한 경우, 보고서에 "외부 검증 미수행" 사유와 리스크를 기록.
> 상세 절차: `cli-tool-routing.md`의 "CLI 실행 실패 시 필수 절차" 참조

## 필수 원칙

1. **각 단계별 보고서 작성 필수** - 단계를 건너뛸 수 없음
2. **다중 승인 체계** - PM이 최종 결정하되, 기술/품질/배포 게이트는 해당 전문가가 승인
3. **외부 CLI 결과 무조건 수용 금지** - Agent Team이 분석 후 유효한 제안만 채택
4. **Codex 수정은 diff 아카이빙 + 파트별 평가 필수** - 되돌리기 전 diff 보존, `git restore` 사용
5. **사용자 승인** - 기획 완료 후 구현 전환 시 반드시 사용자 승인
6. **보고서 버전 관리** - 수정 시 이전 내용 덮어쓰되, 주요 변경 이력은 보고서 하단에 기록
7. **외부 CLI 실패 시 중단 우선** - 기본은 중단+재시도, 건너뛰기는 사용자 명시 승인 + 리스크 기록
8. **파트별 전문성 존중** - 각 파트의 담당 영역은 해당 역할의 전문가가 1차 판단, PM이 최종 조율
9. **보안 사전 검증 필수** - 외부 CLI 호출 전 다단계 보안 스캔 수행, QA 엔지니어 담당
10. **의존성 보안 점검** - 새 라이브러리 도입 시 QA/DevOps가 `pip-audit`/`npm audit`로 확인 후 채택
