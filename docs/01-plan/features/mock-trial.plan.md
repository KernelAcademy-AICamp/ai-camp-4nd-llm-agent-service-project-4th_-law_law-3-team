# Mock Trial (모의 법정) Planning Document

> **Summary**: Phaser.js 픽셀아트 법정 시뮬레이션 — 다중 AI 에이전트(판사/검사/변호사/피고)가 도트 게임 스타일 법정에서 재판을 진행하는 인터랙티브 시뮬레이터
>
> **Project**: law-3-team (법률 서비스 플랫폼)
> **Author**: Claude
> **Date**: 2026-02-12
> **Status**: Draft (v0.6)

---

## 1. Overview

### 1.1 Purpose

사용자가 **도트(픽셀아트) 게임 스타일의 가상 법정**에서 AI 에이전트들과 함께 모의재판을 체험할 수 있는 기능을 구현한다. Stanford의 Generative Agents(Smallville) 논문에서 영감을 받은 2D 픽셀아트 법정 환경에서, SimCourt/Court_Agent 논문의 다중 에이전트 재판 절차를 결합한다.

### 1.2 Background

- **한국 형사 공판절차**: 인정신문 → 검사 모두진술 → 피고인 의견진술 → 증거조사(증거동의/부동의 포함) → 피고인신문 → 검사 구형(의견진술, §302) → 변호인 변론 → 피고인 최후진술(§303) → 판결선고 (형사소송법 제275조~제321조)
- **한국 민사 변론절차**: 변론준비절차(쟁점정리) → 원고 청구원인 진술 → 피고 답변/항변 → 증거조사(서증, 증인신문) → 변론 → 변론종결 → 판결선고 (민사소송법 제134조~)
- **Generative Agents (2304.03442)**: Phaser.js 기반 2D 픽셀아트 샌드박스에서 25개 AI 에이전트가 상호작용하는 시뮬레이션. 스프라이트 캐릭터, 타일맵 환경, Memory/Reflection/Planning 아키텍처
- **SimCourt** (참고): 중국 형사 재판 5단계 LLM 에이전트 시뮬레이션. Profile/Memory/Strategy 모듈, Legal Retriever 아키텍처를 참고 (재판 단계는 한국법 적용)
- **Court_Agent (leehan32)** (참고): LangGraph 상태 머신 기반 모의재판. 변호사 적대적 학습, 벡터 DB 활용 기술을 참고
- 기존 법률 플랫폼의 LangGraph 멀티에이전트 시스템이 검증되어 있음

### 1.3 비전

```
┌─────────────────────────────────────────────────────────────┐
│                 🏛️ 픽셀아트 법정 (Phaser.js)                  │
│                                                              │
│    ┌──────┐                                    ┌──────┐     │
│    │ 👨‍⚖️   │  "피고인은 과실이 인정됩니다..."    │  📋  │     │
│    │ 판사  │  ← 말풍선 (LLM 스트리밍)           │ 기록 │     │
│    └──┬───┘                                    └──────┘     │
│       │                                                      │
│  ┌────┴────────────────────────────────┐                    │
│  │         법정 테이블 (타일맵)          │                    │
│  └────┬────────────────────────┬───────┘                    │
│       │                        │                             │
│  ┌────┴───┐              ┌────┴───┐                         │
│  │ 👨‍💼     │              │ 👩‍💼     │                         │
│  │ 검사   │              │ 변호사  │  ← 사용자 역할          │
│  └────────┘              └────────┘                         │
│                                                              │
│  ┌────────┐                                                  │
│  │ 🧑     │  ← 피고인                                       │
│  │ 피고   │                                                  │
│  └────────┘                                                  │
│                                                              │
│  ───────────────────────────────────────────────────         │
│  [대화 패널] [증거 패널] [법령 검색] [다음 단계 →]            │
│  "검사 측 주장을 입력하세요..."                                │
└─────────────────────────────────────────────────────────────┘
```

### 1.4 핵심 사용자 시나리오

**형사 재판 시나리오**:
```
1. 사용자가 /mock-trial 페이지 접속
2. 픽셀아트 법정 로비에서 사건 유형(형사) + 역할(검사/변호사) 선택
3. 사건 개요 입력 → 법정 씬으로 전환
4. 픽셀아트 캐릭터 배치 (판사, 검사, 변호사, 피고인, 서기)
5. [인정신문] 재판장이 피고인 인적사항 확인, 진술거부권 고지
6. [모두진술] 검사 공소사실 요지 진술 → 피고인/변호인 의견 진술
7. [증거조사] RAG 검색된 판례/법령 제시 + 증인신문
8. [피고인신문] 검사/변호인이 피고인에게 질문
9. [구형 및 최후진술] 검사 의견진술(구형) → 변호인 변론 → 피고인 최후진술
10. [판결선고] AI 판사 판결문 낭독 (말풍선 스트리밍)
11. 결과 피드백: 강점/약점 분석 + 인용 판례/법령 표시
```

**민사 재판 시나리오**:
```
1. 사용자가 /mock-trial 페이지 접속
2. 픽셀아트 법정 로비에서 사건 유형(민사) + 역할(원고/피고) 선택
3. 사건 개요 입력 → 법정 씬으로 전환
4. 픽셀아트 캐릭터 배치 (판사, 원고측, 피고측, 서기)
5. [변론준비] 쟁점 정리, 증거 목록 확인
6. [주장/답변] 원고 청구원인 진술 → 피고 답변/항변
7. [증거조사] RAG 검색 판례/법령 + 서증 제출 + 증인신문
8. [변론] 양측 주장/반박 교환 (2-3 라운드)
9. [변론종결] 양측 최종 주장 정리
10. [판결선고] AI 판사 판결문 낭독 (말풍선 스트리밍)
11. 결과 피드백: 강점/약점 분석 + 인용 판례/법령 표시
```

### 1.5 참고 자료

| 자료 | 핵심 차용 요소 |
|------|---------------|
| [Generative Agents (2304.03442)](https://arxiv.org/abs/2304.03442) | Phaser.js 픽셀아트 2D 환경, 스프라이트 캐릭터, 타일맵 법정 |
| [Court_Agent (leehan32)](https://github.com/leehan32/Court_Agent) | LangGraph 상태 머신, 변호사 적대적 학습, 벤치마크 평가 |
| [SimCourt Review](https://www.themoonlight.io/ko/review/chinese-court-simulation-with-llm-based-agent-system) | Profile/Memory/Strategy 모듈, Legal Retriever (재판 단계는 한국법 적용) |
| [대한민국 법원 형사소송절차](https://help.scourt.go.kr/nm/min_9/min_9_1/index.html) | 한국 형사 공판절차 (인정신문→모두진술→증거조사→피고인신문→최종변론→판결선고) |
| [찾기쉬운 생활법령 - 민사소송](https://easylaw.go.kr/CSP/CnpClsMain.laf?popMenu=ov&csmSeq=568&ccfNo=5&cciNo=3&cnpClsNo=4) | 한국 민사 변론절차 (변론준비→주장/답변→증거조사→변론→변론종결→판결선고) |

### 1.6 Related Documents

- `backend/app/multi_agent/subgraphs/small_claims.py` - interrupt 기반 서브그래프 패턴
- `backend/app/multi_agent/agents/base_chat.py` - BaseChatAgent 추상 클래스
- `backend/app/multi_agent/graph.py` - LangGraph StateGraph 빌드
- `backend/app/multi_agent/router.py` - AgentType enum + INTENT_PATTERNS + RulesRouter
- `backend/app/multi_agent/nodes.py` - AGENT_NODE_MAP (에이전트→노드 매핑)
- `frontend/src/features/small-claims/hooks/useWizardState.ts` - 위자드 상태 관리 훅 패턴

### 1.7 기존 구현 현황 (v0.4 추가)

아래 파일은 **이미 초안이 구현**되어 있으며, Plan/Design 보강 후 리팩토링 대상:

| 파일 | 상태 | 내용 |
|------|------|------|
| `backend/app/multi_agent/subgraphs/mock_trial.py` | 초안 구현 | MockTrialState + 노드 함수 스켈레톤 |
| `backend/app/multi_agent/subgraphs/mock_trial_agents.py` | 초안 구현 | CourtAgent dataclass (generate/reflect/to_state/from_state) |
| `backend/app/multi_agent/subgraphs/mock_trial_prompts.py` | 초안 구현 | 7개 시스템 프롬프트 + AGENT_CONFIGS + SYSTEM_PROMPTS 매핑 |
| `backend/app/multi_agent/router.py` | 등록 완료 | AgentType.MOCK_TRIAL + INTENT_PATTERNS + ROLE_AGENTS |
| `backend/app/multi_agent/nodes.py` | 등록 완료 | AGENT_NODE_MAP["mock_trial"] = "mock_trial_subgraph" |
| `backend/app/multi_agent/graph.py` | 등록 완료 | add_node("mock_trial_subgraph", build_mock_trial_subgraph()) |
| `frontend/src/lib/modules.ts` | 등록 완료 | mock-trial 모듈 정의 |
| `frontend/src/lib/api.ts` | 등록 완료 | endpoints.mockTrial = '/mock-trial' |

**미구현 항목** (Do 단계에서 구현 필요):

| 카테고리 | 미구현 항목 |
|---------|-----------|
| Backend 모듈 | `modules/mock_trial/router/__init__.py` (전용 엔드포인트) |
| Backend 모듈 | `modules/mock_trial/schema/__init__.py` (Pydantic 스키마) |
| Backend 서비스 | `services/service_function/mock_trial_service.py` (RAG 검색) |
| Backend 에이전트 | `agents/mock_trial_agent.py` (MockTrialAgent: BaseChatAgent 상속) |
| Frontend 게임 | `features/mock-trial/game/` (Phaser.js 법정 씬 전체) |
| Frontend 컴포넌트 | `features/mock-trial/components/` (React 오버레이 전체) |
| Frontend 훅 | `features/mock-trial/hooks/useTrialState.ts` (상태 관리) |
| Frontend 서비스 | `features/mock-trial/services/index.ts` (API 호출) |
| Frontend 타입 | `features/mock-trial/types/index.ts` (TypeScript 타입) |
| Frontend 페이지 | `app/mock-trial/page.tsx` (페이지 엔트리) |
| 에셋 | `public/assets/mock-trial/` (픽셀아트 타일맵, 스프라이트) |
| 설정 | `next.config.js` rewrites (API 프록시) |

---

## 2. Scope

### 2.1 In Scope

**Frontend — 픽셀아트 법정 UI (Phaser.js)**
- [ ] Phaser.js 기반 2D 픽셀아트 법정 씬
- [ ] 법정 타일맵 (판사석, 검사석, 변호인석, 피고인석, 방청석)
- [ ] 스프라이트 캐릭터 5종 (판사, 검사, 변호사, 피고인, 서기)
- [ ] 캐릭터 말풍선 (LLM 스트리밍 텍스트 표시)
- [ ] 캐릭터 애니메이션 (대기, 발언, 반응)
- [ ] 하단 UI 패널 (대화 입력, 증거 목록, 법령 검색, 단계 표시)
- [ ] Next.js 내 Phaser.js 통합 (`<canvas>` + React 오버레이)

**Backend — 다중 AI 에이전트 재판 시스템**
- [ ] 5개 에이전트 역할: 판사, 검사(원고측), 변호사(피고측), 피고인, 서기
- [ ] LangGraph 서브그래프 — 한국 법정 절차 기반 6단계 (형사/민사 분기)
- [ ] 에이전트별 Profile/Memory/Strategy 모듈 (SimCourt 아키텍처 참고)
- [ ] 법률 검색 도구 (Legal Article Retriever + Case Retriever)
- [ ] `mock_trial` 모듈 (라우터, 스키마)

**RAG — 독립 검색 파이프라인**
- [ ] 모의재판 전용 검색 인터페이스 (기존 RAG와 분리 가능)
- [ ] 판례 검색 + 법령 검색 + 관련도 점수 반환
- [ ] 에이전트가 Tool로 호출하는 구조

### 2.2 Out of Scope

- 음성 입력/출력 (STT/TTS)
- 사용자 간 대전 모드 (PvP)
- 모의재판 결과 DB 저장/이력 관리 (향후 확장)
- 배심원 제도 시뮬레이션 (※ JurorSprite.ts, JuryPanel.ts 등 장식적 UI 요소는 구현됨. 배심원 평의/평결 로직은 미구현)
- 3D 법정 환경 (2D 픽셀아트만)
- 모바일 최적화 (데스크톱 우선)

---

## 3. Requirements

### 3.1 Functional Requirements

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| **픽셀아트 법정 UI** | | | |
| FR-01 | Phaser.js 2D 법정 씬 렌더링 (타일맵 + 스프라이트) | High | Done |
| FR-02 | 5종 캐릭터 스프라이트 (판사/검사/변호사/피고/서기) + 대기/발언 애니메이션 | High | Done |
| FR-03 | 캐릭터 말풍선 — LLM 스트리밍 텍스트를 타이핑 효과로 표시 | High | Done |
| FR-04 | 재판 단계별 씬 전환 효과 (페이드, 카메라 이동) | Medium | Done |
| FR-05 | 하단 React 오버레이 UI (대화 입력, 증거 패널, 단계 표시) | High | Done |
| **설정 단계** | | | |
| FR-06 | 사건 유형 선택 (민사: 손해배상/계약, 형사: 폭행/사기/횡령 등) | High | Done |
| FR-07 | 역할 선택 — 형사: 검사 또는 변호사 / 민사: 원고측 또는 피고측 | High | Done |
| FR-08 | 사건 개요 자유 입력 (텍스트, 500자 이내) | High | Done |
| **형사 공판절차 (한국 형사소송법 기반 6단계)** | | | |
| FR-09 | 1단계 인정신문: 재판장이 피고인 인적사항 확인 + 진술거부권 고지 | High | Done |
| FR-10 | 2단계 모두진술: 검사 공소사실 요지 진술 → 피고인/변호인 의견 진술 (인정/부인) | High | Done |
| FR-11 | 3단계 증거조사: RAG 검색 판례/법령 제시 + 서증 제출 + 증인신문 시뮬레이션 | High | Done |
| FR-12 | 4단계 피고인신문: 검사/변호인이 피고인에게 질문 (사용자 참여) | High | Done |
| FR-13 | 5단계 구형 및 최후진술: 검사 의견진술(구형, §302) → 변호인 변론 → 피고인 최후진술(§303) | High | Done |
| FR-13a | 6단계 판결선고: AI 판사 판결문 생성 (한국 판결문 형식: 주문→이유→판사명) + 피드백 | High | Done |
| **민사 변론절차 (한국 민사소송법 기반 6단계)** | | | |
| FR-14a | 1단계 변론준비: 쟁점 정리 + 증거 목록 확인 + 양측 입장 정리 | High | Done |
| FR-14b | 2단계 주장/답변: 원고 청구원인 진술 → 피고 답변/항변 | High | Done |
| FR-14c | 3단계 증거조사: RAG 검색 판례/법령 + 서증 제출 + 증인신문 | High | Done |
| FR-14d | 4단계 변론: 양측 주장/반박 교환 (2-3 라운드) | High | Done |
| FR-14e | 5단계 변론종결: 양측 최종 주장 정리 | High | Done |
| FR-14f | 6단계 판결선고: AI 판사 판결문 생성 + 피드백 | High | Done |
| **에이전트 시스템** | | | |
| FR-15 | 에이전트별 Profile 모듈 — 역할 정의, 성향, 전문 분야 | High | Done |
| FR-16 | 에이전트별 Memory 모듈 — 단기(현재 단계), 장기(이전 단계 요약) | Medium | Done |
| FR-17 | 에이전트별 Strategy 모듈 — 단계별 전략 동적 조정 | Medium | Done |
| FR-18 | AI 상대측 반론 자동 생성 (LLM 스트리밍) | High | Done |
| FR-19 | LLM 작성 지원 — "주장 보강" 버튼으로 법적 근거 추가 | Medium | In Progress |
| **법률 검색 도구** | | | |
| FR-20 | Legal Article Retriever — 법령 검색 (기존 RAG 활용 또는 독립) | High | Done |
| FR-21 | Legal Case Retriever — 판례 검색 (기존 RAG 활용 또는 독립) | High | Done |
| FR-22 | 검색 결과를 증거 패널에 카드 형태로 표시 | Medium | Done |
| **판결 및 피드백** | | | |
| FR-23 | AI 판사 판결문 생성 (양측 주장 + 증거 종합, 한국 판결문 형식) | High | Done |
| FR-24 | 판결 이유 설명 (인용 판례/법령 명시) | High | Done |
| FR-25 | 사용자 주장의 강점/약점 피드백 제공 | Medium | Done |
| **통합 및 안전** | | | |
| FR-26 | 채팅 위젯에서 "모의재판" 키워드로 진입 가능 | Medium | Done |
| FR-27 | 전용 페이지(`/mock-trial`)에서 직접 시작 | High | Done |
| FR-28 | 면책 고지 표시 ("실제 법률 자문이 아닙니다") + 명시적 동의 절차 | High | Done |
| **법률 정확성 (v0.5 추가)** | | | |
| FR-29 | 형사 증거조사 시 증거동의/부동의 절차 구현 (형사소송법 §318) | High | In Progress |
| FR-30 | 전문법칙(§310-2) 반영: 증거능력 vs 증명력 구분, 증거능력 판단 | Medium | In Progress |
| FR-31 | 한국 판결문 정형 형식: 형사(주문→범죄사실→증거요지→법령적용→양형이유), 민사(주문→이유→결론) | High | Done |
| FR-32 | 양형위원회 양형기준 참조: 범죄군별 권고형 범위(감경/기본/가중), 양형인자 | Medium | In Progress |
| FR-33 | 입증책임 원칙: 형사(검찰 입증, 무죄추정), 민사(주장자 입증, 변론주의) | High | Done |
| FR-34 | 에이전트 법정 어투: 실제 한국 법정 관례 발화 패턴 (판사 인정신문, 진술거부권 고지, 검사 구형 등) | Medium | Done |
| **보안 (v0.5 추가)** | | | |
| FR-35 | 프롬프트 인젝션 방어: 시스템 프롬프트 역할 바운더리 + case_summary 필터링 | High | In Progress |
| FR-36 | LLM 출력 안전성: 편향/혐오 표현 필터링, 실존 인물 비방 방지 | High | In Progress |
| FR-37 | 세션 Rate Limiting: 세션당 LLM 호출 최대 50회, max_rounds 서버 강제 | Medium | Pending |
| FR-38 | 사용자 입력 검증 강화: case_type/user_role 화이트리스트, 단계별 입력 1000자 제한 | Medium | Done |

### 3.2 Non-Functional Requirements

| Category | Criteria | Measurement Method |
|----------|----------|-------------------|
| Performance | Phaser.js 법정 씬 로드 3초 이내 | 페이지 로드 측정 |
| Performance | AI 반론 스트리밍 첫 토큰 5초 이내 | 응답 시간 측정 |
| Performance | RAG 증거 검색 3초 이내 | 검색 시간 측정 |
| UX | 60fps 이상 애니메이션 (Phaser.js) | FPS 카운터 |
| UX | 단계 전환 시 캐릭터 애니메이션 자연스러움 | 사용자 테스트 |
| Safety | 면책 고지 상시 표시 | UI 확인 |
| Compatibility | Chrome, Edge, Firefox 최신 버전 | 브라우저 테스트 |
| Accessibility | Phaser canvas ARIA live region (에이전트 발언 미러링), ChatPanel 키보드 포커스 | 스크린리더 테스트 |
| Security | 프롬프트 인젝션 방어율 > 95% (테스트 세트 기반) | 보안 테스트 |
| Security | 모든 사용자 입력/LLM 출력 텍스트 이스케이프 (XSS 0건) | 코드 리뷰 |
| Data Privacy | 체크포인터 세션 데이터 24시간 후 자동 삭제 | 배치 삭제 cron |
| Minimum Resolution | 최소 1280x720 지원, 1024px 이하 2단 레이아웃 전환 | 반응형 테스트 |

---

## 4. Success Criteria

### 4.1 Definition of Done

- [ ] Phaser.js 법정 씬 렌더링 + 5종 캐릭터 스프라이트 표시
- [ ] 캐릭터 말풍선에 LLM 스트리밍 텍스트 표시
- [ ] 한국 법정 절차 6단계 전체 흐름 동작 (형사: 인정신문→판결선고 / 민사: 변론준비→판결선고)
- [ ] Backend: 5개 에이전트 + LangGraph 서브그래프 구현
- [ ] RAG 검색 연동 (판례 + 법령)
- [ ] 정적 검증: `ruff check`, `mypy`, `npm run build` 통과
- [ ] E2E: 설정 → 6단계 재판 → 판결 전체 흐름 동작 (형사/민사 각각)

### 4.2 Quality Criteria

- [ ] 픽셀아트 법정이 시각적으로 일관된 도트 스타일
- [ ] 캐릭터 애니메이션이 자연스러움 (대기↔발언 전환)
- [ ] AI 에이전트 반론이 논리적이고 사건 맥락에 맞음
- [ ] 판결문이 양측 주장과 증거를 인용하여 근거 기반 판결

---

## 5. Risks and Mitigation

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Phaser.js + Next.js 통합 복잡성 | High | Medium | `dynamic import` + `useEffect` 패턴으로 SSR 회피. react-phaser-fiber 등 래퍼 검토 |
| 픽셀아트 에셋 제작 공수 | High | High | 무료 에셋 활용 (itch.io, OpenGameArt) + 최소 스프라이트로 시작 |
| LLM 환각 — 존재하지 않는 판례/법령 인용 | High | Medium | RAG 검색 결과만 인용하도록 프롬프트 제약 + 출처 검증 |
| 실제 법률 자문으로 오해 | High | Medium | 면책 고지 강제 표시, 판결문에 "모의재판" 명시 |
| 다중 에이전트 호출로 LLM 비용 증가 | Medium | High | 에이전트별 토큰 제한, 서기 에이전트는 요약만 |
| 6단계 재판이 너무 길어 이탈 | Medium | Medium | 각 단계 시간/라운드 제한. ※ "빠른 재판" 모드는 향후 확장으로 분류 (현재 FR에 미포함) |
| Phaser.js 번들 크기 (500KB+) | Medium | Low | dynamic import로 lazy load 구현 완료. 법정 씬 진입 시만 로드 |
| 프롬프트 인젝션 (case_summary 통한 5개 에이전트 동시 공격) | High | Medium | 시스템 프롬프트 역할 바운더리 + case_summary 정규식 필터링 (FR-35) |
| LLM 편향/혐오 출력 (AI 판사 편향 판결, 혐오 표현) | High | Low | 출력 사후 필터링 + 시스템 프롬프트 안전 규칙 (FR-36) |
| sessionStorage 5MB 초과 (court_record 무한 증가) | Medium | Medium | 설정 정보만 저장, court_record는 메모리 전용 + Backend 체크포인터 복원 |
| EventBus 이벤트 유실 (Phaser 씬 전환 중 SSE 응답 도착) | Medium | High | bufferedEmit 패턴 또는 scene.sleep/launch 전략 |
| 법률 용어/절차 부정확 (교육 목적 훼손) | High | Medium | 법률 전문가 리뷰, 양형기준/판결문 형식/입증책임 설계 반영 (FR-29~34) |
| 접근성(A11y) 미달 (Phaser canvas 스크린리더 접근 불가) | Medium | High | ARIA live region + ChatPanel 텍스트 미러링 |

---

## 6. Architecture Considerations

### 6.1 Project Level Selection

| Level | Characteristics | Recommended For | Selected |
|-------|-----------------|-----------------|:--------:|
| **Starter** | Simple structure | Static sites | |
| **Dynamic** | Feature-based modules | Web apps | |
| **Enterprise** | Strict layer separation | High-traffic systems | **V** |

### 6.2 Key Architectural Decisions

| Decision | Options | Selected | Rationale |
|----------|---------|----------|-----------|
| 법정 UI 엔진 | HTML/CSS / Canvas / **Phaser.js** / PixiJS | **Phaser.js** | Generative Agents 논문 기술, 타일맵+스프라이트+애니메이션 통합 |
| 에셋 스타일 | 실사 / 일러스트 / **픽셀아트** | **픽셀아트 (도트)** | 논문 비전, 개발 비용 절감, 독특한 UX |
| Next.js 통합 | iframe / **dynamic import** / 별도 페이지 | **dynamic import** | SSR 회피, React 오버레이 UI 연동 |
| 에이전트 수 | 2명 / 3명 / **5명** | **5명** | 한국 법정 구성: 판사/검사(원고측)/변호사(피고측)/피고인/서기 |
| 재판 절차 | 자유 대화 / 3단계 / **6단계** | **6단계** | 한국 법정 절차 기반 (형사: 형사소송법, 민사: 민사소송법) |
| 에이전트 모듈 | 단순 프롬프트 / **Profile+Memory+Strategy** | **P+M+S** | SimCourt 아키텍처, 단계별 전략 업데이트 |
| LLM 클라이언트 | Solar / OpenAI / Gemini | **get_chat_model()** (`app.tools.llm`) | 기존 통합 LLM 클라이언트 사용, 모델 교체 유연 |
| RAG | 기존 파이프라인 공유 / **독립 인터페이스** | **독립 인터페이스** | 모의재판 전용 검색 전략 가능, 기존 RAG와 다를 수 있음 |
| 상태 저장 | InMemory / PostgreSQL | **PostgreSQL** | 기존 체크포인터, 세션 복원 |

### 6.3 프론트엔드 아키텍처: Phaser.js + Next.js 하이브리드

```
┌─────────────────────────────────────────────────────────────┐
│                    Next.js Page (/mock-trial)                 │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │               Phaser.js Canvas (법정 씬)                │ │
│  │                                                         │ │
│  │  ┌──────────────────────────────────────────────────┐  │ │
│  │  │ Tilemap Layer: 법정 바닥, 벽, 가구 (타일셋)       │  │ │
│  │  │ Sprite Layer: 판사, 검사, 변호사, 피고, 서기       │  │ │
│  │  │ UI Layer: 말풍선, 이름표, 단계 표시               │  │ │
│  │  └──────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │          React Overlay UI (하단 패널)                   │ │
│  │                                                         │ │
│  │  [💬 대화 입력]  [📋 증거 목록]  [📖 법령 검색]        │ │
│  │  [⏩ 재판 단계: 3/6 증거조사]  [⚖️ 역할: 검사]        │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌───── Phaser ↔ React 통신 ─────┐                         │
│  │ EventBus (CustomEvent)         │                         │
│  │ Phaser → React: agent_speak    │                         │
│  │ React → Phaser: user_input     │                         │
│  │ React → Phaser: stage_change   │                         │
│  └────────────────────────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

### 6.4 백엔드 에이전트 아키텍처 (한국 법정 절차 기반)

**형사 공판절차 서브그래프** (형사소송법 기반):
```
┌─────────────── LangGraph StateGraph (형사) ──────────────┐
│                                                           │
│  START                                                    │
│    │                                                      │
│    ▼                                                      │
│  setup_node ─(interrupt)─► 사건유형(형사) + 역할 선택     │
│    │                                                      │
│    ▼                                                      │
│  identity_node ─► [인정신문]                              │
│    │   재판장: 피고인 인적사항 확인                        │
│    │   재판장: 진술거부권 고지                             │
│    │   서기: 기록 시작                                    │
│    ▼                                                      │
│  opening_node ─► [모두진술]                               │
│    │   검사: 공소사실 요지 진술                            │
│    │   ─(interrupt)─► 피고인/변호인: 의견 진술 (인정/부인) │
│    ▼                                                      │
│  evidence_node ─► [증거조사]                              │
│    │   RAG 검색 (판례 + 법령) → 증거 자동 제시            │
│    │   ─(interrupt)─► 사용자 증거 선택/제출               │
│    │   증인신문 시뮬레이션 (선택)                          │
│    ▼                                                      │
│  examination_node ─► [피고인신문]                         │
│    │   ─(interrupt)─► 검사/변호인이 피고인에게 질문        │
│    │   피고인 AI: 답변 생성                               │
│    ▼                                                      │
│  criminal_closing_node ─► [구형 및 최후진술]               │
│    │   검사: 의견진술/구형 (LLM 생성)                     │
│    │   ─(interrupt)─► 변호인: 변론                        │
│    │   피고인: 최후진술 (§303)                             │
│    ▼                                                      │
│  verdict_node ─► [판결선고]                               │
│    │   AI 판사: 판결문 생성 (유죄/무죄 + 양형 이유)       │
│    │   서기: 판결 기록                                    │
│    │   피드백 생성 (강점/약점 분석)                        │
│    ▼                                                      │
│  END                                                      │
└───────────────────────────────────────────────────────────┘
```

**민사 변론절차 서브그래프** (민사소송법 기반):
```
┌─────────────── LangGraph StateGraph (민사) ──────────────┐
│                                                           │
│  START                                                    │
│    │                                                      │
│    ▼                                                      │
│  setup_node ─(interrupt)─► 사건유형(민사) + 역할 선택     │
│    │                                                      │
│    ▼                                                      │
│  pretrial_node ─► [변론준비]                              │
│    │   재판장: 쟁점 정리                                  │
│    │   양측: 주장 요지 및 증거 목록 확인                   │
│    │   서기: 기록 시작                                    │
│    ▼                                                      │
│  claims_node ─► [주장/답변]                               │
│    │   ─(interrupt)─► 원고: 청구원인 진술                 │
│    │   피고 AI: 답변/항변 생성                             │
│    ▼                                                      │
│  evidence_node ─► [증거조사]                              │
│    │   RAG 검색 (판례 + 법령) → 증거 자동 제시            │
│    │   ─(interrupt)─► 사용자 증거 선택/제출               │
│    │   서증 검토 + 증인신문 시뮬레이션 (선택)              │
│    ▼                                                      │
│  argument_node ─► [변론] (2-3 라운드 루프)                │
│    │   ─(interrupt)─► 사용자 측 주장                      │
│    │   AI 상대측: 반박 (LLM 스트리밍)                     │
│    │   라운드 체크 → 계속 or 변론종결로                    │
│    ▼                                                      │
│  civil_closing_node ─► [변론종결]                          │
│    │   ─(interrupt)─► 양측 최종 주장 정리                 │
│    │   재판장: 변론 종결 선언                              │
│    ▼                                                      │
│  verdict_node ─► [판결선고]                               │
│    │   AI 판사: 판결문 생성 (청구 인용/기각 + 이유)       │
│    │   서기: 판결 기록                                    │
│    │   피드백 생성 (강점/약점 분석)                        │
│    ▼                                                      │
│  END                                                      │
└───────────────────────────────────────────────────────────┘
```

**형사/민사 공통 구조**: `setup_node`, `evidence_node`, `verdict_node`는 공유하고 중간 노드만 분기 처리.

```python
# 라우팅 분기
def route_by_case_type(state: MockTrialState) -> str:
    if state["case_type"].startswith("형사"):
        return "identity_node"      # 형사: 인정신문부터
    else:
        return "pretrial_node"      # 민사: 변론준비부터
```

### 6.5 에이전트별 모듈 구조 (SimCourt 참고)

```python
@dataclass
class CourtAgent:
    """각 법정 역할 에이전트의 공통 구조 (이미 mock_trial_agents.py에 초안 구현)"""

    # Profile Module — 역할 정의
    role: str              # "judge" | "prosecutor" | "attorney" | "defendant" | "clerk"
    name: str              # 표시 이름 ("재판장 김법관")
    system_prompt: str     # 역할별 시스템 프롬프트
    temperature: float     # LLM 온도 (에이전트별 차등)

    # Memory Module — 기억 관리
    short_term: list[str]  # 현재 단계 발언 기록
    long_term: list[str]   # 이전 단계 요약 (reflect() 결과)

    # Strategy Module — 전략 관리
    strategy: str          # 현재 단계 전략
    # 각 단계 종료 시 reflect() → 단기→장기 이동 + update_strategy()

    # Tools — 외부 도구
    tools: list[str]       # ["case_retriever", "article_retriever"]

    # LLM 호출: get_chat_model(temperature=self.temperature) 사용
    # 직렬화: to_state() ↔ from_state() (TypedDict AgentState 변환)
```

| 에이전트 | 형사 역할 | 민사 역할 | Profile 핵심 | Strategy 핵심 | Tools |
|---------|----------|----------|-------------|--------------|-------|
| 판사 | 재판장 | 재판장 | 공정, 절차 통제, 진술거부권 고지 | 양측 균형 유지, 쟁점 정리 | - |
| 검사/원고측 | 검사 | 원고 대리인 | 공소 유지 / 청구원인 입증 | 범죄 입증 / 손해 입증, 판례 인용 | Case Retriever, Article Retriever |
| 변호사/피고측 | 변호인 | 피고 대리인 | 피고 방어, 반박 | 무죄/감형 / 청구 기각 논증 | Case Retriever, Article Retriever |
| 피고인/당사자 | 피고인 | 상대 당사자 | 사건 당사자, 감정+사실 표현 | 자신의 입장 표현 | - |
| 서기 | 법원서기 | 법원서기 | 기록 작성, 중립 | 요약 기록, 단계별 진행 정리 | - |

### 6.6 모듈/파일 구조

```
# Backend
backend/app/
├── multi_agent/
│   ├── agents/mock_trial_agent.py          # MockTrialAgent (BaseChatAgent 상속)
│   ├── subgraphs/mock_trial.py             # MockTrialState + 6단계 서브그래프 (형사/민사 분기)
│   ├── subgraphs/mock_trial_agents.py      # CourtAgent 클래스 (Profile/Memory/Strategy)
│   ├── subgraphs/mock_trial_prompts.py     # 에이전트별 시스템 프롬프트
│   ├── router.py                           # AgentType.MOCK_TRIAL 추가
│   ├── nodes.py                            # mock_trial_node() + AGENT_NODE_MAP
│   └── graph.py                            # add_node("mock_trial_subgraph", ...)
├── modules/mock_trial/
│   ├── __init__.py
│   ├── router/__init__.py                  # /api/mock-trial 엔드포인트
│   └── schema/__init__.py                  # MockTrialSetup, CourtEvent, Judgment 등
└── services/service_function/
    └── mock_trial_service.py               # 증거 검색, 에이전트 반론, 판결문 생성

# Frontend
frontend/src/
├── app/mock-trial/page.tsx                 # 페이지 엔트리 (dynamic import Phaser)
├── features/mock-trial/
│   ├── game/                               # Phaser.js 게임 코드
│   │   ├── CourtScene.ts                   # 메인 법정 씬
│   │   ├── LobbyScene.ts                   # 로비/설정 씬
│   │   ├── config.ts                       # Phaser 게임 설정
│   │   ├── sprites/                        # 스프라이트 클래스
│   │   │   ├── JudgeSprite.ts
│   │   │   ├── LawyerSprite.ts
│   │   │   └── CharacterBase.ts            # 공통 캐릭터 베이스
│   │   ├── ui/                             # Phaser 내부 UI
│   │   │   ├── SpeechBubble.ts             # 말풍선
│   │   │   └── StageIndicator.ts           # 단계 표시
│   │   └── EventBus.ts                     # Phaser ↔ React 통신
│   ├── components/                         # React 오버레이 컴포넌트
│   │   ├── MockTrialGame.tsx               # Phaser 게임 래퍼
│   │   ├── ChatPanel.tsx                   # 하단 대화 입력 패널
│   │   ├── EvidencePanel.tsx               # 증거 목록 패널
│   │   ├── StageProgress.tsx               # 재판 단계 표시
│   │   └── JudgmentDisplay.tsx             # 판결문 모달
│   ├── services/index.ts                   # API 호출 함수
│   └── types/index.ts                      # TypeScript 타입 정의
├── public/assets/mock-trial/               # 픽셀아트 에셋
│   ├── tilemap/                            # 법정 타일맵 (Tiled 에디터)
│   │   ├── courtroom.json                  # 타일맵 데이터
│   │   └── courtroom_tiles.png             # 타일셋 이미지
│   ├── sprites/                            # 캐릭터 스프라이트시트
│   │   ├── judge.png                       # 판사 (idle, speak 프레임)
│   │   ├── prosecutor.png                  # 검사
│   │   ├── attorney.png                    # 변호사
│   │   ├── defendant.png                   # 피고
│   │   └── clerk.png                       # 서기
│   └── ui/                                 # UI 에셋
│       ├── speech_bubble.png               # 말풍선 9-patch
│       └── panel_bg.png                    # 패널 배경
├── lib/modules.ts                          # mock-trial 모듈 등록
└── lib/api.ts                              # mockTrial endpoint 추가
```

---

## 7. Convention Prerequisites

### 7.1 Existing Project Conventions

- [x] `CLAUDE.md` — 모듈 구조, API 경로 규칙 정의됨
- [x] `.claude/rules/coding-style.md` — 코딩 스타일 규칙
- [x] `.claude/rules/code-verification.md` — 검증 프로토콜
- [x] BaseChatAgent 추상 클래스 패턴
- [x] LangGraph interrupt 서브그래프 패턴 (소액소송)
- [x] 모듈 자동 등록 (ModuleRegistry)

### 7.2 API 경로 규칙

| Backend 모듈명 | API 경로 | Frontend 경로 |
|----------------|----------|---------------|
| `mock_trial` | `/api/mock-trial` | `/mock-trial` |

### 7.3 Dependencies (추가 필요)

| Package | 위치 | 용도 |
|---------|------|------|
| `phaser` | Frontend (npm) | 2D 게임 엔진 (타일맵, 스프라이트, 애니메이션) |
| - | Frontend | Tiled 에디터 (타일맵 제작, 별도 도구) |

Backend는 기존 의존성으로 구현 가능:
- LangGraph: 서브그래프, interrupt, Command
- `app.tools.llm.get_chat_model()`: 에이전트별 LLM 호출 (temperature 파라미터)
- LanceDB / 기존 RAG: 법률 검색 도구 (EvidenceSearcher Protocol로 래핑)
- `base_chat.py`: BaseChatAgent 추상 클래스 (MockTrialAgent 상속)
- `base_chat.py`: ChatAction/ActionType (UI 액션 버튼 표준 타입)

---

## 8. Implementation Plan

### 8.1 구현 순서

> **참고**: Step 표기에 `[완료]`는 이미 초안이 구현된 항목, `[보강]`은 리팩토링 필요 항목

| Phase | Step | 작업 | 핵심 파일 | 의존성 | 상태 |
|-------|------|------|----------|--------|------|
| **A. 에셋** | 1 | 픽셀아트 법정 타일맵 제작 (Tiled) | `public/assets/mock-trial/tilemap/` | - | 미착수 |
| | 2 | 캐릭터 스프라이트시트 제작/조달 | `public/assets/mock-trial/sprites/` | - | 미착수 |
| **B. Backend 코어** | 3 | Pydantic 스키마 정의 | `modules/mock_trial/schema/` | - | [완료] |
| | 4 | CourtAgent 클래스 보강 (generate→get_chat_model) | `subgraphs/mock_trial_agents.py` | Step 3 | [완료] |
| | 5 | 에이전트 프롬프트 보강 (한국법 상세화) | `subgraphs/mock_trial_prompts.py` | Step 4 | [완료] |
| | 6 | 서브그래프 노드 함수 완성 (형사 6단계 + 민사 6단계) | `subgraphs/mock_trial.py` | Step 4-5 | [완료] |
| | 7 | MockTrialAgent(BaseChatAgent) 구현 | `agents/mock_trial_agent.py` | Step 6 | [완료] |
| | 8 | RAG 검색 서비스 (EvidenceSearcher Protocol) | `services/mock_trial_service.py` | Step 6 | [완료] |
| | 9 | 모듈 라우터 (전용 엔드포인트) | `modules/mock_trial/router/__init__.py` | Step 3, 8 | [완료] |
| | 10 | 기존 시스템 통합 점검 | `router.py`, `nodes.py`, `graph.py` | Step 6 | [완료] |
| **C. Frontend 코어** | 11 | TypeScript 타입 + 상수 정의 | `features/mock-trial/types/index.ts` | - | [완료] |
| | 12 | Phaser.js 설치 + Next.js dynamic import | `package.json`, `MockTrialGame.tsx` | Step 1-2 | [완료] |
| | 13 | EventBus (Phaser ↔ React 통신) | `game/EventBus.ts` | Step 12 | [완료] |
| | 14 | useTrialState 훅 (sessionStorage 동기화) | `hooks/useTrialState.ts` | Step 11 | 미구현 (컴포넌트 직접 관리) |
| | 15 | 법정 씬 구현 (타일맵 + 캐릭터 배치) | `game/CourtScene.ts` | Step 12 | [완료] |
| | 16 | 말풍선 + 캐릭터 애니메이션 | `game/ui/SpeechBubble.ts`, sprites | Step 15 | [완료] |
| | 17 | React 오버레이 UI (ChatPanel, EvidencePanel 등) | `components/` | Step 13-14 | [완료] |
| | 18 | API 서비스 함수 | `services/index.ts` | Step 11 | [완료] |
| **D. 통합** | 19 | next.config.js rewrites (API 프록시) | `next.config.js` | Step 9 | [완료] |
| | 20 | Frontend ↔ Backend SSE 연동 | 전체 | Step 17, 9 | [완료] |
| | 21 | 모듈 등록 확인 + 정적 검증 | `modules.ts`, `api.ts`, 린트 | Step 20 | [완료] |
| **E. 보안/품질 (v0.5 추가)** | 22 | 프롬프트 인젝션 방어 (역할 바운더리 + 필터링) | `subgraphs/mock_trial_prompts.py` | Step 5 | In Progress |
| | 23 | LLM 출력 안전성 (편향/혐오 필터링) | `subgraphs/mock_trial.py` | Step 6 | In Progress |
| | 24 | 세션 Rate Limiting (50회/세션) | `subgraphs/mock_trial.py` | Step 6 | 미착수 |
| | 25 | 증거동의/부동의 절차 (§318) | `subgraphs/mock_trial.py` | Step 6 | In Progress |

### 8.2 서브그래프 상태 스키마

```python
class MockTrialState(TypedDict, total=False):
    # 부모 그래프에서 전달
    message: str
    history: list[dict[str, str]]
    session_data: dict[str, Any]

    # 설정
    case_type: str              # "민사_손해배상", "형사_사기" 등
    user_role: str              # "prosecutor" (검사) | "attorney" (변호사)
    case_summary: str           # 사건 개요 (사용자 입력)

    # 에이전트 상태
    agents: dict[str, dict]     # 역할별 Profile/Memory/Strategy 상태
    # { "judge": { "profile": ..., "memory": ..., "strategy": ... }, ... }

    # 증거
    evidence_cases: list[dict[str, Any]]     # 판례 검색 결과
    evidence_articles: list[dict[str, Any]]  # 법령 검색 결과
    selected_evidence: list[str]             # 사용자 선택 증거 ID

    # 재판 진행
    stage: str                  # 형사: "setup" | "identity" | "opening" |
                                #       "evidence" | "examination" | "closing" | "verdict"
                                # 민사: "setup" | "pretrial" | "claims" |
                                #       "evidence" | "argument" | "closing" | "verdict"
    current_round: int          # 변론 라운드 (1-based, 민사 변론/형사 증거조사)
    max_rounds: int             # 최대 라운드 수 (기본 3)
    court_record: list[dict]    # 서기 기록 (발언자, 내용, 단계, 시각)

    # 출력
    response: str               # 현재 에이전트 발언
    speaking_agent: str         # 현재 발언 중인 에이전트 역할
    actions: list[dict[str, Any]]
    judgment: Optional[str]     # 최종 판결문
    feedback: Optional[str]     # 강점/약점 피드백
    is_complete: bool
    agent_used: str
```

### 8.3 LLM 프롬프트 전략

| 에이전트 | 프롬프트 핵심 | Temperature |
|---------|-------------|-------------|
| 판사 | 한국 법정 절차 진행 (형사: 진술거부권 고지, 인정신문 / 민사: 쟁점정리), 공정한 절차 통제, 양측 균형 | 0.3 |
| 검사/원고측 | [형사] 공소사실 입증, 구형 근거 제시, 판례 인용 / [민사] 청구원인 입증, 손해 산정 | 0.7 |
| 변호사/피고측 | [형사] 무죄/감형 논증, 증거 탄핵, 유리 판례 인용 / [민사] 청구 기각, 항변 사유 | 0.7 |
| 피고인/당사자 | 자신의 입장 감정적+사실적 표현, 신문 시 답변 | 0.8 |
| 서기 | 발언 요약 기록, 중립적 서술, 단계별 진행 정리 | 0.2 |

### 8.3.1 한국 법정 절차 상세

**형사 공판절차 (형사소송법 기반)**:

| 단계 | 노드 | 법적 근거 | 게임 내 동작 |
|------|------|----------|-------------|
| 1. 인정신문 | `identity_node` | 형사소송법 §284(피고인진술), §283-2(진술거부권고지) | 재판장이 피고인 인적사항 확인, 진술거부권 고지 (자동) |
| 2. 모두진술 | `opening_node` | 형사소송법 §285~§286 | 검사: 공소사실 요지 / 피고인·변호인: 의견 (사용자 입력) |
| 3. 증거조사 | `evidence_node` | 형사소송법 §290~§313, §318(증거동의) | RAG 검색 판례/법령 제시, **증거동의/부동의**, 서증 제출, 증인신문 |
| 4. 피고인신문 | `examination_node` | 형사소송법 §296-2 | 검사/변호인이 피고인에게 질문 (사용자 참여) |
| 5. 구형 및 최후진술 | `criminal_closing_node` | 형사소송법 §302(검사 의견진술), §303(최후진술) | 검사 구형 → 변호인 변론 → 피고인 최후진술 (사용자 입력) |
| 6. 판결선고 | `verdict_node` | 형사소송법 §43(판결선고방식), §39(판결선고기일), §318(유죄이유) | AI 판사 판결문 생성 (한국 판결문 형식: 주문→이유→양형) |

**민사 변론절차 (민사소송법 기반)**:

| 단계 | 노드 | 법적 근거 | 게임 내 동작 |
|------|------|----------|-------------|
| 1. 변론준비 | `pretrial_node` | 민사소송법 §258~§268 | 재판장 쟁점 정리, 양측 입장·증거 목록 확인 (자동) |
| 2. 주장/답변 | `claims_node` | 민사소송법 §256~§257 | 원고 청구원인 진술 → 피고 답변/항변 (사용자 입력) |
| 3. 증거조사 | `evidence_node` | 민사소송법 §288~§344 | RAG 검색 판례/법령, 서증 제출, 증인신문 시뮬레이션 |
| 4. 변론 | `argument_node` | 민사소송법 §134~§148 | 양측 주장/반박 교환 2-3 라운드 (사용자 입력) |
| 5. 변론종결 | `civil_closing_node` | 민사소송법 §200 | 양측 최종 주장 정리, 재판장 변론종결 선언 (사용자 입력) |
| 6. 판결선고 | `verdict_node` | 민사소송법 §206~§208 | AI 판사 판결문 (청구 인용/기각 + 이유) |

### 8.4 Phaser.js ↔ React 통신 프로토콜

```typescript
// EventBus 이벤트 타입
interface CourtEvents {
    // Phaser → React
    "agent:speak": { agent: string; text: string; streaming: boolean };
    "stage:change": { from: string; to: string };
    "evidence:presented": { evidence: Evidence[] };

    // React → Phaser
    "user:input": { text: string };
    "user:select_evidence": { evidenceIds: string[] };
    "game:advance_stage": {};
    "agent:animate": { agent: string; animation: "idle" | "speak" | "react" };
}
```

---

## 9. Next Steps

1. [x] Plan 리뷰 및 승인
2. [x] Design 문서 작성 (`/pdca design mock-trial`)
3. [x] Plan/Design 보강 (v0.4~v0.6) — 기존 코드 현황 반영, 실제 패턴 정합성, 문서 동기화
4. [x] Analysis 작성 및 보강 (v0.2) — Match Rate 산출 근거, 개선 로드맵 추가
5. [ ] Error Handling 보강 — LLM 타임아웃, Canvas 폴백, EventBus 큐 (Analysis 70% → 90%)
6. [ ] Security 보강 — Rate Limiting, 프롬프트 인젝션 방어 구현 (Analysis 80% → 95%)
7. [ ] Frontend 미구현 항목 구현 — useTrialState 훅 분리, 증거 패널 고도화
8. [ ] 통합 테스트 및 Gap Re-analysis (`/pdca analyze mock-trial`)

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-02-12 | Initial draft | Claude |
| 0.2 | 2026-02-12 | 참고자료 반영 대폭 개편: Phaser.js 픽셀아트 UI, 5개 에이전트 역할, 5단계 재판 절차, Profile/Memory/Strategy 모듈, RAG 독립 인터페이스 | Claude |
| 0.3 | 2026-02-12 | 한국 법정 절차 적용: 형사 공판절차 6단계(인정신문→판결선고) + 민사 변론절차 6단계(변론준비→판결선고), 형사/민사 분기 서브그래프, 법적 근거(형사소송법/민사소송법) 명시, FR 재구성(FR-28개) | Claude |
| 0.4 | 2026-02-21 | 기존 구현 현황 반영 (초안 코드 8개 파일 + 미구현 12개 항목), LLM 클라이언트 정정 (get_solar_response_stream→get_chat_model), Implementation Plan 상태 표시 추가, Dependencies 정확도 개선 | Claude |
| 0.6 | 2026-02-21 | 문서 동기화 보강: (1) FR 상태 38건 Pending→Done/In Progress 업데이트, (2) Implementation Step 상태 반영 + Phase E 추가, (3) §42→§43+§39 조문 정정, (4) closing_node→criminal/civil 분리 반영, (5) Next Steps 현행화, (6) 배심원 Out of Scope 명확화, (7) 리스크 완화 전략 현행화 | Claude |
| 0.5 | 2026-02-21 | 5개 관점 에이전트 팀 리뷰 반영: (1) 법률 용어 정정 — "최종변론"→"구형 및 최후진술", 형사소송법 조문 번호 정정(§318-4→§42~43), 인정신문에 §283-2 추가, (2) FR 10개 추가 — 증거동의/부동의(FR-29), 전문법칙(FR-30), 판결문 형식(FR-31), 양형기준(FR-32), 입증책임(FR-33), 법정어투(FR-34), 프롬프트인젝션방어(FR-35), LLM출력필터링(FR-36), Rate Limiting(FR-37), 입력검증(FR-38), (3) NFR 6개 추가 — 접근성, 보안(2), 데이터보존, 최소해상도, (4) Risks 7개 추가 — 프롬프트인젝션, LLM편향, sessionStorage, EventBus유실, 법률정확성, 접근성 | Claude |
