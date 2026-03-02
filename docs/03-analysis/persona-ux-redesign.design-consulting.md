# 외부 컨설팅 설계 리뷰 보고서 — 페르소나 UX 개선

> **검증 대상**: `docs/02-design/features/persona-ux-redesign.design.md` v1.0
> **검증 도구**: Codex CLI (External Consultant, gpt-5.3-codex)
> **일시**: 2026-02-28

---

## 외부 컨설팅 설계 리뷰 보고서

### 1. 격차 분석 (현재 설계 vs 업계 최고)

**총평:** 현재 설계는 "초기 전환 UX + 실패 복구"는 준수하지만, 업계 최고 수준 대비 핵심 격차는 **운영형 에이전트 기능(지속 실행/관측성/거버넌스/모델 라우팅)** 입니다.

#### 강점
1. Track1 실패 시 Track2 자동 전환, draft 보존, 세션 복원은 실사용 안정성에 유리.
2. `EvidenceSnippet` 도입은 법률 도메인 신뢰 UX에 적합.
3. `messages=[]` 하드코딩 제거 계획은 품질 리스크를 정확히 짚음.

#### 격차 (Notion AI/Jasper/Copy.ai 대비)
1. **Agent 운영성 부족**: 최고 서비스들은 다단계 워크플로·에이전트 오케스트레이션·관리자 관측 기능을 전면 제공하는데, 현재는 단일 `analyze` 중심.
2. **모델-agnostic 계층 부족**: 모델 선택/라우팅/실패 시 대체 모델 정책이 스펙에 없음.
3. **엔터프라이즈 거버넌스 부족**: 감사 로그, 정책 제어, 팀 단위 AI 사용 분석이 미정.
4. **통합 생태계 부족**: 외부 툴/문서 시스템과의 표준화된 연결(MCP 등) 전략이 없음.
5. **비동기 처리 한계**: 30초 타임아웃 기반 동기 호출은 장문/혼잡 시 UX 저하 가능.

---

### 2. 기술 트렌드 반영 평가

**평가:** 부분 반영 (중간 수준)

#### 반영된 트렌드
1. LangGraph 기반 상태 관리 + human fallback.
2. 실패 시 우회 플로우(Track2) 설계.
3. 신뢰성 요소(Evidence, PII 마스킹, Rate limiting) 고려.

#### 미반영/보완 필요 (2025-2026 핵심)
1. **Durable execution + 체크포인트 재개**: 클라이언트 세션 복원만으로는 부족, 서버 런 단위 재개 필요.
2. **Model-agnostic orchestration**: 모델 라우터(품질/지연/비용 정책), 폴백 체인 필요.
3. **표준 도구 연결 계층**: MCP 기반 도구/지식 연결로 벤더 종속 완화.
4. **Observability/EvalOps**: trace, tool-call 로그, 실패 분류, 오프라인 평가셋 기반 회귀검증 필요.

---

### 3. UX 최적화 제안

1. **분석 대기시간 체감 축소**
   - `track1_analyzing` 중 Track2 입력을 병렬 허용("분석 중 미리 입력").
   - 예상 소요시간/진행률 근거 표시(전문직 사용자는 예측 가능성을 선호).

2. **신뢰 UX 강화**
   - `EvidenceSnippet`에 출처 신뢰도/시점/적용 근거를 구조화 표시.
   - HybridReview에서 "수정 전/후 diff" 제공(법률 문맥 검토 효율↑).

3. **전환율 최적화**
   - KPI를 명시적으로 운영: `Welcome→Start`, `Start→Ready`, `Time-to-Ready`, Track1 이탈률.
   - 최소 목표: Ready 전환율 +10~15%, Time-to-Ready -20% (A/B로 검증).

4. **전문직 워크플로 정합**
   - "즉시 사용 기본안 + 정교화 나중" 버튼을 상단 고정.
   - 실수 복구(undo/restore)와 법률용 템플릿 추천을 리뷰 단계에 통합.

---

### 4. 상태머신 설계 평가

**평가:** 구조는 양호, 전이 안정성/복구성에서 보강 필요.

1. **전이 안전성**
   - `useReducer`만으로는 불법 전이 방지가 약함.
   - 권장: 상태별 허용 이벤트 매트릭스 + reducer 가드(불법 전이 reject + telemetry).

2. **경쟁 상태(race condition)**
   - `ANALYSIS_SUCCESS`가 늦게 도착해 Track2 입력을 덮어쓸 위험.
   - 권장: `request_id`/`analysis_job_id` 비교 후 최신 응답만 반영.

3. **복구 가능성**
   - SessionStorage는 디바이스/브라우저 범위 복구만 제공.
   - 권장: 서버 체크포인트 저장(유저+세션+state_version), 재로그인 복원 지원.

4. **확장성**
   - 현재 6개 상태는 관리 가능하나 기능 확장 시 폭증 가능.
   - 권장: 상위 상태(`onboarding`, `analysis`, `ready`) + 하위 상태(분석 retry/cancel/timedout)로 계층화.

---

### 5. API 설계 평가

**평가:** 기능 단위 API는 적절, 계약 안정성/버전전략 보완 필요.

1. **REST/계약 안정성**
   - `POST /persona/analyze` 응답 변경은 **breaking change** 가능성 큼.
   - 권장: `/api/v1` 명시 + 신규 필드 additive 원칙 또는 `/persona/analyze-v2`.

2. **에러 계약**
   - 현재 HTTP 코드는 실무적으로 타당하나, 클라이언트 일관 처리를 위해 표준 에러 바디 필요.
   - 권장 필드: `type`, `title`, `status`, `detail`, `error_code`, `retryable`, `trace_id`.

3. **장기 실행 작업 패턴**
   - 30초 동기 처리 대신:
     - `POST /persona/analysis-jobs` → `202 + job_id`
     - `GET /persona/analysis-jobs/{id}` (poll/SSE)
   - 타임아웃·재시도·중복 제출 제어가 쉬워짐.

4. **운영/보안**
   - `429` 응답에 `Retry-After`와 제한 정책 노출.
   - 모든 에러 응답에 `trace_id` 포함(관측성/CS 대응).
   - IDOR/JWT 검증은 명시되어 있어 방향은 적절.

---

## 출처
- Notion AI/Agents: https://www.notion.com/product/ai
- Notion 3.2 릴리스(모바일 AI/모델 선택): https://www.notion.com/releases/2026-01-20
- Jasper Platform(Agentic/Trust/거버넌스): https://www.jasper.ai/platform
- Copy.ai Agents: https://www.copy.ai/agents
- LangGraph Durable Execution: https://docs.langchain.com/oss/javascript/langgraph/durable-execution
- MCP Specification/Versioning: https://modelcontextprotocol.io/specification/
- OpenAI Responses API (tools/MCP): https://platform.openai.com/docs/api-reference/responses
- AWS Bedrock Multi-agent GA: https://aws.amazon.com/about-aws/whats-new/2025/03/amazon-bedrock-multi-agent-collaboration/
- Azure AI Foundry Agent Service: https://learn.microsoft.com/en-us/azure/ai-foundry/agents/whats-new
