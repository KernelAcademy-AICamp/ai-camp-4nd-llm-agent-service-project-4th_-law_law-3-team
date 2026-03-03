# Red Team 설계 리뷰 보고서 — 페르소나 UX 개선

> **검증 대상**: `docs/02-design/features/persona-ux-redesign.design.md` v1.0
> **검증 도구**: Gemini CLI (Red Team)
> **일시**: 2026-02-28

---

## Red Team 설계 리뷰 보고서

본 보고서는 **페르소나 UX 개선 설계서** 및 관련 백엔드/프론트엔드 코드를 기반으로 보안 취약점 및 아키텍처 약점을 분석한 결과입니다.

---

### 1. 보안 취약점 (Critical/High/Medium/Low)

*   **[Critical] 인증 및 인가 결여 (H-01/IDOR):**
    *   `content_marketing/router/__init__.py`에 `TEMP_USER_ID = "temp_user_001"`이 하드코딩되어 있습니다. 설계상 JWT 도입이 Phase D로 예정되어 있으나, 현재 구조에서는 타인의 `persona_id`를 추측하여 `update`하거나 `feedback`을 조작할 수 있는 **IDOR(Insecure Direct Object Reference)** 취약점에 무방비합니다.
*   **[High] PII(개인정보) 노출 및 유출 위험:**
    *   `_fetch_chat_history()`를 통해 실제 대화 이력을 조회하여 LLM에 전송합니다. 법률 상담 특성상 의뢰인의 성명, 사건 번호 등 민감 정보(PII)가 포함될 가능성이 매우 높으나, 설계서상 마스킹 처리가 백엔드 구현 단계(Phase D)로 밀려 있어 개발 중 실데이터 유출 위험이 존재합니다.
*   **[Medium] SessionStorage를 통한 상태 조작:**
    *   `persona_gate_draft`를 `SessionStorage`에 저장합니다. 클라이언트 사이드에서 `ready` 상태로 강제 변경하거나, `draftState` 내부의 `confidence` 점수 등을 조작하여 시스템 로직(예: 온보딩 스킵)을 우회할 가능성이 있습니다.
*   **[Medium] LLM 프롬프트 인젝션:**
    *   `PersonaAnalysisResponse`에 포함되는 `evidence_snippets` 생성 시, 대화 이력이 정제 없이 프롬프트에 포함될 경우 LLM의 시스템 프롬프트를 무력화하거나 오작동을 유도하는 공격이 가능합니다.

---

### 2. 아키텍처 개선 제안

*   **비동기 분석 패턴 도입 (Long-polling/SSE):**
    *   현재 설계는 `/persona/analyze` API에 30초 타임아웃을 설정한 동기(Sync) 방식입니다. 브라우저/Nginx의 기본 타임아웃 설정과 충돌할 가능성이 높고, 분석 중 네트워크 단절 시 상태 복구가 어렵습니다. 작업을 `Task ID` 기반으로 생성하고 상태를 폴링하는 방식으로 전환을 권장합니다.
*   **상태 전이 무결성 검증 로직 강화:**
    *   프론트엔드 `useReducer`에서 `ready` 상태로 전이할 때, 반드시 서버의 `LawyerPersona` 존재 여부를 재검증(Revalidation)하는 로직이 필요합니다. 클라이언트 상태만 믿고 대시보드를 노출할 경우 데이터 불일치가 발생할 수 있습니다.
*   **HybridReview 컴포넌트 결합도 분리:**
    *   `PersonaHybridReview`가 AI 추천 확인과 인라인 수정을 동시에 처리하므로 복잡도가 높습니다. 데이터 수정을 위한 `EditablePersonaCard`와 같은 원자적 컴포넌트로 분리하여 재사용성을 높여야 합니다.

---

### 3. API 계약 정합성 검토

*   **타입 불일치 주의:**
    *   백엔드(Python)의 `dict[str, float]`와 프론트엔드(TS)의 `Record<string, number>` 간 매핑은 안전해 보이나, `TrendCategory` Enum 값이 키로 사용될 때 양측의 Enum 정의가 동기화되지 않으면 런타임 에러가 발생합니다. (공통 스키마 생성 도구 사용 권장)
*   **누락된 필드:**
    *   `LawyerPersona` 스키마(v2.0)에는 `insights` 필드가 포함되어 있지 않습니다. 설계서의 `PersonaAnalysisResponse`가 기존 `LawyerPersona`를 상속하는지, 아니면 Composition 구조인지 명확히 정의되어야 합니다. (현재 코드상으로는 `insights`가 보이지 않음)

---

### 4. 성능/확장성 제안

*   **분석 결과 캐싱 미비:**
    *   사용자가 `Track1` 분석을 반복 요청할 경우 매번 LLM 비용이 발생합니다. 대화 이력의 `hash` 값을 기반으로 일정 기간 분석 결과를 캐싱하여 응답 속도를 개선하고 비용을 절감해야 합니다.
*   **번들 사이즈 최적화:**
    *   `PersonaAnalysisProgress`에 복잡한 애니메이션 라이브러리(Lottie 등)가 포함될 경우, 설정한 50KB 제한을 초과할 수 있습니다. `framer-motion` 등을 활용한 경량 구현이 필요합니다.

---

### 5. 엣지 케이스 및 장애 시나리오

*   **분석 중 브라우저 새로고침:**
    *   `SessionStorage`에 `track1_analyzing` 상태를 저장하더라도, 백엔드에서 해당 작업이 진행 중인지 알 수 있는 `Job ID`가 없으므로 다시 API를 호출하게 됩니다. 중복 요청 방지 로직이 필수적입니다.
*   **동시 요청(Race Condition):**
    *   온보딩 완료(`COMPLETE_ONBOARDING`) 요청을 연타할 경우, DB에 동일 유저의 페르소나가 중복 생성될 위험이 있습니다. `user_id`에 대한 Unique 제약 조건 또는 분산 락 검토가 필요합니다.
*   **네트워크 단절 시 복구:**
    *   `SessionStorage`에 저장된 `draftOnboarding` 데이터가 부분적으로 손상되었을 경우를 대비한 유효성 검사 로직이 프론트엔드 진입 시점에 존재해야 합니다.

---

**Red Team 결론:**
전반적인 UX 흐름은 논리적이나, **동기식 LLM 호출에 따른 타임아웃 위험**과 **임시 유저 ID 사용에 따른 보안 구멍**이 가장 시급한 개선 과제입니다. 특히 PII 마스킹은 설계 이후가 아닌 구현 즉시 적용되어야 할 Critical 요소입니다.
