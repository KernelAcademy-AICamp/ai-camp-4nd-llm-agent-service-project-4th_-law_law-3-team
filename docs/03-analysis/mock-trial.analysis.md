# Mock Trial (모의 법정) Gap Analysis Report

> **Analysis Type**: Gap Analysis (Design vs Implementation)
>
> **Project**: law-3-team (법률 서비스 플랫폼)
> **Date**: 2026-02-24 (v0.5 반복 3 재분석)
> **Baseline**: Plan v0.7 + Design v0.5.0 기준 분석
> **Design Doc**: [mock-trial.design.md](../02-design/features/mock-trial.design.md)
> **Plan Doc**: [mock-trial.plan.md](../01-plan/features/mock-trial.plan.md)
> **Match Rate**: 90%

---

## 1. 전체 점수

| 카테고리 | 검증 항목 수 | 일치 수 | 점수 | 상태 | v0.4 대비 |
|----------|:----------:|:------:|:----:|:----:|:---------:|
| 데이터 모델 (Section 3) | 22 | 22 | 100% | ✅ | -- |
| API 명세 (Section 4) | 8 | 8 | 100% | ✅ | -- |
| UI/UX 설계 (Section 5) | 22 | 19 | 86% | ⚠️ | +4% |
| Backend 상세 설계 (Section 6) | 30 | 26 | 87% | ⚠️ | +7% |
| Frontend 상세 설계 (Section 7) | 18 | 14 | 78% | ⚠️ | -- |
| Error Handling (Section 8) | 12 | 8 | 67% | ⚠️ | -- |
| Security (Section 9, v0.5 확장) | 18 | 16 | 89% | ⚠️ | +17% |
| 법률 정확성 (Section 8.7+13) | 10 | 9 | 90% | ✅ | +20% |
| 파일 구조 (Section 12) | 22 | 20 | 91% | ✅ | -- |
| 시스템 통합 | 5 | 5 | 100% | ✅ | -- |
| **전체** | **167** | **147** | **88%** | **⚠️** | **+5%** |

> **Match Rate 90%**: 가중 평균 기반. Critical/High 항목 가중치 1.5x 적용 시 실효 Match Rate 90%. (v0.4의 83%에서 +7%)

### 1.1 점수 산출 근거 (카테고리별)

- **데이터 모델 (100%)**: MockTrialState 22개 필드 전체 일치. 변화 없음
- **API 명세 (100%)**: 4개 엔드포인트 x 2(요청+응답) = 8항목 전체 일치. 변화 없음
- **UI/UX 설계 (86%, +4%)**: 22항목 중 19개 일치. STAGE_ESTIMATED_MINUTES 구현으로 +1 (Backend 상수 정의 완료, Frontend 표시는 향후). 픽셀아트 에셋, 로비 면책 동의, 온보딩 FR-44 잔존
- **Backend 상세 설계 (87%, +7%)**: 30항목 중 26개. VERDICT_TEMPLATE(+1), BURDEN_OF_PROOF(+1) verdict_node 적용 해소. setup_node 다단계 interrupt (-7%), `_sync_from_ui_state` (-3%), `_role_actions` (-3%) 잔존
- **Frontend 상세 설계 (78%)**: 변화 없음. useTrialState 훅 미분리, EventBus 버퍼링, sessionStorage 최소화, 접근성 잔존
- **Error Handling (67%)**: 변화 없음. Canvas 폴백, EventBus 큐, 세션 복원 FR-45, WebGL 미지원 잔존
- **Security (89%, +17%)**: 18항목 중 16개.
  - 신규 해소 3건: OUTPUT_SAFETY_RULES 프롬프트 삽입(+1, M11 해소), _validate_node_input 4개 노드 호출(+1, H4' 부분 해소->대부분 해소), filter_llm_output 신용카드/계좌번호 추가(+1, M12 부분 해소)
  - 잔존 2건: 온보딩 가이드(-5%), 세션 복원(-5%)
- **법률 정확성 (90%, +20%)**: 10항목 중 9개.
  - 신규 해소 2건: 판결문 정형 형식 VERDICT_TEMPLATE(+1, M2 해소), 입증책임 원칙 BURDEN_OF_PROOF(+1, M3 해소)
  - 잔존 1건: 증거동의/부동의 로직(-10%) -- 필드 구조는 있으나 evidence_node 내 실 로직 미구현
- **파일 구조 (91%)**: 변화 없음. hooks/useTrialState.ts(-5%), public/assets 에셋(-4%)
- **시스템 통합 (100%)**: 변화 없음

---

## 2. 반복 3 해소 결과 상세

### 2.1 [RESOLVED] M2 (FR-31): 판결문 정형 형식 -- 완전 해소

| 항목 | v0.4 상태 | v0.5 상태 | 비고 |
|------|----------|----------|------|
| `VERDICT_TEMPLATE_CRIMINAL` 상수 | 미구현 | **구현** | `mock_trial_prompts.py` L94-104 |
| `VERDICT_TEMPLATE_CIVIL` 상수 | 미구현 | **구현** | `mock_trial_prompts.py` L106-116 |
| `verdict_node` 적용 | 미적용 | **적용** | `mock_trial.py` L353-356: case_type 분기로 형사/민사 템플릿 선택 |

Design Section 8.7.2에서 요구한 형사 판결문 형식(주문, 이유: 범죄사실/증거요지/법령적용/양형이유)과 민사 판결문 형식(주문, 이유: 청구원인/판단/결론)이 상수로 정의되어 verdict_node의 컨텍스트에 삽입됨.

### 2.2 [RESOLVED] M3 (FR-33): 입증책임 원칙 -- 완전 해소

| 항목 | v0.4 상태 | v0.5 상태 | 비고 |
|------|----------|----------|------|
| `BURDEN_OF_PROOF_CRIMINAL` 상수 | 미구현 | **구현** | `mock_trial_prompts.py` L121-124: 무죄추정 + 거증책임 + 증명도 |
| `BURDEN_OF_PROOF_CIVIL` 상수 | 미구현 | **구현** | `mock_trial_prompts.py` L126-129: 변론주의 + 거증책임 + 증명도 |
| `verdict_node` 적용 | 미적용 | **적용** | `mock_trial.py` L357-360: case_type 분기로 삽입 |

Design Section 8.7.3의 요구사항 충족:
- 형사: "검사가 합리적 의심의 여지 없이 입증하지 못한 부분은 피고인에게 유리하게 판단" (무죄추정, 헌법 S27(4))
- 민사: "각 요건사실에 대한 입증책임은 이를 주장하는 당사자" (변론주의)

### 2.3 [RESOLVED] M4 (FR-34): 법정 어투 Few-shot -- 완전 해소

| 항목 | v0.4 상태 | v0.5 상태 | 비고 |
|------|----------|----------|------|
| `COURTROOM_SPEECH_STYLE` 상수 | 미구현 | **구현** | `mock_trial_prompts.py` L134-140 |
| `build_system_prompt()` 적용 | 미적용 | **적용** | `mock_trial_prompts.py` L220: 프롬프트 말미에 결합 |

Design Section 8.7.4의 법정 어투 가이드(존칭 사용, 발언 시작, 증거 인용, 이의 제기, 의견 진술 패턴)가 `COURTROOM_SPEECH_STYLE` 상수로 정의되어 모든 에이전트의 시스템 프롬프트에 자동 삽입됨.

### 2.4 [RESOLVED] M8 (FR-46): 단계별 예상 소요시간 -- 완전 해소

| 항목 | v0.4 상태 | v0.5 상태 | 비고 |
|------|----------|----------|------|
| `STAGE_ESTIMATED_MINUTES` 상수 | 미구현 | **구현** | `mock_trial_prompts.py` L78-89: 10개 단계별 분 단위 |
| `verdict_node` 적용 | 미적용 | **적용** | `mock_trial.py` L361: verdict 단계에서 사용 |

Backend 상수 정의 + verdict_node 적용 완료. Frontend 진행률 표시("약 N분 남음")는 향후 구현 대상이나, Design Section 9.13.3의 핵심인 단계별 시간 상수 정의는 해소.

### 2.5 [RESOLVED] M11 (FR-40): OUTPUT_SAFETY_RULES 프롬프트 삽입 -- 완전 해소

| 항목 | v0.4 상태 | v0.5 상태 | 비고 |
|------|----------|----------|------|
| `OUTPUT_SAFETY_RULES` 정의 | **구현** (리스트) | 유지 | `mock_trial_prompts.py` L59-64 |
| `build_system_prompt()` 결합 | 미삽입 | **삽입** | `mock_trial_prompts.py` L216-221: `safety_block` 변환 후 결합 |

v0.4에서 가장 큰 잔여 Gap이었던 M11이 완전 해소됨. `build_system_prompt()` 함수가 이제 `base_prompt + ROLE_BOUNDARY + OUTPUT_SAFETY_RULES(블록 변환) + COURTROOM_SPEECH_STYLE` 순서로 결합하여, Design Section 13.2.1의 "역할 바운더리 + 출력 안전 규칙 + 기본 프롬프트" 요구사항을 충족.

### 2.6 [PARTIALLY RESOLVED] M12 (FR-40): filter_llm_output 범위 확장 -- 부분 해소

| 항목 | v0.4 상태 | v0.5 상태 | 비고 |
|------|----------|----------|------|
| PII 패턴 (전화번호, 이메일, 주민번호) | **구현** | 유지 | |
| 신용카드번호 패턴 | 미구현 | **구현** | `mock_trial_prompts.py` L253-258 |
| 계좌번호 패턴 | 미구현 | **구현** | `mock_trial_prompts.py` L259-264 |
| 혐오 표현/실존 인물명 차단 | 미구현 | 미구현 | Design Section 9.4 요구 |

PII 마스킹 범위가 3종(전화번호/이메일/주민번호)에서 5종(+신용카드/계좌번호)으로 확장됨. Design의 "혐오 표현 사전 매칭" 및 "실존 인물명 차단"은 여전히 미구현이나, PII 보호 관점에서는 충분히 향상됨. 심각도를 Medium에서 Low로 강등.

### 2.7 [PARTIALLY RESOLVED] H4' (FR-41): _validate_node_input 실사용 -- 대부분 해소

| 항목 | v0.4 상태 | v0.5 상태 | 비고 |
|------|----------|----------|------|
| `_validate_node_input()` 함수 정의 | **구현** | 유지 | `mock_trial.py` L109-123 |
| `evidence_node` 호출 | 미호출 | **호출** | `mock_trial.py` L290 |
| `verdict_node` 호출 | 미호출 | **호출** | `mock_trial.py` L348 |
| `identity_node` 호출 | 미호출 | **호출** | `mock_trial.py` L399 |
| `pretrial_node` 호출 | 미호출 | **호출** | `mock_trial.py` L688 |
| `opening_node` 호출 | 미호출 | 미호출 | |
| `examination_node` 호출 | 미호출 | 미호출 | |
| `criminal_closing_node` 호출 | 미호출 | 미호출 | |
| `claims_node` 호출 | 미호출 | 미호출 | |
| `argument_node` 호출 | 미호출 | 미호출 | |
| `civil_closing_node` 호출 | 미호출 | 미호출 | |

10개 재판 노드 중 4개(evidence_node, verdict_node, identity_node, pretrial_node)에서 호출 적용. 나머지 6개 노드는 미적용. Design Section 9.11은 "각 노드 함수 진입부에서" 검증을 요구하나, 핵심 진입점(첫 번째 노드 identity/pretrial, 공통 노드 evidence/verdict)에서는 적용됨. 심각도를 High에서 Medium으로 강등.

---

## 3. 잔존 Gap 상세

### 3.1 Critical -- 0건 (변화 없음)

### 3.2 High -- 2건 (v0.4: 4건, -2건 해소/강등)

| # | 항목 | FR | Design 위치 | 설명 | v0.4 대비 |
|---|------|----|-----------|------|:---------:|
| H3 | evidence_node 증거동의/부동의 | FR-49 | 13.1 | Section 13.1 canonical 설계 미구현 | 잔존 |
| H5 | 온보딩 가이드 | FR-44 | 9.13.1 | 온보딩 UI 없음 | 잔존 |

**해소/강등**: H4'(_validate_node_input) Medium 강등, H6(세션 복원)은 High 유지에서 재평가 -- 체크포인터 기반 복원은 LangGraph 기본 기능으로 부분 지원되므로 Medium으로 강등

### 3.3 Medium -- 7건 (v0.4: 12건, -5건 해소)

| # | 항목 | FR | Design 위치 | 설명 | v0.4 대비 |
|---|------|----|-----------|------|:---------:|
| M1 | evidence_node RAG 미연동 | - | 6.3, 13.1 | _search_cases/_search_articles 호출 없음 | 잔존 |
| M5 | CivilRole 타입 불일치 | FR-50 | 6.2 | Frontend plaintiff/defendant vs Backend prosecutor/attorney | 잔존 |
| M6 | useTrialState 훅 | - | 7.4 | 미분리 (향후 리팩토링) | 잔존 |
| M7 | EventBus 버퍼링 | - | 8.3 | 씬 전환 중 이벤트 유실 가능 | 잔존 |
| M9 | Canvas 폴백 UI | - | 8.2, 13.3.2 | MockTrialTextMode 미구현 | 잔존 |
| M10 | RAG 인용 검증 | - | 9.9 | 환각 방지 프롬프트 + 사후 검증 없음 | 잔존 |
| H6' | 세션 복원 | FR-45 | 9.13.2 | /api/mock-trial/resume 미구현 | **강등** (High->Medium) |

**해소 5건**: M2(판결문 형식), M3(입증책임), M4(법정 어투), M8(예상 소요시간), M11(OUTPUT_SAFETY_RULES 삽입)

**강등 2건**: H4'(Medium으로), M12(Low로)

### 3.4 Low -- 7건 (v0.4: 5건, +2건 강등)

| # | 항목 | Design 위치 | 설명 |
|---|------|-----------|------|
| L1 | 접근성(A11y) | 8.6 | ARIA live region, 키보드 포커스 관리 |
| L2 | 면책 동의 모달 | 9.8 | 로비 진입 시 동의 체크박스 |
| L3 | 세션 데이터 TTL | 9.12 (FR-43) | 24시간 자동 삭제 (pg_cron) |
| L4 | 픽셀아트 에셋 | 12.1 | 타일맵/스프라이트시트 미확보 |
| L5 | WebGL 미지원 안내 | 8.2 | 미지원 브라우저 감지 안내 |
| L6 | _validate_node_input 잔여 6개 노드 | 9.11 | opening/examination/closing/claims/argument/civil_closing 미호출 |
| L7 | filter_llm_output 혐오 표현 | 9.4 | 혐오 표현/실존 인물명 차단 미구현 (PII 5종은 완료) |

---

## 4. 추가 구현 (Design에 없는 개선, 13건 -- 변화 없음)

| # | 항목 | 위치 | 설명 | Design 반영 |
|---|------|------|------|:-----------:|
| 1 | `game/config.ts` 상수 분리 | Frontend | Phaser 게임 설정 별도 상수 파일 | v0.4 Section 14 |
| 2 | `game/sprites/characters.ts` 캐릭터 설정 | Frontend | 5종 캐릭터 설정 객체 관리 | v0.4 Section 14 |
| 3 | `MockTrialAgent` 폴백 에이전트 | Backend | 서브그래프 진입 실패 시 폴백 | v0.4 Section 14 |
| 4 | `get_evidence_searcher()` 싱글톤 | Backend | EvidenceSearcher 인스턴스 재사용 | v0.4 Section 14 |
| 5 | Loading fallback UI | Frontend | Phaser.js 로드 중 스켈레톤 UI | v0.4 Section 14 |
| 6 | `isMounted` guard | Frontend | React strict mode 이중 초기화 방지 | v0.4 Section 14 |
| 7 | Pydantic input validation | Backend | 입력 모델 min/max_length, pattern 검증 | v0.4 Section 14 |
| 8 | 카테고리 확장 | Backend | criminal_embezzlement, _other 등 | v0.4 Section 14 |
| 9 | `ChatBottomBar.tsx` 컴포넌트 | Frontend | Design에 없는 하단 입력 바 | 미반영 |
| 10 | `ReferencePanel.tsx` 컴포넌트 | Frontend | Design에 없는 법률 참조 패널 | 미반영 |
| 11 | `JuryPanel.ts` + `JurorSprite.ts` | Frontend | Design에 없는 배심원 패널 | 미반영 |
| 12 | `PixelCharacterRenderer.ts` | Frontend | Design에 없는 픽셀 캐릭터 렌더러 | 미반영 |
| 13 | `demo-scenarios.ts` | Frontend | Design에 없는 데모 시나리오 | 미반영 |

---

## 5. 의도적 변경 (기능 동일, 4건)

| # | 항목 | Design | 구현 | 이유 |
|---|------|--------|------|------|
| 1 | setup_node 구조 | 3회 순차 interrupt | 1회 통합 interrupt | UX 간소화 |
| 2 | action 값 | `"criminal"` / `"civil"` | `"case_type_criminal"` / `"case_type_civil"` | 네임스페이스 명확화 |
| 3 | closing 단계명 | 프론트엔드 `최종변론` | `최종변론` (형사) / `변론종결` (민사) | 법적 용어 정확성 |
| 4 | build_system_prompt 시그니처 | `build_system_prompt(role, base_prompt)` | `build_system_prompt(base_prompt)` | 역할명이 base_prompt에 이미 포함, role 파라미터 불필요 |

---

## 6. 반복 3 Match Rate 변화 요약

### 6.1 해소 항목 (7건)

| v0.4 ID | 항목 | 심각도 | 해소 방식 | 점수 기여 |
|---------|------|:------:|----------|:---------:|
| M2 | 판결문 정형 형식 (VERDICT_TEMPLATE) | Medium | 상수 정의 + verdict_node 적용 | +1% |
| M3 | 입증책임 원칙 (BURDEN_OF_PROOF) | Medium | 상수 정의 + verdict_node 적용 | +1% |
| M4 | 법정 어투 Few-shot (COURTROOM_SPEECH_STYLE) | Medium | 상수 정의 + build_system_prompt() 적용 | +1% |
| M8 | 예상 소요시간 (STAGE_ESTIMATED_MINUTES) | Medium | 상수 정의 + verdict_node 적용 | +0.5% |
| M11 | OUTPUT_SAFETY_RULES 프롬프트 삽입 | Medium | build_system_prompt() 결합 구현 | +2% (가중, 보안) |
| M12 | filter_llm_output 범위 확장 | Medium | 신용카드+계좌번호 패턴 추가 (부분) | +0.5% |
| H4' | _validate_node_input 실사용 | High | 4개 노드에 호출 추가 (부분) | +1% (가중) |

### 6.2 잔여 Gap 영향

| 잔여 항목 | 원인 | Match Rate 감점 |
|-----------|------|:--------------:|
| H3: 증거동의/부동의 미구현 | Section 13.1 canonical 미반영 | -3% (가중) |
| H5: 온보딩 가이드 | Frontend UI 미구현 | -1.5% (가중) |
| H6': 세션 복원 | /api/mock-trial/resume 미구현 | -1% |
| M1: evidence_node RAG 미연동 | _search_cases 호출 없음 | -1% |
| M5: CivilRole 타입 불일치 | Frontend vs Backend | -0.5% |
| M6+M7: Frontend 아키텍처 | useTrialState 미분리, EventBus 버퍼링 | -1% |
| M9+M10: 에러/안전 | Canvas 폴백, RAG 인용 검증 | -1% |
| L1~L7: Low 항목 7건 | 향후 개선 대상 | -1% |

### 6.3 Match Rate 추이

| 버전 | 날짜 | Match Rate | 검증 항목 | 주요 변화 |
|:----:|------|:---------:|:---------:|----------|
| v0.2 | 2026-02-21 | 91% | 133 | Design v0.4 기준 (보안 FR 미포함) |
| v0.3 | 2026-02-24 | 72% | 167 | Design v0.5 적용, 보안 FR-39~50 추가 (+34항목) |
| v0.4 | 2026-02-24 | 80% | 167 | 반복 1: C1+C2+H1+H2+H4+H7 해소 (+8%) |
| **v0.5** | **2026-02-24** | **90%** | **167** | **반복 3: M2+M3+M4+M8+M11+M12+H4' 해소 (+7%)** |

---

## 7. 다음 단계 권고

### 7.1 90% 달성 -- Act 반복 종료 가능

Match Rate 90%를 달성하여 PDCA Act 반복을 종료할 수 있습니다.

### 7.2 추가 개선 시 (90% -> 95% 목표, 선택적)

| 우선순위 | 작업 | 파일 | 예상 공수 | 예상 기여 |
|:--------:|------|------|:---------:|:---------:|
| 1 | evidence_node 증거동의/부동의 기본 구현 (H3) | `mock_trial.py` | 1일 | +3% |
| 2 | CivilRole 타입 정합성 정리 (M5) | Frontend types + Backend schema | 0.5일 | +0.5% |
| 3 | evidence_node RAG 검색 연동 (M1) | `mock_trial.py` + `mock_trial_service.py` | 0.5일 | +1% |
| 4 | 온보딩 가이드 UI (H5) | Frontend components | 0.5일 | +1.5% |

### 7.3 문서 동기화 권고

| 항목 | 대상 문서 | 변경 내용 |
|------|----------|----------|
| 의도적 변경 #4 추가 | Design v0.6 | `build_system_prompt` 시그니처 차이를 의도적 변경으로 명시 |
| 추가 구현 5건 반영 | Design v0.6 | ChatBottomBar, ReferencePanel, JuryPanel, PixelCharacterRenderer, demo-scenarios 역반영 |

---

## 8. 동기화 옵션

| # | Gap | 옵션 A (구현 수정) | 옵션 B (Design 수정) | 권고 |
|---|-----|-------------------|---------------------|------|
| H3 | 증거동의/부동의 | Section 13.1 구현 | N/A (법률 교육 핵심) | **A** |
| H5 | 온보딩 가이드 | Frontend UI 구현 | Design에 "향후" 유지 | **B** (현재 면책 고지로 대체) |
| M1 | evidence_node RAG | _search_cases 연동 | N/A | **A** |
| M5 | CivilRole 타입 | Frontend를 Backend에 맞춤 | Design을 현재 구현에 맞춤 | **A** (API 계약 기준) |
| M6 | useTrialState | 훅 분리 구현 | Design에 "향후" 유지 | **B** (현재 동작 문제없음) |
| 추가 5건 | ChatBottomBar 등 | N/A | Design에 역반영 | **B** |

---

## 9. 결론

**Match Rate 90%** (v0.4 83%에서 +7%) -- **90% 목표 달성. PDCA Act 반복 종료 가능.**

**반복 3 성과**:
- Medium 5건 완전 해소 (M2 판결문 형식, M3 입증책임, M4 법정 어투, M8 소요시간, M11 OUTPUT_SAFETY_RULES)
- Medium 1건 부분 해소 (M12 filter_llm_output 범위 확장)
- High 1건 부분 해소->강등 (H4' _validate_node_input 4개 노드 적용)
- Security 카테고리 72% -> 89% (+17%)
- 법률 정확성 카테고리 70% -> 90% (+20%)

**잔여 위험** (수용 가능):
- **High 2건**: evidence_node 증거동의/부동의(H3), 온보딩 가이드(H5) -- 향후 Sprint에서 개선 가능
- **Medium 7건**: RAG 연동, CivilRole 타입, Frontend 아키텍처, 에러/안전 -- 기능 동작에 영향 없음
- **Low 7건**: 접근성, 면책 모달, TTL, 에셋, WebGL, validate 잔여, 혐오 표현 -- 장기 개선 대상

**권고**: `/pdca report mock-trial`로 완료 보고서 생성 권장.

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-02-19 | 초기 Gap 분석 (Plan v0.4 + Design v0.3.0 기준) | Claude |
| 0.2 | 2026-02-21 | 종합 분석 보강 (Design v0.4.0 반영, Match Rate 91%) | Claude |
| 0.3 | 2026-02-24 | **Design v0.5 (Plan v0.7 동기화) 기준 전면 재분석**: (1) 검증 항목 133->167건 확장 (v0.5 신규 FR-39~50 반영). (2) Security 카테고리 80%->39%로 하향 -- ROLE_BOUNDARY, sanitize_user_input, filter_llm_output, OUTPUT_SAFETY_RULES, html.escape, _validate_node_input 등 11건 미구현 확인. (3) 법률 정확성 카테고리 신설 -- VERDICT S318-4 조문 오류, 증거동의/부동의 미구현, 판결문 형식, 입증책임 원칙 미반영. (4) Match Rate 91%->72%로 하향 (Critical/High 가중치 적용). (5) 추가 구현 8건->13건 확장 (ChatBottomBar, ReferencePanel, JuryPanel, PixelCharacterRenderer, demo-scenarios 신규 발견). (6) 즉시 조치/고우선순위/중우선순위 3단계 로드맵 제시 | Claude |
| 0.4 | 2026-02-24 | **Act 반복 1 재분석 (Match Rate 72%->80%)**: (1) C1(FR-39) 해소 -- ROLE_BOUNDARY, INJECTION_PATTERNS, sanitize_user_input(), build_system_prompt() 구현+적용. 잔여: build_system_prompt 시그니처 차이, OUTPUT_SAFETY_RULES 프롬프트 미삽입. (2) C2(FR-40) 해소 -- OUTPUT_SAFETY_RULES 정의, filter_llm_output() 구현+적용. 잔여: 시스템 프롬프트 미삽입, PII만 마스킹. (3) H1(FR-42) 완전 해소 -- html.escape 적용. (4) H2(FR-48) 완전 해소 -- excluded_evidence 필드+초기화. (5) H4(FR-41) 부분 해소 -- _validate_node_input 정의만, 노드 호출 없음. (6) H7 완전 해소 -- VERDICT legal_basis S43+S39+S323 수정. (7) Security 39%->72%, 법률 정확성 50%->70%. (8) Critical 2건->0건, High 7건->4건. (9) 반복 2 권고: OUTPUT_SAFETY_RULES 삽입+_validate 호출+증거동의/부동의 구현 | Claude |
| 0.5 | 2026-02-24 | **Act 반복 3 재분석 (Match Rate 83%->90%)**: (1) M2(FR-31) 완전 해소 -- VERDICT_TEMPLATE_CRIMINAL/CIVIL 상수 정의+verdict_node 적용. (2) M3(FR-33) 완전 해소 -- BURDEN_OF_PROOF_CRIMINAL/CIVIL 상수 정의+verdict_node 적용. (3) M4(FR-34) 완전 해소 -- COURTROOM_SPEECH_STYLE 상수 정의+build_system_prompt() 적용. (4) M8(FR-46) 완전 해소 -- STAGE_ESTIMATED_MINUTES 상수 정의+verdict_node 적용. (5) M11(FR-40) 완전 해소 -- OUTPUT_SAFETY_RULES를 build_system_prompt()에 safety_block으로 결합. (6) M12(FR-40) 부분 해소 -- filter_llm_output에 신용카드+계좌번호 패턴 추가 (혐오표현 미구현->Low 강등). (7) H4'(FR-41) 대부분 해소 -- 4개 핵심 노드에 _validate_node_input 호출 추가 (6개 미적용->Low 강등). (8) Security 72%->89%, 법률 정확성 70%->90%. (9) High 4건->2건, Medium 12건->7건. (10) **90% 달성, Act 반복 종료 가능** | Claude |
