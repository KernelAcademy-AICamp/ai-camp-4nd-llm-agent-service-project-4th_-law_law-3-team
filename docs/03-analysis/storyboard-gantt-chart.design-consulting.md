# 외부 컨설팅 보고서: 스토리보드 간트차트 설계

> 검증 대상: `docs/02-design/features/storyboard-gantt-chart.design.md`
> 검증 도구: Codex CLI (External Consultant, gpt-5.3-codex)
> 검증일: 2026-03-01

---

## 외부 컨설팅 보고서

기준일: 2026년 3월 1일 (문서 + 현재 코드 동시 검토)

### 1. 격차 분석 (현재 설계 vs 목표)

| 항목 | 현재 수준 | Everlaw/Relativity/CoCounsel 수준 | 격차 |
|---|---|---|---|
| 증거-사실 연결 | 설계상 목표는 명확하나 구현은 아직 카드형 타임라인 중심 | 사실(Fact) 단위 + 증거 링크 + 검토 워크플로우 결합 | 큼 |
| AI 신뢰성 | 요약/추출 중심, 근거 검증 체계는 부분 설계 | 근거 인용, 설명, 검증 루프(예: citation verification) 내장 | 큼 |
| 에이전트 운영 방식 | 단일 요청 처리 중심, 대규모 비동기/거버넌스는 범위 외 | 다단계 위임형(agentic) + human oversight 내장 | 큼 |
| 엔터프라이즈 거버넌스 | 감사·승인·보존정책 일부 범위 외 | 감사추적/권한/정책 준수 운영이 기본 | 매우 큼 |

핵심 관찰:
1. 설계 문서의 목표 기능(`analyze-batch`, `merge`, `EvidenceFile`)과 현재 구현 사이 간극이 큽니다.
2. 핵심 보안/거버넌스 항목이 계획에서 범위 외로 분리되어 있습니다.
3. API-Frontend 계약 규칙이 문서 내부에서 상충됩니다(snake_case 고정 vs camelCase 정정).

### 2. 기술 고도화 제안

1. `agentic workflow + human checkpoint`를 1급 기능으로 승격하세요. 단순 생성이 아니라 "검토 가능한 실행" 모델로 가야 합니다.
2. 모든 LLM 출력을 `strict schema` 기반 구조화 출력으로 통제하고, 근거 span/증거 ID를 필수 필드로 강제하세요.
3. 검색은 Dense 단독이 아니라 `Dense + BM25/FTS + citation graph` 하이브리드로 표준화하세요.
4. 대용량 파일 처리는 API 서버 직처리보다 `큐 기반 비동기 워커`로 분리하세요.
5. `eval-driven` 운영(회귀 평가 + trace grading + 배포 게이트)을 CI/CD에 붙이세요.

### 3. 데이터 모델 개선 제안

1. `Case / TimelineEvent / Evidence / EventEvidenceLink(N:M)` 정규화 모델을 영속 DB로 승격하세요.
2. `evidence_ids[]` 같은 배열 연결보다 링크 테이블에 `source_span`, `link_confidence`, `link_type`를 두세요.
3. 이벤트 버전(`event_revision`)과 감사로그(`who/when/why`)를 추가해 법률 업무 추적성을 확보하세요.
4. 현재 ingest 테이블 다종 분화는 유지하되, 조회용으로는 canonical 뷰/슈퍼타입을 도입해 API 복잡도를 낮추세요.
5. 검색 규모 대비 파티셔닝/인덱스 전략(날짜, 사건유형, 조직단위)을 초기에 확정하세요.

### 4. API 설계 개선 제안

1. 전역 버저닝 `/api/v1/...`을 즉시 도입하세요. 현재는 `/api/...` 고정이며 버전 경로가 없습니다.
2. 표준 에러 스키마(`code`, `message`, `details`, `request_id`)를 공통 적용하세요. 현재는 `detail` 중심 HTTPException이 모듈별로 산재합니다.
3. 페이지네이션은 `cursor` 기반을 기본값으로 하세요. 현재 다수 API가 `limit`만 있고 `next_cursor`가 없습니다.
4. 배치 분석 API는 `idempotency-key`와 job resource(`GET /jobs/{id}`)를 정식 계약화하세요.
5. OpenAPI에 오류 코드 매트릭스(4xx/5xx)와 예제 payload를 모듈 공통으로 명시하세요.

### 5. UX 개선 제안

1. 카드 뷰 중심에서 `간트 + 원문/증거 패널` 2-패널 작업공간으로 전환하세요.
2. 이벤트 클릭 시 증거 원문 하이라이트, 증거 선택 시 이벤트 후보 역탐색을 기본 UX로 넣으세요.
3. 접근성: 토글/상태 버튼에 `aria-label`, `aria-expanded`, `aria-controls`를 추가하고 키보드 포커스 흐름을 명시 설계하세요.
4. 모바일은 "조회 전용 모드"를 독립 UI로 설계하고 편집 기능은 데스크톱 우선 분리 유지가 맞습니다.
5. 신뢰도/충돌/누락 인사이트를 시각 레이어로 분리(색상/아이콘/필터)하세요.

### 6. 테스트 전략 개선 제안

1. 현재 프론트엔드 테스트 스크립트가 없고 수동 검증 중심입니다.
2. 스토리보드 관련 테스트는 존재하지만(총 3개 파일) 신규 핵심 기능(batch upload/merge/security) 커버리지가 비어 있습니다.
3. 권장 체계:
   1. 단위: parser/merger/validator/prompt-guard.
   2. 계약: FE-BE schema contract test.
   3. 통합: 파일 업로드-분석-병합-SSE.
   4. E2E: 대표 사용자 시나리오(직장 괴롭힘, 순서 뒤섞임).
4. CI 게이트로 `coverage threshold`, `LLM eval regression`, `a11y smoke`를 추가하세요.

### 7. 보안 강화 제안

1. 인증 기본값을 "비활성"에서 "필수"로 바꾸세요(운영 환경). 현재 `API_KEY` 비어 있으면 인증 우회됩니다.
2. 업로드 보안은 확장자+크기 검증을 넘어 `magic number`, MIME 일치, 샌드박스 스캔까지 확대하세요.
3. 증거 접근은 UUID만으로 충분하지 않습니다. 사건/사용자/역할 기반 ACL을 강제하세요.
4. Rate limiting은 메모리 스토리지 기본값 대신 Redis 기반 분산 제한으로 전환하세요.
5. PII 마스킹, 보존/삭제 정책, 감사로그를 기능 범위 밖이 아니라 MVP 보안요건으로 재정의하세요.

### 8. 구현 전략 제안

1. **Phase 0 (2주)**: API 표준(버전/에러), 보안기반(ACL, 업로드 게이트), 데이터 canonical 모델 확정.
2. **Phase 1 (3주)**: 다중 증거 ingest + batch analyze + job orchestration + 최소 eval 파이프라인.
3. **Phase 2 (3주)**: 간트 UX(양방향 탐색, 충돌/누락 인사이트, 신뢰도 배지) + 접근성 자동검증.
4. **Phase 3 (2주)**: 운영화(비용 가드레일, 모니터링, 회귀평가, 롤백 전략).
5. 리스크 관리:
   1. 모델 품질 하락: 배포 전 eval gate.
   2. 비용 폭증: 사용자/사건 단위 quota + circuit breaker.
   3. 보안사고: 업로드 샌드박스 + 감사지표.
   4. 일정지연: 기능 플래그로 단계 배포.

---

### 참고 외부 근거 (2026-03-01 기준)

- [Everlaw Fact Timelines (2025-12)](https://support.everlaw.com/hc/en-us/articles/42469701435803-Storybuilder-Fact-Timelines)
- [Relativity aiR for Review](https://help.relativity.com/RelativityOne/Content/Relativity/aiR_for_Review/aiR_for_Review.htm)
- [Thomson Reuters CoCounsel Legal (2025-08)](https://www.thomsonreuters.com/en/press-releases/2025/august/thomson-reuters-launches-cocounsel-legal-transforming-legal-work-with-agentic-ai-and-deep-research)
- [EU AI Act](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai)
- [OpenAI Evaluation Best Practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices)
- [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
- [LangGraph Checkpoint/HITL](https://langchain-ai.github.io/langgraphjs/reference/modules/langgraph-checkpoint.html)
