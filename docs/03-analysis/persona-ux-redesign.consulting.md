# 외부 컨설팅 보고서 — 페르소나 선택 UX 개선

> **검증 대상**: `docs/01-plan/features/persona-ux-redesign.plan.md` v1.0
> **검증 도구**: Codex CLI (External Consultant, gpt-5.3-codex)
> **일시**: 2026-02-28

---

## 1. 격차 분석 (현재 vs 목표)

**총평**: 현재 기획은 "온보딩 UX 개선"으로는 방향이 좋지만, 상위 서비스(Notion AI/Jasper/Copy.ai)가 이미 제공하는 **컨텍스트 레이어·워크플로우 자동화·거버넌스·성과 측정** 대비 범위가 좁습니다.

### 강점
- Track 1/2 통합, 하이브리드 리뷰, 인라인 편집 설계는 실사용 전환율 개선 가능성이 큼
- 법률 도메인 톤을 의식한 문제 정의(P5)가 명확함

### 주요 격차

| ID | 격차 | 설명 |
|----|------|------|
| G1 | 컨텍스트 지속성 | 페르소나를 "초기 설정값"으로 보며, 상위 제품처럼 조직 지식/브랜드/사용자 맥락을 지속 반영하는 계층이 약함 |
| G2 | 신뢰성 UX | 신뢰도 차트는 있으나, 법률 도메인에서 중요한 근거 출처/검증 가능성(citation, provenance) 수준이 부족 |
| G3 | 운영 관점 | 관리자 제어, 권한, 감사로그, ROI 대시보드 등 B2B 확장 핵심이 빠짐 |
| G4 | 자동화 깊이 | 온보딩 이후 반복 작업 자동화(에이전트/워크플로우) 연결이 약함 |
| G5 | 측정체계 | 목표 KPI는 있으나 실험 설계(A/B, 코호트, 실패원인 taxonomy)가 미정 |

---

## 2. 기술 고도화 제안

### 2025-2026 트렌드 반영 우선순위
1. **Agentic architecture** 도입: 페르소나 설정을 단발 UX가 아니라 이후 콘텐츠 생성 파이프라인까지 연결
2. **Model-agnostic** 설계: 모델 교체/혼합 가능 구조(비용·품질·규제 대응)
3. **외부 컨텍스트 표준 연결**(MCP 등): CRM/문서/이전 상담기록과 안전하게 연결
4. **거버넌스·컴플라이언스**: 권한/감사/데이터 경계 명시(특히 법률·B2B)

### 구현 권고(실행형)
- Persona 서비스 레이어 분리: `persona_profile`, `evidence`, `confidence`, `last_validated_at`
- Track 전환을 상태머신으로 고정: `draft-preserve`, `rollback`, `error-recovery`
- 관측성: `start→complete`, `edit depth`, `AI accept/reject reason`, `time-to-first-value`

---

## 3. UX 개선 제안

### 핵심 UX 수정
- 레이더 차트 단독 사용 지양: 법률 사용자에게는 "점수"보다 **근거 카드 + 최근 사례/질문 출처**가 더 신뢰를 줌
- Hybrid Review에 "왜 이런 추천인지"를 1클릭 펼침으로 제공
- "처음부터 설정" 대신 "추천값 유지하고 일부만 수정" 기본 경로 강조

### 전환율 최적화
- Welcome에서 즉시 가치 증명: 샘플 결과 1개 + 예상 소요시간 + 데이터 사용범위
- 실패 UX 표준화: 분석 실패 시 자동 Track2 전환 + 이미 수집한 값 자동 채움
- 90초 목표 달성을 위해 필수입력 2개(전문분야/톤)만 선완료 처리, 나머지는 사후 미세조정

---

## 4. AI/ML 고도화 제안

### 모델링
- 페르소나 추출을 단순 분류가 아닌 `추론 + 근거검색(RAG)` 결합으로 전환
- 확률값은 캘리브레이션 후 노출(과신 방지)

### 학습 루프
- 사용자 수정 로그를 라벨로 축적: `AI 제안 대비 수정량` 기반 개인화
- 오프라인 평가셋(법률 도메인) + 온라인 실험 병행

### 신뢰성
- 근거 출처 링크, 최신성 신호, 충돌 감지(과거 페르소나 vs 최근 행동) 제공
- 민감정보 필터링/마스킹 파이프라인을 persona 생성 전에 강제

---

## 5. 비즈니스 전략 제안

### 수익화 구조
- **Free**: 기본 페르소나 수동 설정
- **Pro**: AI 자동분석 + 하이브리드 편집 + 개인화 학습
- **Team/Enterprise**: 조직 공통 페르소나, 권한/감사로그, 성과 대시보드, 외부 시스템 연동

### 확장 전략
- 법률 특화 템플릿 번들(형사/민사/가사)로 초기 리텐션 강화
- 로펌/법무팀 대상 "조직 지식 레이어" 판매(좌석+사용량 혼합 과금)
- KPI를 매출지표와 직접 연결: `설정완료율`보다 `콘텐츠 생성 전환율`, `재사용률`, `고객유지율` 중심으로 관리

---

## 출처
- Notion AI: https://www.notion.com/product/ai
- Jasper IQ/브랜드보이스: https://www.jasper.ai/jasper-iq
- Copy.ai: https://www.copy.ai/
- OpenAI Agents API: https://platform.openai.com/docs/changelog
- MCP: https://docs.anthropic.com/en/docs/mcp
- Harvey AI Legal Workflows: https://www.harvey.ai/blog/introducing-workflow-builder
- Thomson Reuters CoCounsel: https://www.thomsonreuters.com
