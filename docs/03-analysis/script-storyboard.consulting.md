# 외부 컨설팅 보고서 — 대본 + 웹툰 스토리보드

> **검증 대상**: `docs/01-plan/features/script-storyboard.plan.md` (v0.2)
> **검증 도구**: Codex CLI (External Consultant, gpt-5.3-codex)
> **검증일**: 2026-02-28

---

## 1. 격차 분석 (현재 vs 목표)

| 항목 | 현재 기획 수준 | 업계 상위권 기준 | 격차 |
|------|--------------|----------------|------|
| 제품 정의 | "대본+스토리보드 동시 생성" 핵심 가치 명확 | 가치 + KPI(제작시간, 전환율, 재방문율)까지 명시 | KPI/성공지표 부재 |
| 생성 품질 제어 | 패널 수 규칙(2~3/4~8/2~3)만 정의 | 캐릭터 일관성, 스타일 레퍼런스, shot continuity 제어 | 품질 일관성 제어 설계 부족 |
| 협업 기능 | 1인 제작 흐름 중심 | 코멘트, 승인, 버전관리, 팀 워크스페이스 | 협업/리뷰 루프 없음 |
| 운영 안정성 | SSE/Job 구조 초안 존재 | 큐 기반 비동기 처리, 재시도, idempotency, 장애복구 | 대량 처리/실패 복구 설계 미흡 |
| 거버넌스/컴플라이언스 | 거의 없음 | AI 생성물 라벨링, 저작권/출처 기록, 정책 감사로그 | 법률/브랜드 리스크 대응 부족 |
| 수익화 설계 | 기능 중심 | 요금제/크레딧/팀 플랜/사용량 과금 | 비즈니스 모델 미정의 |

> 진단: 아이디어와 기본 UX 구조는 좋지만, "프로덕션급(상위권)"으로 가기 위한 운영/품질/수익화 축이 빠져 있음.

---

## 2. 기술 고도화 제안

- **모델 버전 리스크 대응**: `gemini-3-pro-image-preview`는 preview 성격. 프로덕션은 `gemini-2.5-flash-image`를 기본, `3-pro-image-preview`를 실험 슬롯으로 분리 권장
- **Chain 3 재정의**: "생성 + 검증 + 재시도" 3단계로 확장
- **큐 기반 파이프라인**: `API → Queue → Worker → Object Storage → SSE/Webhook` 전환
- **스키마 메타데이터**: `model_version`, `prompt_version`, `seed`, `cost_ms`, `safety_flags` 추가로 재현성/비용 추적
- **SSE 폴백**: 모바일/불안정 네트워크용 상태 폴링 + resume 토큰 설계
- **관측성**: OpenTelemetry GenAI semantic convention 기반 통합

---

## 3. UX 개선 제안

- **3단계 진행 UI**: 대본→패널 계획→이미지 단계 + ETA 명확 노출
- **패널 카드 액션**: 재생성, 스타일 고정, 대사 편집, 샷 변경(클로즈업/와이드) 바로 제공
- **스타일 레퍼런스 이미지 업로드**: 브랜드 톤 일관성 확보
- **협업 UX**: 패널별 코멘트, 승인 상태, 변경 이력(diff)
- **모바일 역할 분리**: 생성 모니터링 중심 단순화, 편집은 태블릿/데스크톱

---

## 4. AI/ML 고도화 제안

- **자동 품질 평가**: 장면-대본 정합성, 법률 키워드 정확도, CTA 적합성 루브릭 기반 평가 → 배포 게이트
- **프롬프트 최적화**: 멀티암드 밴딧/온라인 실험으로 템플릿 지속 개선
- **법률 도메인 안전장치**: 환각 탐지, 금지 표현 필터, 법률 조언 고지문 자동 삽입
- **캐릭터/세계관 일관성 모델**: 독립 분리
- **학습 데이터 플라이휠**: 사용자 수정 행동(재생성/편집/삭제)을 피드백 신호로 수집 → 프롬프트/장면 분할 개선

---

## 5. 비즈니스 전략 제안

- **3단계 요금제**: Starter(월 패널 제한) / Pro(고해상도/스타일락) / Team(협업/승인/브랜드킷)
- **크레딧 기반 과금**: "대본 1건" 대신 "생성 크레딧(패널/재생성/해상도)"로 수익 예측성 향상
- **법률 특화 확장**: 사건유형별 템플릿(이혼/상속/노무) 마켓플레이스
- **B2B2C 파트너십**: 법률사무소 SaaS/마케팅 대행사에 API 제공
- **핵심 KPI**: 제작시간 단축률, 영상 게시율, 조회수/상담전환율, 크레딧 ARPU

---

## 참고 소스

- [Google Vertex AI 이미지 생성 모델](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/image-generation)
- [Gemini 이미지 생성 quickstart](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/start/quickstarts/try-gen-ai)
- [Vertex GenAI Evaluation Service](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/evaluation-overview)
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Adobe Firefly Storyboard](https://www.adobe.com/eg_en/products/firefly/features/storyboard.html)
- [EU AI Act timeline](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai)
