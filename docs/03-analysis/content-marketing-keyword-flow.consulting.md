# 외부 컨설팅 보고서 - Content Marketing Keyword Flow

> **Reviewer**: Codex CLI (External Consultant, gpt-5.3-codex)
> **Date**: 2026-02-25
> **Target**: `docs/01-plan/features/content-marketing-keyword-flow.plan.md` v0.2

---

## 1. 격차 분석

기준일: **2026-02-25**

핵심 진단: v0.2는 "속도/비용/제어권"은 개선됐지만, 업계 상위권 대비 중요한 가치사슬(검증 가능한 인사이트, 제작 실행성, 성과 피드백 루프)이 비어 있습니다.

### 업계 최고 수준 대비 큰 GAP
- **강점**: 5단계 일괄 파이프라인을 2단계 인터랙션으로 단순화한 판단은 정확
- **약점**: `LegalGateScorer`/`IssueSummarizer` 제거로 "법률 콘텐츠 제작에 바로 쓰는 깊이"가 약해짐
- **약점**: "키워드 → 뉴스 목록"까지만 있고 "검증된 주장 포인트 → 스크립트 초안 → 썸네일/타이틀 실험"으로 이어지는 제작 체인 부재
- **약점**: 객관식 점수(4차원)는 있으나 재현성/감사 추적(모델 버전, 프롬프트 버전, 근거 링크)이 약함

### 법률 유튜버 실제 워크플로우 GAP
- 현장 워크플로우: `이슈 탐색 → 사실검증(다중 기사/원문 판결문) → 법적 쟁점 정리 → 리스크 검토(명예훼손/사건진행중) → 대본 제작 → 성과 분석`
- 현재안은 사실상 1~2단계만 다루고, 법률 채널의 핵심인 "법적 정확성/리스크 관리" 단계가 누락

---

## 2. 기술 고도화 제안

### 2025-2026 트렌드 반영 보강
- LLM JSON 파싱 대신 **Structured Outputs(JSON Schema 강제)**로 전환
- 프롬프트/모델/룰셋을 버전 관리(`score_model_version`, `prompt_version`)해 재현성 확보
- 온라인 파이프라인에 **관측성(Tracing + 비용/지연/실패율)** 기본 탑재

### Tavily 현실성 보완
- `include_domains`는 Tavily에서 공식 지원되므로 방향은 타당
- 다만 한국 커뮤니티는 로그인/동적렌더링/robots 정책으로 커버리지 편차가 큼. 예: `dcinside.com/robots.txt`는 일반 봇에 `Disallow: /`
- **권장**: Tavily 단독이 아니라 `Tavily + (Naver/Google 뉴스) + 선택적 퍼스트파티 크롤러`의 하이브리드 수집으로 안정화

### 데이터 파이프라인
- 이벤트 기반 파이프라인으로 분리: `collect_job`, `keyword_rank_job`, `news_enrich_job`
- 캐시는 인메모리에서 빠르게 시작하되, 운영 전환 전 Redis(+분산락)로 이동
- 품질 메트릭 정의: 중복률, 무관 키워드율, 법적 관련성 정밀도, 클릭 후 스크립트 전환율

---

## 3. UX 개선 제안

### 점수 설명가능성 추가
- 카드에 "왜 86.2점인지" 근거 2~3개(게시글 수, 증감률, 법적 쟁점 문장) 노출
- 신뢰도 배지(`confidence`) 함께 표기

### 제작 플로우 연결 강화
- 뉴스 리스트에서 끝내지 말고 "근거팩 생성(핵심 쟁점/반대견해/주의표현)" 버튼 추가
- Step 2 완료 직후 "스크립트 초안 생성"으로 원클릭 연결

### 로딩 UX
- SSE 도입은 적절
- 재시도 UX(부분 실패 소스 표시, 다시 시도 시 실패 소스만 재호출)를 넣어 체감 성능 개선

---

## 4. AI/ML 고도화 제안

### 키워드 스코어링 객관성/재현성
- 현재 점수는 LLM 주관 편향 위험이 큼
- **개선안**: 고정 Rubric + 예시셋 + 온도 0 + 스키마 강제 + 사후 규칙검증
- 오프라인 벤치셋(법률 전문가 라벨)으로 월간 캘리브레이션

### 모델 구조
- 1차: 규칙/통계 기반 후보 추출(빈도, 급증률, 엔티티)
- 2차: LLM 재랭킹(법적 연관성/콘텐츠 적합성)
- 3차: 휴먼 피드백(클릭/스크립트 채택률)로 가중치 자동 업데이트

### 제거된 법률 레이어의 부분 복원
- 완전 요약은 빼더라도, `lightweight legal enrichment`(관련 법령 키워드 1~2개, 쟁점 라벨)는 **반드시 복원 권고**

---

## 5. 비즈니스 전략 제안

### 수익화
- `Free`: 키워드 수집 제한
- `Pro`: 소스 확장, 스크립트 초안, 히스토리
- `Team/Law Firm`: 협업, 승인흐름, 감사로그, API 제공

### 확장
- 법률 유튜버에서 시작해 노무/세무/부동산 전문가 채널로 수평 확장 가능
- B2B로 로펌 마케팅팀/법률미디어 대행사에 화이트라벨 공급 가능

### KPI
- 핵심 KPI를 "조회수"보다 "스크립트 채택률, 발행 전환율, 발행 후 7일 유지율"로 재설계

---

## 6. API 설계 개선

### REST 정합성
- `POST /keywords/{id}/news`는 액션형이므로 `GET /keywords/{id}/news`(조회) 또는 `POST /news:search`(검색 액션)로 명확히 분리 권장
- SSE는 `GET /keyword-collections/{collection_id}/events` 형태가 더 자원지향적

### 에러 핸들링
- RFC 9457 `application/problem+json`으로 통일
- 에러 본문에 `type`, `title`, `status`, `detail`, `instance`, `trace_id` 표준화
- 소스 부분 실패는 `partial_failures` 배열로 명시해 디버깅 가능성 향상

### Rate Limit
- 현재 Step1/Step2 공유 5회/시간은 탐색 UX를 과도하게 제약할 수 있음
- **분리 버킷(예: collect 5/h, news 30/h)과 플랜별 차등 정책 권장**

---

## 7. 종합 평가 및 우선순위

**종합 평점: B (방향성 우수, 제품 완성도는 아직 중간 수준)**

### 우선순위 (실행 순서)
1. **P0 (즉시)**: Structured Outputs 전환, RFC 9457 에러 표준화, 스코어 재현성(버전/로그)
2. **P1 (2~4주)**: Tavily 하이브리드 수집, 키워드 점수 근거 노출, 부분 실패 재시도 UX
3. **P2 (4~8주)**: 경량 법률 enrichment 복원, 성과 피드백 루프(채택률 기반 재랭킹)
4. **P3 (8주+)**: Redis/분산 아키텍처, 팀/엔터프라이즈 요금제, 화이트라벨 API

---

## 검증 출처
- Tavily Search API (`include_domains`): https://docs.tavily.com/documentation/api-reference/endpoint/search
- OpenAI Structured Outputs: https://platform.openai.com/docs/guides/structured-outputs/supported-schemas
- IETF Problem Details(RFC 9457): https://www.ietf.org/rfc/rfc9457.html
- YouTube Inspiration Tab/Creator updates: https://support.google.com/youtube/answer/15575509?hl=en
- dcinside robots.txt 확인: https://www.dcinside.com/robots.txt
