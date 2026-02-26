# Content Marketing (콘텐츠 마케팅 자동화) Planning Document

> **Summary**: 변호사를 위한 콘텐츠 마케팅 자동화 모듈 — 실시간 트렌드 분석(모듈 A)과 AI 유튜브 대본 생성(모듈 B)을 결합하여, 변호사가 법률 이슈 기반의 전문 콘텐츠를 효율적으로 생산할 수 있도록 지원
>
> **Project**: law-3 (Legal President / 법률 대통령)
> **Author**: Claude
> **Date**: 2026-02-20
> **Status**: Draft (v0.1)

---

## 1. Overview

### 1.1 Purpose

변호사가 **대중의 관심사(실시간 트렌드)**를 선제적으로 파악하고, 이를 바탕으로 **신뢰감 있는 유튜브 대본**을 자동 생성할 수 있는 콘텐츠 마케팅 자동화 모듈을 구현한다.

### 1.2 Background

- 법률 유튜브 채널이 변호사 마케팅의 핵심 수단으로 부상
- 사회적 이슈(사건/사고, 법적 분쟁)에 대한 법률 해석 콘텐츠가 높은 조회수를 기록
- 현재 변호사들은 이슈 발굴 → 법적 분석 → 대본 작성의 전 과정을 수동으로 수행
- 기존 법률 플랫폼의 RAG 시스템(판례/법령 검색)을 활용하면 법적 분석의 자동화가 가능
- 트렌드 수집 + 법적 해석 가능성 스코어링 + 대본 구조화를 결합하면 콘텐츠 생산 시간을 대폭 단축 가능

### 1.3 비전

```
┌─────────────────────────────────────────────────────────────────────┐
│                  콘텐츠 마케팅 자동화 모듈                              │
│                                                                      │
│  ┌─── 모듈 A: 실시간 트렌드 분석기 ───┐                               │
│  │                                      │                             │
│  │  [Tavily] [Perplexity] [Naver]       │                             │
│  │  [Google Trends] [YouTube API]       │                             │
│  │           │                          │                             │
│  │           ▼                          │                             │
│  │  ┌──────────────────┐               │                             │
│  │  │  스코어링 엔진    │               │                             │
│  │  │  언급량 × 법적    │               │                             │
│  │  │  해석 가능성      │               │                             │
│  │  └────────┬─────────┘               │                             │
│  │           ▼                          │                             │
│  │  ┌──────────────────┐               │                             │
│  │  │  트렌드 대시보드  │               │                             │
│  │  │  - 순위별 이슈    │               │                             │
│  │  │  - 핵심 쟁점 3줄  │               │                             │
│  │  │  - 관련 법령/판례 │               │                             │
│  │  └────────┬─────────┘               │                             │
│  └───────────┼──────────────────────────┘                             │
│              │ 이슈 선택                                               │
│              ▼                                                        │
│  ┌─── 모듈 B: AI 유튜브 대본 엔진 ────┐                               │
│  │                                      │                             │
│  │  ┌──────────────────┐               │                             │
│  │  │  대본 구조화      │               │                             │
│  │  │  1. 도입(Hooking)│               │                             │
│  │  │  2. 본론(Analysis)│               │                             │
│  │  │  3. 결론(CTA)    │               │                             │
│  │  └────────┬─────────┘               │                             │
│  │           ▼                          │                             │
│  │  ┌──────────────────┐               │                             │
│  │  │  페르소나 설정    │               │                             │
│  │  │  - 전문가 톤     │               │                             │
│  │  │  - 구어체 옵션   │               │                             │
│  │  └────────┬─────────┘               │                             │
│  │           ▼                          │                             │
│  │  ┌──────────────────┐               │                             │
│  │  │  메타데이터 생성  │               │                             │
│  │  │  - 영상 설명문   │               │                             │
│  │  │  - 상담 CTA 링크 │               │                             │
│  │  │  - SEO 태그      │               │                             │
│  │  └──────────────────┘               │                             │
│  └──────────────────────────────────────┘                             │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.4 핵심 사용자 시나리오

**시나리오 1: 트렌드 기반 콘텐츠 발굴**
```
1. 변호사가 /content-marketing 페이지 접속
2. 대시보드에 최근 24시간 트렌드 이슈 목록 표시
   - 각 이슈: 제목, 언급량, 법적 해석 가능성 점수, 핵심 쟁점 3줄 요약
3. 이슈를 클릭하면 상세 정보 + 관련 법령/판례 미리보기
4. "대본 생성" 버튼 클릭 → 모듈 B로 이동
```

**시나리오 2: AI 대본 생성**
```
1. 선택한 이슈 또는 직접 입력한 주제로 대본 생성 시작
2. 대본 옵션 설정:
   - 톤: 전문가 / 친근한 구어체
   - 길이: 5분 / 10분 / 15분 분량
   - 구조: 3단 (도입-본론-결론) 기본 구조
3. AI가 RAG 검색(판례/법령)을 활용하여 대본 초안 생성
4. 대본 미리보기 + 편집 기능
5. 메타데이터(영상 설명문, SEO 태그, CTA 문구) 자동 생성
6. 최종 대본 + 메타데이터 다운로드/복사
```

**시나리오 3: 채팅 위젯 연동**
```
1. 기존 채팅 위젯에서 "요즘 핫한 법률 이슈 알려줘" 입력
2. TrendAnalysisAgent가 트렌드 요약 응답
3. "이 주제로 유튜브 대본 만들어줘" → ScriptGeneratorAgent 연동
```

### 1.5 Related Documents

- `backend/app/multi_agent/agents/base_chat.py` - BaseChatAgent 추상 클래스
- `backend/app/multi_agent/graph.py` - LangGraph StateGraph 빌드
- `backend/app/services/rag/pipeline.py` - 검색 파이프라인
- `backend/app/core/registry.py` - 모듈 자동 등록

---

## 2. Scope

### 2.1 In Scope

**모듈 A: 실시간 트렌드 분석기 (Backend + Frontend)**
- [ ] 멀티소스 데이터 수집기 (Tavily, Perplexity, Naver Search API, Google Trends, YouTube Data API)
- [ ] 트렌드 스코어링 엔진 (언급량 x 법적 해석 가능성 결합 점수)
- [ ] 핵심 쟁점 3줄 요약 (LLM 기반)
- [ ] 관련 법령/판례 자동 매칭 (기존 RAG 활용)
- [ ] 트렌드 대시보드 UI (카드 목록, 정렬/필터, 상세 뷰)

**모듈 B: AI 유튜브 대본 엔진 (Backend + Frontend)**
- [ ] 3단 구조 대본 생성 (Hooking → Legal Analysis → Advice & CTA)
- [ ] 페르소나 설정 ('신뢰감 있는 변호사' 톤 + 구어체 옵션)
- [ ] RAG 연동 법적 분석 (판례/법령 인용)
- [ ] 메타데이터 생성 (영상 설명문, SEO 태그, 상담 CTA 링크)
- [ ] 대본 편집 및 미리보기 UI
- [ ] 대본 내보내기 (클립보드 복사, TXT/MD 다운로드)

**에이전트 통합**
- [ ] TrendAnalysisAgent (채팅 위젯 연동)
- [ ] ScriptGeneratorAgent (채팅 위젯 연동)
- [ ] LangGraph 멀티에이전트 시스템 통합

### 2.2 Out of Scope

- YouTube API를 통한 직접 업로드
- 썸네일 이미지 자동 생성
- 영상 편집/자막 생성
- TTS(음성 합성) 연동
- 유료 API 과금 관리 대시보드
- 대본 이력 DB 저장 (향후 확장)
- 멀티 플랫폼(블로그, 인스타그램) 대본 생성

---

## 3. Requirements

### 3.1 Functional Requirements

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| **모듈 A: 실시간 트렌드 분석기** | | | |
| FR-01 | Tavily Search API 연동 — 최근 24시간 뉴스/이슈 수집 | High | Pending |
| FR-02 | Perplexity API 연동 — 심층 이슈 분석 및 요약 | Medium | Pending |
| FR-03 | Naver Search API 연동 — 국내 뉴스/블로그/카페 트렌드 수집 | High | Pending |
| FR-04 | Google Trends API 연동 — 키워드 검색량 추이 데이터 | Medium | Pending |
| FR-05 | YouTube Data API 연동 — 법률 관련 인기 영상/키워드 수집 | Medium | Pending |
| FR-06 | 트렌드 스코어링 엔진 — 언급량(조회수/공유수) x 법적 해석 가능성 결합 점수 | High | Pending |
| FR-07 | LLM 기반 핵심 쟁점 3줄 요약 — 각 이슈별 논란 핵심 포인트 | High | Pending |
| FR-08 | RAG 연동 법령/판례 자동 매칭 — 이슈별 관련 법령/판례 추천 | High | Pending |
| FR-09 | 트렌드 대시보드 UI — 카드형 이슈 목록, 점수 순 정렬, 카테고리 필터 | High | Pending |
| FR-10 | 이슈 상세 뷰 — 전체 요약, 관련 법령/판례, 대본 생성 버튼 | High | Pending |
| FR-11 | 자동 갱신 — 주기적 트렌드 업데이트 (수동 새로고침 + 자동 폴링 옵션) | Low | Pending |
| **모듈 B: AI 유튜브 대본 엔진** | | | |
| FR-12 | 3단 대본 구조 생성 — 도입(Hooking) + 본론(Legal Analysis) + 결론(Advice & CTA) | High | Pending |
| FR-13 | 도입부 — 시청자 관심을 끄는 강렬한 사례/질문 제시 | High | Pending |
| FR-14 | 본론 — 관련 법 조항 및 판례 기반 전문 분석 (RAG 검색 결과 활용) | High | Pending |
| FR-15 | 결론 — 실질적 조언 + 법률 상담 유도 CTA 문구 | High | Pending |
| FR-16 | 페르소나 설정 — '신뢰감 있는 변호사' 톤 기본, 유튜브 구어체 옵션 | High | Pending |
| FR-17 | 대본 길이 옵션 — 5분/10분/15분 분량 선택 | Medium | Pending |
| FR-18 | 메타데이터 자동 생성 — 영상 설명문(description), SEO 태그, 상담 CTA 링크 | High | Pending |
| FR-19 | 대본 미리보기 — 마크다운 렌더링, 섹션별 구분 | High | Pending |
| FR-20 | 대본 편집 — 인라인 텍스트 에디터 (섹션별 수정 가능) | Medium | Pending |
| FR-21 | 대본 내보내기 — 클립보드 복사, TXT/MD 파일 다운로드 | Medium | Pending |
| FR-22 | 인용 출처 표시 — 대본에 사용된 판례/법령 출처 명시 | High | Pending |
| **에이전트 통합** | | | |
| FR-23 | TrendAnalysisAgent — 채팅 위젯에서 트렌드 질의 처리 | Medium | Pending |
| FR-24 | ScriptGeneratorAgent — 채팅 위젯에서 대본 생성 요청 처리 | Medium | Pending |
| FR-25 | LangGraph 라우터 통합 — AgentType.CONTENT_MARKETING 추가 | Medium | Pending |
| **안전 및 고지** | | | |
| FR-26 | 면책 고지 — "AI 생성 콘텐츠이며, 법률 자문이 아닙니다" 표시 | High | Pending |
| FR-27 | 인용 정확성 — RAG 검색 결과만 인용, 환각 방지 프롬프트 제약 | High | Pending |

### 3.2 Non-Functional Requirements

| Category | Criteria | Measurement Method |
|----------|----------|-------------------|
| Performance | 트렌드 수집 및 스코어링 30초 이내 (5개 소스 병렬) | API 응답 시간 측정 |
| Performance | 대본 생성 스트리밍 첫 토큰 5초 이내 | 응답 시간 측정 |
| Performance | RAG 관련 법령/판례 검색 3초 이내 | 검색 시간 측정 |
| Scalability | 동시 5명 이상 트렌드 조회 지원 | 부하 테스트 |
| UX | 대본 생성 중 스트리밍 표시 (로딩 UX) | 사용자 테스트 |
| Cost | API 호출 비용 월 $50 이내 (Tavily/Perplexity 기본 플랜) | 비용 추적 |
| Safety | 면책 고지 상시 표시 | UI 확인 |
| Safety | 개인정보/민감정보 필터링 | 출력 검증 |

---

## 4. Success Criteria

### 4.1 Definition of Done

- [ ] 5개 데이터 소스에서 트렌드 수집 동작 (최소 2개 필수: Tavily + Naver)
- [ ] 트렌드 스코어링 + 3줄 요약 생성 동작
- [ ] RAG 연동 관련 법령/판례 매칭 동작
- [ ] 트렌드 대시보드 UI 렌더링 + 상세 뷰
- [ ] 3단 구조 대본 생성 + 스트리밍 표시
- [ ] 메타데이터(설명문, SEO 태그, CTA) 생성 동작
- [ ] 대본 편집 + 내보내기(복사/다운로드) 동작
- [ ] 정적 검증: `ruff check`, `mypy`, `npm run build` 통과
- [ ] E2E: 트렌드 조회 → 이슈 선택 → 대본 생성 → 내보내기 전체 흐름

### 4.2 Quality Criteria

- [ ] 트렌드 이슈가 실제 최근 뉴스와 일치
- [ ] 법적 해석 가능성 점수가 법률 관련 이슈에 높게 책정
- [ ] 대본이 3단 구조를 명확히 갖추고, 인용 판례/법령이 실재
- [ ] 페르소나 톤이 설정에 따라 구분됨 (전문가 vs 구어체)

---

## 5. Risks and Mitigation

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| 외부 API 비용 증가 (Tavily, Perplexity 등) | High | Medium | 캐싱 전략(24시간 TTL), 호출 횟수 제한, 필수 소스(Tavily+Naver) 우선 구현 |
| 외부 API Rate Limit / 장애 | Medium | Medium | 그레이스풀 디그레이드 — 실패 소스 건너뛰고 나머지로 결과 반환, 재시도 로직 |
| LLM 환각 — 존재하지 않는 판례/법령 인용 | High | Medium | RAG 검색 결과만 인용하도록 프롬프트 제약, 인용 출처 검증 후처리 |
| 트렌드 스코어링의 법적 해석 가능성 판단 부정확 | Medium | High | LLM 기반 법적 관련성 판단 + 법률 키워드 사전 매칭 병행, 사용자 피드백 반영 구조 |
| 대본 품질 불균일 (주제별 편차) | Medium | Medium | 구조화 프롬프트로 일관성 확보, 법률 도메인 특화 few-shot 예시 포함 |
| 저작권 이슈 (뉴스 기사 원문 무단 사용) | High | Low | 요약만 사용(원문 인용 X), 출처 링크 제공, 자체 분석/의견으로 변환 |
| 다중 API 호출로 응답 지연 | Medium | Medium | 병렬 호출(asyncio.gather), 프로그레시브 로딩 (소스별 결과 순차 표시) |
| YouTube API 할당량 초과 | Low | Medium | YouTube는 선택적 소스, 일일 할당량 모니터링, 없어도 핵심 기능 동작 |

---

## 6. Architecture Considerations

### 6.1 Project Level Selection

| Level | Characteristics | Recommended For | Selected |
|-------|-----------------|-----------------|:--------:|
| **Starter** | Simple structure | Static sites | |
| **Dynamic** | Feature-based modules | Web apps | **V** |
| **Enterprise** | Strict layer separation | High-traffic systems | |

**선택 이유**: 기존 모듈 시스템(registry 기반 자동 등록)에 맞춰 Dynamic 레벨로 구현. 마이크로서비스 분리 불필요.

### 6.2 Key Architectural Decisions

| Decision | Options | Selected | Rationale |
|----------|---------|----------|-----------|
| 트렌드 수집 방식 | 실시간 API 호출 / **스케줄 + 캐시** / 크롤링 | **요청 시 API 호출 + 캐시** | MVP 단계에서 단순화. 캐싱(24h TTL)으로 비용 절감 |
| 트렌드 API 우선순위 | 모두 필수 / **단계적 구현** | **단계적** | Phase 1: Tavily + Naver, Phase 2: Perplexity + Google Trends + YouTube |
| 스코어링 엔진 | 규칙 기반 / **LLM + 규칙 하이브리드** | **하이브리드** | 언급량은 수치 기반, 법적 해석 가능성은 LLM 판단 + 법률 키워드 매칭 |
| 대본 생성 LLM | Solar / OpenAI / **설정 가능** | **기존 LLM 설정 따름** | `settings.LLM_PROVIDER` 활용, 교체 용이 |
| 대본 구조 | 자유 형식 / **3단 구조 고정** | **3단 구조** | 도입-본론-결론 일관성 보장, 유튜브 콘텐츠 표준 |
| RAG 연동 | 독립 검색 / **기존 RAG 파이프라인 공유** | **기존 RAG 공유** | 판례/법령 검색 이미 구현됨, 별도 인터페이스 래핑 |
| 에이전트 수 | 1개 통합 / **2개 분리** | **2개** | TrendAnalysisAgent + ScriptGeneratorAgent (관심사 분리) |
| Frontend 프레임워크 | 별도 SPA / **기존 Next.js 모듈** | **기존 Next.js** | 모듈 자동 등록, 기존 UI 패턴 재사용 |

### 6.3 기술 스택

| 영역 | 기술 | 용도 |
|------|------|------|
| **트렌드 수집** | Tavily Search API | 웹 검색 + 뉴스 집계 (최근 24시간) |
| | Naver Search API (뉴스/블로그) | 국내 트렌드 수집 |
| | Perplexity API (Phase 2) | 심층 이슈 분석/요약 |
| | Google Trends (pytrends, Phase 2) | 키워드 검색량 추이 |
| | YouTube Data API v3 (Phase 2) | 법률 관련 인기 영상 분석 |
| **스코어링** | LLM (Solar/OpenAI/Gemini) | 법적 해석 가능성 판단 |
| | 법률 키워드 사전 | 법률 관련성 사전 매칭 |
| **대본 생성** | LLM (기존 설정) | 3단 구조 대본 + 메타데이터 |
| | RAG Pipeline (LanceDB) | 판례/법령 검색 및 인용 |
| **Backend** | FastAPI + LangGraph | API + 멀티에이전트 |
| | httpx (async) | 외부 API 호출 |
| **Frontend** | Next.js + Tailwind CSS | 대시보드 + 대본 에디터 |
| | React Query | 서버 상태 관리 |

### 6.4 데이터 흐름도 (Data Flow)

```
┌─────────────────────────────────────────────────────────────────┐
│                    데이터 흐름 (Data Flow)                        │
│                                                                  │
│  [사용자 요청: 트렌드 조회]                                       │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────────────────────────┐                        │
│  │  TrendCollector (병렬 수집)            │                        │
│  │                                       │                        │
│  │  ┌─────────┐  ┌───────────┐          │                        │
│  │  │ Tavily  │  │  Naver    │  ...      │                        │
│  │  │ Search  │  │  Search   │           │                        │
│  │  └────┬────┘  └─────┬─────┘          │                        │
│  │       └──────┬──────┘                 │                        │
│  │              ▼                        │                        │
│  │  ┌──────────────────┐                │                        │
│  │  │  중복 제거 + 병합  │                │                        │
│  │  │  (제목/URL 기준)   │                │                        │
│  │  └────────┬─────────┘                │                        │
│  └───────────┼───────────────────────────┘                        │
│              ▼                                                    │
│  ┌──────────────────────────────────────┐                        │
│  │  TrendScorer (점수 산정)               │                        │
│  │                                       │                        │
│  │  1. 언급량 점수 (조회수/공유수 정규화)   │                        │
│  │  2. 법적 해석 가능성 점수               │                        │
│  │     - LLM 판단 (0~1)                  │                        │
│  │     - 법률 키워드 매칭 보정             │                        │
│  │  3. 종합 점수 = α×언급량 + β×법적해석   │                        │
│  │     (α=0.4, β=0.6 기본)               │                        │
│  └───────────┬───────────────────────────┘                        │
│              ▼                                                    │
│  ┌──────────────────────────────────────┐                        │
│  │  IssueSummarizer (LLM 요약)           │                        │
│  │                                       │                        │
│  │  - 핵심 쟁점 3줄 요약                  │                        │
│  │  - RAG 검색 → 관련 법령/판례 매칭       │                        │
│  └───────────┬───────────────────────────┘                        │
│              ▼                                                    │
│  ┌──────────────────────────────────────┐                        │
│  │  트렌드 대시보드 (Frontend)            │                        │
│  │  → 이슈 카드 목록, 정렬, 상세 뷰        │                        │
│  └───────────┬───────────────────────────┘                        │
│              │ [이슈 선택 + "대본 생성"]                            │
│              ▼                                                    │
│  ┌──────────────────────────────────────┐                        │
│  │  ScriptGenerator (대본 생성)           │                        │
│  │                                       │                        │
│  │  입력: 이슈 요약 + 관련 법령/판례       │                        │
│  │        + 페르소나 설정 + 길이 옵션       │                        │
│  │                                       │                        │
│  │  1. RAG 심화 검색 (판례/법령 상세)      │                        │
│  │  2. 도입부 생성 (Hooking)              │                        │
│  │  3. 본론 생성 (Legal Analysis)         │                        │
│  │  4. 결론 생성 (Advice & CTA)           │                        │
│  │  5. 메타데이터 생성                    │                        │
│  │     - 영상 설명문                      │                        │
│  │     - SEO 태그                        │                        │
│  │     - 상담 CTA 링크                   │                        │
│  └───────────┬───────────────────────────┘                        │
│              ▼                                                    │
│  ┌──────────────────────────────────────┐                        │
│  │  대본 에디터 (Frontend)                │                        │
│  │  → 미리보기, 편집, 내보내기             │                        │
│  └──────────────────────────────────────┘                        │
└─────────────────────────────────────────────────────────────────┘
```

### 6.5 백엔드 구조 설계

```
backend/app/
├── modules/content_marketing/           # 모듈 (자동 등록)
│   ├── __init__.py
│   ├── router/
│   │   └── __init__.py                  # /api/content-marketing 엔드포인트
│   └── schema/
│       └── __init__.py                  # Pydantic 스키마
│
├── services/service_function/
│   └── content_marketing_service.py     # 비즈니스 로직 통합
│
├── tools/trend/                         # 트렌드 수집 도구
│   ├── __init__.py
│   ├── collector.py                     # TrendCollector (멀티소스 수집기)
│   ├── scorer.py                        # TrendScorer (스코어링 엔진)
│   ├── summarizer.py                    # IssueSummarizer (LLM 요약)
│   └── sources/                         # 개별 데이터 소스 어댑터
│       ├── __init__.py
│       ├── tavily_source.py             # Tavily Search API
│       ├── naver_source.py              # Naver Search API
│       ├── perplexity_source.py         # Perplexity API (Phase 2)
│       ├── google_trends_source.py      # Google Trends (Phase 2)
│       └── youtube_source.py            # YouTube Data API (Phase 2)
│
├── tools/script/                        # 대본 생성 도구
│   ├── __init__.py
│   ├── generator.py                     # ScriptGenerator (대본 생성기)
│   ├── templates.py                     # 대본 구조 템플릿/프롬프트
│   └── metadata.py                      # 메타데이터 생성기
│
└── multi_agent/
    ├── agents/
    │   ├── trend_analysis_agent.py      # TrendAnalysisAgent
    │   └── script_generator_agent.py    # ScriptGeneratorAgent
    ├── router.py                        # AgentType.CONTENT_MARKETING 추가
    ├── nodes.py                         # content_marketing_node 추가
    └── graph.py                         # 노드 등록
```

### 6.6 프론트엔드 구조 설계

```
frontend/src/
├── app/content-marketing/
│   └── page.tsx                         # 페이지 엔트리
│
├── features/content-marketing/
│   ├── components/
│   │   ├── TrendDashboard.tsx           # 트렌드 대시보드 (메인)
│   │   ├── TrendCard.tsx                # 이슈 카드
│   │   ├── TrendDetailView.tsx          # 이슈 상세 뷰
│   │   ├── TrendFilters.tsx             # 카테고리/날짜 필터
│   │   ├── ScriptGenerator.tsx          # 대본 생성 메인 컴포넌트
│   │   ├── ScriptEditor.tsx             # 대본 편집기
│   │   ├── ScriptPreview.tsx            # 대본 미리보기
│   │   ├── MetadataPanel.tsx            # 메타데이터 표시/편집
│   │   ├── PersonaSelector.tsx          # 페르소나/톤 선택
│   │   └── ExportButton.tsx             # 내보내기 (복사/다운로드)
│   ├── hooks/
│   │   ├── useTrends.ts                 # 트렌드 데이터 훅
│   │   └── useScriptGeneration.ts       # 대본 생성 훅
│   ├── services/
│   │   └── index.ts                     # API 호출 함수
│   └── types/
│       └── index.ts                     # TypeScript 타입
│
├── lib/modules.ts                       # content-marketing 모듈 등록
└── lib/api.ts                           # contentMarketing endpoint 추가
```

### 6.7 UI/UX 워크플로우

```
┌─────────────────────────────────────────────────────┐
│          /content-marketing 페이지 레이아웃           │
│                                                      │
│  ┌───────────────────────────────────────────────┐  │
│  │  헤더: 콘텐츠 마케팅 자동화                      │  │
│  │  [트렌드 분석] [대본 생성]  ← 탭 네비게이션       │  │
│  └───────────────────────────────────────────────┘  │
│                                                      │
│  ═══════════════════════════════════════════════════  │
│                                                      │
│  [탭 1: 트렌드 분석]                                  │
│  ┌────────────────────────────────────────────────┐ │
│  │  필터: [전체 ▼] [24시간 ▼] [새로고침 🔄]        │ │
│  │                                                 │ │
│  │  ┌──────────────────────┐ ┌──────────────────┐ │ │
│  │  │ 🔥 1위               │ │ 📊 2위            │ │ │
│  │  │ "XX 사건 손해배상"     │ │ "YY법 개정 논란"  │ │ │
│  │  │ 점수: 92 / 100       │ │ 점수: 87 / 100   │ │ │
│  │  │ 언급량: ████████░░   │ │ 언급량: ██████░░░ │ │ │
│  │  │ 법적해석: █████████░ │ │ 법적해석: ████████░│ │ │
│  │  │                      │ │                   │ │ │
│  │  │ 쟁점:                │ │ 쟁점:             │ │ │
│  │  │ 1. 과실 인정 여부     │ │ 1. 소급적용 가능성 │ │ │
│  │  │ 2. 손해배상 범위      │ │ 2. 기본권 제한    │ │ │
│  │  │ 3. 기업 책임 한계     │ │ 3. 입법 취지 검토 │ │ │
│  │  │                      │ │                   │ │ │
│  │  │ [상세보기] [대본생성]  │ │ [상세보기] [대본생성]│ │ │
│  │  └──────────────────────┘ └──────────────────┘ │ │
│  │                                                 │ │
│  │  ┌──────────────────────┐ ┌──────────────────┐ │ │
│  │  │ 3위...               │ │ 4위...            │ │ │
│  │  └──────────────────────┘ └──────────────────┘ │ │
│  └────────────────────────────────────────────────┘ │
│                                                      │
│  [탭 2: 대본 생성]                                    │
│  ┌────────────────────────────────────────────────┐ │
│  │  주제: [선택된 이슈 또는 직접 입력]               │ │
│  │                                                 │ │
│  │  옵션:                                          │ │
│  │  톤:    [전문가 ▼]   길이: [10분 ▼]             │ │
│  │  구조:  [3단(기본) ▼]                            │ │
│  │                                                 │ │
│  │  [대본 생성하기 ▶]                               │ │
│  │                                                 │ │
│  │  ┌─────────────── 대본 미리보기 ───────────────┐ │ │
│  │  │                                             │ │ │
│  │  │  ## 도입 (Hooking)                          │ │ │
│  │  │  "최근 XX 사건이 큰 논란이 되고 있습니다..."   │ │ │
│  │  │                                             │ │ │
│  │  │  ## 본론 (Legal Analysis)                   │ │ │
│  │  │  "민법 제750조에 따르면..."                   │ │ │
│  │  │  [📋 인용: 대법원 2024다12345]               │ │ │
│  │  │                                             │ │ │
│  │  │  ## 결론 (Advice & CTA)                     │ │ │
│  │  │  "이런 상황에서는 전문 변호사와..."            │ │ │
│  │  │                                             │ │ │
│  │  └─────────────────────────────────────────────┘ │ │
│  │                                                 │ │
│  │  ┌─ 메타데이터 ──┐                              │ │
│  │  │ 설명문: ...    │  [복사 📋] [다운로드 ⬇]     │ │
│  │  │ SEO태그: ...   │                             │ │
│  │  │ CTA: ...       │                             │ │
│  │  └────────────────┘                             │ │
│  └────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

### 6.8 API 엔드포인트 설계

```
POST /api/content-marketing/trends
  Body: { "time_range": "24h", "category": "all", "limit": 10 }
  Response: { "trends": [TrendIssue], "collected_at": "...", "sources_used": [...] }

GET  /api/content-marketing/trends/{trend_id}
  Response: TrendIssueDetail (요약 + 관련 법령/판례 + 원본 링크)

POST /api/content-marketing/script/generate
  Body: {
    "topic": "...",
    "trend_id": "..." (optional),
    "persona": "professional" | "casual",
    "duration": 5 | 10 | 15,
    "related_laws": [...],
    "related_cases": [...]
  }
  Response: SSE stream → ScriptResult { sections: [...], metadata: {...} }

POST /api/content-marketing/script/metadata
  Body: { "script_content": "...", "topic": "..." }
  Response: { "description": "...", "tags": [...], "cta": "..." }
```

### 6.9 Pydantic 스키마 (핵심)

```python
class TrendSource(str, Enum):
    TAVILY = "tavily"
    NAVER = "naver"
    PERPLEXITY = "perplexity"
    GOOGLE_TRENDS = "google_trends"
    YOUTUBE = "youtube"

class TrendIssue(BaseModel):
    id: str
    title: str
    summary: str                    # LLM 생성 요약
    key_points: list[str]           # 핵심 쟁점 3줄
    score: float                    # 종합 점수 (0~100)
    mention_score: float            # 언급량 점수
    legal_relevance_score: float    # 법적 해석 가능성 점수
    sources: list[TrendSource]      # 데이터 출처
    source_urls: list[str]          # 원본 URL
    related_laws: list[str]         # 관련 법령명
    related_cases: list[str]        # 관련 판례번호
    collected_at: datetime

class TrendRequest(BaseModel):
    time_range: str = "24h"         # "24h" | "48h" | "7d"
    category: str = "all"           # "all" | "criminal" | "civil" | "labor" 등
    limit: int = 10

class ScriptRequest(BaseModel):
    topic: str                      # 주제 (직접 입력 또는 트렌드 요약)
    trend_id: str | None = None     # 연결된 트렌드 ID
    persona: str = "professional"   # "professional" | "casual"
    duration: int = 10              # 5, 10, 15 (분)
    related_laws: list[str] = []    # 사전 매칭된 법령
    related_cases: list[str] = []   # 사전 매칭된 판례

class ScriptSection(BaseModel):
    section_type: str               # "hooking" | "analysis" | "advice_cta"
    title: str
    content: str
    citations: list[str]            # 인용 판례/법령

class ScriptResult(BaseModel):
    sections: list[ScriptSection]
    metadata: ScriptMetadata
    word_count: int
    estimated_duration: int         # 예상 분량 (분)

class ScriptMetadata(BaseModel):
    description: str                # 영상 설명문
    tags: list[str]                 # SEO 태그
    cta_text: str                   # 상담 유도 문구
```

---

## 7. Convention Prerequisites

### 7.1 Existing Project Conventions

- [x] `CLAUDE.md` — 모듈 구조, API 경로 규칙 정의됨
- [x] `.claude/rules/coding-style.md` — 코딩 스타일 규칙
- [x] `.claude/rules/code-verification.md` — 검증 프로토콜
- [x] BaseChatAgent 추상 클래스 패턴
- [x] 모듈 자동 등록 (ModuleRegistry)
- [x] RAG 검색 파이프라인 (`services/rag/`)

### 7.2 API 경로 규칙

| Backend 모듈명 | API 경로 | Frontend 경로 |
|----------------|----------|---------------|
| `content_marketing` | `/api/content-marketing` | `/content-marketing` |

### 7.3 Dependencies (추가 필요)

| Package | 위치 | 용도 | 필수 여부 |
|---------|------|------|----------|
| `tavily-python` | Backend (uv) | Tavily Search API 클라이언트 | 필수 (Phase 1) |
| `httpx` | Backend (uv) | 비동기 HTTP 클라이언트 (Naver API 등) | 기존 설치 확인 |
| `pytrends` | Backend (uv) | Google Trends 비공식 클라이언트 | 선택 (Phase 2) |
| `google-api-python-client` | Backend (uv) | YouTube Data API | 선택 (Phase 2) |

### 7.4 환경 변수 (추가 필요)

```bash
# backend/.env
TAVILY_API_KEY=""                   # Tavily Search API 키 (필수)
NAVER_CLIENT_ID=""                  # Naver Search API 클라이언트 ID
NAVER_CLIENT_SECRET=""              # Naver Search API 시크릿
PERPLEXITY_API_KEY=""               # Perplexity API 키 (Phase 2)
YOUTUBE_API_KEY=""                   # YouTube Data API 키 (Phase 2)

# 트렌드 스코어링 가중치
TREND_MENTION_WEIGHT=0.4            # 언급량 가중치
TREND_LEGAL_WEIGHT=0.6              # 법적 해석 가능성 가중치

# 콘텐츠 마케팅 캐시 TTL (초)
CONTENT_MARKETING_CACHE_TTL=86400   # 24시간
```

---

## 8. Implementation Plan

### 8.1 단계적 구현 순서

| Phase | Step | 작업 | 핵심 파일 | 의존성 |
|-------|------|------|----------|--------|
| **Phase 1: 핵심 인프라** | | | | |
| | 1 | `content_marketing` 모듈 스캐폴딩 (스키마, 라우터) | `modules/content_marketing/` | - |
| | 2 | Pydantic 스키마 정의 (Trend, Script 모델) | `modules/content_marketing/schema/` | Step 1 |
| | 3 | 환경 변수 추가 (config.py) | `core/config.py`, `.env.example` | - |
| **Phase 2: 트렌드 수집 (모듈 A 백엔드)** | | | | |
| | 4 | TrendSource 어댑터 인터페이스 정의 | `tools/trend/sources/__init__.py` | Step 2 |
| | 5 | Tavily Source 구현 | `tools/trend/sources/tavily_source.py` | Step 4 |
| | 6 | Naver Source 구현 | `tools/trend/sources/naver_source.py` | Step 4 |
| | 7 | TrendCollector (병렬 수집 + 중복 제거) | `tools/trend/collector.py` | Step 5-6 |
| | 8 | TrendScorer (언급량 x 법적 해석 가능성) | `tools/trend/scorer.py` | Step 7 |
| | 9 | IssueSummarizer (LLM 3줄 요약 + RAG 매칭) | `tools/trend/summarizer.py` | Step 8 |
| | 10 | 서비스 함수 + API 엔드포인트 (trends) | `service_function/`, `router/` | Step 9 |
| **Phase 3: 대본 생성 (모듈 B 백엔드)** | | | | |
| | 11 | 대본 구조 템플릿/프롬프트 정의 | `tools/script/templates.py` | Step 2 |
| | 12 | ScriptGenerator (3단 구조 + RAG 인용) | `tools/script/generator.py` | Step 11 |
| | 13 | MetadataGenerator (설명문, SEO 태그, CTA) | `tools/script/metadata.py` | Step 12 |
| | 14 | API 엔드포인트 (script/generate, SSE 스트리밍) | `router/` | Step 12-13 |
| **Phase 4: 에이전트 통합** | | | | |
| | 15 | TrendAnalysisAgent 구현 | `multi_agent/agents/` | Step 10 |
| | 16 | ScriptGeneratorAgent 구현 | `multi_agent/agents/` | Step 14 |
| | 17 | LangGraph 라우터/노드/그래프 통합 | `router.py`, `nodes.py`, `graph.py` | Step 15-16 |
| **Phase 5: 프론트엔드** | | | | |
| | 18 | 모듈 등록 (modules.ts, api.ts, next.config.js) | `lib/` | - |
| | 19 | 트렌드 대시보드 UI (TrendDashboard, TrendCard) | `features/content-marketing/` | Step 18 |
| | 20 | 이슈 상세 뷰 (TrendDetailView) | `features/content-marketing/` | Step 19 |
| | 21 | 대본 생성 UI (ScriptGenerator, PersonaSelector) | `features/content-marketing/` | Step 18 |
| | 22 | 대본 편집/미리보기 (ScriptEditor, ScriptPreview) | `features/content-marketing/` | Step 21 |
| | 23 | 메타데이터 패널 + 내보내기 | `features/content-marketing/` | Step 22 |
| **Phase 6: 통합 및 검증** | | | | |
| | 24 | Frontend ↔ Backend 연동 테스트 | 전체 | Step 14, 23 |
| | 25 | 정적 검증 (ruff, mypy, npm run build) | - | Step 24 |
| | 26 | E2E 흐름 테스트 | - | Step 25 |

### 8.2 Phase 2 확장 (추가 소스)

Phase 1 완료 후 점진적으로 추가:

| 소스 | 파일 | 우선순위 |
|------|------|---------|
| Perplexity API | `tools/trend/sources/perplexity_source.py` | Medium |
| Google Trends | `tools/trend/sources/google_trends_source.py` | Low |
| YouTube Data API | `tools/trend/sources/youtube_source.py` | Low |

---

## 9. Next Steps

1. [ ] Plan 리뷰 및 승인
2. [ ] Tavily API 키 발급 및 `.env` 설정
3. [ ] Naver Search API 클라이언트 ID/시크릿 발급
4. [ ] Design 문서 작성 (`/pdca design content-marketing`)
5. [ ] 구현 시작 (Phase 1 → Phase 5 순차)

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-02-20 | Initial draft — 모듈 A(트렌드 분석기) + 모듈 B(대본 엔진) Plan 수립, 기술 스택/데이터 흐름도/백엔드-프론트엔드 구조 설계, UI/UX 워크플로우 제안 | Claude |
