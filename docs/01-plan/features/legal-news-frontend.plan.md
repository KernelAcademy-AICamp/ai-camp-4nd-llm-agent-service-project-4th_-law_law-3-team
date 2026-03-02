# 법률 뉴스 프론트엔드 기획서

> **Version**: v1.0.0
> **Status**: Draft
> **Date**: 2026-02-26
> **Author**: Agent Team (PM + UI/UX + Frontend + QA)
> **Related**: `legal-news-pipeline.plan.md` (백엔드 v0.3.0)

## 1. 개요

### 1.1 배경

법률 뉴스 파이프라인(legal-news-pipeline)의 백엔드가 v0.3.0으로 완전 구현되었으나,
원래 기획서에서 프론트엔드를 명시적으로 제외("백엔드 전용")했기 때문에 프론트엔드 UI/UX가 부재.

### 1.2 목적

- 수집/요약된 법률 뉴스 기사를 사용자에게 제공
- 하이브리드 검색(Vector + FTS + 리랭커)으로 법률 뉴스 탐색
- 기존 모듈 패턴(content-marketing) 재사용으로 일관된 UX 유지

### 1.3 현재 상태

| 항목 | 상태 |
|------|------|
| `modules.ts` | `legal-news` 등록됨 (`enabled: false`) |
| `api.ts` | `legalNews: '/legal-news'` 등록됨 |
| `next.config.js` | rewrites 등록됨 |
| `features/legal-news/` | **미존재** |
| `app/legal-news/page.tsx` | **미존재** |

## 2. 사용자 요구사항

### 2.1 대상 사용자

- **변호사(lawyer)**: 법률 뉴스 트렌드 파악, 관련 기사 검색
- **일반 사용자(user)**: 법률 뉴스 열람, 관심 키워드 검색

> 역할 차이 없음: lawyer/user 동일 뷰 제공

### 2.2 핵심 기능

| # | 기능 | API | 설명 |
|---|------|-----|------|
| F1 | 뉴스 목록 조회 | `GET /list` | 최신 뉴스 카드 그리드 + 페이지네이션 |
| F2 | 소스/날짜 필터 | `GET /list?source=&published_date=` | lawtimes/naver 소스 필터, 날짜 필터 |
| F3 | 뉴스 상세 보기 | `GET /{article_id}` | 슬라이드 패널(우측 드로어)로 기사 전문 표시 |
| F4 | 뉴스 검색 | `POST /search` | 하이브리드 검색 (검색바 + 결과 목록) |
| F5 | 면책 안내 | - | 상단 고정 DisclaimerBanner |

## 3. UI/UX 설계

### 3.1 레이아웃: 탭 기반 (content-marketing 패턴)

```
┌────────────────────────────────────────────┐
│ [DisclaimerBanner]                          │
├────────────────────────────────────────────┤
│ [← BackButton] 법률 뉴스                   │
│ 법률 뉴스 수집·요약 및 하이브리드 검색      │
├────────────────────────────────────────────┤
│ [뉴스 목록] [뉴스 검색]  ← 탭              │
├────────────────────────────────────────────┤
│                                            │
│  탭 1: 뉴스 목록                           │
│  ┌────────────────────────────────────┐    │
│  │ [소스필터▼] [날짜필터▼]            │    │
│  │                                    │    │
│  │ ┌──────┐ ┌──────┐ ┌──────┐       │    │
│  │ │카드1 │ │카드2 │ │카드3 │       │    │
│  │ └──────┘ └──────┘ └──────┘       │    │
│  │ ┌──────┐ ┌──────┐ ┌──────┐       │    │
│  │ │카드4 │ │카드5 │ │카드6 │       │    │
│  │ └──────┘ └──────┘ └──────┘       │    │
│  │                                    │    │
│  │ [< 이전] 1/5 [다음 >]             │    │
│  └────────────────────────────────────┘    │
│                                            │
│  탭 2: 뉴스 검색                           │
│  ┌────────────────────────────────────┐    │
│  │ [검색어 입력...] [🔍]              │    │
│  │ [소스필터▼] [결과수▼]              │    │
│  │                                    │    │
│  │ 검색 결과 (5건)                    │    │
│  │ ┌────────────────────────────┐    │    │
│  │ │ 결과1: 제목 / chunk_text  │    │    │
│  │ │ 출처 | 발행일 | 점수      │    │    │
│  │ └────────────────────────────┘    │    │
│  │ ...                               │    │
│  └────────────────────────────────────┘    │
│                                            │
└────────────────────────────────────────────┘
```

### 3.2 상세 보기: 우측 슬라이드 패널

```
┌─────────────────────┬──────────────────┐
│ 뉴스 목록           │ 기사 상세        │
│                     │                  │
│ [카드1] ← selected  │ 제목             │
│ [카드2]             │ 출처 | 발행일    │
│ [카드3]             │                  │
│                     │ [한줄 요약]      │
│                     │                  │
│                     │ 본문             │
│                     │ ...              │
│                     │                  │
│                     │ [관련 법령]      │
│                     │ [관련 판례]      │
│                     │ [관련 기관]      │
│                     │ [시사점]         │
│                     │                  │
│                     │ [면책 고지]      │
│                     │ [원문 링크]      │
│                     │       [X 닫기]   │
└─────────────────────┴──────────────────┘
```

### 3.3 탭 선택 이유

- 목록과 검색이 **다른 API/데이터 모델** 사용
  - 목록: `NewsArticleSummary` (GET, 페이지네이션)
  - 검색: `NewsSearchResult` (POST, chunk 기반)
- content-marketing 모듈과 동일한 탭 패턴으로 **UX 일관성** 유지

## 4. 컴포넌트 구조

### 4.1 파일 트리

```
frontend/src/features/legal-news/
├── components/
│   ├── index.ts              # 배럴 export
│   ├── DisclaimerBanner.tsx   # 면책 배너
│   ├── NewsCard.tsx           # 뉴스 카드 (목록용)
│   ├── NewsListPanel.tsx      # 뉴스 목록 패널 (필터+카드그리드+페이지네이션)
│   ├── NewsDetailPanel.tsx    # 뉴스 상세 슬라이드 패널
│   ├── SearchPanel.tsx        # 검색 패널 (검색바+결과목록)
│   ├── SearchResultCard.tsx   # 검색 결과 카드
│   ├── NewsSourceBadge.tsx    # 소스 뱃지 (lawtimes/naver)
│   └── TagList.tsx            # 태그 목록
├── hooks/
│   ├── useNewsList.ts         # 목록 조회 + 필터 + 상세 선택
│   └── useNewsSearch.ts       # 검색 쿼리 + 결과 관리
├── services/
│   └── index.ts               # API 호출 함수
├── types/
│   └── index.ts               # TypeScript 타입 (백엔드 1:1 매핑)
└── utils/
    └── formatDate.ts          # 날짜 포맷 유틸

frontend/src/app/legal-news/
└── page.tsx                   # 메인 페이지 (탭 기반)
```

### 4.2 컴포넌트 설명

| 컴포넌트 | 크기 | 역할 |
|----------|------|------|
| `DisclaimerBanner` | 원자 | AI 생성 면책 안내 |
| `NewsSourceBadge` | 원자 | 소스 뱃지 (색상: lawtimes=파랑, naver=초록) |
| `TagList` | 원자 | 태그 목록 (접힘/펼침) |
| `NewsCard` | 카드 | 뉴스 목록용 카드 (제목, 요약, 소스, 날짜, 태그) |
| `SearchResultCard` | 카드 | 검색 결과 카드 (제목, chunk_text, 점수) |
| `NewsListPanel` | 패널 | 목록 탭 전체 (필터 + 카드 그리드 + 페이지네이션) |
| `NewsDetailPanel` | 패널 | 상세 슬라이드 패널 (전문, 요약, 관련 법령 등) |
| `SearchPanel` | 패널 | 검색 탭 전체 (검색바 + 결과 목록) |

## 5. API 연동

### 5.1 엔드포인트 매핑

| 프론트엔드 함수 | 메서드 | 경로 | 요청 | 응답 |
|----------------|--------|------|------|------|
| `fetchNewsList` | GET | `/legal-news/list` | query params | `NewsListResponse` |
| `fetchNewsDetail` | GET | `/legal-news/{article_id}` | path param | `NewsArticleResponse` |
| `searchNews` | POST | `/legal-news/search` | `NewsSearchRequest` | `NewsSearchResponse` |

### 5.2 타입 매핑 (snake_case 유지)

| 백엔드 Pydantic | 프론트엔드 TypeScript |
|-----------------|---------------------|
| `NewsArticleSummary` | `NewsArticleSummary` |
| `NewsArticleResponse` | `NewsArticleResponse` |
| `NewsListResponse` | `NewsListResponse` |
| `NewsSearchRequest` | `NewsSearchRequest` |
| `NewsSearchResult` | `NewsSearchResult` |
| `NewsSearchResponse` | `NewsSearchResponse` |
| `datetime \| None` | `string \| null` (ISO8601) |

## 6. 기존 패턴 재사용

| 신규 파일 | 참조 패턴 | 이유 |
|----------|----------|------|
| types/ | content-marketing/types | snake_case 유지, 인터페이스 구조 |
| services/ | content-marketing/services | axios + endpoints 패턴 |
| useNewsList | content-marketing/hooks/useTrends | useState + useCallback 패턴 |
| useNewsSearch | case-precedent/hooks/useCaseSearch | 검색 상태 관리 패턴 |
| DisclaimerBanner | content-marketing/DisclaimerBanner | 동일 메시지 구조 |
| NewsCard | content-marketing/TrendCard | 카드 레이아웃 구조 |
| page.tsx | content-marketing/page.tsx | 탭 기반 레이아웃 + BackButton |

## 7. 구현 순서

1. **기반 레이어** (병렬): types/index.ts + utils/formatDate.ts
2. **서비스 레이어**: services/index.ts
3. **훅 레이어** (병렬): useNewsList.ts + useNewsSearch.ts
4. **컴포넌트 레이어** (병렬): 원자 → 카드 → 패널
5. **페이지**: app/legal-news/page.tsx
6. **활성화**: modules.ts enabled: true
7. **검증**: npm run build

## 8. 검증 계획

- `npm run build` — 타입/린트 에러 0
- 모듈 동기화 4곳 확인 (modules.ts, api.ts, next.config.js, backend router)
- TypeScript 타입 ↔ Pydantic 스키마 1:1 매칭 확인

---

## 변경 이력

| 버전 | 날짜 | 변경 내용 |
|------|------|----------|
| v1.0.0 | 2026-02-26 | 초기 기획서 작성 |
