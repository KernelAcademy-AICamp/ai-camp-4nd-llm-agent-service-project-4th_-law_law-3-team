# PDCA 완료 보고서: legal-news-frontend

> **Author**: bkit-report-generator
> **Created**: 2026-02-26
> **Status**: Completed
> **Match Rate**: 98%

---

## 1. 개요

| 항목 | 내용 |
|------|------|
| Feature 이름 | legal-news-frontend |
| 목적 | 법률 뉴스 파이프라인(legal-news-pipeline) 백엔드 v0.3.0의 수집·요약 데이터를 사용자에게 제공하는 프론트엔드 UI 구현 |
| 대상 사용자 | 변호사(lawyer), 일반 사용자(user) — 동일 뷰 |
| PDCA 시작일 | 2026-02-26 |
| PDCA 완료일 | 2026-02-26 |
| 주요 기술 | Next.js 14 (App Router), TypeScript, Tailwind CSS, axios |

### 배경

법률 뉴스 파이프라인 백엔드가 완전 구현된 상태였으나, 원래 기획에서 프론트엔드를 명시적으로 제외("백엔드 전용")했기 때문에 사용자 인터페이스가 부재하였다. 이를 해소하기 위해 기존 content-marketing 모듈 패턴을 재사용하여 일관된 UX를 유지하면서 신규 프론트엔드 모듈을 구현하였다.

---

## 2. Plan 요약

### 2.1 핵심 기능 (F1~F5)

| # | 기능 | API | 설명 |
|---|------|-----|------|
| F1 | 뉴스 목록 조회 | `GET /legal-news/list` | 최신 뉴스 카드 그리드 + 페이지네이션 |
| F2 | 소스/날짜 필터 | `GET /legal-news/list?source=&published_date=` | lawtimes/naver 소스 필터, 날짜 필터 |
| F3 | 뉴스 상세 보기 | `GET /legal-news/{article_id}` | 우측 슬라이드 패널로 기사 전문 표시 |
| F4 | 뉴스 검색 | `POST /legal-news/search` | 하이브리드 검색 (Vector + FTS + 리랭커) |
| F5 | 면책 안내 | — | 상단 고정 DisclaimerBanner |

### 2.2 대상 사용자

- **변호사(lawyer)**: 법률 뉴스 트렌드 파악, 관련 기사 검색
- **일반 사용자(user)**: 법률 뉴스 열람, 관심 키워드 검색
- 역할 차이 없음: 두 역할 모두 동일한 뷰 제공

### 2.3 UI/UX 설계 요약

- **탭 기반 레이아웃**: content-marketing 모듈과 동일한 패턴으로 UX 일관성 유지
  - 탭 1: 뉴스 목록 (소스/날짜 필터 + 3열 카드 그리드 + 페이지네이션)
  - 탭 2: 뉴스 검색 (검색바 + 소스/결과수 필터 + 검색 결과 목록)
- **상세 보기**: 우측 슬라이드 패널 (배경 오버레이, ESC 닫기, 접근성 role="dialog")
- **기존 패턴 재사용**: content-marketing/TrendCard → NewsCard, content-marketing/page.tsx → page.tsx 등 7개 패턴 재사용

---

## 3. 구현 결과 (Do)

### 3.1 생성된 파일 목록

| 파일 | 역할 |
|------|------|
| `features/legal-news/components/index.ts` | 배럴 export |
| `features/legal-news/components/DisclaimerBanner.tsx` | AI 생성 면책 안내 배너 (amber 스타일) |
| `features/legal-news/components/NewsCard.tsx` | 뉴스 목록용 카드 (제목, 요약, 소스, 날짜, 태그) |
| `features/legal-news/components/NewsListPanel.tsx` | 목록 탭 전체 (필터 + 카드 그리드 + 페이지네이션) |
| `features/legal-news/components/NewsDetailPanel.tsx` | 상세 슬라이드 패널 (전문, 요약, 관련 법령 등) |
| `features/legal-news/components/SearchPanel.tsx` | 검색 탭 전체 (검색바 + 결과 목록) |
| `features/legal-news/components/SearchResultCard.tsx` | 검색 결과 카드 (제목, chunk_text, 점수) |
| `features/legal-news/components/NewsSourceBadge.tsx` | 소스 뱃지 (lawtimes=파랑, naver=초록) |
| `features/legal-news/components/TagList.tsx` | 태그 목록 (maxVisible + 더보기 카운트) |
| `features/legal-news/hooks/useNewsList.ts` | 목록 조회 + 필터 상태 관리 |
| `features/legal-news/hooks/useNewsSearch.ts` | 검색 쿼리 + 결과 관리 |
| `features/legal-news/hooks/useNewsDetail.ts` | 상세 조회 로직 (관심사 분리, 설계 외 추가) |
| `features/legal-news/services/index.ts` | API 호출 함수 (fetchNewsList, fetchNewsDetail, searchNews) |
| `features/legal-news/types/index.ts` | TypeScript 타입 (백엔드 Pydantic 1:1 매핑) |
| `features/legal-news/utils/formatDate.ts` | 날짜 포맷 유틸 (formatDate, formatDateShort) |
| `features/legal-news/utils/url.ts` | URL 안전 검증 유틸 (isSafeUrl — 갭 분석 반영으로 분리) |
| `features/legal-news/utils/constants.ts` | 공유 상수 (SOURCE_OPTIONS, LIMIT_OPTIONS, DEFAULT_FILTERS — 설계 외 추가) |
| `app/legal-news/page.tsx` | 메인 페이지 (탭 기반 레이아웃, BackButton, DisclaimerBanner) |

### 3.2 코드 통계

| 항목 | 수량 |
|------|:----:|
| 총 파일 수 | 18개 |
| 컴포넌트 | 9개 (index.ts 포함) |
| 커스텀 훅 | 3개 |
| 서비스 함수 | 3개 (fetchNewsList, fetchNewsDetail, searchNews) |
| 유틸 파일 | 3개 (formatDate, url, constants) |
| 타입 정의 | 6개 인터페이스 (NewsArticleSummary, NewsArticleResponse, NewsListResponse, NewsSearchRequest, NewsSearchResult, NewsSearchResponse) |
| 페이지 | 1개 |

### 3.3 빌드 결과

| 항목 | 결과 |
|------|------|
| 빌드 상태 | 성공 (npm run build) |
| TypeScript 에러 | 0건 |
| ESLint 에러 | 0건 |
| 번들 사이즈 | 6.32 kB (First Load JS) |

### 3.4 모듈 동기화 4곳 상태

| 항목 | 파일 | 값 | 상태 |
|------|------|---|:----:|
| `modules.ts` enabled | `frontend/src/lib/modules.ts:113` | `enabled: true` | 완료 |
| `modules.ts` roles | `frontend/src/lib/modules.ts:114` | `roles: ['lawyer', 'user']` | 완료 |
| `api.ts` endpoint | `frontend/src/lib/api.ts:23` | `legalNews: '/legal-news'` | 완료 |
| `next.config.js` rewrites | `frontend/next.config.js:46-49` | `/api/legal-news/:path*` → backend | 완료 |

> 시작 시점에 `enabled: false`로 설정되어 있던 것이 구현 완료 후 `enabled: true`로 정상 활성화됨.

---

## 4. 품질 검증 결과 (Check)

### 4.1 Gap Analysis 결과

| 카테고리 | 점수 |
|----------|:----:|
| 파일 구조 정합성 (15/15) | 100% |
| 컴포넌트 정합성 (8/8) | 100% |
| 기능 정합성 F1~F5 (5/5) | 100% |
| API 연동 정합성 (3/3) | 100% |
| 타입 정합성 (전 필드) | 100% |
| 모듈 동기화 4곳 (4/4) | 100% |
| 코드 리뷰 피드백 반영 (6/6) | 100% |
| 아키텍처 준수 | 95% |
| **종합 Match Rate** | **98%** |

- 누락 기능: 0건
- 설계 초과 추가: 4건 (모두 품질 향상 목적)
- 경미한 이슈: 1건 (`isSafeUrl` 위치 → 갭 분석 이후 `utils/url.ts`로 분리 완료)

### 4.2 Red Team 리뷰 (Gemini CLI)

| 버전 | 점수 | 주요 내용 |
|------|:----:|----------|
| v1 | 8.5/10 | AbortController 적용 권장, URL 검증 분리, ESC 키, useNewsDetail 분리 요청 |
| v2 | **9.3/10** | "Best Practice에 부합하는 견고한 아키텍처" — 모든 v1 피드백 완벽 반영 확인 |

v2 핵심 평가:
- XSS 방지: `dangerouslySetInnerHTML` 미사용, React 기본 이스케이프 활용
- URL 검증: `utils/url.ts`의 `isSafeUrl`로 http/https만 허용, Tabnabbing 방지
- Race Condition 해결: 3개 비동기 훅 모두 AbortController 적용
- 타입 안전성: `NewsSource` 유니온 타입으로 컴파일 타임 검증

### 4.3 External Consultant 리뷰 (Codex CLI)

| 버전 | 점수 | 주요 내용 |
|------|:----:|----------|
| v1 | 6.8/10 | selectedArticle 초기화 누락, 오버레이 에러 조건, NewsSourceBadge 타입, 서비스 타입 불일치 |
| v2 | **8.0/10** | 주요 이슈 반영 확인, ID 계약 명확화, 타입 강화 |

v2 잔존 권장 사항 (미적용 — 기능 오류 없음):
- Focus Trap (`react-focus-lock`): 접근성 향상용
- 런타임 API 응답 검증 (Zod): 안전성 강화용
- DetailOverlay 공통 컴포넌트 추출: 코드 중복 제거용

### 4.4 발견된 이슈 및 수정 이력

| 이슈 | 발견 시점 | 처리 여부 | 비고 |
|------|:--------:|:--------:|------|
| `useNewsDetail` 훅 미분리 | Red Team v1 | 완료 | 관심사 분리 목적 |
| AbortController 미적용 | Red Team v1 | 완료 | 3개 훅 전부 적용 |
| URL 검증 함수 미존재 | Red Team v1 | 완료 | `utils/url.ts` 생성 |
| ESC 키 닫기 미구현 | Red Team v1 | 완료 | useEffect로 구현 |
| `utils/constants.ts` 상수 중복 | Red Team v1 | 완료 | 공유 상수 파일 생성 |
| `NewsSource` 유니온 타입 미적용 | Red Team v1 | 완료 | types/index.ts 추가 |
| `isSafeUrl`이 formatDate.ts에 공존 | Gap Analysis | 완료 | `utils/url.ts`로 분리 |
| `selectedArticle` 초기화 누락 | Consultant v1 | 완료 | selectArticle 시작 시 null 처리 |
| 오버레이 에러 조건 미포함 | Consultant v1 | 완료 | `detailError` 조건 추가 |
| `NewsSourceBadge` 타입 느슨 | Consultant v1 | 완료 | `NewsSource` 타입 적용 |
| 서비스 계층 타입 불일치 | Consultant v1 | 완료 | `NewsSource | null` 타입 강화 |

총 반복 개선 횟수: 3회 (Red Team v1 반영 → Gap 분석 반영 → Consultant v2 반영)

---

## 5. 개선 이력 (Act)

### 5.1 1차 피드백 반영 — Red Team v1 + External Consultant v1

Red Team v1(8.5/10) 및 External Consultant v1(6.8/10) 피드백 기반으로 아래 항목을 구현에 반영:

- `hooks/useNewsDetail.ts` 신규 분리: `useNewsList`에서 상세 조회 로직 추출하여 관심사 분리
- AbortController 전면 적용: `useNewsList.ts`, `useNewsSearch.ts`, `useNewsDetail.ts` 모두 적용
- URL 프로토콜 검증 추가: `isSafeUrl` 함수 생성 (http/https만 허용), `NewsDetailPanel`의 원문 링크에 적용
- ESC 키 닫기: `NewsDetailPanel.tsx`에 `useEffect`로 keydown 이벤트 리스너 구현
- `utils/constants.ts` 생성: `SOURCE_OPTIONS`, `LIMIT_OPTIONS`, `DEFAULT_FILTERS` 공유 상수로 추출
- `NewsSource` 유니온 타입: `'lawtimes' | 'naver'`로 정의하여 컴파일 타임 검증 강화
- `selectedArticle` 초기화: 새 상세 조회 시작 시 기존 기사 즉시 null 처리
- 오버레이 에러 조건 추가: `selectedArticle || detailLoading || detailError` 조건으로 에러 시에도 모달 컨텍스트 유지
- `NewsSourceBadge` props 타입 강화: `source: string` → `source: NewsSource`

### 5.2 Gap 분석 피드백 반영

Gap Analysis(98%) 결과에서 권장된 경미한 이슈 1건 처리:

- `isSafeUrl` 함수 위치 이동: `utils/formatDate.ts`에 공존하던 URL 검증 함수를 `utils/url.ts`로 분리
  - 단일 책임 원칙 강화
  - `formatDate.ts`: 날짜 관련 함수만 유지 (`formatDate`, `formatDateShort`)
  - `url.ts`: URL 검증 함수만 유지 (`isSafeUrl`)

### 5.3 2차 피드백 반영 — Red Team v2 + External Consultant v2

Red Team v2(9.3/10)에서 전체 1차 반영 확인 완료.
External Consultant v2(8.0/10)에서 아래 추가 항목 반영:

- 서비스 계층 타입 강화: `source?: string | null` → `source?: NewsSource | null` (fetchNewsList 파라미터)
- ID 계약 명확화: `fetchNewsDetail` 주석에 `doc_id`로 조회함을 명시

---

## 6. 최종 품질 지표

| 지표 | 값 |
|------|-----|
| Match Rate | 98% |
| Red Team 최종 점수 | 9.3/10 |
| External Consultant 최종 점수 | 8.0/10 |
| 빌드 상태 | 성공 (TypeScript 에러 0, ESLint 에러 0) |
| 총 파일 수 | 18개 |
| 번들 사이즈 | 6.32 kB (First Load JS) |
| 누락 기능 | 0건 |
| 미해결 Critical 이슈 | 0건 |
| 개선 반복 횟수 | 3회 |

---

## 7. 향후 개선 권장 사항

아래 항목은 현재 기능 오류가 없으나 향후 품질 고도화를 위해 backlog으로 추가를 권장한다.

### 7.1 접근성 강화 (Red Team 제안)

- **Focus Trap 구현**: `react-focus-lock` 라이브러리 활용 → 상세 패널 열림 시 포커스를 패널 내로 가두어 키보드 접근성 WCAG 2.1 AA 수준 충족
- 우선순위: 중간 (접근성 요건이 강해질 경우 선행 적용)

### 7.2 검색 UX 강화 (Red Team 제안)

- **검색어 하이라이팅**: `SearchResultCard`의 `chunk_text` 내 검색어를 굵게 또는 배경색으로 강조
- **Skeleton UI**: 목록/검색 결과 로딩 중 카드 스켈레톤 표시로 CLS(Cumulative Layout Shift) 감소
- 우선순위: 낮음 (로딩 스피너로 현재 충분히 처리됨)

### 7.3 컴포넌트 리팩토링 (Consultant 제안)

- **DetailOverlay 공통 컴포넌트 추출**: `NewsListPanel`과 `SearchPanel`에 동일하게 반복되는 오버레이+패널 구조를 공통 컴포넌트로 분리
- 우선순위: 낮음 (동작에 영향 없는 코드 중복)

### 7.4 테스트 추가 (Consultant 제안)

- **훅 단위 테스트**: `useNewsList`, `useNewsSearch`, `useNewsDetail` — MSW(Mock Service Worker) 기반 테스트
- **유틸 함수 테스트**: `isSafeUrl`, `formatDate` — 경계값 테스트 (빈 문자열, 잘못된 URL 등), 회귀 방지 효과 큼
- **컴포넌트 테스트**: `NewsCard`, `NewsSourceBadge` — Vitest + React Testing Library
- 우선순위: 중간 (안정성 요건 강화 시 선행 적용)

### 7.5 런타임 검증 (Consultant 제안)

- **Zod 스키마 검증**: API 응답에 대한 런타임 타입 검증 추가 → 백엔드 스키마 변경 시 조기 오류 감지
- 우선순위: 낮음 (TypeScript 컴파일 타임 검증이 현재 충분)

---

## 8. PDCA 사이클 요약

```
[Plan] ✅ → [Design] ⏭ → [Do] ✅ → [Check] ✅ 98% → [Act] ✅ → [Report] ✅
```

| 단계 | 상태 | 문서 | 비고 |
|------|:----:|------|------|
| Plan | 완료 | `docs/01-plan/features/legal-news-frontend.plan.md` | v1.0.0 |
| Design | 생략 | — | Plan 문서가 설계 수준까지 상세 기술하여 별도 Design 문서 불필요로 판단 |
| Do | 완료 | `frontend/src/features/legal-news/`, `frontend/src/app/legal-news/page.tsx` | 18개 파일 구현 |
| Check | 완료 | `docs/03-analysis/legal-news-frontend.analysis.md` | 98% Match Rate |
| Act (1차) | 완료 | — | Red Team v1 + Consultant v1 피드백 반영 |
| Act (2차) | 완료 | — | Gap 분석 반영 (isSafeUrl 분리) |
| Act (3차) | 완료 | — | Red Team v2 + Consultant v2 피드백 반영 |
| Report | 완료 | `docs/04-report/legal-news-frontend.report.md` | 본 문서 |

---

## 관련 문서

| 문서 유형 | 경로 |
|----------|------|
| Plan | `docs/01-plan/features/legal-news-frontend.plan.md` |
| Gap Analysis | `docs/03-analysis/legal-news-frontend.analysis.md` |
| Red Team 코드 리뷰 v2 | `docs/03-analysis/legal-news-frontend.code-review-redteam.md` |
| External Consultant 코드 리뷰 v2 | `docs/03-analysis/legal-news-frontend.code-review-consulting.md` |
| 백엔드 파이프라인 Plan | `docs/01-plan/features/legal-news-pipeline.plan.md` |

---

## 변경 이력

| 버전 | 날짜 | 변경 내용 |
|------|------|----------|
| v1.0 | 2026-02-26 | PDCA 완료 보고서 초안 작성 |
