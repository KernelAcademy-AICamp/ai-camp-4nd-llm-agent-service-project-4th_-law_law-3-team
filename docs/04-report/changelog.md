# Changelog

모든 주요 기능 완료 사항을 문서화합니다.

---

## [2026-02-27] - 키워드 탐색 시간 범위 선택 기능 추가

### Added

- **키워드 수집 시간 범위 선택 UI**: 48시간/일주일/2주일/한달 드롭다운 (힌트 텍스트 포함)
  - 사용자 선택값 localStorage 영속화 (재방문 시 복원)
- **Backend `time_range` 파라미터**: SSE 스트리밍 엔드포인트에 쿼리 파라미터 추가
  - FastAPI Enum 자동 검증 (무효값 → 422 에러)
- **`TimeRange.DAYS_14`**: Backend Enum + Frontend union type에 14d 추가
- **시간 범위별 캐시 분리**: 캐시 키 `f"{user_id}:{time_range}:keywords"`

### Fixed

- **Critical: 캐시 키 불일치 버그** (Red Team 발견): `search_keyword_news`가 기존 키로 조회하여 뉴스 검색 항상 실패
  - 수정: 모든 TimeRange enum 값을 순회하여 keyword_id 탐색

### Changed

- `backend/app/modules/content_marketing/schema/__init__.py` — TimeRange enum 확장
- `backend/app/modules/content_marketing/router/__init__.py` — time_range 쿼리 파라미터
- `backend/app/services/service_function/content_marketing_service.py` — 캐시 키 정교화, 하드코딩 제거
- `backend/app/tools/trend/collector.py` — time_range 파라미터 전달
- `frontend/src/features/content-marketing/types/index.ts` — TimeRange 타입 확장
- `frontend/src/features/content-marketing/services/index.ts` — timeRange API 전달
- `frontend/src/features/content-marketing/hooks/useKeywordFlow.ts` — 상태 관리 + localStorage
- `frontend/src/features/content-marketing/components/KeywordCollector.tsx` — 드롭다운 UI

### Quality Metrics

- **정적 검증**: ruff + mypy + npm run build 모두 통과
- **3중 검증**: Agent Team + Red Team (Gemini CLI) + External Consultant (Codex CLI)
- **E2E 검증**: 시간 범위 선택 → 키워드 수집 → 뉴스 검색 전체 흐름 확인

---

## [2026-02-27] - 법률신문 크롤러 v2.1 수정 완료 (ND소프트 CMS 대응)

### Fixed

- **법률신문 크롤러 전면 복구**: 2026-02 ND소프트 CMS 전환으로 0건 수집 상태였던 크롤러를 HTML 직접 파싱 방식으로 완전히 재작성
  - RSS 피드 제거 (301→403 차단), 기존 URL 패턴 모두 교체 (404 대응)
  - ND소프트 CMS 전용 CSS 셀렉터 적용 (`altlist-webzine`, `altlist-subject`, `altlist-info` 등)
  - 브라우저 유사 헤더 적용으로 봇 차단(403) 우회
  - 서버 사이드 날짜 필터링 파라미터 발견 및 적용 (`sc_sdate`/`sc_edate`)

### Changed

- `backend/app/tools/news_pipeline/sources/lawtimes_source.py` (v2.0→v2.1, 318줄)
  - robots.txt UA 일관성 수정: `_ROBOT_UA = _BROWSER_HEADERS["User-Agent"]`로 통일 (Red Team + Consultant 공통 지적)
  - 페이지네이션 최대 페이지 제한 추가: `_MAX_PAGES_PER_SECTION = 20` (무한 루프 방지)

### Quality Metrics

- **파이프라인 실행**: 법률신문 30건 수집, 29건 저장, 67 청크 생성
- **수집 성공률**: 100%, 요약 성공률: 100%, 에러: 0건
- **정적 검증**: ruff All checks passed, mypy no issues found
- **3중 검증**: Red Team (Gemini CLI) + External Consultant (Codex CLI) 완료
  - robots.txt UA 일관성 수정 (즉시 적용)
  - 페이지네이션 상한 추가 (즉시 적용)
  - asyncio.gather 동시 수집 (향후 과제)
  - robots.txt fail-closed (불채택 — 수집 안정성 우선)

### Documentation

- `docs/03-analysis/lawtimes-crawler-fix.redteam.md` (Red Team 검증 보고서)
- `docs/03-analysis/lawtimes-crawler-fix.consulting.md` (External Consultant 검증 보고서)
- `docs/04-report/lawtimes-crawler-fix.report.md` (완료 보고서)

---

## [2026-02-26] - Legal News Frontend Module Completion

### Added

- **법률 뉴스 프론트엔드 모듈 (legal-news-frontend)**: 뉴스 목록 + 검색 + 상세 보기 UI
  - 프론트엔드: 18개 파일 (컴포넌트 9, 훅 3, 서비스 1, 타입 1, 유틸 3, 페이지 1)
  - 모듈 활성화: modules.ts `enabled: true`, api.ts, next.config.js 동기화 완료

- **핵심 기능** (F1~F5 전부 구현)
  - F1/F2: 뉴스 목록 조회 + 소스/날짜 필터 + 페이지네이션 (`NewsListPanel`, `useNewsList`)
  - F3: 우측 슬라이드 패널 상세 보기 (`NewsDetailPanel`, `useNewsDetail`)
  - F4: 하이브리드 검색 (Vector + FTS + 리랭커) (`SearchPanel`, `useNewsSearch`)
  - F5: DisclaimerBanner (상단 고정 면책 안내)

- **보안/안정성 강화**
  - AbortController: 3개 비동기 훅 전부 적용 (Race Condition 방지)
  - URL 검증: `utils/url.ts`의 `isSafeUrl` (http/https만 허용, Tabnabbing 방지)
  - `NewsSource` 유니온 타입으로 컴파일 타임 타입 안전성 확보

- **접근성**
  - ESC 키 패널 닫기, `role="dialog"` ARIA 속성 적용

### Changed

- `frontend/src/lib/modules.ts`: legal-news 모듈 `enabled: false` → `enabled: true`

### Quality Metrics

- **정적 검증**: npm run build 성공 (TypeScript 에러 0, ESLint 에러 0)
- **설계 준수도**: 98% (Match Rate) — 기준 90% 초과
- **Red Team 점수**: v1 8.5/10 → v2 9.3/10
- **External Consultant 점수**: v1 6.8/10 → v2 8.0/10
- **반복 횟수**: 3회 (피드백 반영 사이클)
- **번들 사이즈**: 6.32 kB (First Load JS)

### Documentation

- `docs/01-plan/features/legal-news-frontend.plan.md` (v1.0.0)
- `docs/03-analysis/legal-news-frontend.analysis.md` (98% Match Rate)
- `docs/03-analysis/legal-news-frontend.code-review-redteam.md` (v2, 9.3/10)
- `docs/03-analysis/legal-news-frontend.code-review-consulting.md` (v2, 8.0/10)
- `docs/04-report/legal-news-frontend.report.md` (완료 보고서)

---

## [2026-02-20] - Content Marketing Module Completion

### Added

- **콘텐츠 마케팅 모듈 (content-marketing)**: 실시간 트렌드 분석 + AI 유튜브 대본 생성
  - 백엔드: 17개 파일 (모듈, 트렌드 수집, 대본 생성, 에이전트, 환경 설정)
  - 프론트엔드: 16개 파일 (컴포넌트, 훅, 서비스, 타입, 페이지)
  - 모듈 등록: modules.ts, api.ts, next.config.js, multi_agent 통합

- **API 엔드포인트** (5개)
  - POST /api/content-marketing/trends: 트렌드 조회
  - GET /api/content-marketing/trends/{id}: 이슈 상세 조회
  - POST /api/content-marketing/script/generate: 대본 생성 (SSE 스트리밍)
  - POST /api/content-marketing/script/metadata: 메타데이터 생성

- **트렌드 수집 기능**
  - Tavily Search API 통합
  - Naver Search API 통합 (국내 뉴스/블로그)
  - Strategy 패턴 기반 데이터 소스 어댑터 (Phase 2 확장 용이)
  - 병렬 수집 + 중복 제거
  - LLM 기반 법적 해석 가능성 점수 산정
  - RAG 연동 관련 법령/판례 자동 매칭

- **대본 생성 기능**
  - 3단 구조 대본 자동 생성 (도입 → 본론 → 결론)
  - SSE 스트리밍 응답 (실시간 토큰 출력)
  - RAG 검색 결과 직접 인용
  - 페르소나/톤 선택 (전문가 vs 구어체)
  - 길이 옵션 (5분/10분/15분)
  - 메타데이터 자동 생성 (영상 설명문, SEO 태그, 상담 CTA)

- **프론트엔드 UI**
  - 트렌드 대시보드 (카드형 목록, 점수 순 정렬, 필터)
  - 이슈 상세 뷰 (요약, 관련 법령/판례, 대본 생성 버튼)
  - 대본 생성 양식 (주제, 페르소나, 길이)
  - 대본 미리보기 (마크다운 렌더링)
  - 메타데이터 패널
  - 내보내기 기능 (클립보드 복사, TXT 다운로드)

- **에이전트 통합**
  - ContentMarketingAgent (BaseChatAgent 상속)
  - LangGraph 멀티에이전트 라우팅 (INTENT_PATTERNS, AGENT_NODE_MAP)
  - 채팅 위젯 연동 가능

- **환경 변수** (8개)
  - TAVILY_API_KEY, NAVER_CLIENT_ID, NAVER_CLIENT_SECRET
  - PERPLEXITY_API_KEY, YOUTUBE_API_KEY (Phase 2)
  - TREND_MENTION_WEIGHT, TREND_LEGAL_WEIGHT (스코어링 가중치)
  - CONTENT_MARKETING_CACHE_TTL (캐시 TTL)

### Changed

- `backend/app/core/config.py`: 콘텐츠 마케팅 관련 환경 변수 8개 추가
- `backend/app/multi_agent/router.py`: AgentType.CONTENT_MARKETING, INTENT_PATTERNS 추가
- `backend/app/multi_agent/nodes.py`: AGENT_NODE_MAP에 ContentMarketingAgent 등록, 노드 함수 추가
- `backend/app/multi_agent/graph.py`: 콘텐츠 마케팅 노드/엣지 등록
- `frontend/src/lib/modules.ts`: content-marketing 모듈 등록 (enabled: true)
- `frontend/src/lib/api.ts`: contentMarketing endpoint 추가
- `frontend/next.config.js`: /api/content-marketing rewrite 프록시 규칙 추가

### Fixed

- Backend mypy: 3개 타입 에러 수정 (Tools 타입, Union 반환값)
- Frontend TypeScript: content-marketing 타입 일관성 검증

### Quality Metrics

- **정적 검증**: ruff ✅, mypy ✅, npm run build ✅
- **설계 준수도**: 93% (Match Rate)
- **반복 횟수**: 0 (설계 → 구현 1회 통과)
- **커버리지**: 필수 기능 100%, 선택 기능 85% (ScriptEditor, CitationList 제외)

### Tech Stack

| 영역 | 기술 |
|------|------|
| 트렌드 수집 | Tavily + Naver API |
| 스코어링 | LLM + 규칙 하이브리드 |
| 대본 생성 | LLM (Solar/OpenAI) + RAG |
| Backend | FastAPI + LangGraph |
| Frontend | Next.js + Tailwind CSS |

### Documentation

- ✅ `docs/01-plan/features/content-marketing.plan.md` (27개 FR, 기술 스택)
- ✅ `docs/02-design/features/content-marketing.design.md` (상세 아키텍처, API 스펙)
- ✅ `docs/03-analysis/content-marketing.analysis.md` (Gap 분석, 93% Match Rate)
- ✅ `docs/04-report/features/content-marketing.report.md` (완료 보고서)

### Breaking Changes

None

### Known Limitations

- ScriptEditor (마크다운 인라인 에디터): 미구현 (메타데이터 문자열 수정으로 대체)
- CitationList (별도 컴포넌트): 미구현 (ScriptPreview에 통합 표시)
- Phase 2 추가 소스 (Perplexity, Google Trends, YouTube): 미구현 (다음 단계)

### Next Steps

1. **Phase 2 확장**: Perplexity, Google Trends, YouTube API 통합 (3-4 시간)
2. **대본 이력 저장**: PostgreSQL 저장 기능 추가
3. **성능 모니터링**: API 호출 비용, 응답 시간 추적
4. **사용자 피드백 수집**: 대본 품질 개선 데이터 수집

---

## [TEMPLATE] How to Use This Changelog

### Format

```markdown
## [YYYY-MM-DD] - Feature/Module Name

### Added
- New feature description

### Changed
- Modified behavior or API

### Fixed
- Bug fixes

### Deprecated
- Features to be removed

### Removed
- Deleted features

### Security
- Security-related changes
```

### Guidelines

- Use present tense ("Add", "Change", not "Added", "Changed")
- Group related changes together
- Reference issue numbers or commit hashes when applicable
- Link to related documentation

---

**Maintained by**: Claude (AI Assistant)
**Last Updated**: 2026-02-20
