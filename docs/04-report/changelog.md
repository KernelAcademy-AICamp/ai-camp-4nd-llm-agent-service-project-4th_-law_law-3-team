# Changelog

모든 주요 기능 완료 사항을 문서화합니다.

---

## [2026-03-01] - Storyboard Module Completion

### Added

- **스토리보드 모듈 (storyboard)**: 멀티모달 입력 → AI 사건 타임라인 추출 → 스토리보드 이미지 생성 → 영상 합성
  - 백엔드: FastAPI 모듈 (timeline 추출, 이미지 생성, 비디오 합성, STT, Gemini Vision, JobManager)
  - 프론트엔드: Next.js 컴포넌트/훅/서비스 (TimelineCard, TimelineItemEditor, VideoGenerationModal, MultiInputPanel 등)

- **API 엔드포인트** (9개)
  - POST /api/storyboard/extract-timeline: 텍스트 기반 타임라인 추출
  - POST /api/storyboard/extract-from-audio: 음성 기반 타임라인 추출 (Whisper STT)
  - POST /api/storyboard/extract-from-image: 이미지 기반 타임라인 추출 (Gemini Vision)
  - POST /api/storyboard/validate-timeline: 타임라인 검증
  - POST /api/storyboard/generate-image: 단일 이미지 생성
  - POST /api/storyboard/generate-images-batch: 배치 이미지 생성 (비동기 JobManager)
  - GET /api/storyboard/job-status/{job_id}: 배치 작업 폴링
  - GET /api/storyboard/job-stream/{job_id}: 배치 작업 SSE 스트리밍
  - POST /api/storyboard/generate-video: 영상 합성 (moviepy)

- **타임라인 추출 기능**
  - OpenAI GPT-4o-mini 기반 구조화 텍스트 분석 (date, time, location, participants, evidence 등 12개 필드)
  - Whisper STT 음성 인식 → 타임라인 추출 파이프라인
  - Gemini Vision 이미지 분석 → 타임라인 추출 파이프라인
  - 한국 법률 사건 도메인 특화 프롬프트

- **스토리보드 이미지 생성**
  - Gemini 2.0 Flash 이미지 생성
  - 배치 처리: Semaphore(2) + asyncio.gather() 병렬화
  - SSE + 폴링 이중 진행 모니터링
  - 1시간 자동 Job 정리 (메모리 누수 방지)

- **영상 합성**
  - moviepy 기반 이미지 시퀀스 → 영상 변환
  - `/media/` 정적 파일 서빙

- **프론트엔드 UI**
  - MultiInputPanel: 텍스트/음성/이미지 탭 전환 입력
  - TimelineView: 날짜 그룹 기반 타임라인 시각화
  - TimelineCard: 6가지 참여자 역할 색상 코딩, 이미지 생성 인라인
  - TimelineItemEditor: 타임라인 아이템 인라인 편집
  - VideoGenerationModal: 아이템 선택 → 영상 생성

### Changed

- `useImageGeneration`, `useVideoGeneration` 서브 훅 분리 (useTimelineState 391줄 → 291줄)
- `schema/__init__.py` → `schema/models.py` + `schema/responses.py` 파일 분리
- `utils/generateId.ts` 유틸 분리
- `transformParticipant()`, `transformTimelineItem()` 런타임 타입 가드 도입

### Fixed

- C-1: 동기 OpenAI 클라이언트 → AsyncOpenAI 전환 (이벤트 루프 차단 해소)
- C-2: JobManager 메모리 누수 → 1시간 자동 정리
- C-3: React setState updater 남용 → useRef 참조 패턴 (Strict Mode 호환)
- C-4: 마이크 스트림 리소스 누수 수정
- H-1: 파일 업로드 크기 검증 (DoS 방지, 25MB/20MB 제한)
- H-3: assert 문 → ValueError 명시 교체
- H-4: vision.py 동기 Gemini 호출 → 비동기 전환
- H-8: 이미지 URL 검증 (XSS 방지)
- H-13: camelCase→snake_case 변환 누락 수정 (API 계약 동기화)
- M-4/M-7: UUID 형식 검증 추가

### Quality Metrics

- **정적 검증**: ruff PASS / mypy PASS (type:ignore 2건) / npm run build PASS
- **설계 준수도**: 94.2% (목표 90% 초과)
- **이슈 수정**: 39/43건 (Critical 100%, High 100%, Medium 93.8%, Low 70%)

### Tech Stack

| 영역 | 기술 |
|------|------|
| 텍스트 추출 | OpenAI GPT-4o-mini |
| 음성 인식 | OpenAI Whisper |
| 이미지 분석 | Google Gemini 2.0 Flash (Vision) |
| 이미지 생성 | Google Gemini 2.0 Flash |
| 영상 합성 | moviepy |
| Backend | FastAPI + asyncio |
| Frontend | Next.js + Tailwind CSS |

### Documentation

- 분석: `docs/03-analysis/storyboard.analysis.md` (43건 이슈 + Gap Analysis 재검증)
- 완료: `docs/04-report/features/storyboard.report.md`

### Known Limitations

- L-8: `types-requests` mypy 스텁 미설치 (기능 영향 없음)
- L-9: Gemini `generate_content()` 인자 타입 `type: ignore` 억제
- M-12: 앱 전체 다크 테마 미구현 (별도 PR 예정)
- N-2: `video_generation.py` MEDIA_DIR 하드코딩 미수정 (`image_generation.py`와 불일치)

### Next Steps

1. N-2: `video_generation.py` MEDIA_DIR `_resolve_media_dir()` 패턴 통일
2. N-1: `useVideoGeneration.ts` generateId import 교체
3. L-8: `types-requests` 스텁 설치
4. 타임라인 저장/불러오기 (PostgreSQL 연동) — 다음 스프린트

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
