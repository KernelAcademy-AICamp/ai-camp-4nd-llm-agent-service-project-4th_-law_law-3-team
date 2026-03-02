# 콘텐츠 마케팅 대본 + 웹툰 스토리보드 기획서

> **기능명**: 대본 생성 시 웹툰 스토리보드 동시 생성
> **PDCA Phase**: Plan
> **작성일**: 2026-02-28
> **작성자**: PM (Agent Team 7명 통합)
> **상태**: 초안 — Red Team / External Consultant 검증 전

---

## 1. 개요

### 1.1 목적
콘텐츠 마케팅 기능에서 "선택된 뉴스로 대본 생성" 시, 좌측에 대본을 표시하고 우측에 나노바나나(Gemini 3 Pro Image) 기반 웹툰 스토리보드를 동시 생성하여, 변호사가 유튜브 영상 구성을 시각적으로 확인할 수 있게 한다.

### 1.2 대상 사용자
- 법률 유튜브 채널 운영 변호사
- 법률 콘텐츠 마케팅 담당자

### 1.3 핵심 가치
- 대본만으로는 파악하기 어려운 **영상 구성을 웹툰으로 시각화**
- 유튜브 영상 제작 전 **씬별 구성 사전 검토** 가능
- 웹툰 형식으로 **비전문가도 영상 흐름 이해** 가능

---

## 2. 기능 요구사항

### 2.1 사용자 플로우

```
[키워드 수집] → [뉴스 선택] → [선택한 뉴스로 대본 생성] 버튼 클릭
        │
        ▼
┌──────────────────────────────────────────────┐
│  Phase 1: 대본 SSE 스트리밍 (기존)            │
│  → 좌측 패널에 실시간 대본 표시               │
│  → hooking → analysis → advice_cta 순차 생성  │
├──────────────────────────────────────────────┤
│  Phase 2: 스토리보드 패널 SSE 스트리밍 (신규)  │
│  → 대본 완성 후 자동 트리거                   │
│  → 장면 분할 → 패널 목록 실시간 전달          │
├──────────────────────────────────────────────┤
│  Phase 3: 웹툰 이미지 생성 (신규)             │
│  → 패널별 나노바나나 이미지 순차 생성          │
│  → 우측 패널에 실시간 이미지 표시              │
└──────────────────────────────────────────────┘
```

### 2.2 화면 레이아웃

```
[데스크톱 — md 이상]
┌─────────────────────┬──────────────────────────┐
│  대본 (좌측 45%)     │  스토리보드 (우측 55%)     │
│                     │                          │
│  ┌───────────────┐  │  ┌────────────────────┐  │
│  │ 도입 (Hooking)│  │  │ 패널 1: 훅 장면    │  │
│  │ ────────────  │  │  │ [웹툰 이미지]      │  │
│  │ 텍스트 내용   │  │  │ "프롬프트 인젝션"  │  │
│  └───────────────┘  │  ├────────────────────┤  │
│  ┌───────────────┐  │  │ 패널 2: 훅 질문    │  │
│  │ 본론          │  │  │ [웹툰 이미지]      │  │
│  │ (Analysis)    │  │  ├────────────────────┤  │
│  │ ────────────  │  │  │ 패널 3: 법 설명    │  │
│  │ 텍스트 내용   │  │  │ [웹툰 이미지]      │  │
│  │ ...           │  │  │ ...                │  │
│  └───────────────┘  │  ├────────────────────┤  │
│  ┌───────────────┐  │  │ 패널 N: CTA        │  │
│  │ 결론          │  │  │ [웹툰 이미지]      │  │
│  │ (Advice/CTA)  │  │  │ "변호사 상담"      │  │
│  └───────────────┘  │  └────────────────────┘  │
└─────────────────────┴──────────────────────────┘

[태블릿] 탭 전환: [대본] | [스토리보드]
[모바일] 스택형 + 하단 슬라이드업 시트
```

### 2.3 핵심 인터랙션
- **스크롤 동기화**: 대본 섹션 스크롤 시 해당 스토리보드 패널 자동 하이라이트 (IntersectionObserver)
- **역방향 연동**: 스토리보드 패널 클릭 시 대본 해당 섹션으로 스크롤
- **섹션-패널 매핑**: hooking → 2~3패널, analysis → 4~8패널, advice_cta → 2~3패널
- **패널 재생성**: 개별 패널의 이미지를 재생성 가능
- **일괄 다운로드**: 전체 스토리보드 이미지 ZIP 다운로드

---

## 3. 기술 설계

### 3.1 아키텍처 개요

```
Frontend                          Backend
────────                          ───────
ScriptStoryboardSplitView
  ├── ScriptPanel (좌측)
  │   └── useScript (기존)  ──── POST /script/generate (SSE, 기존)
  │
  └── StoryboardPanel (우측)
      └── useStoryboardStream ── POST /script/webtoon (Job 생성)
                                 GET  /script/webtoon/{job_id}/stream (SSE)
                                 GET  /script/webtoon/{job_id} (폴링)
                                      │
                                      ▼
                               WebtoonPipeline
                                 ├── Chain 1: ScriptToSceneSplitter (Solar Pro2)
                                 ├── Chain 2: SceneToImagePromptBuilder (텍스트 처리)
                                 └── Chain 3: generate_webtoon_panel (Gemini 3 Pro Image)
```

### 3.2 Backend API 엔드포인트 (신규 3개)

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/api/content-marketing/script/webtoon` | POST | 웹툰 스토리보드 Job 생성, job_id 즉시 반환 |
| `/api/content-marketing/script/webtoon/{job_id}/stream` | GET | SSE로 패널별 생성 진행률 스트리밍 |
| `/api/content-marketing/script/webtoon/{job_id}` | GET | 작업 상태 폴링 조회 |

### 3.3 요청/응답 스키마

```python
# 요청
class WebtoonGenerateRequest(BaseModel):
    topic: str
    sections: dict[str, str]  # {"hooking": "...", "analysis": "...", "advice_cta": "..."}
    persona: PersonaType
    persona_id: str | None = None
    panel_count: int | None = None  # None이면 자동 결정 (8~14)

# 응답 (Job 생성)
class WebtoonJobResponse(BaseModel):
    job_id: str
    status: str  # "accepted"
    estimated_panels: int

# SSE 이벤트
class WebtoonStreamEvent(BaseModel):
    event: str  # panel_start | panel_complete | panel_failed | all_done | error
    panel_number: int | None = None
    total_panels: int | None = None
    section: str | None = None  # hooking | analysis | advice_cta
    caption: str | None = None
    image_url: str | None = None
    error: str | None = None

# 웹툰 패널
class WebtoonPanel(BaseModel):
    panel_number: int
    section: str  # hooking | analysis | advice_cta
    scene_type: str  # hook_shock | legal_explanation | case_example | ...
    script_excerpt: str
    scene_description: str
    location: str
    time_of_day: str
    characters: list[str]
    emotion: str
    visual_focus: str
    camera_angle: str
    legal_keyword: str
    image_prompt: str | None = None
    image_url: str | None = None
    image_status: str = "pending"  # pending | generating | completed | error
    error_message: str | None = None
    # 메타데이터 (External Consultant 제안 — 채택)
    model_version: str | None = None      # 사용된 모델 버전
    prompt_version: str | None = None     # 프롬프트 템플릿 버전
    generation_cost_ms: int | None = None # 생성 소요 시간 (ms)
    safety_flags: list[str] | None = None # 안전 필터 플래그
```

### 3.4 AI 파이프라인 (3-Chain)

**Chain 1: 대본 → 장면 분할** (ScriptToSceneSplitter)
- LLM: Upstage Solar Pro2 (기존 `get_chat_model` 재사용)
- 입력: 3섹션 대본 텍스트 + 주제
- 출력: WebtoonPanel 목록 (8~14개)
- 패널 수 결정: HOOKING 2~3 / ANALYSIS 4~8 / ADVICE_CTA 2~3

**Chain 2: 장면 → 이미지 프롬프트** (SceneToImagePromptBuilder)
- LLM 불필요 (텍스트 조합)
- 입력: WebtoonPanel + LawyerCharacterProfile
- 출력: Gemini 전용 영문 이미지 프롬프트
- 웹툰 스타일 프롬프트 템플릿 적용 (한국 웹툰 만화풍, 법률 드라마 톤)

**Chain 3: 이미지 생성** (generate_webtoon_panel)
- 모델: `gemini-3-pro-image-preview` (나노바나나)
- 첫 패널: 레퍼런스 없이 생성 → 저장
- 이후 패널: 첫 패널을 image_reference로 전달 (캐릭터 일관성)
- 순차 처리 (캐릭터 일관성 필수)
- 패널당 최대 2회 재시도 → 실패 시 PIL 플레이스홀더

### 3.5 캐릭터 일관성 전략

```python
@dataclass
class LawyerCharacterProfile:
    gender: str = "male"
    age_range: str = "35-45"
    hair: str = "neat black hair, side-parted"
    suit: str = "dark navy pinstripe suit, white dress shirt, burgundy tie"
    build: str = "lean, professional posture"
    expression_default: str = "calm, authoritative, reassuring"
```

- **방법 A (기본)**: 모든 패널에 동일 텍스트 프롬프트 주입
- **방법 B (권장)**: 첫 패널 이미지를 이후 생성 시 `image_reference`로 전달

### 3.6 웹툰 스타일 정의

- 스타일: Korean webtoon manhwa art style, professional legal drama
- 색상: navy blue 기조, gold accent, cel shading
- 포맷: cinematic widescreen 16:9 패널, full-bleed illustration
- 섹션별 오버라이드:
  - HOOKING: bold high-contrast, 빨간 액센트 (충격/긴장)
  - ANALYSIS: clean informative, 네이비 블루 (권위/전문성)
  - ADVICE_CTA: warm golden light (희망/해결)

### 3.7 씬 타입 정의 (8종)

| 씬 타입 | 섹션 | 설명 |
|---------|------|------|
| `hook_shock` | hooking | 충격적 사실/숫자 강조 |
| `hook_question` | hooking | 질문 유도 |
| `legal_explanation` | analysis | 법 조문/법령 설명 |
| `case_example` | analysis | 판례/사례 설명 |
| `conflict_drama` | analysis | 갈등/분쟁 장면 |
| `document_closeup` | analysis | 서류/계약서 클로즈업 |
| `lawyer_advice` | advice_cta | 변호사 직접 조언 |
| `cta_subscribe` | advice_cta | 구독/상담 유도 CTA |

---

## 4. Frontend 설계

### 4.1 컴포넌트 구조

**신규 컴포넌트:**
- `ScriptStoryboardSplitView.tsx` — 최상위 분할 뷰 컨테이너
- `ScriptSectionList.tsx` — 대본 섹션별 분리 렌더링 (ref 연결)
- `StoryboardPanel.tsx` — 우측 패널 (패널 목록 + 뷰어)
- `StoryboardPanelCard.tsx` — 웹툰 패널 카드 (이미지 + 캡션)
- `StoryboardPanelSkeleton.tsx` — 로딩 스켈레톤

**신규 훅:**
- `useStoryboardStream.ts` — SSE 스트리밍 + 이미지 상태 관리

**수정 컴포넌트:**
- `ScriptGenerator.tsx` → `ScriptStoryboardSplitView`에서 로직 분리
- `ScriptPreview.tsx` → `ScriptSectionList`로 교체 (ref 추가)
- `page.tsx` → 스토리보드 상태 관리 추가

### 4.2 SSE 처리 전략

```
Phase 1: useScript.generate() — 대본 SSE (기존, 변경 없음)
    ↓ (done 이벤트 시 자동 트리거)
Phase 2: useStoryboardStream.streamPanels() — 패널 목록 SSE
    ↓ (all_done 이벤트 시 또는 사용자 수동)
Phase 3: useStoryboardStream.generateAllImages() — 이미지 생성
```

- `useScript`와 `useStoryboardStream`은 완전 분리 (독립 abort/reset)
- 각 Phase는 독립적인 `AbortController`로 취소 가능

### 4.3 스크롤 동기화

- IntersectionObserver로 좌측 대본 섹션 가시성 감지
- 해당 섹션의 첫 번째 패널로 우측 자동 스크롤 (`scrollIntoView`)
- 역방향: 패널 클릭 시 `panel.section` 기준 대본 섹션으로 스크롤

### 4.4 상태별 렌더링

| 상태 | 좌측 (대본) | 우측 (스토리보드) |
|------|-----------|----------------|
| 초기 | 주제 입력 + 설정 | "대본 생성 후 스토리보드가 표시됩니다" |
| 대본 생성 중 | SSE 실시간 텍스트 | 빈 상태 또는 스켈레톤 |
| 대본 완료 → 패널 분석 중 | 완성된 대본 | 스켈레톤 카드 (패널 수만큼) |
| 이미지 생성 중 | 완성된 대본 | 생성된 이미지 + 나머지 스피너 |
| 완료 | 완성된 대본 | 전체 웹툰 패널 표시 |
| 에러 | 에러 메시지 | 개별 패널 "다시 시도" 버튼 |

---

## 5. 인프라 및 비용

### 5.1 비용 분석

| 항목 | 비용 |
|------|------|
| 이미지 1장 (Gemini 3 Pro) | ~$0.12 |
| 대본 1개 (4패널) | ~$0.48 |
| 대본 1개 (6패널) | ~$0.72 |

| 사용자 규모 | 일 대본 | 월 비용 |
|------------|--------|--------|
| 소규모 (10명) | 20 | $288~$432 |
| 중규모 (50명) | 100 | $1,440~$2,160 |

### 5.2 비용 절감 전략
- **해시 기반 캐싱**: script_id + panel_index 해시 → 동일 대본 재생성 방지 → **50~70% 절감**
- **WebP 변환**: PNG → WebP → 저장소 30~50% 절감
- **TTL 7일**: 오래된 캐시 자동 정리

### 5.3 동시성 제어
- `asyncio.Semaphore(3)`: 최대 3개 동시 이미지 생성
- Rate Limit 재시도: exponential backoff (2^attempt초)
- MVP 단계: asyncio 방식으로 ~10명 동시 대응 가능
- 성장기: taskiq 워커 큐 도입 (이미 의존성 존재)

### 5.4 이미지 저장
- 경로: `data/media/webtoon/images/{script_id}/panel_NNN.webp`
- StaticFiles 마운트: `/media/webtoon/images` → 직접 서빙
- 추가 의존성: **없음** (google-genai, Pillow, taskiq 모두 기존 존재)

### 5.5 환경변수 추가

```
STORYBOARD_IMAGE_MODEL=gemini-3-pro-image-preview
STORYBOARD_MAX_PANELS=14
STORYBOARD_IMAGE_FORMAT=webp
STORYBOARD_CACHE_TTL=604800
STORYBOARD_MAX_CONCURRENT=3
```

---

## 6. 보안 요구사항 (QA 엔지니어 Critical 발견)

### 6.1 프롬프트 인젝션 방지 (Critical)
- 사용자 입력(topic, news_articles)이 이미지 생성 프롬프트에 삽입됨
- **대응**: 입력 sanitize 함수 도입, 특수문자/지시어 필터링
- Chain 1 LLM 출력도 검증 (scene_description 길이 제한 등)

### 6.2 Path Traversal 방지 (Critical)
- script_id, panel_number가 파일 경로에 사용됨
- **대응**: UUID 기반 script_id 강제, panel_number 정수 범위 검증
- `Path.resolve()` 후 기준 디렉토리 하위인지 확인

### 6.3 배치 DoS 방지 (Critical)
- panel_count 최대값 제한: `max_panels = 14`
- 동시 요청 제한: 사용자당 1개 Job만 허용

### 6.4 Resource Exhaustion / API Cost Attack 방지 (Critical — Red Team 발견)
- 공격자가 `/webtoon` POST를 루프로 난사 시 API 비용 폭증
- **대응**: Rate Limiting (사용자당 분당 3회), IP 기반 스로틀링
- 이상 징후 시 Circuit Breaker 자동 차단

### 6.5 IDOR 방지 (High — Red Team 발견)
- `job_id`가 유추 가능한 패턴이면 타 사용자 스토리보드 탈취 가능
- **대응**: UUID v4 강제, 사용자 소유권 검증 (job_id ↔ user_id 매핑)

### 6.6 SSE Connection Leak 방지 (Medium — Red Team 발견)
- HTTP/1.1 브라우저의 도메인당 동시 연결 제한(6개)으로 서비스 먹통 가능
- **대응**: HTTP/2 사용 권장, SSE 연결 타임아웃 설정

### 6.7 이미지 저장 보안
- 이미지 파일 MIME 타입 검증 (image/png, image/webp만 허용)
- 저장 경로 whitelist 검증

### 6.8 Audit Logging (Red Team 제안 — 채택)
- 누가/언제/어떤 키워드로 이미지를 생성했는지 전수 로그
- 저작권 분쟁/부적절 콘텐츠 생성 시 소명 자료 활용

---

## 7. 품질 기준 (KPI)

### 7.1 기술 KPI

| 지표 | 목표 | 측정 방법 |
|------|------|----------|
| 이미지 생성 성공률 | ≥85% (최솟값 70%) | 성공 패널 / 전체 패널 |
| 단건 이미지 생성 시간 | ≤15초 | API 호출~응답 |
| 전체 스토리보드 생성 (4패널) | ≤90초 | Job 시작~완료 |
| SSE 연결 성공률 | ≥99% | 연결 성공 / 시도 |
| 캐릭터 일관성 | 동일 인물 인식 가능 | 수동 검증 |
| 대본-스토리보드 매칭도 | 내용 일치 | 수동 검증 |

### 7.2 비즈니스 KPI (External Consultant 제안 — 채택)

| 지표 | 목표 | 설명 |
|------|------|------|
| 스토리보드 생성 활용률 | ≥60% | 대본 생성 시 스토리보드도 함께 생성하는 비율 |
| 제작 시간 단축률 | ≥30% | 스토리보드 없이 영상 구성 vs 스토리보드 활용 시 |
| 패널 재생성률 | ≤30% | 낮을수록 첫 생성 품질이 높음 |
| 기능 재방문률 | ≥50% | 1회 사용 후 재사용 비율 |

---

## 8. 파일 구조 (신규/수정)

### 8.1 Backend 신규 파일

```
backend/app/tools/webtoon/
├── __init__.py
├── panel_planner.py          # Chain 1: 장면 분할 (ScriptToSceneSplitter)
├── prompt_builder.py         # Chain 2: 이미지 프롬프트 생성
└── image_generator.py        # Chain 3: 나노바나나 이미지 생성

backend/app/services/service_function/
└── webtoon_service.py        # 웹툰 파이프라인 서비스 함수
```

### 8.2 Backend 수정 파일

```
backend/app/modules/content_marketing/
├── router/__init__.py        # 3개 엔드포인트 추가
└── schema/__init__.py        # WebtoonPanel, WebtoonGenerateRequest 등 추가

backend/app/core/config.py    # STORYBOARD_* 환경변수 추가
backend/app/main.py           # StaticFiles 마운트 추가 (/media/webtoon)
```

### 8.3 Frontend 신규 파일

```
frontend/src/features/content-marketing/
├── components/
│   ├── ScriptStoryboardSplitView.tsx
│   ├── ScriptSectionList.tsx
│   ├── StoryboardPanel.tsx
│   ├── StoryboardPanelCard.tsx
│   └── StoryboardPanelSkeleton.tsx
└── hooks/
    └── useStoryboardStream.ts
```

### 8.4 Frontend 수정 파일

```
frontend/src/features/content-marketing/
├── components/ScriptGenerator.tsx  # SplitView로 로직 분리
├── types/index.ts                  # StoryboardPanel 등 타입 추가
└── services/index.ts               # streamStoryboardPanels 서비스 추가

frontend/src/app/content-marketing/page.tsx  # 스토리보드 상태 추가
frontend/src/lib/api.ts                       # 웹툰 엔드포인트 (불필요, 기존 base 활용)
frontend/next.config.js                       # 프록시 (SSE용 API Route 방식)
```

---

## 9. 구현 우선순위

### Phase 1: 핵심 파이프라인 (MVP)
1. Backend: WebtoonPanel 스키마 + 환경변수 추가
2. Backend: Chain 1 장면 분할 (panel_planner.py)
3. Backend: Chain 2 프롬프트 빌더 (prompt_builder.py)
4. Backend: Chain 3 이미지 생성 (image_generator.py)
5. Backend: 웹툰 서비스 + Job Manager + 3개 라우터
6. Frontend: 타입 정의 + useStoryboardStream 훅
7. Frontend: StoryboardPanelCard + StoryboardPanel
8. Frontend: ScriptStoryboardSplitView (분할 레이아웃)

### Phase 2: UX 개선
9. 스크롤 동기화 (IntersectionObserver)
10. 모바일/태블릿 반응형 (탭 전환)
11. 패널 개별 재생성
12. 일괄 다운로드 (ZIP)

### Phase 3: 최적화 및 고급 기능
13. 해시 기반 캐싱 (비용 절감)
14. WebP 변환 (저장소 절감)
15. 보안 강화 (프롬프트 인젝션, Path Traversal 방지)
16. 모니터링 (구조화 로깅, API 비용 추적, Cost Dashboard)
17. 1번 패널 → 나머지 병렬 생성 (성능 30초 단축 — Red Team 제안)
18. Dead Letter Queue (실패 패널 관리 — Red Team 제안)
19. Inter-Panel Editing / In-painting (고급 기능 — Red Team 제안)
20. Voice-over Preview / TTS 결합 (고급 기능 — Red Team 제안)
21. Brand Style Transfer / 워터마킹 (고급 기능 — Red Team 제안)

---

## 10. 리스크 및 대응

| 리스크 | 영향 | 대응 |
|--------|------|------|
| Gemini 3 Pro Image API 불안정 | 이미지 생성 실패 | 재시도 2회 + PIL 플레이스홀더 |
| 캐릭터 일관성 부족 | UX 품질 저하 | image_reference 전달 + 프로필 고정 |
| 이미지 생성 비용 증가 | 운영비 부담 | 캐싱 도입 (50~70% 절감) |
| 프롬프트 인젝션 | 보안 취약 | 입력 sanitize + 길이 제한 |
| SSE 연결 끊김 | UX 저하 | 폴링 폴백 + 자동 재연결 |
| Rate Limit 초과 | 서비스 중단 | Semaphore(3) + exponential backoff |

---

## 11. 기존 코드 재사용 요약

| 기존 코드 | 재사용 방식 |
|----------|-----------|
| `useScript` 훅 | 변경 없이 그대로 사용 |
| `StageProgress`, `ExportButton` 컴포넌트 | 그대로 재사용 |
| `genai.Client` + `generate_content` 패턴 | 모델명만 변경 |
| `JobManager` 클래스 | 별도 인스턴스 `webtoon_job_manager` |
| `get_chat_model(provider="upstage")` | Chain 1 장면 분할에 사용 |
| `SectionType`, `ImageStatus` enum | 그대로 활용 |
| `IMAGES_DIR` 패턴 | 경로만 `webtoon/images/`로 변경 |
| google-genai, Pillow, taskiq 의존성 | 추가 설치 불필요 |

---

## 변경 이력

| 날짜 | 버전 | 변경 내용 |
|------|------|----------|
| 2026-02-28 | v0.1 | 초안 작성 (Agent Team 7명 통합) |
| 2026-02-28 | v0.2 | Red Team 피드백 반영 (보안 5건 추가, 성능 최적화, 고급 기능 로드맵) |
| 2026-02-28 | v1.0 | External Consultant 피드백 반영 (비즈니스 KPI, 스키마 메타데이터, 법률 안전장치) — 최종본 |
