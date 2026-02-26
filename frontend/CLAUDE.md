# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
npm install          # 의존성 설치
npm run dev          # 개발 서버 (localhost:3000)
npm run build        # 프로덕션 빌드
npm run start        # 프로덕션 서버
npm run lint         # ESLint 실행
```

## Architecture

### Next.js App Router

`src/app/` 폴더 구조가 URL 라우팅과 직접 매핑됩니다.
- `src/app/page.tsx` → `/`
- `src/app/lawyer-finder/page.tsx` → `/lawyer-finder`

### 모듈 시스템

**모듈 정의**: `src/lib/modules.ts`
```typescript
// 현재 등록 모듈: lawyer-finder, lawyer-stats, case-precedent,
// law-search, storyboard, law-study, statute-hierarchy,
// small-claims, mock-trial
export const modules: Module[] = [
  { id: 'lawyer-finder', name: '...', enabled: true, ... },
]
export const getEnabledModules = (role?) => modules.filter((m) => m.enabled && ...)
```

**API endpoints**: `src/lib/api.ts`
```typescript
export const endpoints = {
  lawyerFinder: '/lawyer-finder',
  // 모듈 추가 시 여기에 endpoint 추가
}
```

### Feature 구조

```
src/features/<module-name>/
├── components/     # 모듈 전용 컴포넌트
├── hooks/          # 커스텀 훅
├── services/
│   └── index.ts    # API 호출 함수
└── types/          # TypeScript 타입
```

### ChatWidget (통합 채팅 위젯)

`src/components/ChatWidget.tsx` — SSE 스트리밍 채팅, 에이전트 응답 후 자동 네비게이션.

**네비게이션 우선순위:** NAVIGATE 액션 (좌표/파라미터 포함) > AGENT_PAGE_MAP (기본 페이지 이동)
- NAVIGATE 액션: 에이전트가 `nav_params` (lat, lng, radius, zoom, category, sigungu)를 포함하여 URL 생성
- AGENT_PAGE_MAP: NAVIGATE 액션이 없을 때 에이전트 유형별 기본 페이지로 이동

### lawyer-finder (변호사 찾기)

**경로:** `src/features/lawyer-finder/`

**페이지 (`src/app/lawyer-finder/page.tsx`):**
- URL searchParams에서 `lat`, `lng`, `radius`, `zoom`, `category`, `sigungu` 파라미터 파싱
- `useState` 초기화 함수로 URL 파라미터를 동기적으로 반영 (첫 렌더 시 올바른 위치 표시)

**KakaoMap (`components/KakaoMap.tsx`):**
- `initialLevel` prop (optional, 기본값 5) — 초기 줌 레벨
- `prevInitialLevelRef`로 줌 레벨 변경 감지, 드래그 시 리셋 방지

### API 프록시

`next.config.js`의 rewrites 설정으로 `/api/*` 요청이 백엔드(localhost:8000)로 프록시됩니다.

## Features

### lawyer-stats (변호사 통계 대시보드)

**경로:** `src/features/lawyer-stats/`

**컴포넌트:**
- `RegionGeoMap` - 대한민국 시군구 지도 (TopoJSON), 수요 모드 시 법원 마커 표시
- `RegionDetailList` - 지역 목록, 예측 상세 뷰, 법원 상세 뷰 (사건 수/변호사 수/부담지수/관할 지역)
- `CrossAnalysisHeatmap` - 지역×전문분야 히트맵
- `SpecialtyBarChart` - 전문분야별 바 차트
- `StickyTabNav` - 스크롤 연동 탭

**IndicatorGroup:**
- `supply` (공급): 변호사 수, 인구 대비 밀도, 향후 예측
- `demand` (수요): 법원별 사건 접수 수

**ViewMode:**
- `count` - 변호사 수
- `density` - 인구 대비 밀도 (현재)
- `prediction` - 향후 예측 (2030/2035/2040)
- `case_count` - 사건 접수 수 (수요 모드)

**타입:**
- `CourtDemandMarker` - 법원 단위 수요 데이터 (좌표, 사건 수, 변호사 수, 부담지수, 관할 지역)

### mock-trial (모의 법정)

**경로:** `src/features/mock-trial/`

**페이지 흐름:** `setup` → `briefing` (데모 시나리오) → `trial` → `verdict`

**컴포넌트:**
- `MockTrialGame` - Phaser.js 기반 픽셀아트 법정 게임
- `MockTrialSetup` - 사건 입력 및 게임 설정
- `ScenarioBriefing` - 데모 시나리오 브리핑 (사건 개요, 등장인물, 목표, 증거 미리보기)
- `ChatPanel` - AI 에이전트 채팅 (검사/변호사/판사), 감정 이모지 표시
- `ChatBottomBar` - 하단 입력바, 최근 AI 발언 + 감정 이모지
- `DialogueControls` - 대화 속도(1x/2x/4x/즉시) + 스킵 버튼 + Space 안내
- `ReferencePanel` - 법률 참조 패널
- `EvidencePanel` - 증거 표시 (물적 증거 + RAG 판례/법령 + 사용자 힌트)
- `JudgmentDisplay` - 판결 결과
- `StageProgress` - 재판 단계 진행 표시
- `DisclaimerBanner` - 면책 안내

**게임 구조:** `src/features/mock-trial/game/`

씬 체인: `PreloadScene` → `LobbyScene` → `CourtScene`

- `PreloadScene` - 에셋 로딩 (스프라이트시트, 배경, 오디오, 타일맵) + 애니메이션 등록
- `CourtScene` - 법정 씬 (타일맵 → 배경이미지 → Graphics 3단계 fallback), DialogueController 통합
- `LobbyScene` - 로비 씬 (대법원 배경 + 국기 펄럭임 오버레이 + 캐릭터 입장 시퀀스)
- `DialogueController` - 대화 큐 관리, 속도 제어(normal/fast/faster/instant), 스킵, Space 키 바인딩
- `EventBus` - 이벤트 시스템 (`agent:speak` + `dialogue:enqueue/set_speed/advance/skip/queue:empty`)
- `AssetConfig` - 에셋 키/경로/프레임 크기 중앙 관리, `hasTexture()` fallback 헬퍼
- `AudioManager` - BGM/SFX 관리 (에셋 없으면 무음 fallback)
- `config` - 게임 레이아웃 상수 (캐릭터 위치, 국기 위치/스케일, 입장 시퀀스, 말풍선 높이 제한)
- `sprites/LpcSpriteConfig` - LPC 스프라이트시트 설정 (832x1344, 13열x21행, 64px 프레임)
- `sprites/CharacterBase` - LPC 스프라이트 캐릭터 (walk/speak/react/idle + 감정 아이콘)
- `sprites/EmotionIconRenderer` - 8x8 도트 스프라이트 감정 아이콘 (PIXEL_SIZE=3, 24x24px)
- `ui/SpeechBubble` - 말풍선 (NineSlice/Graphics fallback + 타이핑 + 페이지 분할 + ▼ 인디케이터)
- `ui/StageIndicator` - 재판 단계 진행 표시

**에셋 파이프라인:** `public/assets/mock-trial/`
- `sprites/` - LPC 캐릭터 6종 (832x1344 PNG) + 국기 스프라이트시트 2종 (2816x1536, 4프레임)
- `backgrounds/` - 로비 배경 (대법원 픽셀아트), 법정 내부 배경
- `tilesets/` - Tiled 타일맵 (court-map.json + court-tiles.png)
- `audio/` - BGM (lobby, court) + SFX (gavel, typing, objection, stage-change)
- `ui/` - 말풍선 NineSlice + 꼬리 이미지
- `effects/` - 감정 아이콘 스프라이트시트

**데모 시나리오:** `src/features/mock-trial/demo/`
- `DemoScenario` - 시나리오 정의 (setup, characters, objectives, evidence, stages)
- `DemoCharacter` - 등장인물 (role, name, description)
- `DemoStage` - 단계별 사용자 입력 + mock AI 응답

**물적 증거 시스템:**
- `PhysicalEvidence` - 시나리오 내장 증거물 (document, video, financial, photo, testimony, other)
- `PHYSICAL_EVIDENCE_TYPE_LABEL` - 증거 유형별 아이콘/라벨 상수
- 증거조사 단계에서 왼쪽 패널에 RAG 검색 결과와 함께 표시, 선택/제출 가능

**감정 표현 시스템 (FR-51):**
- `EmotionType` - 8가지 감정: neutral, angry, thinking, sad, confident, stern, recording, judging
- `DEFAULT_ROLE_EMOTION` - 역할별 기본 감정 (judge→stern, prosecutor→confident, attorney→thinking 등)
- `CharacterBase.setEmotion/clearEmotion` - 감정 아이콘 스프라이트시트 또는 Graphics 도트 렌더링

**의존성:** `phaser` (package.json)

## Conventions

- 컴포넌트: 함수형 컴포넌트 + TypeScript
- 스타일링: Tailwind CSS
- 상태관리: Zustand (전역), React Query (서버 상태)
- 클라이언트 컴포넌트는 파일 최상단에 `'use client'` 명시

### Path Aliases

```typescript
import { api } from '@/lib/api'           // src/lib/
import { Button } from '@/components/ui'   // src/components/
import { useAuth } from '@/features/auth'  // src/features/
```
