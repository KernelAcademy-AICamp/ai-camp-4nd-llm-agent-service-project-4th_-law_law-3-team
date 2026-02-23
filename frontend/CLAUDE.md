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

**컴포넌트:**
- `MockTrialGame` - Phaser.js 기반 픽셀아트 법정 게임
- `MockTrialSetup` - 사건 입력 및 게임 설정
- `ChatPanel` - AI 에이전트 채팅 (검사/변호사/판사), 감정 이모지 표시
- `ChatBottomBar` - 하단 입력바, 최근 AI 발언 + 감정 이모지
- `ReferencePanel` - 법률 참조 패널
- `EvidencePanel` - 증거 표시
- `JudgmentDisplay` - 판결 결과
- `StageProgress` - 재판 단계 진행 표시
- `DisclaimerBanner` - 면책 안내

**게임 구조:** `src/features/mock-trial/game/`
- `CourtScene` - 법정 씬 (Phaser), `agent:speak` 이벤트에서 감정 이모지 전달
- `LobbyScene` - 로비 씬 (대법원 배경)
- `EventBus` - 이벤트 시스템 (`agent:speak`에 `emotion` 필드 포함)
- `sprites/` - 픽셀아트 캐릭터, 배심원 패널, 도트 감정 아이콘 렌더러
- `sprites/EmotionIconRenderer` - 8x8 도트 스프라이트 감정 아이콘 (PIXEL_SIZE=3, 24x24px)
- `ui/` - 말풍선 (`SpeechBubble`에 게임풍 감정 심볼 표시), 단계 표시

**감정 표현 시스템 (FR-51):**
- `EmotionType` - 8가지 감정: neutral💬, angry💢, thinking❓, sad💧, confident✨, stern❗, recording✏️, judging🔨
- `EMOTION_EMOJI` - 감정→게임풍 심볼 매핑 상수
- `DEFAULT_ROLE_EMOTION` - 역할별 기본 감정 (judge→stern, prosecutor→confident, attorney→thinking 등)
- `CourtEvent.emotion` - 선택적 감정 필드 (Backend LLM이 `[EMOTION:태그]`로 생성)
- `EmotionIconRenderer` - 8x8 도트 스프라이트로 캐릭터 머리 위 감정 아이콘 렌더링 (pop-in + floating 애니메이션)
- `CharacterBase.setEmotion/clearEmotion` - 발언 시 도트 아이콘 표시/숨기기

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
