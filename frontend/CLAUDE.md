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
export const modules: Module[] = [
  { id: 'lawyer-finder', name: '...', enabled: true, ... },
]
export const getEnabledModules = () => modules.filter((m) => m.enabled)
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
- `RegionGeoMap` - 대한민국 시군구 지도 (TopoJSON)
- `RegionDetailList` - 지역 목록 및 예측 상세 뷰
- `CrossAnalysisHeatmap` - 지역×전문분야 히트맵
- `SpecialtyBarChart` - 전문분야별 바 차트
- `StickyTabNav` - 스크롤 연동 탭

**ViewMode:**
- `count` - 변호사 수
- `density` - 인구 대비 밀도 (현재)
- `prediction` - 향후 예측 (2030/2035/2040)

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
