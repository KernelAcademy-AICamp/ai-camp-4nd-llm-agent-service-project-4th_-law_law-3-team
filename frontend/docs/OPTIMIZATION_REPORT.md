# Frontend 최적화 보고서

**날짜:** 2026-01-23
**기준:** Vercel React Best Practices
**커밋:** 2ce775e

---

## 요약

Vercel React Best Practices 45개 규칙을 기준으로 프론트엔드 코드를 분석하고 최적화를 수행했습니다.

### 번들 크기 변화

| 페이지 | 이전 | 이후 | 절감률 |
|--------|------|------|--------|
| case-precedent | 7.15 kB | 2.75 kB | **-61%** |
| small-claims | 8.54 kB | 5.5 kB | **-35%** |
| lawyer-finder | 13.4 kB | 11.9 kB | **-11%** |
| First Load JS (shared) | 87.3 kB | 87.5 kB | - |

**추가 효과:**
- Framer Motion 제거로 초기 로드 시 ~28KB 절감
- ChatWidget 지연 로드로 LCP(Largest Contentful Paint) 개선
- 코드 스플리팅으로 필요한 시점에만 컴포넌트 로드

---

## 수정 내용

### 1. Dynamic Import 적용 (CRITICAL)

#### `src/app/layout.tsx`
```typescript
// Before
import ChatWidget from '@/components/ChatWidget'

// After
const ChatWidget = dynamic(() => import('@/components/ChatWidget'), {
  ssr: false,
})
```
- ChatWidget을 동적 로드하여 초기 번들에서 제외
- SSR 비활성화로 클라이언트에서만 렌더링

#### `src/app/template.tsx`
```typescript
// Before
import { motion } from 'framer-motion'
// ~28KB 라이브러리 로드

// After
import '../styles/page-transition.css'
// CSS 애니메이션으로 대체 (~1KB)
```
- Framer Motion 의존성 제거
- CSS @keyframes 애니메이션으로 동일한 효과 구현
- `prefers-reduced-motion` 미디어 쿼리 지원

#### `src/app/lawyer-finder/page.tsx`
```typescript
// Before
import { KakaoMap } from '@/features/lawyer-finder/components/KakaoMap'

// After
const KakaoMap = dynamic(
  () => import('@/features/lawyer-finder/components/KakaoMap').then((m) => m.MemoizedKakaoMap),
  {
    ssr: false,
    loading: () => <MapLoadingSkeleton />,
  }
)
```
- KakaoMap 동적 로드 + 로딩 스켈레톤
- MemoizedKakaoMap 사용으로 불필요한 리렌더링 방지

#### `src/app/case-precedent/page.tsx`
```typescript
const LawyerView = dynamic(
  () => import('@/features/case-precedent/components/LawyerView').then((m) => m.LawyerView),
  { ssr: false }
)

const UserView = dynamic(
  () => import('@/features/case-precedent/components/UserView').then((m) => m.UserView),
  { ssr: false }
)
```

#### `src/app/small-claims/page.tsx`
```typescript
const CaseInfoStep = dynamic(...)
const EvidenceStep = dynamic(...)
const DocumentStep = dynamic(...)
const RelatedCases = dynamic(...)
```
- 위저드 스텝별 코드 스플리팅
- 현재 단계에 필요한 컴포넌트만 로드

---

### 2. React Query 설정 개선 (HIGH)

#### `src/app/providers.tsx`
```typescript
// Before
const [queryClient] = useState(() => new QueryClient())

// After
function createQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: 1000 * 60 * 5,    // 5분
        gcTime: 1000 * 60 * 10,       // 10분
        retry: 1,
        refetchOnWindowFocus: false,
      },
    },
  })
}
```
- 캐시 적중률 향상
- 불필요한 재요청 방지
- 에러 시 1회만 재시도

---

### 3. Suspense Boundaries 추가 (HIGH)

#### `src/app/case-precedent/page.tsx`
```typescript
<Suspense fallback={<ViewSkeleton />}>
  {userRole === 'lawyer' ? <LawyerView /> : <UserView />}
</Suspense>
```

#### `src/app/small-claims/page.tsx`
```typescript
<Suspense fallback={<StepSkeleton />}>
  {renderStep()}
</Suspense>

<Suspense fallback={<div className="w-80 bg-white animate-pulse" />}>
  <RelatedCases ... />
</Suspense>
```
- 스트리밍 렌더링 활성화
- 스켈레톤 UI로 지각 성능(Perceived Performance) 개선

---

### 4. 리렌더 최적화 (MEDIUM)

#### `src/app/page.tsx`
```typescript
// 아이콘 컴포넌트 memo 적용
const LawyerIcon = memo(() => (...))
LawyerIcon.displayName = 'LawyerIcon'

// enabledModules useMemo 적용
const enabledModules = useMemo(
  () => getEnabledModules(role || undefined),
  [role]
)
```

#### `src/context/UIContext.tsx`
```typescript
// Before
const toggleChat = () => setIsChatOpen((prev) => !prev)

// After
const toggleChat = useCallback(() => setIsChatOpen((prev) => !prev), [])
const setChatOpen = useCallback((isOpen: boolean) => setIsChatOpen(isOpen), [])
const setChatMode = useCallback((mode: 'split' | 'floating') => setChatModeState(mode), [])
```

#### `src/features/storyboard/hooks/useTimelineState.ts`
```typescript
// Before - items가 의존성에 포함되어 매번 재생성
const generateItemImage = useCallback(async (itemId: string) => {
  const item = items.find((i) => i.id === itemId)
  ...
}, [items])

// After - functional update로 최신 상태 참조
const generateItemImage = useCallback(async (itemId: string) => {
  let targetItem: TimelineItem | undefined
  setItems((currentItems) => {
    targetItem = currentItems.find((i) => i.id === itemId)
    return currentItems
  })
  ...
}, [])  // 의존성 제거
```

#### `src/features/case-precedent/components/CaseCard.tsx`
```typescript
// Before
export function CaseCard(...) { ... }

// After
function CaseCardComponent(...) { ... }
export const CaseCard = memo(CaseCardComponent)
```

---

### 5. JavaScript 성능 최적화 (LOW)

#### `src/features/lawyer-finder/components/KakaoMap.tsx`
```typescript
// Before - 매번 DOM 요소 생성
function escapeHtml(text: string): string {
  const div = document.createElement('div')
  div.textContent = text
  return div.innerHTML
}

// After - RegExp 기반 (더 빠름)
const HTML_ESCAPE_MAP: Record<string, string> = {
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&#39;',
}

function escapeHtml(text: string): string {
  return text.replace(/[&<>"']/g, (char) => HTML_ESCAPE_MAP[char])
}
```

---

## 추가 권장 사항 (미적용)

다음 항목들은 이번 최적화에서 제외되었으나 추후 고려할 수 있습니다:

### 높은 우선순위
1. **React Query 도입 확대**
   - `lawyer-finder/page.tsx`의 수동 Promise 처리를 React Query로 마이그레이션
   - 자동 캐싱, 중복 제거, 백그라운드 리프레시 활용

2. **이미지 최적화**
   - `next/image` 컴포넌트 사용
   - 자동 WebP 변환 및 지연 로드

### 중간 우선순위
3. **긴 목록 가상화**
   - `react-window` 또는 `@tanstack/react-virtual` 도입
   - 변호사 목록, 판례 목록에 적용

4. **content-visibility CSS 적용**
   ```css
   .list-item {
     content-visibility: auto;
     contain-intrinsic-size: auto 100px;
   }
   ```

### 낮은 우선순위
5. **Portal을 사용한 모달 렌더링**
   - z-index 스태킹 컨텍스트 이슈 방지

---

## 참고 자료

- [Vercel React Best Practices](https://vercel.com/blog/how-we-optimized-package-imports-in-next-js)
- [Next.js Dynamic Imports](https://nextjs.org/docs/pages/building-your-application/optimizing/lazy-loading)
- [React.memo](https://react.dev/reference/react/memo)
- [useCallback](https://react.dev/reference/react/useCallback)

---

## 변경된 파일 목록

```
src/app/case-precedent/page.tsx
src/app/lawyer-finder/page.tsx
src/app/layout.tsx
src/app/page.tsx
src/app/providers.tsx
src/app/small-claims/page.tsx
src/app/template.tsx
src/components/ChatWidget.tsx
src/context/UIContext.tsx
src/features/case-precedent/components/CaseCard.tsx
src/features/lawyer-finder/components/KakaoMap.tsx
src/features/small-claims/hooks/useWizardState.ts
src/features/storyboard/hooks/useTimelineState.ts
src/styles/page-transition.css (신규)
```
