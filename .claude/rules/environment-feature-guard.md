# Environment Feature Guard Rules

Claude는 개발 전용 기능과 프로덕션 기능을 구분할 때 이 규칙들을 **항상(ALWAYS)** 따라야 합니다.

## 1. 분류 기준

| 분류 | 설명 | 예시 |
|------|------|------|
| **개발 전용 (Dev-Only)** | 개발/테스트 편의를 위한 기능. 프로덕션에서 절대 노출 안 됨 | 더미 데이터 삽입 버튼, 디버그 패널, 테스트 데이터 채우기 |
| **데모/체험 (Demo)** | 사용자가 체험할 수 있는 기능. 프로덕션에서도 노출 | 모의 법정 데모 시나리오, 자동 입력 버튼 |

## 2. 구현 패턴

### 개발 전용 → `NODE_ENV === 'development'`

```tsx
{process.env.NODE_ENV === 'development' && (
  <button onClick={fillTestData}>
    [DEV] 테스트 데이터 채우기
  </button>
)}
```

- Next.js 프로덕션 빌드 시 **코드 자체가 tree-shaking으로 제거**됨
- 별도 환경변수 설정 불필요
- `[DEV]` 접두사로 시각적 구분 권장

### 데모/체험 → 가드 없이 유지

```tsx
// 데모 시나리오는 사용자 기능이므로 조건 없이 렌더링
<DemoScenarioSelector scenarios={DEMO_SCENARIOS} onSelect={handleDemoStart} />
```

- 프로덕션에서도 사용자에게 노출
- 데모 데이터 파일(`demo/demo-scenarios.ts` 등)은 번들에 포함

## 3. 현재 적용 현황

| 모듈 | 기능 | 분류 | 가드 방식 |
|------|------|------|----------|
| **small-claims** | `[DEV] 테스트 데이터 채우기` 버튼 | 개발 전용 | `NODE_ENV === 'development'` |
| **mock-trial** | 데모 시나리오 2개 (형사 사기, 민사 손해배상) | 데모/체험 | 가드 없음 (의도적) |
| **mock-trial** | 자동 입력 버튼 | 데모/체험 | 가드 없음 (의도적) |

## 4. 새 페이지/모듈 필수 세팅 (체크리스트)

새 페이지나 모듈을 만들 때 **반드시** 아래 두 가지를 기본 적용한다:

### 4-1. `BackButton` (뒤로가기)

헤더에 `@/components/ui/BackButton` 컴포넌트를 배치한다. 홈으로 이동하며 세션을 초기화한다.

```tsx
import { BackButton } from '@/components/ui/BackButton'

<header>
  <div className="flex items-center gap-3">
    <BackButton />
    <div>
      <h1>페이지 제목</h1>
    </div>
  </div>
</header>
```

### 4-2. 챗봇 플로팅 모드

`src/components/ChatWidget.tsx`의 `FLOATING_MODE_PATHS`에 새 페이지 경로를 추가한다.
하위 경로까지 포함하려면 `supportsFloatingMode` 조건에 `pathname.startsWith()` 추가.

```tsx
// ChatWidget.tsx
const FLOATING_MODE_PATHS = new Set([
  '/lawyer-finder',
  '/small-claims',
  // ... 새 경로 추가
])
// 하위 경로 포함: pathname.startsWith('/workspace')
```

## 5. 새 기능 추가 시 판단 기준

새로운 더미 데이터/테스트 기능을 추가할 때:

1. **"프로덕션 사용자가 이 기능을 사용하는가?"** → Yes: 데모/체험, No: 개발 전용
2. 개발 전용이면 반드시 `NODE_ENV === 'development'` 가드 적용
3. 데모/체험이면 가드 없이 유지하되, `demo/` 디렉토리에 데이터 파일 분리

## 6. 금지 사항

- 개발 전용 기능을 가드 없이 커밋하지 않음
- `NEXT_PUBLIC_` 환경변수로 개발 전용 기능을 제어하지 않음 (빌드 설정 실수 위험)
- 프로덕션 빌드에 `console.log`, `debugger`, 디버그 패널이 포함되지 않도록 확인
- 데모/체험 기능에 `NODE_ENV` 가드를 걸어 프로덕션에서 제거하지 않음
