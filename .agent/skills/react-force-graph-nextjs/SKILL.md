---
name: react-force-graph-nextjs
description: Next.js App Router 환경에서 react-force-graph 라이브러리를 안전하게 사용하는 방법 (SSR 비활성화 및 Ref 처리)
---

# React Force Graph in Next.js Guidelines

`react-force-graph` 라이브러리는 브라우저 전용 API(Window, Canvas 등)를 사용하므로 Next.js의 Server Side Rendering(SSR) 환경에서 직접 로드하면 에러가 발생합니다. 또한 `next/dynamic`과 `Ref`를 함께 사용할 때 주의가 필요합니다.

## 1. 핵심 원칙

1.  **SSR 비활성화 필수**: `next/dynamic`을 사용하여 `ssr: false` 옵션으로 로드해야 합니다.
2.  **Ref 전달 주의**: `dynamic`으로 로드된 컴포넌트에 바로 `ref`를 전달할 때, 라이브러리의 export 방식(default vs named)이나 `forwardRef` 지원 여부에 따라 `Function components cannot be given refs` 에러가 발생할 수 있습니다.
3.  **중간 래퍼(Wrapper) 패턴 권장**: 라이브러리를 직접 `dynamic` import 하는 대신, **일반적인 방식으로 라이브러리를 import 하는 중간 컴포넌트**를 만들고, 그 중간 컴포넌트를 `dynamic` import 하는 것이 가장 안정적입니다.

## 2. 구현 패턴 (Recommended)

### Step 1: 래퍼 컴포넌트 생성 (`ForceGraphWrapper.tsx`)
이 파일은 클라이언트 사이드에서만 렌더링될 것이므로 일반적인 import를 사용합니다.

```tsx
// ForceGraphWrapper.tsx
import React, { forwardRef } from 'react';
import ForceGraph2D, { ForceGraphMethods, ForceGraphProps } from 'react-force-graph-2d';

// Props 타입 정의 (필요에 따라 확장)
interface WrapperProps extends ForceGraphProps {
  // 추가 커스텀 props
}

// forwardRef를 사용하여 부모로부터 ref를 받아 라이브러리 컴포넌트에 전달
const ForceGraphWrapper = forwardRef<ForceGraphMethods, WrapperProps>((props, ref) => {
  return <ForceGraph2D ref={ref} {...props} />;
});

ForceGraphWrapper.displayName = 'ForceGraphWrapper';

export default ForceGraphWrapper;
```

### Step 2: 부모 컴포넌트에서 동적 임포트 (`ParentComponent.tsx`)

```tsx
// ParentComponent.tsx
import dynamic from 'next/dynamic';
import { useRef } from 'react';
import type { ForceGraphMethods } from 'react-force-graph-2d';

// 래퍼 컴포넌트를 SSR 없이 동적 로드
const ForceGraph = dynamic(() => import('./ForceGraphWrapper'), {
  ssr: false,
  loading: () => <div>Loading Graph...</div>
});

export function ParentComponent() {
  const fgRef = useRef<ForceGraphMethods>();

  return (
    <div className="w-full h-full">
      <ForceGraph
        ref={fgRef}
        graphData={...}
        // ... other props
      />
    </div>
  );
}
```

## 3. 트러블슈팅

### "Function components cannot be given refs"
- **원인**: `next/dynamic`이 반환하는 로더블 컴포넌트에 `ref`를 전달했는데, 내부적으로 타겟 컴포넌트가 `forwardRef`로 감싸져 있지 않거나 `default` export를 제대로 찾지 못할 때 발생.
- **해결**: 위에서 제안한 **래퍼 패턴**을 사용하면 확실하게 해결됩니다. 래퍼 컴포넌트에서 `forwardRef`를 명시적으로 사용하기 때문입니다.

### 다이내믹 임포트 경로 문제 ("Module not found")
- `import` 경로가 정확한지 확인하세요. default export를 명시적으로 지정해야 할 수도 있습니다 (`.then(mod => mod.default)`). 하지만 래퍼 패턴을 쓰면 이 문제는 래퍼 파일 내부의 정적 import에서 처리되므로 `dynamic` 호출부는 단순해집니다.
