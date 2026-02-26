# Phaser.js + Next.js 통합 패턴

Phaser.js v3.90.0과 Next.js App Router를 결합하여 Canvas 기반 인터랙티브 UI를 구현하는 패턴.
Phaser 코드 작성/수정의 전제 조건으로, 이 스킬의 규칙을 먼저 확인한다.

> **적용 시점**: Phaser 게임 코드 수정, Canvas 기반 UI 추가, 픽셀아트 캐릭터 작업 시
> **전제 스킬**: `react-nextjs-frontend` (일반 React 패턴)
> **후속 스킬**: `court-eventbus-patterns` (Phaser ↔ React 통신), `court-dialog-system` (게임 로직)

---

## 1. Next.js SSR 회피 패턴

Phaser는 `window`, `document`, `HTMLCanvasElement`에 의존하므로 **서버 사이드에서 절대 import하면 안 된다**.

### 1.1 Dynamic Import (필수)

```typescript
// MockTrialGame.tsx
'use client'

import { useEffect, useRef, useState } from 'react'
import type { Game as PhaserGame } from 'phaser'

export function MockTrialGame() {
  const gameRef = useRef<PhaserGame | null>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    let isMounted = true

    const initGame = async (): Promise<void> => {
      // 반드시 useEffect 내부에서 dynamic import
      const Phaser = (await import('phaser')).default
      const { CourtScene } = await import('../game/CourtScene')
      const { LobbyScene } = await import('../game/LobbyScene')

      if (!isMounted || !containerRef.current || gameRef.current) return
      // ... Game 생성
    }

    initGame()
    return () => { isMounted = false; /* cleanup */ }
  }, [])
}
```

### 1.2 필수 가드 3개

| 가드 | 목적 | 코드 |
|------|------|------|
| `'use client'` | 서버 컴포넌트 제외 | 파일 최상단 |
| `useEffect` 내 초기화 | SSR 시 실행 방지 | `useEffect(() => { ... }, [])` |
| `isMounted` 플래그 | StrictMode 이중 실행 방지 | `let isMounted = true` → cleanup에서 `false` |

### 1.3 금지 사항

```typescript
// BAD: 트리 레벨 import → SSR 크래시
import Phaser from 'phaser'

// BAD: 조건부 import도 번들러가 서버에 포함시킴
if (typeof window !== 'undefined') {
  const Phaser = require('phaser')
}

// GOOD: useEffect 내 dynamic import만 허용
const Phaser = (await import('phaser')).default
```

---

## 2. Game 인스턴스 관리

### 2.1 useRef 패턴

```typescript
const gameRef = useRef<PhaserGame | null>(null)
```

- `useState`가 아닌 `useRef` 사용 (리렌더 방지)
- 초기값 `null`, 생성 후 할당
- cleanup에서 `destroy(true)` 호출 + `null` 재할당

### 2.2 이중 생성 방지

```typescript
// 이미 존재하면 생성하지 않음
if (!isMounted || !containerRef.current || gameRef.current) return
```

React 18 StrictMode에서 `useEffect`가 2번 실행되므로 반드시 체크.

### 2.3 Cleanup 패턴

```typescript
return () => {
  isMounted = false
  gameRef.current?.destroy(true)  // true = DOM 요소까지 제거
  gameRef.current = null
}
```

`destroy(true)` 파라미터:
- `true`: Canvas DOM 요소 + WebGL 컨텍스트 + 모든 텍스처 해제
- `false`: Game 객체만 정리, DOM은 유지 (거의 사용하지 않음)

---

## 3. Game Config 표준

### 3.1 현재 프로젝트 표준 설정

```typescript
// 참조: MockTrialGame.tsx
gameRef.current = new Phaser.Game({
  type: Phaser.AUTO,           // WebGL 우선, Canvas fallback
  parent: containerRef.current, // React ref로 연결
  width: 800,                  // GAME_WIDTH (config.ts)
  height: 480,                 // GAME_HEIGHT (config.ts)
  pixelArt: true,              // 안티앨리어싱 비활성화
  roundPixels: true,           // 서브픽셀 렌더링 방지
  scene: [LobbyScene, CourtScene],
  scale: {
    mode: Phaser.Scale.FIT,          // 컨테이너에 맞춤
    autoCenter: Phaser.Scale.CENTER_BOTH, // 중앙 정렬
  },
  backgroundColor: '#f5f0e8',  // COURT_BACKGROUND_COLOR
})
```

### 3.2 설정값 의미

| 설정 | 값 | 이유 |
|------|---|------|
| `type` | `Phaser.AUTO` | WebGL 먼저 시도, 실패 시 Canvas 2D |
| `pixelArt` | `true` | 4px 그리드 픽셀아트에 필수 (블러 방지) |
| `roundPixels` | `true` | 서브픽셀 위치 반올림 → 선명한 픽셀 |
| `scale.mode` | `Scale.FIT` | 비율 유지하며 컨테이너 크기에 맞춤 |
| `scale.autoCenter` | `CENTER_BOTH` | 수평+수직 모두 중앙 정렬 |

### 3.3 설정 변경 시 주의

- `width`/`height` 변경 시 → `config.ts`의 `GAME_WIDTH`/`GAME_HEIGHT`도 동기화
- `pixelArt: false`로 변경 금지 (모든 캐릭터 렌더링 깨짐)
- Scene 배열 순서 = 첫 번째 씬이 자동 시작

---

## 4. Scene 생명주기

### 4.1 생명주기 메서드

```
init(data) → preload() → create() → update(time, delta) → shutdown()
```

| 메서드 | 용도 | 현재 사용 |
|--------|------|----------|
| `init(data)` | Scene 전환 시 데이터 수신 | `CourtScene.init()` → caseType 설정 |
| `preload()` | 에셋 로드 (이미지, 오디오) | 미사용 (Graphics API로 직접 그리기) |
| `create()` | 게임 오브젝트 생성 | 배경, 캐릭터, 말풍선, 이벤트 구독 |
| `update(time, delta)` | 매 프레임 호출 | 미사용 (이벤트 기반 업데이트) |
| `shutdown()` | Scene 종료 시 정리 | 이벤트 구독 해제, 타이머 정리 |

### 4.2 Scene 전환

```typescript
// LobbyScene → CourtScene (데이터 전달)
this.scene.start('CourtScene', {
  caseType: data.caseType,
  userRole: data.userRole,
  caseSummary: data.caseSummary,
})
```

- `scene.start(key, data)`: 현재 씬 종료 → 대상 씬의 `init(data)` 호출
- `scene.launch(key)`: 현재 씬 유지 + 병렬 씬 시작 (사용하지 않음)

### 4.3 Cleanup 패턴 (shutdown)

```typescript
// 참조: CourtScene.ts:175-180
shutdown(): void {
  this.unsubscribers.forEach((unsub) => unsub())
  this.unsubscribers = []
  this.juryPanel?.destroy()
  this.juryPanel = null
}
```

**반드시 정리할 항목**:
1. EventBus 구독 (`unsubscribers` 배열)
2. `Phaser.Time.TimerEvent` (delayedCall, addEvent)
3. Tween (`killTweensOf`)
4. 자식 GameObject (`destroy()`)

---

## 5. Graphics API 픽셀아트 패턴

이 프로젝트는 이미지 에셋 없이 **Graphics API로 픽셀아트를 직접 그린다**.

### 5.1 핵심 타입과 상수

```typescript
// 참조: config.ts, PixelCharacterRenderer.ts
export const PIXEL_SIZE = 4  // 1 논리 픽셀 = 4x4 실제 픽셀

type PixelGrid = number[][]  // 2D 색상 배열 (0 = 투명)

const GRID_WIDTH = 12   // 캐릭터 그리드 너비 (12x4 = 48px)
const GRID_HEIGHT = 16  // 캐릭터 그리드 높이 (16x4 = 64px)
```

### 5.2 그리드 규격

| 대상 | 그리드 크기 | 실제 크기 | 파일 |
|------|-----------|----------|------|
| 메인 캐릭터 | 12x16 | 48x64px | `PixelCharacterRenderer.ts` |
| 배심원 | 8x10 | 32x40px | `JurorSprite.ts` |

### 5.3 렌더링 함수

```typescript
// 참조: PixelCharacterRenderer.ts:243-270
export function drawPixelCharacter(
  graphics: Phaser.GameObjects.Graphics,
  role: string,
  state: CharacterState
): void {
  graphics.clear()  // 이전 그리기 제거

  const grid = CHARACTER_GRIDS[role][state]
  const offsetX = -(GRID_WIDTH * PIXEL_SIZE) / 2  // 중앙 정렬
  const offsetY = -(GRID_HEIGHT * PIXEL_SIZE) / 2

  for (let row = 0; row < grid.length; row++) {
    for (let col = 0; col < grid[row].length; col++) {
      const color = grid[row][col]
      if (color === 0) continue  // 투명 픽셀 건너뛰기
      graphics.fillStyle(color, 1)
      graphics.fillRect(
        offsetX + col * PIXEL_SIZE,
        offsetY + row * PIXEL_SIZE,
        PIXEL_SIZE, PIXEL_SIZE
      )
    }
  }
}
```

### 5.4 색상 팔레트

```typescript
// 공용 색상 (모든 캐릭터)
const SKIN = 0xffcc99
const SKIN_DARK = 0xdba876
const HAIR_BLACK = 0x2c2c2c
const WHITE = 0xffffff
const BLACK = 0x000000

// 역할별 복장 색상
const NAVY = 0x1a237e       // 검사 정장
const RED_TIE = 0xb71c1c    // 검사 넥타이
const CHARCOAL = 0x37474f   // 변호사 정장
const GREEN_TIE = 0x1b5e20  // 변호사 넥타이
const GRAY_CLOTH = 0x78909c // 피고인 옷
const CLERK_SUIT = 0x546e7a // 서기 정장
```

### 5.5 새 PixelGrid 추가 규칙

1. 그리드 크기는 반드시 `GRID_WIDTH x GRID_HEIGHT` (12x16 또는 8x10)
2. `0`은 투명 (해당 셀을 그리지 않음)
3. 색상은 `0xRRGGBB` 형식의 number
4. `idle`과 `speak` 두 상태 필수 정의
5. `react` 상태는 `idle`과 동일하게 해도 무방

---

## 6. Container 패턴

### 6.1 기본 구조

```typescript
// 참조: CharacterBase.ts
export class CharacterBase extends Phaser.GameObjects.Container {
  constructor(scene: Phaser.Scene, x: number, y: number, role: string) {
    super(scene, x, y)

    // 자식 오브젝트들은 Container 기준 상대 좌표
    this.characterGraphics = scene.add.graphics()  // (0, 0) 기준
    this.label = scene.add.text(0, LABEL_OFFSET_Y, name, { ... })

    this.add([this.characterGraphics, this.label])
    scene.add.existing(this)  // 씬에 등록
  }
}
```

### 6.2 Container 규칙

- 자식 좌표는 **Container 원점 기준 상대 좌표**
- Container의 `x, y`를 이동하면 모든 자식이 함께 이동
- `setScale()`, `setAngle()`, `setAlpha()` 등도 자식에 전파
- `scene.add.existing(this)` 호출 필수 (씬 렌더링 트리에 등록)

### 6.3 Container vs 개별 배치

| 방식 | 사용 시점 |
|------|----------|
| Container | 여러 오브젝트가 함께 이동/변환 (캐릭터+라벨+이펙트) |
| 개별 배치 | 독립적으로 움직이는 오브젝트 (배경 요소) |

---

## 7. Tween 관리

### 7.1 기본 Tween 패턴

```typescript
// 참조: CharacterBase.ts:49-56 (발언 시 확대 애니메이션)
this.scene.tweens.add({
  targets: this,
  scaleX: 1.05,
  scaleY: 1.05,
  yoyo: true,     // 원래값으로 복귀
  repeat: -1,     // 무한 반복
  duration: 400,  // ms
})
```

### 7.2 Breath Animation (호흡 효과)

```typescript
// 참조: CharacterBase.ts:79-91
private startBreathAnimation(): void {
  if (this.breathTween) {
    this.breathTween.destroy()
  }
  this.breathTween = this.scene.tweens.add({
    targets: this.characterGraphics,
    y: -2,                    // 2px 위로
    yoyo: true,
    repeat: -1,
    duration: 1000,           // 1초 주기
    ease: 'Sine.easeInOut',   // 부드러운 이징
  })
}
```

### 7.3 Tween Cleanup (필수)

```typescript
// 상태 전환 시 기존 Tween 정리
this.scene.tweens.killTweensOf(this)  // 해당 타겟의 모든 Tween 제거
this.setScale(1, 1)                    // 원래 값 복원

// 개별 Tween 참조 정리
if (this.breathTween) {
  this.breathTween.destroy()
  this.breathTween = null
}
```

**Tween 누수 방지 체크리스트**:
- [ ] `repeat: -1` 사용 시 반드시 `killTweensOf()` 또는 `destroy()` 호출
- [ ] 상태 전환 시 이전 Tween 제거 후 새 Tween 시작
- [ ] Scene `shutdown()`에서 모든 활성 Tween 정리
- [ ] `yoyo: true` 사용 시 중간에 destroy되면 원래값 수동 복원

---

## 8. TypeScript 타입 안전성

### 8.1 Phaser 타입 네임스페이스

```typescript
import Phaser from 'phaser'
import type { Game as PhaserGame } from 'phaser'

// Scene 정의
class MyScene extends Phaser.Scene { ... }

// GameObject 타입
private graphics: Phaser.GameObjects.Graphics
private text: Phaser.GameObjects.Text
private circle: Phaser.GameObjects.Arc
private tween: Phaser.Tweens.Tween | null = null
private timer: Phaser.Time.TimerEvent | null = null

// Config 타입
const config: Phaser.Types.Core.GameConfig = { ... }
```

### 8.2 Scene 데이터 타입

```typescript
// Scene 전환 데이터에 인터페이스 정의
interface CourtSceneData {
  caseType: string
  userRole: string
  caseSummary: string
}

// init에서 타입 적용
init(data: CourtSceneData): void {
  this.caseType = data.caseType || 'criminal'
}
```

### 8.3 CharacterState 타입

```typescript
// 참조: PixelCharacterRenderer.ts:10
export type CharacterState = 'idle' | 'speak' | 'react'
```

역할과 상태의 조합으로 PixelGrid를 선택:
```typescript
const CHARACTER_GRIDS: Record<string, Record<CharacterState, PixelGrid>> = {
  judge: { idle: JUDGE_IDLE, speak: JUDGE_SPEAK, react: JUDGE_IDLE },
  // ...
}
```

---

## 9. WebGL/Canvas Fallback

### 9.1 Phaser.AUTO 동작

1. WebGL 2.0 시도
2. 실패 시 WebGL 1.0 시도
3. 실패 시 Canvas 2D fallback
4. 모두 실패 시 에러 발생

### 9.2 WebGL 미지원 감지 (미구현 - 향후 추가)

```typescript
// 권장 구현 패턴
function checkWebGLSupport(): boolean {
  try {
    const canvas = document.createElement('canvas')
    return !!(
      canvas.getContext('webgl2') || canvas.getContext('webgl')
    )
  } catch {
    return false
  }
}

// React 컴포넌트에서 사전 체크
if (!checkWebGLSupport()) {
  return <div>WebGL을 지원하지 않는 브라우저입니다. Chrome/Firefox를 사용해주세요.</div>
}
```

### 9.3 Canvas 렌더링 실패 시 React Fallback (미구현 - 향후 추가)

```typescript
// Error Boundary로 Canvas 오류 포착
<ErrorBoundary fallback={<MockTrialTextMode />}>
  <MockTrialGame />
</ErrorBoundary>
```

---

## 10. 성능 최적화

### 10.1 번들 사이즈

- Phaser.js는 ~1MB (gzip ~300KB)
- `dynamic import`로 초기 로드에서 분리 (코드 스플리팅)
- Game 페이지 진입 시에만 로드

### 10.2 Graphics API 최적화

- `graphics.clear()` 후 다시 그리기 (상태 변경 시)
- 변경 없는 정적 배경은 한 번만 그리고 유지
- 배심원 4명은 개별 Graphics 객체 (독립 업데이트)

### 10.3 Tween 재사용

- 동일 패턴 Tween은 `destroy()` + 새로 생성 (재사용 불가)
- `killTweensOf()`로 일괄 정리 후 재시작

### 10.4 이벤트 기반 업데이트

- `update()` 루프 미사용 → CPU 절약
- EventBus 이벤트 수신 시에만 화면 갱신
- Timer/DelayedCall로 시간 기반 로직 처리

---

## 11. 트러블슈팅

### 11.1 React StrictMode 이중 실행

**증상**: Game 인스턴스가 2개 생성되어 Canvas가 겹침

**해결**:
```typescript
// isMounted 가드 + gameRef.current 체크
if (!isMounted || !containerRef.current || gameRef.current) return
```

### 11.2 HMR (Hot Module Replacement) 충돌

**증상**: 코드 수정 시 기존 Game이 파괴되지 않고 새 Game이 추가

**해결**: cleanup 함수에서 확실히 파괴
```typescript
return () => {
  isMounted = false
  gameRef.current?.destroy(true)
  gameRef.current = null
}
```

### 11.3 Canvas 메모리 누수

**증상**: Scene 전환 반복 시 메모리 증가

**원인/해결**:
- EventBus 구독 미해제 → `shutdown()`에서 `unsubscribers` 전부 호출
- Tween 미정리 → `killTweensOf()` 호출
- TimerEvent 미파괴 → `timer.destroy()` 호출

### 11.4 "Cannot read properties of null (reading 'tweens')"

**증상**: Scene 종료 후 비동기 콜백에서 `this.scene` 접근

**해결**: 비동기 콜백 내에서 씬 존재 여부 확인
```typescript
this.scene.time.delayedCall(delay, () => {
  if (!this.scene || !this.scene.tweens) return
  // ... tween 코드
})
```

### 11.5 픽셀아트 흐릿하게 보임

**원인**: `pixelArt: true` 누락 또는 CSS transform 적용

**해결**:
- Game Config에 `pixelArt: true`, `roundPixels: true` 확인
- Canvas 요소에 `image-rendering: pixelated` CSS 적용 확인

---

## 12. 파일 구조 참조

```
frontend/src/features/mock-trial/
├── components/
│   └── MockTrialGame.tsx          # Game 초기화, React 래퍼
├── game/
│   ├── config.ts                  # 상수 (크기, 색상, 좌표, 말풍선 높이 제한)
│   ├── EventBus.ts                # Phaser ↔ React 통신 (dialogue 이벤트 포함)
│   ├── DialogueController.ts      # 대화 큐/속도 제어/스킵 컨트롤러
│   ├── CourtScene.ts              # 메인 법정 씬 (DialogueController 통합)
│   ├── LobbyScene.ts              # 로비/설정 씬
│   ├── sprites/
│   │   ├── CharacterBase.ts       # 캐릭터 Container
│   │   ├── PixelCharacterRenderer.ts  # 픽셀 그리드 렌더링
│   │   ├── JuryPanel.ts           # 배심원단 관리
│   │   └── JurorSprite.ts         # 개별 배심원
│   └── ui/
│       ├── SpeechBubble.ts        # 말풍선 (페이지 분할 + advance/setTypingSpeed)
│       └── StageIndicator.ts      # 단계 표시 바
├── demo/
│   └── demo-scenarios.ts          # 데모 시나리오 데이터
└── types/
    └── index.ts                   # 타입 + 상수 정의
```

---

## 13. 새 Scene 추가 체크리스트

1. `game/` 폴더에 Scene 클래스 파일 생성
2. `extends Phaser.Scene` + `constructor`에서 `super({ key: 'SceneName' })`
3. `shutdown()` 메서드에서 리소스 정리
4. `MockTrialGame.tsx`의 `scene` 배열에 추가
5. 필요 시 EventBus에 새 이벤트 타입 추가
6. Scene 전환 로직에서 `this.scene.start('SceneName', data)` 호출

---

## 14. 새 GameObject 추가 체크리스트

1. `Phaser.GameObjects.Container` 확장 (여러 요소 조합 시)
2. 또는 `Phaser.GameObjects.Graphics` 직접 사용 (단순 그래픽)
3. `scene.add.existing(this)` 호출 (씬에 등록)
4. `destroy()` 오버라이드로 리소스 정리
5. Tween 사용 시 참조 변수에 저장 + cleanup 코드 추가
