# Phaser ↔ React 이벤트 통신 패턴

모의 법정의 Phaser Canvas와 React UI 간 양방향 통신(EventBus) 패턴과 에러 복구 가이드.
LLM 타임아웃(408), Canvas fallback, WebGL 감지 등 미완성 항목의 구현 패턴을 포함한다.

> **적용 시점**: EventBus 수정, 새 이벤트 추가, Phaser ↔ React 통신 에러 처리 시
> **전제 스킬**: `phaser-nextjs-integration` (Phaser 기본 패턴)
> **관련 스킬**: `error-handling-patterns` (HTTP/React Error Boundary - 기존), 이 스킬은 Canvas/EventBus 고유 패턴

---

## 1. EventBus 아키텍처

### 1.1 구조

```
React UI (MockTrialGame.tsx)
    ↕ eventBus.emit() / eventBus.on()
CourtEventBus (EventBus.ts)  ← EventTarget 기반 싱글턴
    ↕ eventBus.emit() / eventBus.on()
Phaser Scene (CourtScene.ts, LobbyScene.ts)
```

### 1.2 핵심 구현 (38줄)

```typescript
// 참조: EventBus.ts
class CourtEventBus {
  private target = new EventTarget()

  emit<K extends keyof EventMap>(event: K, data: EventMap[K]): void {
    this.target.dispatchEvent(new CustomEvent(event, { detail: data }))
  }

  on<K extends keyof EventMap>(event: K, handler: (data: EventMap[K]) => void): () => void {
    const listener = (e: Event): void => {
      handler((e as CustomEvent).detail as EventMap[K])
    }
    this.target.addEventListener(event, listener)
    return () => this.target.removeEventListener(event, listener)
  }
}

export const eventBus = new CourtEventBus()
```

**설계 특징**:
- `EventTarget` 기반 (브라우저 네이티브, 외부 의존성 없음)
- `on()` 반환값 = unsubscribe 함수 (React cleanup 패턴 호환)
- 싱글턴 인스턴스 (`export const eventBus`)

---

## 2. EventMap 타입 시스템

### 2.1 현재 이벤트 정의 (16개)

```typescript
// 참조: EventBus.ts
export interface EventMap {
  // Phaser → React (6개)
  'agent:speak': { agent: string; text: string; streaming: boolean; emotion?: EmotionType }
  'stage:change': { from: string; to: string; stageNumber: number; totalStages: number }
  'evidence:presented': { cases: EvidenceItem[]; articles: EvidenceItem[] }
  'trial:complete': { judgment: string; feedback: string }
  'game:ready': Record<string, never>
  'court:entrance:complete': Record<string, never>
  'dialogue:queue:empty': Record<string, never>

  // React → Phaser (9개)
  'dialogue:enqueue': { agent: string; text: string; emotion?: EmotionType }
  'dialogue:set_speed': { speed: DialogueSpeed }
  'dialogue:advance': Record<string, never>
  'dialogue:skip': Record<string, never>
  'user:input': { text: string }
  'user:select_evidence': { evidenceIds: string[] }
  'game:advance_stage': Record<string, never>
  'agent:animate': { agent: string; animation: 'idle' | 'speak' | 'react' }
  'setup:complete': { caseType: string; userRole: string; caseSummary: string }
}
```

### 2.2 이벤트 네이밍 규칙

| 접두사 | 방향 | 예시 |
|--------|------|------|
| `agent:` | 양방향 | `agent:speak` (P→R), `agent:animate` (R→P) |
| `dialogue:` | 양방향 | `dialogue:enqueue` (R→P), `dialogue:queue:empty` (P→R) |
| `stage:` | P→R | `stage:change` |
| `evidence:` | P→R | `evidence:presented` |
| `trial:` | P→R | `trial:complete` |
| `court:` | P→R | `court:entrance:complete` |
| `game:` | 양방향 | `game:ready` (P→R), `game:advance_stage` (R→P) |
| `user:` | R→P | `user:input`, `user:select_evidence` |
| `setup:` | R→P | `setup:complete` |

### 2.3 새 이벤트 추가 규칙

1. `EventMap` 인터페이스에 타입 추가
2. `접두사:동작` 형식 (`명사:동사_과거분사` 또는 `명사:동사`)
3. 데이터 없는 이벤트는 `Record<string, never>` 사용
4. `EvidenceItem` 등 복합 타입은 `types/index.ts`에서 import

```typescript
// 새 이벤트 추가 예시
export interface EventMap {
  // ... 기존 이벤트

  // 새 이벤트 추가
  'jury:verdict_ready': { guilty: boolean; confidence: number }
}
```

---

## 3. Phaser → React 이벤트 흐름

### 3.1 agent:speak (가장 빈번)

```
CourtScene → EventBus → React 채팅 UI
```

**Phaser 측 emit**:
```typescript
// 백엔드 응답 수신 후
eventBus.emit('agent:speak', {
  agent: 'judge',
  text: '판결을 선고합니다.',
  streaming: false,    // true: 스트리밍 중, false: 완료
})
```

**CourtScene 내부 처리** (DialogueController 경유):
```typescript
// React → dialogue:enqueue → DialogueController.enqueue()
eventBus.on('dialogue:enqueue', (data) => {
  this.dialogueController?.enqueue({
    agent: data.agent,
    text: data.text,
    emotion: data.emotion,
  })
})

// DialogueController.processNext() 내부에서:
// 1. 이전 말풍선 숨기기 + 캐릭터 초기화
// 2. SpeechBubble.show() (페이지 분할 + 타이핑)
// 3. 캐릭터 애니메이션 + 감정 아이콘
// 4. 배심원 반응
// 5. eventBus.emit('agent:speak') → ChatPanel 호환
```

### 3.2 stage:change

```typescript
eventBus.emit('stage:change', {
  from: 'opening',
  to: 'evidence',
  stageNumber: 3,
  totalStages: 6,
})
```

**CourtScene 내부 처리** (참조: CourtScene.ts:157-161):
```typescript
eventBus.on('stage:change', (data) => {
  this.stageIndicator?.setCurrentStage(data.stageNumber - 1)  // 0-indexed
})
```

### 3.3 game:ready

```typescript
// LobbyScene.create() 마지막에 emit
eventBus.emit('game:ready', {} as Record<string, never>)
```

React 측에서 수신하여 로딩 UI 제거 등에 활용.

---

## 4. React → Phaser 이벤트 흐름

### 4.1 setup:complete (Scene 전환 트리거)

```typescript
// React 설정 UI에서 emit
eventBus.emit('setup:complete', {
  caseType: 'criminal',
  userRole: 'prosecutor',
  caseSummary: '사건 요약...',
})
```

**LobbyScene 처리** (참조: LobbyScene.ts:68-75):
```typescript
const unsubscribe = eventBus.on('setup:complete', (data) => {
  this.scene.start('CourtScene', {
    caseType: data.caseType,
    userRole: data.userRole,
    caseSummary: data.caseSummary,
  })
  unsubscribe()  // 1회성 리스너
})
```

### 4.2 agent:animate

```typescript
eventBus.emit('agent:animate', {
  agent: 'prosecutor',
  animation: 'speak',
})
```

**CourtScene 처리** (참조: CourtScene.ts:164-172):
```typescript
eventBus.on('agent:animate', (data) => {
  const character = this.characters.get(data.agent)
  if (character) {
    character.setSpeaking(data.animation === 'speak')
    character.highlight(data.animation === 'react')
  }
})
```

### 4.3 user:input

```typescript
// React 채팅 입력창에서 emit
eventBus.emit('user:input', { text: userMessage })
```

---

## 5. 구독/해제 패턴

### 5.1 배열 기반 Cleanup (권장)

```typescript
// 참조: CourtScene.ts:29, 135-173, 175-180
export class CourtScene extends Phaser.Scene {
  private unsubscribers: (() => void)[] = []

  private setupEventListeners(): void {
    // on() 반환값(unsubscribe 함수)을 배열에 저장
    this.unsubscribers.push(
      eventBus.on('agent:speak', (data) => { /* ... */ })
    )
    this.unsubscribers.push(
      eventBus.on('stage:change', (data) => { /* ... */ })
    )
  }

  shutdown(): void {
    // 모든 구독 일괄 해제
    this.unsubscribers.forEach((unsub) => unsub())
    this.unsubscribers = []
  }
}
```

### 5.2 1회성 리스너

```typescript
// 참조: LobbyScene.ts:68-75
const unsubscribe = eventBus.on('setup:complete', (data) => {
  // 처리 후 즉시 해제
  unsubscribe()
})
```

### 5.3 React 컴포넌트에서의 구독

```typescript
useEffect(() => {
  const unsubscribe = eventBus.on('game:ready', () => {
    setIsLoaded(true)
  })
  return () => unsubscribe()
}, [])
```

### 5.4 구독 누수 방지 체크리스트

- [ ] 모든 `eventBus.on()` 반환값이 저장되어 있는가?
- [ ] `shutdown()` 또는 `useEffect cleanup`에서 해제하는가?
- [ ] 1회성 리스너는 콜백 내에서 `unsubscribe()` 호출하는가?
- [ ] Scene 재시작 시 이전 구독이 남아있지 않은가?

---

## 6. 스트리밍 → 말풍선 연동

### 6.1 streaming 플래그 동작

```typescript
// streaming: true → 타이핑 효과 (글자별 표시)
eventBus.emit('agent:speak', {
  agent: 'judge',
  text: '피고인은...',
  streaming: true,    // 아직 응답 생성 중
})

// streaming: false → 즉시 전체 표시
eventBus.emit('agent:speak', {
  agent: 'judge',
  text: '피고인은 앞으로 나오세요.',
  streaming: false,   // 응답 완료
})
```

### 6.2 SpeechBubble 분기 (참조: SpeechBubble.ts:35-54)

```typescript
showText(text: string, immediate = false): void {
  if (immediate) {
    // streaming=false → 전체 텍스트 즉시 표시
    this.textObject.setText(text)
    this.drawBackground()
    return
  }

  // streaming=true → 타이핑 효과
  this.textObject.setText('')
  this.typingTimer = this.scene.time.addEvent({
    delay: TYPING_SPEED,  // 30ms per character
    callback: this.typeNextCharacter,
    callbackScope: this,
    repeat: text.length - 1,
  })
}
```

### 6.3 스트리밍 중 텍스트 업데이트 패턴

```typescript
// 스트리밍 토큰 누적 시
let accumulated = ''

onStreamToken((token) => {
  accumulated += token
  eventBus.emit('agent:speak', {
    agent: currentAgent,
    text: accumulated,
    streaming: true,
  })
})

onStreamEnd(() => {
  eventBus.emit('agent:speak', {
    agent: currentAgent,
    text: accumulated,
    streaming: false,  // 마지막에 false로 전환
  })
})
```

---

## 7. LLM 타임아웃(408) 에러 처리

### 7.1 문제 상황

LLM API 호출이 타임아웃(408)되면 사용자에게 아무 피드백 없이 멈추는 문제.

### 7.2 권장 구현 패턴

```typescript
// React 측 API 호출 래퍼
async function callLLMWithRetry(
  payload: ChatRequest,
  maxRetries = 2
): Promise<ChatResponse> {
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        body: JSON.stringify(payload),
        signal: AbortSignal.timeout(30_000),  // 30초 타임아웃
      })

      if (response.status === 408) {
        if (attempt < maxRetries) {
          // Exponential backoff
          const delay = 1000 * Math.pow(2, attempt)
          await new Promise((r) => setTimeout(r, delay))
          continue
        }
        throw new Error('LLM_TIMEOUT')
      }

      if (!response.ok) throw new Error(`HTTP_${response.status}`)
      return await response.json()
    } catch (error) {
      if (attempt === maxRetries) throw error
    }
  }
  throw new Error('MAX_RETRIES_EXCEEDED')
}
```

### 7.3 사용자 안내 UI 패턴

```typescript
// EventBus로 에러 상태 전달 (향후 EventMap에 추가)
// 'error:llm_timeout': { message: string; canRetry: boolean }

eventBus.emit('agent:speak', {
  agent: 'clerk',
  text: 'AI 응답 대기 시간이 초과되었습니다. 잠시 후 다시 시도합니다...',
  streaming: false,
})
```

### 7.4 Fallback 메시지 패턴

```typescript
const FALLBACK_MESSAGES: Record<string, string> = {
  judge: '잠시 정리할 시간을 갖겠습니다. 다음 단계로 진행해주세요.',
  prosecutor: '주장을 정리 중입니다.',
  attorney: '변론을 준비 중입니다.',
  defendant: '...',
  clerk: '기록을 정리하고 있습니다.',
}
```

---

## 8. Canvas 렌더링 실패 React Fallback

### 8.1 권장 구현 패턴

```typescript
// ErrorBoundary 컴포넌트
'use client'

import { Component, type ReactNode } from 'react'

interface Props {
  children: ReactNode
  fallback: ReactNode
}

interface State {
  hasError: boolean
}

class GameErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false }

  static getDerivedStateFromError(): State {
    return { hasError: true }
  }

  render(): ReactNode {
    if (this.state.hasError) {
      return this.props.fallback
    }
    return this.props.children
  }
}
```

### 8.2 텍스트 모드 대체 UI

```typescript
// Canvas 실패 시 HTML로 법정 시뮬레이션
function MockTrialTextMode() {
  return (
    <div className="p-6 border rounded-lg bg-gray-50">
      <h2 className="text-lg font-bold text-red-700 mb-2">
        Canvas 렌더링을 사용할 수 없습니다
      </h2>
      <p className="text-sm text-gray-600 mb-4">
        브라우저가 Canvas/WebGL을 지원하지 않거나 오류가 발생했습니다.
        텍스트 모드로 진행합니다.
      </p>
      {/* 텍스트 기반 법정 진행 UI */}
    </div>
  )
}
```

---

## 9. WebGL 미지원 감지 및 안내

### 9.1 권장 감지 패턴

```typescript
function detectRenderingCapability(): 'webgl2' | 'webgl' | 'canvas' | 'none' {
  const canvas = document.createElement('canvas')
  if (canvas.getContext('webgl2')) return 'webgl2'
  if (canvas.getContext('webgl')) return 'webgl'
  if (canvas.getContext('2d')) return 'canvas'
  return 'none'
}
```

### 9.2 안내 메시지 분기

| 결과 | 동작 |
|------|------|
| `webgl2` | 최적 상태, 그대로 진행 |
| `webgl` | 정상 동작, 성능 약간 저하 가능 |
| `canvas` | Canvas 2D fallback, 안내 배너 표시 |
| `none` | 텍스트 모드 전환, 경고 표시 |

### 9.3 브라우저 호환성

| 브라우저 | WebGL 2 | WebGL 1 | 비고 |
|----------|---------|---------|------|
| Chrome 56+ | O | O | 권장 |
| Firefox 51+ | O | O | 권장 |
| Safari 15+ | O | O | iOS도 지원 |
| Edge 79+ | O | O | Chromium 기반 |
| IE 11 | X | 부분 | 미지원 |

---

## 10. 디버깅 가이드

### 10.1 EventBus 로깅

```typescript
// 개발 환경에서 모든 이벤트 로깅
class CourtEventBus {
  emit<K extends keyof EventMap>(event: K, data: EventMap[K]): void {
    if (process.env.NODE_ENV === 'development') {
      console.log(`[EventBus] ${String(event)}`, data)
    }
    this.target.dispatchEvent(new CustomEvent(event, { detail: data }))
  }
}
```

### 10.2 이벤트 흐름 추적

```
[setup:complete] React → LobbyScene
  → scene.start('CourtScene', data)
  → [game:ready 아님 - CourtScene에서는 별도 emit 없음]

[dialogue:enqueue] React → DialogueController.enqueue()
  → processNext() → SpeechBubble.show() (페이지 분할)
  → character.setSpeaking(true) + setEmotion()
  → juryPanel.reactToSpeech()
  → eventBus.emit('agent:speak') (ChatPanel 호환)

[dialogue:advance] React(Space키) → DialogueController.handleAdvance()
  → bubble.advance() → 타이핑 완료 / 다음 페이지 / 다음 대화

[dialogue:set_speed] React → DialogueController.setSpeed()
  → bubble.setTypingSpeed() (현재 말풍선에도 즉시 적용)

[dialogue:skip] React → DialogueController.skipAll()
  → 큐 전체 스킵, agent:speak emit (ChatPanel 기록 유지)

[dialogue:queue:empty] DialogueController → React
  → setIsWaiting(false)

[stage:change] 백엔드/로직 → CourtScene
  → stageIndicator.setCurrentStage()

[agent:animate] React → CourtScene
  → character.setSpeaking() / character.highlight()
```

### 10.3 타이밍 이슈 진단

| 문제 | 원인 | 해결 |
|------|------|------|
| 이벤트 유실 | 구독 전 emit 발생 | Scene `create()` 완료 후 emit |
| 순서 역전 | 비동기 처리 차이 | `delayedCall`로 순서 보장 |
| 이중 실행 | 구독 미해제 후 재구독 | `shutdown()`에서 정리 확인 |

---

## 11. 메시지 큐 패턴 (향후 확장)

### 11.1 이벤트 버퍼링 설계

Scene이 아직 준비되지 않았을 때 이벤트를 버퍼링하여 순서 보장.

```typescript
// 권장 구현 패턴
class BufferedEventBus extends CourtEventBus {
  private buffer: { event: string; data: unknown }[] = []
  private isReady = false

  emit<K extends keyof EventMap>(event: K, data: EventMap[K]): void {
    if (!this.isReady && event !== 'game:ready') {
      this.buffer.push({ event: event as string, data })
      return
    }
    super.emit(event, data)
  }

  setReady(): void {
    this.isReady = true
    this.buffer.forEach(({ event, data }) => {
      super.emit(event as keyof EventMap, data as EventMap[keyof EventMap])
    })
    this.buffer = []
  }
}
```

### 11.2 재전송 로직

```typescript
// 응답 확인(ACK) 기반 재전송
interface PendingEvent {
  id: string
  event: keyof EventMap
  data: EventMap[keyof EventMap]
  timestamp: number
  retries: number
}

// 일정 시간 내 ACK 없으면 재전송
// 최대 3회 재시도 후 에러 처리
```

---

## 12. 트러블슈팅

### 12.1 이벤트가 수신되지 않음

**체크리스트**:
1. `eventBus` import 경로가 동일한 인스턴스인가? (`../game/EventBus`)
2. `on()` 호출이 `emit()` 이전에 실행되었는가?
3. 이벤트 이름에 오타가 없는가? (TypeScript가 체크하지만 확인)
4. `shutdown()`에서 이미 해제된 구독이 아닌가?

### 12.2 이벤트 순서가 꼬임

**원인**: `setTimeout`, `delayedCall` 등 비동기 처리로 인한 순서 역전

**해결**:
```typescript
// 순서가 중요한 이벤트는 동기적으로 처리
eventBus.emit('stage:change', stageData)  // 먼저
eventBus.emit('agent:speak', speakData)   // 바로 다음
```

### 12.3 메모리 누수 (미해제 리스너)

**진단**:
```typescript
// Chrome DevTools → Memory → Heap Snapshot
// EventTarget의 리스너 수 확인

// 또는 카운터 추가
let activeListeners = 0
on(...) {
  activeListeners++
  return () => { activeListeners--; /* 원래 해제 로직 */ }
}
```

### 12.4 CustomEvent detail이 undefined

**원인**: `emit()` 시 data 누락 또는 잘못된 타입

**해결**: TypeScript `EventMap` 타입으로 컴파일 타임 검증. 런타임에도 방어:
```typescript
on<K extends keyof EventMap>(event: K, handler: (data: EventMap[K]) => void): () => void {
  const listener = (e: Event): void => {
    const detail = (e as CustomEvent).detail
    if (detail === undefined) {
      console.warn(`[EventBus] ${String(event)} received with no data`)
      return
    }
    handler(detail as EventMap[K])
  }
  // ...
}
```
