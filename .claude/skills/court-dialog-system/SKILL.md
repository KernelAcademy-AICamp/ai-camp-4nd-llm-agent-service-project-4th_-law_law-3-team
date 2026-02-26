# 모의 법정 대화 시스템

모의 법정의 비주얼 노벨 + 시뮬레이션 대화 시스템 구현 가이드.
말풍선, 캐릭터, 배심원 반응, 단계 진행, 데모 시나리오의 조율 패턴.

> **적용 시점**: 대화/캐릭터 수정, 새 캐릭터 추가, 배심원 반응 조정, 데모 시나리오 추가 시
> **전제 스킬**: `phaser-nextjs-integration`, `court-eventbus-patterns`
> **관련 스킬**: `korean-legal-domain` (법률 절차 지식)

---

## 1. SpeechBubble 패턴

### 1.1 구조 (참조: SpeechBubble.ts)

```typescript
export class SpeechBubble extends Phaser.GameObjects.Container {
  private background: Phaser.GameObjects.Graphics  // 9-patch 유사 배경
  private textObject: Phaser.GameObjects.Text       // 텍스트 표시
  private fullText = ''                              // 전체 텍스트
  private displayedLength = 0                        // 타이핑 진행 위치
  private typingTimer: Phaser.Time.TimerEvent | null = null
}
```

### 1.2 상수

```typescript
const BUBBLE_PADDING = 12    // 내부 여백 (px)
const BUBBLE_RADIUS = 8      // 모서리 라운드 (px)
const MAX_WIDTH = 260         // 최대 너비 (px) → 자동 줄바꿈
const FONT_SIZE = 13          // 글자 크기 (px)

// config.ts
const MAX_BUBBLE_HEIGHT = 120 // 말풍선 최대 높이 (px)
const MAX_TEXT_HEIGHT = 80    // 텍스트 영역 최대 높이 (~4-5줄)
const BASE_TYPING_SPEED = 30  // 타이핑 기본 속도 (ms/글자)
```

### 1.2a 페이지 분할 + 대화 진행

SpeechBubble은 긴 텍스트를 `MAX_TEXT_HEIGHT` 기준으로 페이지로 분할합니다.

```typescript
// 핵심 API
show(name, text, emotion?, immediate?)  // 페이지 분할 후 첫 페이지 표시
advance(): boolean  // 타이핑 완료 → 다음 페이지 → true(모든 페이지 완료)
completeTyping()    // 현재 페이지 타이핑 즉시 완료
setTypingSpeed(ms)  // 속도 동적 변경 (0=instant)
hasNextPage()       // 다음 페이지 존재 여부
isComplete()        // 타이핑 완료 여부
```

다음 페이지가 있을 때 ▼ 인디케이터가 깜빡입니다.

### 1.3 showText 메서드

```typescript
// 참조: SpeechBubble.ts:35-54
showText(text: string, immediate = false): void {
  this.fullText = text
  this.displayedLength = 0
  this.setVisible(true)
  this.stopTyping()

  if (immediate) {
    // 전체 텍스트 즉시 표시 (스트리밍 완료 시)
    this.textObject.setText(text)
    this.drawBackground()
    return
  }

  // 타이핑 효과 (스트리밍 중)
  this.textObject.setText('')
  this.typingTimer = this.scene.time.addEvent({
    delay: TYPING_SPEED,
    callback: this.typeNextCharacter,
    callbackScope: this,
    repeat: text.length - 1,
  })
}
```

### 1.4 배경 그리기 (Graphics API)

```typescript
// 참조: SpeechBubble.ts:67-87
private drawBackground(): void {
  this.background.clear()

  const textWidth = Math.min(this.textObject.width + BUBBLE_PADDING * 2, MAX_WIDTH)
  const textHeight = this.textObject.height + BUBBLE_PADDING * 2

  // 말풍선 본체 (둥근 사각형)
  this.background.fillStyle(0xffffff, 0.95)
  this.background.lineStyle(2, 0x333333, 1)
  this.background.fillRoundedRect(0, 0, textWidth, textHeight, BUBBLE_RADIUS)
  this.background.strokeRoundedRect(0, 0, textWidth, textHeight, BUBBLE_RADIUS)

  // 꼬리 삼각형 (아래쪽)
  const tailX = textWidth / 2
  const tailY = textHeight
  this.background.fillStyle(0xffffff, 0.95)
  this.background.fillTriangle(tailX - 6, tailY, tailX + 6, tailY, tailX, tailY + 10)
  this.background.lineStyle(2, 0x333333, 1)
  this.background.lineBetween(tailX - 6, tailY, tailX, tailY + 10)
  this.background.lineBetween(tailX + 6, tailY, tailX, tailY + 10)
}
```

### 1.5 텍스트 스타일

```typescript
{
  fontSize: '13px',
  color: '#1a1a1a',
  fontFamily: 'sans-serif',
  wordWrap: { width: MAX_WIDTH - BUBBLE_PADDING * 2 },  // 자동 줄바꿈
  lineSpacing: 4,
}
```

### 1.6 말풍선 위치 계산

```typescript
// 참조: CourtScene.ts:112-118
// 캐릭터 위치 기준으로 오프셋
const bubbleY = position.y - 70     // 캐릭터 위 70px
const bubbleX = position.x - 130    // 좌측으로 130px 오프셋
```

---

## 2. 캐릭터 상태 머신

### 2.1 상태 타입

```typescript
// 참조: PixelCharacterRenderer.ts:10
export type CharacterState = 'idle' | 'speak' | 'react'
```

### 2.2 상태 전환 (참조: CharacterBase.ts)

```
             setSpeaking(true)
   idle ─────────────────────► speak
    ▲                            │
    │    setSpeaking(false)       │
    ◄────────────────────────────┘
    │
    │  highlight(true)     highlight(false)
    ├────────────────► react ────────────┘
```

### 2.3 setSpeaking 동작

```typescript
// 참조: CharacterBase.ts:45-62
setSpeaking(speaking: boolean): void {
  this.isSpeaking = speaking
  if (speaking) {
    this.renderState('speak')       // PixelGrid → speak 스프라이트
    this.scene.tweens.add({          // 확대/축소 반복 애니메이션
      targets: this,
      scaleX: 1.05, scaleY: 1.05,
      yoyo: true, repeat: -1, duration: 400,
    })
  } else {
    this.scene.tweens.killTweensOf(this)  // Tween 정리
    this.setScale(1, 1)                     // 원래 크기 복원
    this.renderState('idle')                // PixelGrid → idle 스프라이트
    this.startBreathAnimation()             // 호흡 애니메이션 재시작
  }
}
```

### 2.4 highlight 동작

```typescript
// 참조: CharacterBase.ts:64-71
highlight(isHighlighted: boolean): void {
  if (isHighlighted) {
    this.renderState('react')
  } else if (!this.isSpeaking) {
    this.renderState('idle')  // 발언 중이면 speak 유지
  }
}
```

### 2.5 Breath Animation (호흡 효과)

```typescript
// 참조: CharacterBase.ts:79-91
// idle 상태에서 자연스러운 움직임
this.breathTween = this.scene.tweens.add({
  targets: this.characterGraphics,
  y: -2,                    // 2px 위로 이동
  yoyo: true,               // 원위치 복귀
  repeat: -1,               // 무한 반복
  duration: 1000,            // 1초 주기
  ease: 'Sine.easeInOut',   // 부드러운 이징
})
```

---

## 3. PixelCharacterRenderer

### 3.1 역할별 PixelGrid 정의

| 역할 | 특징 | 소품 | 복장 색상 |
|------|------|------|----------|
| `judge` | 사모 형태 머리장식 | 법봉 (BROWN_WOOD) | 검정 (0x1a1a1a) |
| `prosecutor` | 단정한 머리 | 서류 (PAPER) | 남색 정장 + 빨간 넥타이 |
| `attorney` | 단정한 머리 | 법전 (BOOK) | 차콜 정장 + 초록 넥타이 |
| `defendant` | 수수한 느낌 | 없음 | 회색 옷 (GRAY_CLOTH) |
| `clerk` | 안경 (GLASSES) | 노트북 (NOTEBOOK) | 진회색 정장 (CLERK_SUIT) |

### 3.2 색상 팔레트

```typescript
// 공용 피부/머리
const SKIN = 0xffcc99       // 밝은 살구색
const SKIN_DARK = 0xdba876  // 어두운 살구색 (코/입)
const HAIR_BLACK = 0x2c2c2c // 검은 머리
const WHITE = 0xffffff      // 셔츠, 눈 흰자
const BLACK = 0x000000      // 눈동자

// 역할별 색상
const NAVY = 0x1a237e        // 검사 정장
const RED_TIE = 0xb71c1c     // 검사 넥타이
const CHARCOAL = 0x37474f    // 변호사 정장
const GREEN_TIE = 0x1b5e20   // 변호사 넥타이
const GRAY_CLOTH = 0x78909c  // 피고인 옷
const GRAY_DARK = 0x546e7a   // 피고인 옷 (어두운)
const CLERK_SUIT = 0x546e7a  // 서기 정장
const GLASSES = 0x87ceeb     // 서기 안경
const BROWN_WOOD = 0x8b5e3c  // 판사 법봉
const BOOK = 0x1b5e20        // 변호사 법전
const BOOK_PAGE = 0xfff8e1   // 법전 페이지
const PAPER = 0xfafafa       // 검사 서류
const NOTEBOOK = 0xe0e0e0    // 서기 노트북
```

### 3.3 그리드 → Graphics 렌더링 프로세스

```
1. CHARACTER_GRIDS[role][state] → PixelGrid 선택
2. graphics.clear() → 이전 그리기 제거
3. 중앙 정렬 오프셋 계산: -(GRID_WIDTH * PIXEL_SIZE) / 2
4. 2중 for 루프로 각 셀 순회
5. color === 0 → 건너뛰기 (투명)
6. graphics.fillStyle(color, 1) → 색상 설정
7. graphics.fillRect(x, y, PIXEL_SIZE, PIXEL_SIZE) → 4x4px 사각형 그리기
```

### 3.4 상태별 차이

- **idle**: 기본 자세, 양손 내림
- **speak**: 한 손 들어올림 (검사: 서류 들기, 변호사: 법전 들기, 판사: 법봉 들기)
- **react**: 현재 `idle`과 동일 (향후 확장 가능)

---

## 4. 배심원 반응 시스템

### 4.1 JuryPanel 구조 (참조: JuryPanel.ts)

```typescript
export class JuryPanel {
  private scene: Phaser.Scene
  private jurors: JurorSprite[] = []        // 배심원 4명
  private resetTimers: Phaser.Time.TimerEvent[] = []  // 반응 초기화 타이머
}
```

### 4.2 반응 규칙 (REACTION_RULES)

```typescript
// 참조: JuryPanel.ts:14-20
const REACTION_RULES: Record<string, ReactionRule> = {
  judge:      { probability: 0.50, reactions: ['neutral', 'nod'] },
  prosecutor: { probability: 0.75, reactions: ['nod', 'think', 'surprise'] },
  attorney:   { probability: 0.75, reactions: ['think', 'nod', 'surprise'] },
  defendant:  { probability: 0.75, reactions: ['think', 'nod'] },
  clerk:      { probability: 0.25, reactions: ['neutral'] },
}
```

| 발언자 | 반응 확률 | 반응 타입 | 이유 |
|--------|----------|----------|------|
| 판사 | 50% | 고개 끄덕, 무반응 | 중립적 발언 |
| 검사 | 75% | 끄덕, 생각, 놀람 | 강한 주장 |
| 변호사 | 75% | 생각, 끄덕, 놀람 | 반론 |
| 피고인 | 75% | 생각, 끄덕 | 감정적 반응 |
| 서기 | 25% | 무반응 | 절차적 발언 |

### 4.3 키워드 트리거 (STRONG_KEYWORDS)

```typescript
// 참조: JuryPanel.ts:23-27
const STRONG_KEYWORDS: Record<string, string[]> = {
  prosecutor: ['유죄', '징역', '구형', '범행', '증거'],
  attorney:   ['무죄', '반박', '증거 불충분', '석방', '변론'],
  defendant:  ['반성', '후회', '용서', '죄송', '잘못'],
}
```

키워드 포함 시 → `probability + 0.2` (최대 1.0)
키워드 반응 시 → `neutral` 제외, 더 강한 반응 선택

### 4.4 reactToSpeech 로직 (참조: JuryPanel.ts:47-77)

```
1. 기존 반응 타이머 초기화 (clearResetTimers)
2. 발언자의 REACTION_RULES 조회
3. STRONG_KEYWORDS 포함 여부 → 확률 조정
4. 각 배심원(4명)에 대해:
   a. Math.random() < effectiveProbability → 반응 여부 결정
   b. delay = 300~800ms (랜덤) → 자연스러운 시차
   c. pickReaction(reactions, isStrong) → 반응 타입 선택
   d. juror.react(reaction) → Tween 실행
   e. delay + 2500ms 후 juror.stopReaction() → 자동 초기화
```

### 4.5 배심원 반응 타이밍 상수

```typescript
const REACTION_DELAY_MIN = 300   // 최소 대기 (ms)
const REACTION_DELAY_MAX = 800   // 최대 대기 (ms)
const REACTION_DURATION = 2500   // 반응 지속 시간 (ms)
```

### 4.6 JurorReaction 6종 + Tween 매핑

```typescript
// 참조: JurorSprite.ts:31
export type JurorReaction = 'neutral' | 'nod' | 'shake' | 'surprise' | 'think' | 'whisper'
```

| 반응 | Tween | 의미 |
|------|-------|------|
| `neutral` | 없음 | 무표정 |
| `nod` | y: -4px, yoyo, repeat:1, 300ms | 고개 끄덕 |
| `shake` | x: -3px, yoyo, repeat:1, 200ms | 고개 젓기 |
| `surprise` | scale: 1.1, yoyo, 200ms | 놀람 |
| `think` | angle: 2, yoyo, 400ms | 생각 (머리 기울임) |
| `whisper` | x: +8px, yoyo, 300ms | 옆 사람과 속삭임 |

### 4.7 배심원 스프라이트 (JurorSprite)

```typescript
// 8x10 그리드 = 32x40px (메인 캐릭터의 절반 크기)
const PIXEL = 4
const GRID_W = 8
const GRID_H = 10

// 색상 변형 4종 (배심원 개성)
const JUROR_SUIT_COLORS = [0x5c6bc0, 0x7e57c2, 0x26a69a, 0x8d6e63]

// -1 셀 = suitColor로 치환 (캐릭터별 색상 차별화)
```

---

## 5. 단계 진행 시스템

### 5.1 StageIndicator (참조: StageIndicator.ts)

```typescript
export class StageIndicator extends Phaser.GameObjects.Container {
  private dots: Phaser.GameObjects.Arc[] = []         // 진행 점
  private stageTexts: Phaser.GameObjects.Text[] = []  // 단계명
  private currentIndex = 0
}
```

### 5.2 단계 데이터 (참조: types/index.ts)

**형사 재판 6단계** (`CRIMINAL_STAGES`):

| 순서 | ID | 이름 | 법적 근거 | 사용자 행동 |
|------|----|------|----------|-----------|
| 1 | `identity` | 인정신문 | 형소법 284 | 자동 진행 |
| 2 | `opening` | 모두진술 | 형소법 285~286 | 역할별 진술 입력 |
| 3 | `evidence` | 증거조사 | 형소법 290~313 | 증거 선택/제출 |
| 4 | `examination` | 피고인신문 | 형소법 296-2 | 질문 입력 |
| 5 | `closing` | 최종변론 | 형소법 302~303 | 최후변론 입력 |
| 6 | `verdict` | 판결선고 | 형소법 318-4 | 관전 |

**민사 재판 6단계** (`CIVIL_STAGES`):

| 순서 | ID | 이름 | 법적 근거 | 사용자 행동 |
|------|----|------|----------|-----------|
| 1 | `pretrial` | 변론준비 | 민소법 258~268 | 자동 진행 |
| 2 | `claims` | 주장/답변 | 민소법 256~257 | 역할별 입력 |
| 3 | `evidence` | 증거조사 | 민소법 288~344 | 증거 선택/제출 |
| 4 | `argument` | 변론 | 민소법 134~148 | 주장 입력 (2-3 라운드) |
| 5 | `closing` | 변론종결 | 민소법 200 | 최종 주장 입력 |
| 6 | `verdict` | 판결선고 | 민소법 206~208 | 관전 |

### 5.3 시각적 표현

```
완료(녹색)  현재(금색+흰테두리)  미진행(회색)
   ●──────────●──────────○──────────○
  인정신문   모두진술    증거조사    피고인신문
```

```typescript
// 참조: StageIndicator.ts:63-88
if (index < this.currentIndex) {
  dot.setFillStyle(0x4caf50)     // 완료: 녹색
} else if (index === this.currentIndex) {
  dot.setFillStyle(0xffd700)     // 현재: 금색
  dot.setStrokeStyle(2, 0xffffff)
} else {
  dot.setFillStyle(0x666666)     // 미진행: 회색
}
```

### 5.4 stage:change 이벤트 연동

```typescript
// 단계 전환 시 emit
eventBus.emit('stage:change', {
  from: 'opening',
  to: 'evidence',
  stageNumber: 3,       // 1-indexed
  totalStages: 6,
})

// CourtScene에서 수신
eventBus.on('stage:change', (data) => {
  this.stageIndicator?.setCurrentStage(data.stageNumber - 1)  // 0-indexed 변환
})
```

---

## 6. 법정 배경 레이아웃

### 6.1 좌표 맵 (800x480 기준)

```
                    [대한민국 법원] (400, 20)
        ┌─────────────────────────────────────────┐
        │      서기석        판사석                 │
        │   (120,140)    (400,140)                 │ 배심원석
        │                 ┌──────┐                 │ (686-782,
        │                 │ 판사 │                 │  120-220)
   검사석│                 └──────┘                 │
 (200,260)│                                        │
        │   검사          피고인       변호사       │
        │ (200,260)    (400,340)    (560,260)       │
        │──────────────────────────────────────────│ ← 방청석 구분선 y=400
        │                 방청석                     │
        └─────────────────────────────────────────┘
```

### 6.2 캐릭터 배치 좌표

```typescript
// 참조: config.ts:31-37
export const CHARACTER_POSITIONS: Record<string, { x: number; y: number }> = {
  judge:      { x: 400, y: 140 },  // 중앙 상단
  prosecutor: { x: 200, y: 260 },  // 좌측
  attorney:   { x: 560, y: 260 },  // 우측
  defendant:  { x: 400, y: 340 },  // 중앙 하단
  clerk:      { x: 120, y: 140 },  // 좌측 상단
}

// 참조: config.ts:40-45
export const JURY_POSITIONS: { x: number; y: number }[] = [
  { x: 710, y: 150 },  // 좌상
  { x: 760, y: 150 },  // 우상
  { x: 710, y: 200 },  // 좌하
  { x: 760, y: 200 },  // 우하
]
```

### 6.3 법정 배경 Graphics 구성

```typescript
// 참조: CourtScene.ts:49-101
// 판사석: fillRect(300, 90, 200, 60)  갈색 플랫폼
// 서기석: fillRect(60, 110, 80, 40)
// 검사석: fillRect(120, 230, 160, 10) 나무 책상 라인
// 변호사석: fillRect(480, 230, 160, 10)
// 피고인석: fillRect(330, 310, 140, 10)
// 배심원석 배경: fillRoundedRect(686, 120, 96, 100, 6)
// 방청석 구분선: lineBetween(50, 400, 750, 400)
```

---

## 7. 새 캐릭터 추가 체크리스트

### 7.1 4단계 필수 작업

**Step 1: PixelGrid 정의** (`PixelCharacterRenderer.ts`)
```typescript
// 12x16 그리드, idle + speak 2상태 필수
const NEW_ROLE_IDLE: PixelGrid = [
  // ... 16행 x 12열
]
const NEW_ROLE_SPEAK: PixelGrid = [
  // ... 16행 x 12열
]

// CHARACTER_GRIDS에 추가
const CHARACTER_GRIDS: Record<string, Record<CharacterState, PixelGrid>> = {
  // ... 기존 역할
  new_role: { idle: NEW_ROLE_IDLE, speak: NEW_ROLE_SPEAK, react: NEW_ROLE_IDLE },
}
```

**Step 2: config.ts 업데이트**
```typescript
// CHARACTER_COLORS에 추가
export const CHARACTER_COLORS: Record<string, number> = {
  // ... 기존
  new_role: 0x______,
}

// CHARACTER_NAMES에 추가
export const CHARACTER_NAMES: Record<string, string> = {
  // ... 기존
  new_role: '새역할',
}

// CHARACTER_POSITIONS에 추가
export const CHARACTER_POSITIONS: Record<string, { x: number; y: number }> = {
  // ... 기존
  new_role: { x: ___, y: ___ },
}
```

**Step 3: CourtScene.ts 업데이트**
```typescript
private createCharacters(): void {
  const roles = ['judge', 'prosecutor', 'attorney', 'defendant', 'clerk', 'new_role']
  // ... 나머지 동일
}
```

**Step 4: 배심원 반응 규칙 (필요 시)**
```typescript
// JuryPanel.ts의 REACTION_RULES에 추가
const REACTION_RULES: Record<string, ReactionRule> = {
  // ... 기존
  new_role: { probability: 0.5, reactions: ['neutral', 'nod'] },
}
```

---

## 8. 데모 시나리오 구조

### 8.1 DemoScenario 인터페이스

```typescript
// 참조: demo-scenarios.ts
export interface DemoScenario {
  id: string              // 'criminal-fraud-prosecutor'
  name: string            // '투자 사기 사건 (검사)'
  description: string     // 시나리오 설명
  setup: {
    caseType: CaseType         // 'criminal' | 'civil'
    caseCategory: CaseCategory // 세부 유형
    userRole: UserRole         // 사용자 역할
    caseSummary: string        // 사건 요약문
  }
  stages: DemoStage[]     // 단계별 대화 데이터
}

interface DemoStage {
  stageId: string                              // 'identity', 'opening', ...
  userInputs: string[]                         // 사용자 자동 입력 텍스트
  mockResponses: { speaker: string; content: string }[]  // AI 응답 시뮬레이션
}
```

### 8.2 현재 시나리오 목록

| ID | 유형 | 역할 | 설명 |
|----|------|------|------|
| `criminal-fraud-prosecutor` | 형사 사기 | 검사 | 투자 사기 5천만원 편취 사건 |
| `civil-damages-plaintiff` | 민사 손해배상 | 원고 대리인 | 교통사고 2,500만원 손해배상 |

### 8.3 시나리오 데이터 흐름

```
1. 데모 모드 선택 → DemoScenario 로드
2. setup 데이터 → eventBus.emit('setup:complete', setup)
3. 각 stage 순회:
   a. mockResponses → eventBus.emit('dialogue:enqueue', { agent, text, emotion })
      → DialogueController가 큐 관리 + SpeechBubble 페이지 분할
      → 내부에서 agent:speak emit (ChatPanel 호환)
   b. userInputs → 자동 입력 시뮬레이션
   c. dialogue:queue:empty → setIsWaiting(false)
4. verdict 단계 → 판결 표시 + trial:complete
```

---

## 9. 새 데모 시나리오 추가 가이드

### 9.1 형사 시나리오 템플릿

```typescript
const NEW_CRIMINAL_SCENARIO: DemoScenario = {
  id: 'criminal-{유형}-{역할}',
  name: '{사건명} ({역할})',
  description: '...',
  setup: {
    caseType: 'criminal',
    caseCategory: 'criminal_{유형}',
    userRole: 'prosecutor' | 'attorney',
    caseSummary: '피고인 OOO는 YYYY년 M월부터...',
  },
  stages: [
    // 6단계: identity → opening → evidence → examination → closing → verdict
    {
      stageId: 'identity',
      userInputs: [],  // 자동 진행
      mockResponses: [
        { speaker: 'clerk', content: 'YYYY고단NNNN호 OO 사건을 개정합니다.' },
        { speaker: 'judge', content: '피고인은 앞으로 나오세요...' },
        { speaker: 'defendant', content: '예, 맞습니다.' },
        { speaker: 'judge', content: '피고인에게 진술거부권을 고지합니다...' },
        { speaker: 'defendant', content: '예, 이해했습니다.' },
      ],
    },
    // ... opening, evidence, examination, closing, verdict
  ],
}
```

### 9.2 민사 시나리오 템플릿

```typescript
const NEW_CIVIL_SCENARIO: DemoScenario = {
  id: 'civil-{유형}-{역할}',
  name: '{사건명} ({역할})',
  description: '...',
  setup: {
    caseType: 'civil',
    caseCategory: 'civil_{유형}',
    userRole: 'plaintiff' | 'defendant',
    caseSummary: 'YYYY년 M월 D일...',
  },
  stages: [
    // 6단계: pretrial → claims → evidence → argument → closing → verdict
    {
      stageId: 'pretrial',
      userInputs: [],
      mockResponses: [
        { speaker: 'clerk', content: 'YYYY가단NNNNN호 OO 사건을 개정합니다.' },
        { speaker: 'judge', content: '양측 대리인은 출석하셨습니까?...' },
      ],
    },
    // ... claims, evidence, argument, closing, verdict
  ],
}
```

### 9.3 시나리오 등록

```typescript
// demo-scenarios.ts 하단
export const DEMO_SCENARIOS: DemoScenario[] = [
  CRIMINAL_FRAUD_PROSECUTOR,
  CIVIL_DAMAGES_PLAINTIFF,
  NEW_CRIMINAL_SCENARIO,  // 추가
]
```

### 9.4 시나리오 작성 규칙

- **법률 정확성**: 법적 근거(조문 번호)를 mockResponse에 포함
- **사건번호 형식**: 형사 `YYYY고단NNNN호`, 민사 `YYYY가단NNNNN호`
- **판결 형식**: `【이유】` 섹션 포함, 양형 이유 명시
- **userInputs**: 역할에 맞는 전문적 발언 (3-5문장 권장)
- **mockResponses 순서**: speaker 순서 = 실제 법정 진행 순서

---

## 10. 말풍선 위치 조정

### 10.1 위치 겹침 방지

현재 모든 캐릭터에 동일한 오프셋 적용:
```typescript
const bubbleY = position.y - 70   // 위로 70px
const bubbleX = position.x - 130  // 좌로 130px
```

겹침 발생 시 역할별 개별 조정:
```typescript
const BUBBLE_OFFSETS: Record<string, { x: number; y: number }> = {
  judge:      { x: -130, y: -70 },
  prosecutor: { x: -130, y: -70 },
  attorney:   { x: -130, y: -70 },
  defendant:  { x: -130, y: -70 },
  clerk:      { x: 20, y: -70 },   // 서기는 우측에 표시
}
```

### 10.2 화면 밖 방지

```typescript
// 말풍선이 화면 밖으로 나가지 않도록 clamp
const clampedX = Math.max(0, Math.min(bubbleX, GAME_WIDTH - MAX_WIDTH))
const clampedY = Math.max(0, bubbleY)
```

---

## 11. 트러블슈팅

### 11.1 말풍선이 표시되지 않음

**체크리스트**:
1. `agent` 이름이 `CHARACTER_POSITIONS` 키와 일치하는가?
2. `speechBubbles.get(data.agent)` 반환값이 null이 아닌가?
3. `setVisible(true)`가 호출되는가?
4. 말풍선 좌표가 화면 범위 내인가?

### 11.2 배심원이 반응하지 않음

**체크리스트**:
1. `JuryPanel`이 `create()`에서 초기화되었는가?
2. `reactToSpeech(agent, text)` 호출 확인
3. `REACTION_RULES[agent]`에 해당 역할이 정의되어 있는가?
4. `probability`가 0이 아닌가?
5. `JURY_POSITIONS` 좌표가 화면 범위 내인가?

### 11.3 단계 전환이 반영되지 않음

**체크리스트**:
1. `stage:change` 이벤트의 `stageNumber`가 1-indexed인지 확인
2. `setCurrentStage()`에 전달 시 `-1`로 0-indexed 변환
3. `stageIndicator`가 null이 아닌지 확인
4. `caseType`에 맞는 단계 데이터(`CRIMINAL_STAGES` vs `CIVIL_STAGES`)가 로드되었는지 확인

### 11.4 타이핑 효과가 멈춤

**원인**: 이전 `typingTimer`가 정리되지 않고 새 `showText()` 호출

**해결**: `showText()` 시작 시 `stopTyping()` 호출 확인 (이미 구현됨)

### 11.5 캐릭터 스프라이트가 깨짐

**체크리스트**:
1. PixelGrid 배열의 행 수 = `GRID_HEIGHT` (16)인가?
2. 각 행의 열 수 = `GRID_WIDTH` (12)인가?
3. 색상값이 `0x000000`~`0xffffff` 범위인가? (0은 투명)
4. `CHARACTER_GRIDS`에 해당 role의 3개 상태가 모두 등록되었는가?

---

## 12. 파일 참조 인덱스

| 파일 | 핵심 역할 |
|------|----------|
| `game/ui/SpeechBubble.ts` | 말풍선 UI (페이지 분할, advance, setTypingSpeed, ▼ 인디케이터) |
| `game/DialogueController.ts` | 대화 큐 관리, 속도 제어 (normal/fast/faster/instant), 스킵, Space 키 |
| `game/sprites/CharacterBase.ts` | 캐릭터 Container (상태 머신, Tween) |
| `game/sprites/PixelCharacterRenderer.ts` | 5개 역할 x 2상태 PixelGrid + 렌더링 함수 |
| `game/sprites/JuryPanel.ts` | 배심원단 관리 (반응 규칙, 키워드 트리거) |
| `game/sprites/JurorSprite.ts` | 개별 배심원 (8x10 그리드, 6종 반응 Tween) |
| `game/ui/StageIndicator.ts` | 단계 표시 바 (진행 점, 색상 분기) |
| `game/config.ts` | 상수 (크기, 색상, 좌표, 말풍선 높이 제한, 타이핑 속도) |
| `components/DialogueControls.tsx` | 대화 속도(1x/2x/4x/즉시) + 스킵 버튼 React UI |
| `demo/demo-scenarios.ts` | 데모 시나리오 2개 (형사/민사) |
| `types/index.ts` | 타입 정의 (DialogueSpeed 포함) + 단계 상수 (법적 근거 포함) |
