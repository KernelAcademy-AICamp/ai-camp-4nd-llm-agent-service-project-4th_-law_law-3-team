/** 대화 큐 관리 + 속도 제어 + 스킵 컨트롤러 */

import type { SpeechBubble } from './ui/SpeechBubble'
import type { CharacterBase } from './sprites/CharacterBase'
import type { JuryPanel } from './sprites/JuryPanel'
import { CHARACTER_NAMES, BASE_TYPING_SPEED } from './config'
import { eventBus } from './EventBus'
import type { EmotionType, DialogueSpeed } from '../types'
import { DEFAULT_ROLE_EMOTION } from '../types'

interface DialogueEntry {
  agent: string
  text: string
  emotion?: EmotionType
}

/** 속도별 타이핑 딜레이 (ms/글자) */
const SPEED_MAP: Record<DialogueSpeed, number> = {
  normal: BASE_TYPING_SPEED,
  fast: Math.round(BASE_TYPING_SPEED / 2),
  faster: Math.round(BASE_TYPING_SPEED / 4),
  instant: 0,
}

export class DialogueController {
  private queue: DialogueEntry[] = []
  private isProcessing = false
  private currentBubble: SpeechBubble | null = null
  private currentAgent: string | null = null
  private speed: DialogueSpeed = 'normal'

  private speechBubbles: Map<string, SpeechBubble>
  private characters: Map<string, CharacterBase>
  private juryPanel: JuryPanel | null
  private scene: Phaser.Scene

  private spaceKey: Phaser.Input.Keyboard.Key | null = null

  constructor(
    scene: Phaser.Scene,
    speechBubbles: Map<string, SpeechBubble>,
    characters: Map<string, CharacterBase>,
    juryPanel: JuryPanel | null
  ) {
    this.scene = scene
    this.speechBubbles = speechBubbles
    this.characters = characters
    this.juryPanel = juryPanel
    this.setupKeyboard()
  }

  /** 대화 큐에 추가 */
  enqueue(entry: DialogueEntry): void {
    this.queue.push(entry)
    if (!this.isProcessing) {
      this.processNext()
    }
  }

  /** 큐에서 다음 대화 처리 */
  private processNext(): void {
    if (this.queue.length === 0) {
      this.isProcessing = false
      eventBus.emit('dialogue:queue:empty', {} as Record<string, never>)
      return
    }

    this.isProcessing = true
    const entry = this.queue.shift()!

    // 이전 말풍선/캐릭터 상태 초기화
    this.speechBubbles.forEach((bubble) => bubble.hide())
    this.characters.forEach((char) => {
      char.setSpeaking(false)
      char.clearEmotion()
    })

    const bubble = this.speechBubbles.get(entry.agent)
    const character = this.characters.get(entry.agent)
    const emotion = entry.emotion ?? DEFAULT_ROLE_EMOTION[entry.agent]

    if (bubble) {
      const agentName = CHARACTER_NAMES[entry.agent] ?? entry.agent
      const isInstant = this.speed === 'instant'
      bubble.setTypingSpeed(SPEED_MAP[this.speed])
      bubble.show(agentName, entry.text, emotion, isInstant)
      this.currentBubble = bubble
      this.currentAgent = entry.agent
    }

    if (character) {
      character.setSpeaking(true)
      if (entry.text.length <= 80) {
        character.setEmotion(emotion)
      }
    }

    this.juryPanel?.reactToSpeech(entry.agent, entry.text)

    // ChatPanel 호환: agent:speak 이벤트 emit
    eventBus.emit('agent:speak', {
      agent: entry.agent,
      text: entry.text,
      streaming: false,
      emotion,
    })
  }

  /** 스페이스바 / advance 동작 */
  handleAdvance(): void {
    if (!this.currentBubble) {
      // 말풍선이 없으면 다음 큐 처리
      this.processNext()
      return
    }

    const isAllDone = this.currentBubble.advance()
    if (isAllDone) {
      // 현재 대화 완료 → 다음 큐 아이템으로
      this.currentBubble = null
      this.currentAgent = null
      this.processNext()
    }
  }

  /** 속도 변경 */
  setSpeed(speed: DialogueSpeed): void {
    this.speed = speed
    // 현재 표시 중인 말풍선에도 즉시 적용
    if (this.currentBubble) {
      this.currentBubble.setTypingSpeed(SPEED_MAP[speed])
    }
  }

  /** 큐 전체 스킵 */
  skipAll(): void {
    // 현재 말풍선 완료
    if (this.currentBubble) {
      this.currentBubble.completeTyping()
      this.currentBubble.hide()
      this.currentBubble = null
      this.currentAgent = null
    }

    // 남은 큐의 각 항목에 대해 ChatPanel에 이벤트 emit (기록 유지)
    for (const entry of this.queue) {
      const emotion = entry.emotion ?? DEFAULT_ROLE_EMOTION[entry.agent]
      eventBus.emit('agent:speak', {
        agent: entry.agent,
        text: entry.text,
        streaming: false,
        emotion,
      })
    }

    this.queue = []
    this.isProcessing = false

    // 캐릭터 상태 초기화
    this.characters.forEach((char) => {
      char.setSpeaking(false)
      char.clearEmotion()
    })
    this.speechBubbles.forEach((bubble) => bubble.hide())

    eventBus.emit('dialogue:queue:empty', {} as Record<string, never>)
  }

  /** Space 키 바인딩 */
  private setupKeyboard(): void {
    if (!this.scene.input.keyboard) return
    this.spaceKey = this.scene.input.keyboard.addKey(
      Phaser.Input.Keyboard.KeyCodes.SPACE
    )
    this.spaceKey.on('down', () => {
      this.handleAdvance()
    })
  }

  destroy(): void {
    if (this.spaceKey) {
      this.spaceKey.removeAllListeners()
      this.spaceKey = null
    }
    this.queue = []
    this.currentBubble = null
    this.currentAgent = null
    this.isProcessing = false
  }
}
