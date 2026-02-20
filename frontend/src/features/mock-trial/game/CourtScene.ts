/** 메인 법정 씬 (캐릭터 배치, 말풍선, 단계 표시) */

import Phaser from 'phaser'
import {
  GAME_WIDTH,
  GAME_HEIGHT,
  COURT_BACKGROUND_COLOR,
  CHARACTER_POSITIONS,
} from './config'
import { CharacterBase } from './sprites/CharacterBase'
import { JuryPanel } from './sprites/JuryPanel'
import { SpeechBubble } from './ui/SpeechBubble'
import { StageIndicator } from './ui/StageIndicator'
import { eventBus } from './EventBus'
import { CRIMINAL_STAGES, CIVIL_STAGES } from '../types'

interface CourtSceneData {
  caseType: string
  userRole: string
  caseSummary: string
}

export class CourtScene extends Phaser.Scene {
  private characters: Map<string, CharacterBase> = new Map()
  private speechBubbles: Map<string, SpeechBubble> = new Map()
  private stageIndicator: StageIndicator | null = null
  private juryPanel: JuryPanel | null = null
  private caseType = 'criminal'
  private unsubscribers: (() => void)[] = []

  constructor() {
    super({ key: 'CourtScene' })
  }

  init(data: CourtSceneData): void {
    this.caseType = data.caseType || 'criminal'
  }

  create(): void {
    this.cameras.main.setBackgroundColor(COURT_BACKGROUND_COLOR)
    this.drawCourtBackground()
    this.createCharacters()
    this.createSpeechBubbles()
    this.createStageIndicator()
    this.juryPanel = new JuryPanel(this)
    this.setupEventListeners()
  }

  private drawCourtBackground(): void {
    const graphics = this.add.graphics()

    // 판사석 (상단 플랫폼)
    graphics.fillStyle(0x5d4037, 1)
    graphics.fillRect(300, 90, 200, 60)
    graphics.fillStyle(0x6d4c41, 1)
    graphics.fillRect(310, 80, 180, 15)

    // 서기석
    graphics.fillStyle(0x795548, 1)
    graphics.fillRect(60, 110, 80, 40)

    // 검사석 (좌측)
    graphics.fillStyle(0x795548, 1)
    graphics.fillRect(120, 230, 160, 10)

    // 변호사석 (우측 - 배심원석 공간 확보를 위해 좌표 조정)
    graphics.fillStyle(0x795548, 1)
    graphics.fillRect(480, 230, 160, 10)

    // 피고인석 (중앙 하단)
    graphics.fillStyle(0x795548, 1)
    graphics.fillRect(330, 310, 140, 10)

    // 배심원석 배경 (우측)
    graphics.fillStyle(0x8d6e63, 0.4)
    graphics.fillRoundedRect(686, 120, 96, 100, 6)
    graphics.lineStyle(1, 0x795548, 0.6)
    graphics.strokeRoundedRect(686, 120, 96, 100, 6)

    this.add
      .text(734, 228, '배심원석', {
        fontSize: '9px',
        color: '#6d4c41',
        fontFamily: 'sans-serif',
      })
      .setOrigin(0.5, 0)

    // 방청석 구분선
    graphics.lineStyle(2, 0x8d6e63, 0.5)
    graphics.lineBetween(50, 400, 750, 400)

    // 법정 텍스트
    this.add
      .text(GAME_WIDTH / 2, 20, '대한민국 법원', {
        fontSize: '16px',
        color: '#1a237e',
        fontFamily: 'sans-serif',
        fontStyle: 'bold',
      })
      .setOrigin(0.5, 0)
  }

  private createCharacters(): void {
    const roles = ['judge', 'prosecutor', 'attorney', 'defendant', 'clerk']
    roles.forEach((role) => {
      const position = CHARACTER_POSITIONS[role]
      const character = new CharacterBase(this, position.x, position.y, role)
      this.characters.set(role, character)
    })
  }

  private createSpeechBubbles(): void {
    this.characters.forEach((_, role) => {
      const position = CHARACTER_POSITIONS[role]
      const bubbleY = position.y - 70
      const bubble = new SpeechBubble(this, position.x - 130, bubbleY)
      this.speechBubbles.set(role, bubble)
    })
  }

  private createStageIndicator(): void {
    const stages =
      this.caseType === 'criminal' ? CRIMINAL_STAGES : CIVIL_STAGES
    const stageNames = stages.map((stage) => stage.name)

    this.stageIndicator = new StageIndicator(
      this,
      50,
      GAME_HEIGHT - 34,
      GAME_WIDTH - 100,
      stageNames
    )
  }

  private setupEventListeners(): void {
    // agent:speak -> 말풍선 표시 + 캐릭터 애니메이션 + 배심원 반응
    this.unsubscribers.push(
      eventBus.on('agent:speak', (data) => {
        // 이전 말풍선 숨기기
        this.speechBubbles.forEach((bubble) => bubble.hide())
        this.characters.forEach((char) => char.setSpeaking(false))

        const bubble = this.speechBubbles.get(data.agent)
        const character = this.characters.get(data.agent)
        if (bubble) {
          bubble.showText(data.text, !data.streaming)
        }
        if (character) {
          character.setSpeaking(true)
        }

        this.juryPanel?.reactToSpeech(data.agent, data.text)
      })
    )

    // stage:change -> 단계 표시바 업데이트
    this.unsubscribers.push(
      eventBus.on('stage:change', (data) => {
        this.stageIndicator?.setCurrentStage(data.stageNumber - 1)
      })
    )

    // agent:animate -> 캐릭터 상태 변경
    this.unsubscribers.push(
      eventBus.on('agent:animate', (data) => {
        const character = this.characters.get(data.agent)
        if (character) {
          character.setSpeaking(data.animation === 'speak')
          character.highlight(data.animation === 'react')
        }
      })
    )
  }

  shutdown(): void {
    this.unsubscribers.forEach((unsub) => unsub())
    this.unsubscribers = []
    this.juryPanel?.destroy()
    this.juryPanel = null
  }
}
