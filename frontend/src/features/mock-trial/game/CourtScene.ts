/** 메인 법정 씬 (배경 이미지/타일맵/Graphics fallback, 캐릭터 배치, 말풍선, 단계 표시) */

import Phaser from 'phaser'
import {
  GAME_WIDTH,
  GAME_HEIGHT,
  COURT_BACKGROUND_COLOR,
  CHARACTER_POSITIONS,
  CHARACTER_NAMES,
  CHARACTER_FACING,
  BUBBLE_OFFSETS,
} from './config'
import { CharacterBase } from './sprites/CharacterBase'
import { JuryPanel } from './sprites/JuryPanel'
import { SpeechBubble } from './ui/SpeechBubble'
import { StageIndicator } from './ui/StageIndicator'
import { eventBus } from './EventBus'
import { CRIMINAL_STAGES, CIVIL_STAGES, DEFAULT_ROLE_EMOTION } from '../types'
import { ASSET_KEYS, hasTexture, TILEMAP_TILE_SIZE } from './AssetConfig'
import { AudioManager } from './AudioManager'

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
  private audioManager: AudioManager | null = null

  constructor() {
    super({ key: 'CourtScene' })
  }

  init(data: CourtSceneData): void {
    this.caseType = data.caseType || 'criminal'
  }

  create(): void {
    this.cameras.main.setBackgroundColor(COURT_BACKGROUND_COLOR)
    this.audioManager = new AudioManager(this)

    this.drawCourtBackground()
    this.createCharacters()
    this.createSpeechBubbles()
    this.createStageIndicator()
    this.juryPanel = new JuryPanel(this)

    // Phase 4: BGM
    this.audioManager.playBGM(ASSET_KEYS.BGM_COURT)

    // 판사 입장 연출 (완료 후 이벤트 리스너 등록)
    this.playJudgeEntrance()
  }

  private drawCourtBackground(): void {
    // Phase 3: 타일맵 우선
    if (this.cache.tilemap.exists(ASSET_KEYS.TILEMAP_COURT) && hasTexture(this, ASSET_KEYS.TILESET_COURT)) {
      this.createTilemap()
      return
    }

    // Phase 1: 배경 이미지
    if (hasTexture(this, ASSET_KEYS.BG_COURT)) {
      this.add
        .image(GAME_WIDTH / 2, GAME_HEIGHT / 2, ASSET_KEYS.BG_COURT)
        .setDisplaySize(GAME_WIDTH, GAME_HEIGHT)
        .setDepth(0)

      // 오버레이 (판사석 등 입체 구조물)
      if (hasTexture(this, ASSET_KEYS.BG_COURT_OVERLAY)) {
        this.add
          .image(GAME_WIDTH / 2, GAME_HEIGHT / 2, ASSET_KEYS.BG_COURT_OVERLAY)
          .setDisplaySize(GAME_WIDTH, GAME_HEIGHT)
          .setDepth(11)
      }

      // 최소 텍스트 요소만 유지
      this.addCourtLabels()
      return
    }

    // Fallback: Graphics API
    this.drawCourtBackgroundGraphics()
  }

  /** Phase 3: 타일맵 기반 배경 */
  private createTilemap(): void {
    const map = this.make.tilemap({ key: ASSET_KEYS.TILEMAP_COURT })
    const tileset = map.addTilesetImage('court-tiles', ASSET_KEYS.TILESET_COURT)
    if (!tileset) return

    // 레이어 이름은 Tiled 에디터에서 정의한 것과 일치해야 함
    const floorLayer = map.createLayer('floor', tileset)
    floorLayer?.setDepth(0)

    const furnitureLayer = map.createLayer('furniture', tileset)
    furnitureLayer?.setDepth(1)

    const decorationLayer = map.createLayer('decoration', tileset)
    decorationLayer?.setDepth(2)

    // Object Layer에서 캐릭터 스폰 포인트 읽기
    const spawnLayer = map.getObjectLayer('spawns')
    if (spawnLayer) {
      for (const obj of spawnLayer.objects) {
        const roleProp = obj.properties?.find(
          (p: { name: string }) => p.name === 'role'
        )
        const role = roleProp?.value as string | undefined
        if (role && CHARACTER_POSITIONS[role] && obj.x != null && obj.y != null) {
          CHARACTER_POSITIONS[role] = {
            x: obj.x + TILEMAP_TILE_SIZE / 2,
            y: obj.y,
          }
        }
      }
    }

    this.addCourtLabels()
  }

  /** 법정 내 텍스트 레이블 (배경 방식과 무관하게 공통) */
  private addCourtLabels(): void {
    this.add
      .text(GAME_WIDTH / 2, 20, '대한민국 법원', {
        fontSize: '16px',
        color: '#1a237e',
        fontFamily: 'sans-serif',
        fontStyle: 'bold',
      })
      .setOrigin(0.5, 0)
      .setDepth(15)
  }

  /** Graphics API fallback (에셋 없을 때) */
  private drawCourtBackgroundGraphics(): void {
    const graphics = this.add.graphics()

    // 판사석
    graphics.fillStyle(0x5d4037, 1)
    graphics.fillRect(300, 90, 200, 60)
    graphics.fillStyle(0x6d4c41, 1)
    graphics.fillRect(310, 80, 180, 15)

    // 서기석
    graphics.fillStyle(0x795548, 1)
    graphics.fillRect(60, 110, 80, 40)

    // 검사석
    graphics.fillStyle(0x795548, 1)
    graphics.fillRect(120, 230, 160, 10)

    // 변호사석
    graphics.fillStyle(0x795548, 1)
    graphics.fillRect(480, 230, 160, 10)

    // 피고인석
    graphics.fillStyle(0x795548, 1)
    graphics.fillRect(330, 310, 140, 10)

    // 배심원석 배경
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

    this.addCourtLabels()
  }

  private createCharacters(): void {
    const roles = ['judge', 'prosecutor', 'attorney', 'defendant', 'clerk']
    roles.forEach((role) => {
      const position = CHARACTER_POSITIONS[role]
      const facing = CHARACTER_FACING[role] ?? 'down'
      const character = new CharacterBase(this, position.x, position.y, role, facing)
      character.setDepth(10)

      // 판사는 입장 시퀀스를 위해 화면 밖에서 대기
      if (role === 'judge') {
        character.setPosition(GAME_WIDTH + 60, position.y)
        character.setAlpha(0)
      }

      this.characters.set(role, character)
    })
  }

  private createSpeechBubbles(): void {
    this.characters.forEach((_, role) => {
      const position = CHARACTER_POSITIONS[role]
      const offset = BUBBLE_OFFSETS[role] ?? { x: 0, y: -80 }
      const bubbleY = position.y + offset.y
      const anchorX = position.x + offset.x
      const bubble = new SpeechBubble(this, anchorX, bubbleY, anchorX)
      bubble.setDepth(20)
      this.speechBubbles.set(role, bubble)
    })
  }

  private createStageIndicator(): void {
    const stages = this.caseType === 'criminal' ? CRIMINAL_STAGES : CIVIL_STAGES
    const stageNames = stages.map((stage) => stage.name)

    this.stageIndicator = new StageIndicator(
      this,
      50,
      GAME_HEIGHT - 34,
      GAME_WIDTH - 100,
      stageNames
    )
  }

  /** 판사 입장 연출: 서기 안내 → 판사 걸어서 입장 → 개정 선언 */
  private playJudgeEntrance(): void {
    const clerk = this.characters.get('clerk')
    const judge = this.characters.get('judge')
    const clerkBubble = this.speechBubbles.get('clerk')

    if (!clerk || !judge || !clerkBubble) return

    // Step 1: 서기가 판사 입장을 안내
    clerk.setSpeaking(true)
    clerkBubble.show('서기', '전원 기립! 판사님 입장하십니다.', 'stern', true)

    // Step 2: 판사가 오른쪽에서 걸어 들어옴
    this.time.delayedCall(2500, () => {
      clerkBubble.hide()
      clerk.setSpeaking(false)

      judge.setAlpha(1)
      judge.playWalkAnimation('left')

      const targetPos = CHARACTER_POSITIONS.judge
      this.tweens.add({
        targets: judge,
        x: targetPos.x,
        y: targetPos.y,
        duration: 1500,
        ease: 'Power2',
        onComplete: () => {
          judge.setPosition(targetPos.x, targetPos.y)
          judge.stopWalkAnimation()

          // Step 3: 서기가 개정을 선언
          this.time.delayedCall(500, () => {
            clerk.setSpeaking(true)
            clerkBubble.show('서기', '지금부터 재판을 시작하겠습니다.', 'recording', true)

            this.time.delayedCall(2500, () => {
              clerkBubble.hide()
              clerk.setSpeaking(false)

              // 입장 연출 완료 후 이벤트 리스너 등록 + React에 알림
              this.setupEventListeners()
              eventBus.emit('court:entrance:complete', {} as Record<string, never>)
            })
          })
        },
      })
    })
  }

  private setupEventListeners(): void {
    // agent:speak -> 말풍선 표시 + 캐릭터 애니메이션 + 배심원 반응
    this.unsubscribers.push(
      eventBus.on('agent:speak', (data) => {
        this.speechBubbles.forEach((bubble) => bubble.hide())
        this.characters.forEach((char) => {
          char.setSpeaking(false)
          char.clearEmotion()
        })

        const bubble = this.speechBubbles.get(data.agent)
        const character = this.characters.get(data.agent)
        const emotion = data.emotion ?? DEFAULT_ROLE_EMOTION[data.agent]
        if (bubble) {
          const agentName = CHARACTER_NAMES[data.agent] ?? data.agent
          bubble.show(agentName, data.text, emotion, !data.streaming)
        }
        if (character) {
          character.setSpeaking(true)
          if (data.text.length <= 80) {
            character.setEmotion(emotion)
          }
        }

        this.juryPanel?.reactToSpeech(data.agent, data.text)
      })
    )

    // stage:change -> 단계 표시바 업데이트 + SFX
    this.unsubscribers.push(
      eventBus.on('stage:change', (data) => {
        this.stageIndicator?.setCurrentStage(data.stageNumber - 1)
        this.audioManager?.playStageChange()
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
    this.audioManager?.destroy()
    this.audioManager = null
    this.juryPanel?.destroy()
    this.juryPanel = null
  }
}
