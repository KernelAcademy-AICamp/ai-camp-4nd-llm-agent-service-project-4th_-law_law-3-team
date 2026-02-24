/** 로비/설정 씬 (배경 이미지 또는 Graphics API fallback, 캐릭터 입장 애니메이션) */

import Phaser from 'phaser'
import {
  GAME_WIDTH,
  GAME_HEIGHT,
  LOBBY_CHARACTER_POSITIONS,
  LOBBY_ENTRANCE_SEQUENCE,
  ENTRANCE_WALK_DURATION,
  LOBBY_FLAGS,
} from './config'
import { CharacterBase } from './sprites/CharacterBase'
import { eventBus } from './EventBus'
import { ASSET_KEYS, hasTexture } from './AssetConfig'
import { AudioManager } from './AudioManager'

export class LobbyScene extends Phaser.Scene {
  private lobbyCharacters: CharacterBase[] = []
  private entranceTweens: Phaser.Tweens.Tween[] = []
  private delayedCalls: Phaser.Time.TimerEvent[] = []
  private unsubscribe: (() => void) | null = null
  private audioManager: AudioManager | null = null

  constructor() {
    super({ key: 'LobbyScene' })
  }

  create(): void {
    this.cameras.main.setBackgroundColor(0x87ceeb)
    this.audioManager = new AudioManager(this)

    // Phase 1: 배경 (이미지 또는 Graphics fallback)
    if (hasTexture(this, ASSET_KEYS.BG_LOBBY)) {
      this.add
        .image(GAME_WIDTH / 2, GAME_HEIGHT / 2, ASSET_KEYS.BG_LOBBY)
        .setDisplaySize(GAME_WIDTH, GAME_HEIGHT)
    } else {
      this.drawCourthouseGraphics()
    }

    // 국기 애니메이션 오버레이
    this.createFlags()

    // 하단 반투명 오버레이 (캐릭터 가시성 확보)
    const overlay = this.add.graphics()
    overlay.fillStyle(0x000000, 0.2)
    overlay.fillRect(0, GAME_HEIGHT * 0.6, GAME_WIDTH, GAME_HEIGHT * 0.4)

    // 상단 반투명 오버레이 (타이틀 가독성)
    const topOverlay = this.add.graphics()
    topOverlay.fillStyle(0x000000, 0.25)
    topOverlay.fillRect(0, 0, GAME_WIDTH, 55)

    // 타이틀
    this.add
      .text(GAME_WIDTH / 2, 15, '모의 법정', {
        fontSize: '28px',
        color: '#ffffff',
        fontFamily: 'sans-serif',
        fontStyle: 'bold',
        stroke: '#1a237e',
        strokeThickness: 4,
        shadow: {
          offsetX: 2,
          offsetY: 2,
          color: '#000000',
          blur: 4,
          fill: true,
          stroke: true,
        },
      })
      .setOrigin(0.5, 0)

    // 면책 고지
    this.add
      .text(
        GAME_WIDTH / 2,
        GAME_HEIGHT - 65,
        '본 서비스는 교육 목적의 모의재판이며, 실제 법률 자문이 아닙니다.',
        {
          fontSize: '11px',
          color: '#ffcdd2',
          fontFamily: 'sans-serif',
          align: 'center',
          stroke: '#000000',
          strokeThickness: 2,
        }
      )
      .setOrigin(0.5, 0)

    // 안내 텍스트
    this.add
      .text(
        GAME_WIDTH / 2,
        GAME_HEIGHT - 38,
        '아래 설정을 완료하면 재판이 시작됩니다',
        {
          fontSize: '13px',
          color: '#e0e0e0',
          fontFamily: 'sans-serif',
          stroke: '#000000',
          strokeThickness: 2,
        }
      )
      .setOrigin(0.5, 0)

    // 캐릭터 입장 시퀀스
    this.playEntranceSequence()

    // Phase 4: BGM
    this.audioManager.playBGM(ASSET_KEYS.BGM_LOBBY)

    // EventBus: setup:complete 수신 시 CourtScene 전환
    this.unsubscribe = eventBus.on('setup:complete', (data) => {
      this.scene.start('CourtScene', {
        caseType: data.caseType,
        userRole: data.userRole,
        caseSummary: data.caseSummary,
      })
    })

    // Scene 전환 시 cleanup
    this.events.once('shutdown', this.shutdown, this)

    eventBus.emit('game:ready', {} as Record<string, never>)
  }

  // ── 국기 펄럭임 오버레이 ──

  private readonly FLAG_ASSET_MAP: Record<string, string> = {
    korea: ASSET_KEYS.FLAG_KOREA,
    court: ASSET_KEYS.FLAG_COURT,
  }

  private createFlags(): void {
    for (const flag of LOBBY_FLAGS) {
      const assetKey = this.FLAG_ASSET_MAP[flag.id]
      if (!assetKey || !hasTexture(this, assetKey)) continue

      const sprite = this.add.sprite(flag.x, flag.y, assetKey)
      sprite.setOrigin(0.15, 0)
      sprite.setScale(flag.scale)
      sprite.setDepth(flag.depth)
      sprite.play(`${assetKey}-wave`)
    }
  }

  // ── Graphics API Fallback (에셋 파일 없을 때 사용) ──

  private drawCourthouseGraphics(): void {
    const g = this.add.graphics()

    // 하늘 그라데이션
    g.fillStyle(0x87ceeb, 1)
    g.fillRect(0, 0, GAME_WIDTH, 100)
    g.fillStyle(0x7ec8e3, 1)
    g.fillRect(0, 100, GAME_WIDTH, 90)
    g.fillStyle(0x6bb5d9, 1)
    g.fillRect(0, 190, GAME_WIDTH, 80)

    // 구름
    this.drawCloud(g, 120, 50, 1.0)
    this.drawCloud(g, 580, 35, 0.8)
    this.drawCloud(g, 380, 75, 0.6)

    // 잔디
    g.fillStyle(0x7cb342, 1)
    g.fillRect(0, 270, GAME_WIDTH, GAME_HEIGHT - 270)
    g.fillStyle(0x689f38, 1)
    g.fillRect(0, 340, GAME_WIDTH, GAME_HEIGHT - 340)

    // 건물 본체
    const bx = 180
    const by = 100
    const bw = 440
    const bh = 180

    g.fillStyle(0x000000, 0.08)
    g.fillRect(bx + 4, by + 4, bw, bh)
    g.fillStyle(0xe8e0d0, 1)
    g.fillRect(bx, by, bw, bh)
    g.fillStyle(0x8b7b6b, 1)
    g.fillRect(bx - 10, by - 6, bw + 20, 10)

    // 중앙 페디먼트
    const pedX = GAME_WIDTH / 2 - 70
    g.fillStyle(0xe8e0d0, 1)
    g.fillRect(pedX, by - 30, 140, 30)
    g.fillStyle(0x8b7b6b, 1)
    g.fillTriangle(pedX - 10, by - 30, pedX + 150, by - 30, GAME_WIDTH / 2, by - 55)

    // 기둥 (6개)
    const pillarCount = 6
    const pillarMargin = 30
    const pillarGap = (bw - pillarMargin * 2) / (pillarCount - 1)
    for (let i = 0; i < pillarCount; i++) {
      const px = bx + pillarMargin + pillarGap * i
      g.fillStyle(0xf5ede0, 1)
      g.fillRect(px - 7, by + 15, 14, bh - 25)
      g.fillStyle(0xd5ccc0, 1)
      g.fillRect(px - 9, by + 15, 18, 6)
      g.fillRect(px - 9, by + bh - 14, 18, 6)
    }

    // 창문
    for (let row = 0; row < 2; row++) {
      const wy = by + 35 + row * 60
      for (let col = 0; col < 5; col++) {
        const wx = bx + 55 + col * 85
        g.fillStyle(0x90caf9, 0.6)
        g.fillRect(wx, wy, 20, 30)
        g.lineStyle(1, 0xb0a090, 1)
        g.strokeRect(wx, wy, 20, 30)
      }
    }

    // 정문
    const doorW = 40
    const doorH = 60
    const doorX = GAME_WIDTH / 2 - doorW / 2
    const doorY = by + bh - doorH
    g.fillStyle(0x5d4037, 1)
    g.fillRect(doorX, doorY, doorW, doorH)
    g.lineStyle(1, 0x4e342e, 1)
    g.lineBetween(GAME_WIDTH / 2, doorY, GAME_WIDTH / 2, doorY + doorH)

    // 계단
    const stairColors = [0xd0c8b8, 0xc8c0b0, 0xc0b8a8, 0xb8b0a0]
    for (let i = 0; i < stairColors.length; i++) {
      g.fillStyle(stairColors[i], 1)
      const sw = 100 + i * 30
      g.fillRect(GAME_WIDTH / 2 - sw / 2, by + bh + i * 7, sw, 8)
    }

    // 현판
    g.fillStyle(0x4e342e, 0.85)
    g.fillRoundedRect(GAME_WIDTH / 2 - 42, by + 18, 84, 26, 3)
    this.add
      .text(GAME_WIDTH / 2, by + 31, '대법원', {
        fontSize: '14px',
        color: '#FFD700',
        fontFamily: 'sans-serif',
        fontStyle: 'bold',
      })
      .setOrigin(0.5, 0.5)

    // 나무
    this.drawTree(g, 70, 235)
    this.drawTree(g, 130, 255)
    this.drawTree(g, 730, 235)
    this.drawTree(g, 670, 255)

    // 관목
    g.fillStyle(0x558b2f, 1)
    g.fillEllipse(210, 285, 45, 20)
    g.fillEllipse(590, 285, 45, 20)
    g.fillStyle(0x4caf50, 1)
    g.fillEllipse(170, 290, 35, 16)
    g.fillEllipse(630, 290, 35, 16)

    // 진입로
    g.fillStyle(0xbdb8a8, 1)
    const pathTop = by + bh + 28
    g.beginPath()
    g.moveTo(GAME_WIDTH / 2 - 50, pathTop)
    g.lineTo(GAME_WIDTH / 2 + 50, pathTop)
    g.lineTo(GAME_WIDTH / 2 + 100, GAME_HEIGHT)
    g.lineTo(GAME_WIDTH / 2 - 100, GAME_HEIGHT)
    g.closePath()
    g.fillPath()

    g.lineStyle(1, 0xa8a090, 0.5)
    g.lineBetween(GAME_WIDTH / 2, pathTop + 5, GAME_WIDTH / 2, GAME_HEIGHT)
  }

  private drawCloud(g: Phaser.GameObjects.Graphics, x: number, y: number, scale: number): void {
    g.fillStyle(0xffffff, 0.5)
    g.fillEllipse(x, y, 80 * scale, 28 * scale)
    g.fillEllipse(x + 25 * scale, y - 7 * scale, 50 * scale, 20 * scale)
    g.fillEllipse(x - 20 * scale, y + 3 * scale, 40 * scale, 16 * scale)
  }

  private drawTree(g: Phaser.GameObjects.Graphics, x: number, y: number): void {
    g.fillStyle(0x795548, 1)
    g.fillRect(x - 4, y, 8, 35)
    g.fillStyle(0x4caf50, 1)
    g.fillEllipse(x, y - 8, 45, 35)
    g.fillStyle(0x388e3c, 1)
    g.fillEllipse(x + 3, y - 3, 35, 28)
  }

  // ── 캐릭터 입장 ──

  private playEntranceSequence(): void {
    LOBBY_ENTRANCE_SEQUENCE.forEach((entry) => {
      const targetPos = LOBBY_CHARACTER_POSITIONS[entry.role]
      if (!targetPos) return

      const character = new CharacterBase(this, entry.startX, entry.startY, entry.role)
      character.setAlpha(0)
      this.lobbyCharacters.push(character)

      const walkDirection = this.getWalkDirection(entry.startX, entry.startY, targetPos)

      const timerEvent = this.time.delayedCall(entry.delay, () => {
        character.setAlpha(1)
        character.playWalkAnimation(walkDirection)

        const moveTween = this.tweens.add({
          targets: character,
          x: targetPos.x,
          y: targetPos.y,
          duration: ENTRANCE_WALK_DURATION,
          ease: 'Power2',
          onComplete: () => {
            character.stopWalkAnimation()
          },
        })
        this.entranceTweens.push(moveTween)
      })
      this.delayedCalls.push(timerEvent)
    })
  }

  private getWalkDirection(
    startX: number,
    startY: number,
    target: { x: number; y: number }
  ): 'left' | 'right' | 'up' | 'down' {
    const dx = Math.abs(target.x - startX)
    const dy = Math.abs(target.y - startY)
    if (dy > dx) {
      return target.y > startY ? 'down' : 'up'
    }
    return target.x > startX ? 'right' : 'left'
  }

  shutdown(): void {
    if (this.unsubscribe) {
      this.unsubscribe()
      this.unsubscribe = null
    }
    this.audioManager?.destroy()
    this.audioManager = null
    this.entranceTweens.forEach((tween) => tween.stop())
    this.entranceTweens = []
    this.delayedCalls.forEach((timer) => timer.destroy())
    this.delayedCalls = []
    this.lobbyCharacters.forEach((char) => char.destroy())
    this.lobbyCharacters = []
  }
}
