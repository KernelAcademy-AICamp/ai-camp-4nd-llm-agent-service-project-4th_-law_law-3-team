/** 캐릭터 베이스 클래스 (Graphics API 기반 플레이스홀더) */

import Phaser from 'phaser'
import { CHARACTER_COLORS, CHARACTER_NAMES } from '../config'

const CHARACTER_WIDTH = 48
const CHARACTER_HEIGHT = 64
const LABEL_OFFSET_Y = 12

export class CharacterBase extends Phaser.GameObjects.Container {
  private bodyRect: Phaser.GameObjects.Rectangle
  private label: Phaser.GameObjects.Text
  private role: string
  private isSpeaking = false

  constructor(scene: Phaser.Scene, x: number, y: number, role: string) {
    super(scene, x, y)
    this.role = role

    const color = CHARACTER_COLORS[role] ?? 0x888888
    const name = CHARACTER_NAMES[role] ?? role

    this.bodyRect = scene.add.rectangle(0, 0, CHARACTER_WIDTH, CHARACTER_HEIGHT, color)
    this.bodyRect.setStrokeStyle(2, 0x000000)

    this.label = scene.add.text(0, CHARACTER_HEIGHT / 2 + LABEL_OFFSET_Y, name, {
      fontSize: '12px',
      color: '#333333',
      fontFamily: 'sans-serif',
      align: 'center',
    })
    this.label.setOrigin(0.5, 0)

    this.add([this.bodyRect, this.label])
    scene.add.existing(this)
  }

  getRole(): string {
    return this.role
  }

  setSpeaking(speaking: boolean): void {
    this.isSpeaking = speaking
    if (speaking) {
      this.bodyRect.setStrokeStyle(3, 0xffd700)
      this.scene.tweens.add({
        targets: this.bodyRect,
        scaleX: 1.05,
        scaleY: 1.05,
        yoyo: true,
        repeat: -1,
        duration: 400,
      })
    } else {
      this.scene.tweens.killTweensOf(this.bodyRect)
      this.bodyRect.setScale(1, 1)
      this.bodyRect.setStrokeStyle(2, 0x000000)
    }
  }

  highlight(isHighlighted: boolean): void {
    if (isHighlighted) {
      this.bodyRect.setStrokeStyle(3, 0x2196f3)
    } else if (!this.isSpeaking) {
      this.bodyRect.setStrokeStyle(2, 0x000000)
    }
  }
}
