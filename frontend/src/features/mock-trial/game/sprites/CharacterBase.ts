/** 캐릭터 베이스 클래스 (픽셀아트 기반) */

import Phaser from 'phaser'
import { CHARACTER_NAMES } from '../config'
import { drawPixelCharacter } from './PixelCharacterRenderer'
import type { CharacterState } from './PixelCharacterRenderer'

const LABEL_OFFSET_Y = 44

export class CharacterBase extends Phaser.GameObjects.Container {
  private characterGraphics: Phaser.GameObjects.Graphics
  private label: Phaser.GameObjects.Text
  private role: string
  private isSpeaking = false
  private currentState: CharacterState = 'idle'
  private breathTween: Phaser.Tweens.Tween | null = null

  constructor(scene: Phaser.Scene, x: number, y: number, role: string) {
    super(scene, x, y)
    this.role = role

    const name = CHARACTER_NAMES[role] ?? role

    this.characterGraphics = scene.add.graphics()
    drawPixelCharacter(this.characterGraphics, role, 'idle')

    this.label = scene.add.text(0, LABEL_OFFSET_Y, name, {
      fontSize: '12px',
      color: '#333333',
      fontFamily: 'sans-serif',
      align: 'center',
    })
    this.label.setOrigin(0.5, 0)

    this.add([this.characterGraphics, this.label])
    scene.add.existing(this)

    this.startBreathAnimation()
  }

  getRole(): string {
    return this.role
  }

  setSpeaking(speaking: boolean): void {
    this.isSpeaking = speaking
    if (speaking) {
      this.renderState('speak')
      this.scene.tweens.add({
        targets: this,
        scaleX: 1.05,
        scaleY: 1.05,
        yoyo: true,
        repeat: -1,
        duration: 400,
      })
    } else {
      this.scene.tweens.killTweensOf(this)
      this.setScale(1, 1)
      this.renderState('idle')
      this.startBreathAnimation()
    }
  }

  highlight(isHighlighted: boolean): void {
    if (isHighlighted) {
      this.renderState('react')
    } else if (!this.isSpeaking) {
      this.renderState('idle')
    }
  }

  private renderState(state: CharacterState): void {
    if (this.currentState === state) return
    this.currentState = state
    drawPixelCharacter(this.characterGraphics, this.role, state)
  }

  private startBreathAnimation(): void {
    if (this.breathTween) {
      this.breathTween.destroy()
    }
    this.breathTween = this.scene.tweens.add({
      targets: this.characterGraphics,
      y: -2,
      yoyo: true,
      repeat: -1,
      duration: 1000,
      ease: 'Sine.easeInOut',
    })
  }
}
