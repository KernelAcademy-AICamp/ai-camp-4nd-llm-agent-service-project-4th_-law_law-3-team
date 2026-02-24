/** 개별 배심원 스프라이트 (LPC 스프라이트 + tint 색상 구분) */

import Phaser from 'phaser'
import { JUROR_SCALE } from '../config'
import { CHARACTER_SPRITE_KEYS, animKey } from './LpcSpriteConfig'

/** 배심원 tint 색상 변형 4종 */
const JUROR_TINT_COLORS = [0xaabbff, 0xddaaff, 0xaaffdd, 0xffddaa]

export type JurorReaction = 'neutral' | 'nod' | 'shake' | 'surprise' | 'think' | 'whisper'

export class JurorSprite extends Phaser.GameObjects.Container {
  private sprite: Phaser.GameObjects.Sprite
  private reactionTween: Phaser.Tweens.Tween | null = null

  constructor(scene: Phaser.Scene, x: number, y: number, index: number) {
    super(scene, x, y)

    const spriteKey = CHARACTER_SPRITE_KEYS.juror
    this.sprite = scene.add.sprite(0, 0, spriteKey)
    this.sprite.setOrigin(0.5, 0.5)
    this.sprite.setScale(JUROR_SCALE)
    this.sprite.setTint(JUROR_TINT_COLORS[index % JUROR_TINT_COLORS.length])

    // idle 첫 프레임
    const idleKey = animKey('juror', 'idle')
    if (scene.anims.exists(idleKey)) {
      this.sprite.play(idleKey)
    }

    this.add(this.sprite)
    scene.add.existing(this)
  }

  react(reaction: JurorReaction): void {
    this.stopReaction()

    switch (reaction) {
      case 'nod':
        this.reactionTween = this.scene.tweens.add({
          targets: this,
          y: this.y - 4,
          yoyo: true,
          repeat: 1,
          duration: 300,
        })
        break
      case 'shake':
        this.reactionTween = this.scene.tweens.add({
          targets: this,
          x: this.x - 3,
          yoyo: true,
          repeat: 1,
          duration: 200,
        })
        break
      case 'surprise':
        this.reactionTween = this.scene.tweens.add({
          targets: this,
          scaleX: 1.1,
          scaleY: 1.1,
          yoyo: true,
          duration: 200,
        })
        break
      case 'think':
        this.reactionTween = this.scene.tweens.add({
          targets: this,
          angle: 2,
          yoyo: true,
          duration: 400,
        })
        break
      case 'whisper':
        this.reactionTween = this.scene.tweens.add({
          targets: this,
          x: this.x + 8,
          yoyo: true,
          duration: 300,
        })
        break
      case 'neutral':
      default:
        break
    }
  }

  stopReaction(): void {
    if (this.reactionTween) {
      this.reactionTween.destroy()
      this.reactionTween = null
    }
    this.setScale(1)
    this.setAngle(0)
  }
}
