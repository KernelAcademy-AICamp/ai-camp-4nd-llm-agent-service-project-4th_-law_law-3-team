/** 캐릭터 베이스 클래스 (LPC 스프라이트 + 감정 아이콘 이미지/Graphics 겸용) */

import Phaser from 'phaser'
import { CHARACTER_NAMES } from '../config'
import { drawEmotionIcon } from './EmotionIconRenderer'
import { CHARACTER_SPRITE_KEYS, animKey } from './LpcSpriteConfig'
import { ASSET_KEYS, EMOTION_FRAME_INDEX, hasTexture } from '../AssetConfig'

export type CharacterState = 'idle' | 'speak' | 'react'

const LABEL_OFFSET_Y = 36
const EMOTION_ICON_X = 20
const EMOTION_ICON_Y = -50

export class CharacterBase extends Phaser.GameObjects.Container {
  private characterSprite: Phaser.GameObjects.Sprite
  private emotionGraphics: Phaser.GameObjects.Graphics | null = null
  private emotionSprite: Phaser.GameObjects.Sprite | null = null
  private label: Phaser.GameObjects.Text
  private role: string
  private isSpeaking = false
  private currentState: CharacterState = 'idle'
  private breathTween: Phaser.Tweens.Tween | null = null
  private emotionFloatTween: Phaser.Tweens.Tween | null = null
  private useEmotionSprite: boolean

  constructor(scene: Phaser.Scene, x: number, y: number, role: string) {
    super(scene, x, y)
    this.role = role
    this.useEmotionSprite = hasTexture(scene, ASSET_KEYS.EMOTION_ICONS)

    const name = CHARACTER_NAMES[role] ?? role
    const spriteKey = CHARACTER_SPRITE_KEYS[role] ?? CHARACTER_SPRITE_KEYS.judge

    // LPC 스프라이트 (발 기준 앵커)
    this.characterSprite = scene.add.sprite(0, 0, spriteKey)
    this.characterSprite.setOrigin(0.5, 1.0)

    const idleAnimKey = animKey(role, 'idle')
    if (scene.anims.exists(idleAnimKey)) {
      this.characterSprite.play(idleAnimKey)
    }

    // 감정 아이콘: 스프라이트시트 또는 Graphics fallback
    if (this.useEmotionSprite) {
      this.emotionSprite = scene.add.sprite(
        x + EMOTION_ICON_X,
        y + EMOTION_ICON_Y,
        ASSET_KEYS.EMOTION_ICONS,
        0
      )
      this.emotionSprite.setDepth(30)
      this.emotionSprite.setVisible(false)
      this.emotionSprite.setScale(0)
    } else {
      this.emotionGraphics = scene.add.graphics()
      this.emotionGraphics.setPosition(x + EMOTION_ICON_X, y + EMOTION_ICON_Y)
      this.emotionGraphics.setDepth(30)
      this.emotionGraphics.setVisible(false)
      this.emotionGraphics.setScale(0)
    }

    this.label = scene.add.text(0, LABEL_OFFSET_Y, name, {
      fontSize: '12px',
      color: '#333333',
      fontFamily: 'sans-serif',
      align: 'center',
    })
    this.label.setOrigin(0.5, 0)

    this.add([this.characterSprite, this.label])
    scene.add.existing(this)

    this.startBreathAnimation()
  }

  getRole(): string {
    return this.role
  }

  setSpeaking(speaking: boolean): void {
    if (!this.scene) return
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
    if (!this.scene) return
    if (isHighlighted) {
      this.renderState('react')
    } else if (!this.isSpeaking) {
      this.renderState('idle')
    }
  }

  playWalkAnimation(direction: 'left' | 'right' | 'up' | 'down'): void {
    const walkState = `walk_${direction}`
    const key = animKey(this.role, walkState)
    if (this.scene.anims.exists(key)) {
      this.characterSprite.play(key)
    }
  }

  stopWalkAnimation(): void {
    const idleKey = animKey(this.role, 'idle')
    if (this.scene.anims.exists(idleKey)) {
      this.characterSprite.play(idleKey)
    } else {
      this.characterSprite.stop()
    }
  }

  private renderState(state: CharacterState): void {
    if (this.currentState === state) return
    this.currentState = state
    const key = animKey(this.role, state)
    if (this.scene.anims.exists(key)) {
      this.characterSprite.play(key)
    }
  }

  setEmotion(emotion: string): void {
    if (!this.scene) return

    const target = this.getEmotionTarget()
    if (!target) return

    // 아이콘 내용 설정
    if (this.useEmotionSprite && this.emotionSprite) {
      const frameIndex = EMOTION_FRAME_INDEX[emotion] ?? 0
      this.emotionSprite.setFrame(frameIndex)
    } else if (this.emotionGraphics) {
      drawEmotionIcon(this.emotionGraphics, emotion)
    }

    target.setVisible(true)

    // pop-in 애니메이션 (절대 좌표)
    const absX = this.x + EMOTION_ICON_X
    const absY = this.y + EMOTION_ICON_Y
    this.scene.tweens.killTweensOf(target)
    target.setScale(0)
    target.setPosition(absX, absY)

    this.scene.tweens.add({
      targets: target,
      scaleX: 1,
      scaleY: 1,
      duration: 200,
      ease: 'Back.easeOut',
      onComplete: () => {
        this.startEmotionFloat()
      },
    })
  }

  clearEmotion(): void {
    if (this.emotionFloatTween) {
      this.emotionFloatTween.destroy()
      this.emotionFloatTween = null
    }
    const target = this.getEmotionTarget()
    if (target) {
      target.setVisible(false)
      target.setScale(0)
    }
  }

  private getEmotionTarget(): Phaser.GameObjects.GameObject & {
    setVisible: (v: boolean) => void
    setScale: (x: number, y?: number) => void
    setPosition: (x: number, y: number) => void
    y: number
  } | null {
    if (this.useEmotionSprite && this.emotionSprite) {
      return this.emotionSprite as unknown as ReturnType<typeof this.getEmotionTarget>
    }
    if (this.emotionGraphics) {
      return this.emotionGraphics as unknown as ReturnType<typeof this.getEmotionTarget>
    }
    return null
  }

  private startEmotionFloat(): void {
    if (this.emotionFloatTween) {
      this.emotionFloatTween.destroy()
    }
    const target = this.getEmotionTarget()
    if (!target) return

    const absY = this.y + EMOTION_ICON_Y
    this.emotionFloatTween = this.scene.tweens.add({
      targets: target,
      y: absY - 4,
      yoyo: true,
      repeat: -1,
      duration: 800,
      ease: 'Sine.easeInOut',
    })
  }

  destroy(fromScene?: boolean): void {
    if (this.emotionFloatTween) {
      this.emotionFloatTween.destroy()
      this.emotionFloatTween = null
    }
    this.emotionGraphics?.destroy()
    this.emotionSprite?.destroy()
    super.destroy(fromScene)
  }

  private startBreathAnimation(): void {
    if (this.breathTween) {
      this.breathTween.destroy()
    }
    this.breathTween = this.scene.tweens.add({
      targets: this.characterSprite,
      y: -2,
      yoyo: true,
      repeat: -1,
      duration: 1000,
      ease: 'Sine.easeInOut',
    })
  }
}
