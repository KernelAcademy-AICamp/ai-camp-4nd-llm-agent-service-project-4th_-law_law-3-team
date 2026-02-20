/** 개별 배심원 스프라이트 (8x10 그리드 = 32x40px) */

import Phaser from 'phaser'

const PIXEL = 4
const GRID_W = 8
const GRID_H = 10

const SKIN = 0xffcc99
const SKIN_DARK = 0xdba876
const HAIR_BLACK = 0x2c2c2c
const BLACK = 0x000000

/** 배심원 색상 변형 4종 */
const JUROR_SUIT_COLORS = [0x5c6bc0, 0x7e57c2, 0x26a69a, 0x8d6e63]

/** 배심원 기본 그리드 (색상 슬롯: -1 → 슈트 색상으로 치환) */
const JUROR_GRID: number[][] = [
  [0, 0, HAIR_BLACK, HAIR_BLACK, HAIR_BLACK, HAIR_BLACK, 0, 0],
  [0, HAIR_BLACK, SKIN, SKIN, SKIN, SKIN, HAIR_BLACK, 0],
  [0, SKIN, BLACK, SKIN, SKIN, BLACK, SKIN, 0],
  [0, SKIN, SKIN, SKIN_DARK, SKIN_DARK, SKIN, SKIN, 0],
  [0, 0, SKIN, SKIN, SKIN, SKIN, 0, 0],
  [0, -1, -1, -1, -1, -1, -1, 0],
  [0, -1, -1, -1, -1, -1, -1, 0],
  [-1, -1, -1, -1, -1, -1, -1, -1],
  [0, -1, -1, -1, -1, -1, -1, 0],
  [0, 0, -1, 0, 0, -1, 0, 0],
]

export type JurorReaction = 'neutral' | 'nod' | 'shake' | 'surprise' | 'think' | 'whisper'

export class JurorSprite extends Phaser.GameObjects.Container {
  private graphics: Phaser.GameObjects.Graphics
  private suitColor: number
  private reactionTween: Phaser.Tweens.Tween | null = null

  constructor(scene: Phaser.Scene, x: number, y: number, index: number) {
    super(scene, x, y)
    this.suitColor = JUROR_SUIT_COLORS[index % JUROR_SUIT_COLORS.length]

    this.graphics = scene.add.graphics()
    this.drawJuror()

    this.add(this.graphics)
    scene.add.existing(this)
  }

  private drawJuror(): void {
    this.graphics.clear()
    const offsetX = -(GRID_W * PIXEL) / 2
    const offsetY = -(GRID_H * PIXEL) / 2

    for (let row = 0; row < JUROR_GRID.length; row++) {
      for (let col = 0; col < JUROR_GRID[row].length; col++) {
        const val = JUROR_GRID[row][col]
        if (val === 0) continue
        const color = val === -1 ? this.suitColor : val
        this.graphics.fillStyle(color, 1)
        this.graphics.fillRect(
          offsetX + col * PIXEL,
          offsetY + row * PIXEL,
          PIXEL,
          PIXEL
        )
      }
    }
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
