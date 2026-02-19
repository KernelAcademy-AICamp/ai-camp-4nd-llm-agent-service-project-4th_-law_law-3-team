/** 단계 표시 바 (Phaser Graphics) */

import Phaser from 'phaser'

const BAR_HEIGHT = 28
const PADDING_X = 16
const DOT_RADIUS = 6
const DOT_SPACING = 100

export class StageIndicator extends Phaser.GameObjects.Container {
  private background: Phaser.GameObjects.Graphics
  private stageTexts: Phaser.GameObjects.Text[] = []
  private dots: Phaser.GameObjects.Arc[] = []
  private currentIndex = 0
  private stageNames: string[] = []

  constructor(
    scene: Phaser.Scene,
    x: number,
    y: number,
    width: number,
    stageNames: string[]
  ) {
    super(scene, x, y)
    this.stageNames = stageNames

    this.background = scene.add.graphics()
    this.background.fillStyle(0x263238, 0.85)
    this.background.fillRoundedRect(0, 0, width, BAR_HEIGHT, 4)
    this.add(this.background)

    const totalWidth = (stageNames.length - 1) * DOT_SPACING
    const startX = (width - totalWidth) / 2

    stageNames.forEach((name, index) => {
      const dotX = startX + index * DOT_SPACING
      const dotY = BAR_HEIGHT / 2

      const dot = scene.add.circle(dotX, dotY, DOT_RADIUS, 0x666666)
      this.dots.push(dot)
      this.add(dot)

      const text = scene.add.text(dotX, dotY + DOT_RADIUS + 2, name, {
        fontSize: '9px',
        color: '#aaaaaa',
        fontFamily: 'sans-serif',
        align: 'center',
      })
      text.setOrigin(0.5, 0)
      this.stageTexts.push(text)
      this.add(text)
    })

    this.updateVisual()
    scene.add.existing(this)
  }

  setCurrentStage(index: number): void {
    this.currentIndex = Math.max(0, Math.min(index, this.stageNames.length - 1))
    this.updateVisual()
  }

  private updateVisual(): void {
    this.dots.forEach((dot, index) => {
      if (index < this.currentIndex) {
        dot.setFillStyle(0x4caf50) // 완료: 녹색
      } else if (index === this.currentIndex) {
        dot.setFillStyle(0xffd700) // 현재: 금색
        dot.setStrokeStyle(2, 0xffffff)
      } else {
        dot.setFillStyle(0x666666) // 미진행: 회색
        dot.setStrokeStyle(0)
      }
    })

    this.stageTexts.forEach((text, index) => {
      if (index === this.currentIndex) {
        text.setColor('#ffffff')
        text.setFontStyle('bold')
      } else if (index < this.currentIndex) {
        text.setColor('#4caf50')
        text.setFontStyle('')
      } else {
        text.setColor('#aaaaaa')
        text.setFontStyle('')
      }
    })
  }
}
