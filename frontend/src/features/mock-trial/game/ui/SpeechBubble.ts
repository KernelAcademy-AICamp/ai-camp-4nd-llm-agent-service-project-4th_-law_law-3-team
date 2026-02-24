/** 말풍선 UI (NineSlice 이미지 또는 Graphics API fallback + 타이핑 효과) */

import Phaser from 'phaser'
import { GAME_WIDTH } from '../config'
import { ASSET_KEYS, hasTexture } from '../AssetConfig'

const BUBBLE_PADDING = 12
const BUBBLE_RADIUS = 8
const MAX_WIDTH = 260
const TYPING_SPEED = 30
const FONT_SIZE = 13
const NAME_FONT_SIZE = 11
const SCREEN_MARGIN = 10
const NINE_SLICE_BORDER = 8

const EMOTION_EMOJI_MAP: Record<string, string> = {
  neutral: '💬',
  angry: '💢',
  thinking: '❓',
  sad: '💧',
  confident: '✨',
  stern: '❗',
  recording: '✏️',
  judging: '🔨',
}

export class SpeechBubble extends Phaser.GameObjects.Container {
  private graphicsBg: Phaser.GameObjects.Graphics
  private nineSliceBg: Phaser.GameObjects.NineSlice | null = null
  private tailImage: Phaser.GameObjects.Image | null = null
  private nameText: Phaser.GameObjects.Text
  private textObject: Phaser.GameObjects.Text
  private fullText = ''
  private displayedLength = 0
  private typingTimer: Phaser.Time.TimerEvent | null = null
  private anchorX: number
  private useImageMode: boolean

  constructor(scene: Phaser.Scene, x: number, y: number, anchorX: number) {
    super(scene, x, y)
    this.anchorX = anchorX
    this.useImageMode = hasTexture(scene, ASSET_KEYS.UI_SPEECH_BUBBLE)

    this.graphicsBg = scene.add.graphics()

    if (this.useImageMode) {
      this.nineSliceBg = scene.add.nineslice(
        0,
        0,
        ASSET_KEYS.UI_SPEECH_BUBBLE,
        undefined,
        MAX_WIDTH,
        60,
        NINE_SLICE_BORDER,
        NINE_SLICE_BORDER,
        NINE_SLICE_BORDER,
        NINE_SLICE_BORDER
      )
      this.nineSliceBg.setOrigin(0, 0)

      if (hasTexture(scene, ASSET_KEYS.UI_SPEECH_TAIL)) {
        this.tailImage = scene.add.image(0, 0, ASSET_KEYS.UI_SPEECH_TAIL)
        this.tailImage.setOrigin(0.5, 0)
      }
    }

    this.nameText = scene.add.text(BUBBLE_PADDING, BUBBLE_PADDING, '', {
      fontSize: `${NAME_FONT_SIZE}px`,
      color: '#555555',
      fontFamily: 'sans-serif',
      fontStyle: 'bold',
    })
    this.textObject = scene.add.text(
      BUBBLE_PADDING,
      BUBBLE_PADDING + NAME_FONT_SIZE + 4,
      '',
      {
        fontSize: `${FONT_SIZE}px`,
        color: '#1a1a1a',
        fontFamily: 'sans-serif',
        wordWrap: { width: MAX_WIDTH - BUBBLE_PADDING * 2 },
        lineSpacing: 4,
      }
    )

    const children: Phaser.GameObjects.GameObject[] = [this.graphicsBg]
    if (this.nineSliceBg) children.push(this.nineSliceBg)
    if (this.tailImage) children.push(this.tailImage)
    children.push(this.nameText, this.textObject)
    this.add(children)

    this.setVisible(false)
    scene.add.existing(this)
  }

  show(name: string, text: string, emotion?: string, immediate = false): void {
    if (!this.scene || !this.nameText) return
    const emoji = EMOTION_EMOJI_MAP[emotion ?? 'neutral'] ?? '😐'
    this.nameText.setText(`${name} ${emoji}`)
    this.showText(text, immediate)
  }

  showText(text: string, immediate = false): void {
    if (!this.scene || !this.textObject) return
    this.fullText = text
    this.displayedLength = 0
    this.setVisible(true)
    this.stopTyping()

    if (immediate) {
      this.textObject.setText(text)
      this.drawBackground()
      return
    }

    this.textObject.setText('')
    this.typingTimer = this.scene.time.addEvent({
      delay: TYPING_SPEED,
      callback: this.typeNextCharacter,
      callbackScope: this,
      repeat: text.length - 1,
    })
  }

  hide(): void {
    this.stopTyping()
    this.setVisible(false)
  }

  private typeNextCharacter(): void {
    if (!this.scene || !this.textObject) return
    this.displayedLength += 1
    this.textObject.setText(this.fullText.slice(0, this.displayedLength))
    this.drawBackground()
  }

  private drawBackground(): void {
    if (!this.scene || !this.nameText || !this.textObject) return
    const contentWidth = Math.max(this.nameText.width, this.textObject.width)
    const textWidth = Math.min(contentWidth + BUBBLE_PADDING * 2, MAX_WIDTH)
    const nameHeight = this.nameText.text ? this.nameText.height + 4 : 0
    const textHeight = nameHeight + this.textObject.height + BUBBLE_PADDING * 2

    // 컨테이너 x 정렬
    const idealX = this.anchorX - textWidth / 2
    this.setX(
      Math.max(SCREEN_MARGIN, Math.min(GAME_WIDTH - textWidth - SCREEN_MARGIN, idealX))
    )

    const tailX = Math.max(12, Math.min(textWidth - 12, this.anchorX - this.x))

    if (this.useImageMode && this.nineSliceBg) {
      this.graphicsBg.clear()
      this.nineSliceBg.setSize(textWidth, textHeight)
      this.nineSliceBg.setVisible(true)

      if (this.tailImage) {
        this.tailImage.setPosition(tailX, textHeight)
        this.tailImage.setVisible(true)
      }
    } else {
      this.nineSliceBg?.setVisible(false)
      this.tailImage?.setVisible(false)
      this.drawGraphicsBackground(textWidth, textHeight, tailX)
    }
  }

  /** Graphics API fallback */
  private drawGraphicsBackground(
    textWidth: number,
    textHeight: number,
    tailX: number
  ): void {
    this.graphicsBg.clear()

    this.graphicsBg.fillStyle(0xffffff, 0.95)
    this.graphicsBg.lineStyle(2, 0x333333, 1)
    this.graphicsBg.fillRoundedRect(0, 0, textWidth, textHeight, BUBBLE_RADIUS)
    this.graphicsBg.strokeRoundedRect(0, 0, textWidth, textHeight, BUBBLE_RADIUS)

    const tailY = textHeight
    this.graphicsBg.fillStyle(0xffffff, 0.95)
    this.graphicsBg.fillTriangle(tailX - 6, tailY, tailX + 6, tailY, tailX, tailY + 10)
    this.graphicsBg.lineStyle(2, 0x333333, 1)
    this.graphicsBg.lineBetween(tailX - 6, tailY, tailX, tailY + 10)
    this.graphicsBg.lineBetween(tailX + 6, tailY, tailX, tailY + 10)
  }

  private stopTyping(): void {
    if (this.typingTimer) {
      this.typingTimer.destroy()
      this.typingTimer = null
    }
  }

  destroy(fromScene?: boolean): void {
    this.stopTyping()
    super.destroy(fromScene)
  }
}
