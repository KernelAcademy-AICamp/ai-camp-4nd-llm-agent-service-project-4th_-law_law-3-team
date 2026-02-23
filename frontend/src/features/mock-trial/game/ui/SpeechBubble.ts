/** 말풍선 UI (Graphics API + 타이핑 효과) */

import Phaser from 'phaser'

const BUBBLE_PADDING = 12
const BUBBLE_RADIUS = 8
const MAX_WIDTH = 260
const TYPING_SPEED = 30 // ms per character
const FONT_SIZE = 13
const NAME_FONT_SIZE = 11

/** Phaser.js 파일 내 독립 이모지 맵 (게임풍 유니코드 심볼) */
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
  private background: Phaser.GameObjects.Graphics
  private nameText: Phaser.GameObjects.Text
  private textObject: Phaser.GameObjects.Text
  private fullText = ''
  private displayedLength = 0
  private typingTimer: Phaser.Time.TimerEvent | null = null

  constructor(scene: Phaser.Scene, x: number, y: number) {
    super(scene, x, y)

    this.background = scene.add.graphics()
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

    this.add([this.background, this.nameText, this.textObject])
    this.setVisible(false)
    scene.add.existing(this)
  }

  show(name: string, text: string, emotion?: string, immediate = false): void {
    const emoji = EMOTION_EMOJI_MAP[emotion ?? 'neutral'] ?? '😐'
    this.nameText.setText(`${name} ${emoji}`)
    this.showText(text, immediate)
  }

  showText(text: string, immediate = false): void {
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
    this.displayedLength += 1
    this.textObject.setText(this.fullText.slice(0, this.displayedLength))
    this.drawBackground()
  }

  private drawBackground(): void {
    this.background.clear()

    const contentWidth = Math.max(this.nameText.width, this.textObject.width)
    const textWidth = Math.min(contentWidth + BUBBLE_PADDING * 2, MAX_WIDTH)
    const nameHeight = this.nameText.text ? this.nameText.height + 4 : 0
    const textHeight = nameHeight + this.textObject.height + BUBBLE_PADDING * 2

    // 말풍선 배경
    this.background.fillStyle(0xffffff, 0.95)
    this.background.lineStyle(2, 0x333333, 1)
    this.background.fillRoundedRect(0, 0, textWidth, textHeight, BUBBLE_RADIUS)
    this.background.strokeRoundedRect(0, 0, textWidth, textHeight, BUBBLE_RADIUS)

    // 말풍선 꼬리 (아래 삼각형)
    const tailX = textWidth / 2
    const tailY = textHeight
    this.background.fillStyle(0xffffff, 0.95)
    this.background.fillTriangle(tailX - 6, tailY, tailX + 6, tailY, tailX, tailY + 10)
    this.background.lineStyle(2, 0x333333, 1)
    this.background.lineBetween(tailX - 6, tailY, tailX, tailY + 10)
    this.background.lineBetween(tailX + 6, tailY, tailX, tailY + 10)
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
