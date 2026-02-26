/** 말풍선 UI (NineSlice 이미지 또는 Graphics API fallback + 타이핑 효과 + 페이지 분할) */

import Phaser from 'phaser'
import { GAME_WIDTH, MAX_BUBBLE_HEIGHT, MAX_TEXT_HEIGHT, BASE_TYPING_SPEED } from '../config'
import { ASSET_KEYS, hasTexture } from '../AssetConfig'

const BUBBLE_PADDING = 12
const BUBBLE_RADIUS = 8
const MAX_WIDTH = 260
const FONT_SIZE = 13
const NAME_FONT_SIZE = 11
const LINE_SPACING = 4
const SCREEN_MARGIN = 10
const NINE_SLICE_BORDER = 8

/** 페이지 인디케이터 깜빡임 주기 (ms) */
const INDICATOR_BLINK_INTERVAL = 500

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
  private indicatorText: Phaser.GameObjects.Text
  private fullText = ''
  private displayedLength = 0
  private typingTimer: Phaser.Time.TimerEvent | null = null
  private anchorX: number
  private useImageMode: boolean

  /** 페이지 분할 */
  private pages: string[] = []
  private currentPage = 0

  /** 타이핑 속도 (ms/글자) */
  private typingSpeed = BASE_TYPING_SPEED

  /** 타이핑 완료 여부 */
  private isTypingComplete = false

  /** 인디케이터 깜빡임 타이머 */
  private indicatorTimer: Phaser.Time.TimerEvent | null = null

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
        lineSpacing: LINE_SPACING,
      }
    )

    // 페이지 인디케이터 (▼)
    this.indicatorText = scene.add.text(0, 0, '▼', {
      fontSize: '10px',
      color: '#888888',
      fontFamily: 'sans-serif',
    })
    this.indicatorText.setVisible(false)

    const children: Phaser.GameObjects.GameObject[] = [this.graphicsBg]
    if (this.nineSliceBg) children.push(this.nineSliceBg)
    if (this.tailImage) children.push(this.tailImage)
    children.push(this.nameText, this.textObject, this.indicatorText)
    this.add(children)

    this.setVisible(false)
    scene.add.existing(this)
  }

  show(name: string, text: string, emotion?: string, immediate = false): void {
    if (!this.scene || !this.nameText) return
    const emoji = EMOTION_EMOJI_MAP[emotion ?? 'neutral'] ?? '😐'
    this.nameText.setText(`${name} ${emoji}`)
    this.pages = this.splitIntoPages(text)
    this.currentPage = 0
    this.showCurrentPage(immediate)
  }

  /** 텍스트를 MAX_TEXT_HEIGHT 기준으로 페이지 분할 */
  private splitIntoPages(text: string): string[] {
    if (!this.scene || !this.textObject) return [text]

    // 전체 텍스트를 설정하여 줄 바꿈 계산
    this.textObject.setText(text)
    const wrappedLines = this.textObject.getWrappedText(text)
    this.textObject.setText('')

    if (wrappedLines.length === 0) return [text]

    // 줄 높이 계산
    const lineHeight = FONT_SIZE + LINE_SPACING
    const maxLines = Math.floor(MAX_TEXT_HEIGHT / lineHeight)

    if (maxLines <= 0) return [text]

    const pages: string[] = []
    for (let i = 0; i < wrappedLines.length; i += maxLines) {
      const pageLines = wrappedLines.slice(i, i + maxLines)
      pages.push(pageLines.join('\n'))
    }

    return pages.length > 0 ? pages : [text]
  }

  /** 현재 페이지 텍스트를 표시 */
  private showCurrentPage(immediate: boolean): void {
    if (!this.scene || !this.textObject) return

    const pageText = this.pages[this.currentPage] ?? ''
    this.fullText = pageText
    this.displayedLength = 0
    this.isTypingComplete = false
    this.setVisible(true)
    this.stopTyping()
    this.hideIndicator()

    if (immediate || this.typingSpeed <= 0) {
      this.textObject.setText(pageText)
      this.isTypingComplete = true
      this.drawBackground()
      this.showIndicatorIfNeeded()
      return
    }

    this.textObject.setText('')
    this.typingTimer = this.scene.time.addEvent({
      delay: this.typingSpeed,
      callback: this.typeNextCharacter,
      callbackScope: this,
      repeat: pageText.length - 1,
    })
  }

  /**
   * 대화 진행 (스페이스바 동작)
   * @returns true: 모든 페이지 완료 (다음 대화로 넘어가야 함)
   */
  advance(): boolean {
    if (!this.scene) return true

    // 1) 타이핑 중 → 즉시 완료
    if (!this.isTypingComplete) {
      this.completeTyping()
      return false
    }

    // 2) 다음 페이지 있으면 → 다음 페이지 표시
    if (this.currentPage < this.pages.length - 1) {
      this.currentPage++
      this.showCurrentPage(false)
      return false
    }

    // 3) 마지막 페이지 → 완료 신호
    return true
  }

  /** 현재 페이지 타이핑 즉시 완료 */
  completeTyping(): void {
    if (!this.scene || !this.textObject || this.isTypingComplete) return
    this.stopTyping()
    this.textObject.setText(this.fullText)
    this.displayedLength = this.fullText.length
    this.isTypingComplete = true
    this.drawBackground()
    this.showIndicatorIfNeeded()
  }

  /** 타이핑 속도 동적 변경 */
  setTypingSpeed(speed: number): void {
    this.typingSpeed = speed

    // 현재 타이핑 중이면 새 속도로 재시작
    if (this.typingTimer && !this.isTypingComplete) {
      this.stopTyping()
      if (speed <= 0) {
        // instant 모드
        this.textObject.setText(this.fullText)
        this.displayedLength = this.fullText.length
        this.isTypingComplete = true
        this.drawBackground()
        this.showIndicatorIfNeeded()
      } else {
        const remaining = this.fullText.length - this.displayedLength
        if (remaining > 0) {
          this.typingTimer = this.scene.time.addEvent({
            delay: speed,
            callback: this.typeNextCharacter,
            callbackScope: this,
            repeat: remaining - 1,
          })
        }
      }
    }
  }

  /** 다음 페이지가 있는지 여부 */
  hasNextPage(): boolean {
    return this.currentPage < this.pages.length - 1
  }

  /** 타이핑 완료 여부 */
  isComplete(): boolean {
    return this.isTypingComplete
  }

  hide(): void {
    this.stopTyping()
    this.hideIndicator()
    this.setVisible(false)
  }

  private typeNextCharacter(): void {
    if (!this.scene || !this.textObject) return
    this.displayedLength += 1
    this.textObject.setText(this.fullText.slice(0, this.displayedLength))
    this.drawBackground()

    // 타이핑 완료 시
    if (this.displayedLength >= this.fullText.length) {
      this.isTypingComplete = true
      this.showIndicatorIfNeeded()
    }
  }

  /** 다음 페이지가 있을 때 ▼ 인디케이터 깜빡임 표시 */
  private showIndicatorIfNeeded(): void {
    if (!this.scene || !this.hasNextPage()) return

    this.indicatorText.setVisible(true)
    this.indicatorTimer = this.scene.time.addEvent({
      delay: INDICATOR_BLINK_INTERVAL,
      callback: () => {
        if (this.indicatorText) {
          this.indicatorText.setVisible(!this.indicatorText.visible)
        }
      },
      loop: true,
    })
  }

  private hideIndicator(): void {
    if (this.indicatorTimer) {
      this.indicatorTimer.destroy()
      this.indicatorTimer = null
    }
    this.indicatorText?.setVisible(false)
  }

  private drawBackground(): void {
    if (!this.scene || !this.nameText || !this.textObject) return
    const contentWidth = Math.max(this.nameText.width, this.textObject.width)
    const textWidth = Math.min(contentWidth + BUBBLE_PADDING * 2, MAX_WIDTH)
    const nameHeight = this.nameText.text ? this.nameText.height + 4 : 0
    const rawTextHeight = nameHeight + this.textObject.height + BUBBLE_PADDING * 2
    const textHeight = Math.min(rawTextHeight, MAX_BUBBLE_HEIGHT)

    // 인디케이터 위치 (말풍선 우하단)
    this.indicatorText.setPosition(
      textWidth - BUBBLE_PADDING - 8,
      textHeight - BUBBLE_PADDING - 4
    )

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
    this.hideIndicator()
    super.destroy(fromScene)
  }
}
