/** 8x8 도트 스프라이트 감정 아이콘 렌더러 (픽셀아트 스타일) */

import Phaser from 'phaser'

const PIXEL_SIZE = 3
const GRID_SIZE = 8

/** 8x8 그리드 데이터 (0=투명, 1~9=색상 인덱스) */
type SpriteGrid = number[][]

interface EmotionSprite {
  grid: SpriteGrid
  colors: Record<number, number> // 인덱스 → 0xRRGGBB
}

// ── 감정별 도트 스프라이트 정의 ──

const SPRITES: Record<string, EmotionSprite> = {
  /** 💢 분노 마크 (십자형 파열) */
  angry: {
    grid: [
      [1, 0, 0, 0, 0, 0, 0, 1],
      [0, 1, 0, 0, 0, 0, 1, 0],
      [0, 0, 1, 0, 0, 1, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 1, 0, 0, 1, 0, 0],
      [0, 1, 0, 0, 0, 0, 1, 0],
      [1, 0, 0, 0, 0, 0, 0, 1],
    ],
    colors: { 1: 0xe53935 },
  },

  /** ❓ 물음표 */
  thinking: {
    grid: [
      [0, 0, 1, 1, 1, 0, 0, 0],
      [0, 1, 0, 0, 0, 1, 0, 0],
      [0, 0, 0, 0, 0, 1, 0, 0],
      [0, 0, 0, 0, 1, 0, 0, 0],
      [0, 0, 0, 1, 0, 0, 0, 0],
      [0, 0, 0, 1, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 1, 0, 0, 0, 0],
    ],
    colors: { 1: 0xfdd835 },
  },

  /** 💧 눈물방울 */
  sad: {
    grid: [
      [0, 0, 0, 1, 0, 0, 0, 0],
      [0, 0, 1, 1, 1, 0, 0, 0],
      [0, 0, 1, 2, 1, 0, 0, 0],
      [0, 1, 1, 2, 1, 1, 0, 0],
      [0, 1, 2, 2, 2, 1, 0, 0],
      [0, 1, 1, 2, 1, 1, 0, 0],
      [0, 0, 1, 1, 1, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0],
    ],
    colors: { 1: 0x1e88e5, 2: 0x64b5f6 },
  },

  /** ✨ 반짝 별 */
  confident: {
    grid: [
      [0, 0, 0, 1, 0, 0, 0, 0],
      [0, 0, 0, 1, 0, 0, 0, 0],
      [0, 0, 2, 2, 2, 0, 0, 0],
      [1, 1, 2, 1, 2, 1, 1, 0],
      [0, 0, 2, 2, 2, 0, 0, 0],
      [0, 0, 0, 1, 0, 0, 0, 0],
      [0, 0, 0, 1, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 2, 0],
    ],
    colors: { 1: 0xfdd835, 2: 0xffffff },
  },

  /** ❗ 느낌표 */
  stern: {
    grid: [
      [0, 0, 0, 1, 1, 0, 0, 0],
      [0, 0, 0, 1, 1, 0, 0, 0],
      [0, 0, 0, 1, 1, 0, 0, 0],
      [0, 0, 0, 1, 1, 0, 0, 0],
      [0, 0, 0, 1, 1, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 1, 1, 0, 0, 0],
      [0, 0, 0, 1, 1, 0, 0, 0],
    ],
    colors: { 1: 0xe53935 },
  },

  /** ✏️ 연필 */
  recording: {
    grid: [
      [0, 0, 0, 0, 0, 0, 1, 1],
      [0, 0, 0, 0, 0, 1, 2, 1],
      [0, 0, 0, 0, 1, 2, 1, 0],
      [0, 0, 0, 1, 2, 1, 0, 0],
      [0, 0, 1, 2, 1, 0, 0, 0],
      [0, 1, 2, 1, 0, 0, 0, 0],
      [1, 3, 1, 0, 0, 0, 0, 0],
      [1, 1, 0, 0, 0, 0, 0, 0],
    ],
    colors: { 1: 0x795548, 2: 0xfdd835, 3: 0x333333 },
  },

  /** 🔨 가벨 */
  judging: {
    grid: [
      [0, 0, 1, 1, 1, 1, 0, 0],
      [0, 0, 1, 2, 2, 1, 0, 0],
      [0, 0, 0, 1, 1, 0, 0, 0],
      [0, 0, 0, 1, 0, 0, 0, 0],
      [0, 0, 0, 1, 0, 0, 0, 0],
      [0, 0, 0, 1, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0],
      [0, 1, 1, 1, 1, 1, 1, 0],
    ],
    colors: { 1: 0x795548, 2: 0x8d6e63 },
  },

  /** 💬 말풍선 점(...) */
  neutral: {
    grid: [
      [0, 1, 1, 1, 1, 1, 1, 0],
      [1, 0, 0, 0, 0, 0, 0, 1],
      [1, 0, 2, 0, 2, 0, 2, 1],
      [1, 0, 0, 0, 0, 0, 0, 1],
      [0, 1, 1, 1, 1, 1, 1, 0],
      [0, 0, 1, 0, 0, 0, 0, 0],
      [0, 1, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0],
    ],
    colors: { 1: 0x9e9e9e, 2: 0xffffff },
  },
}

/**
 * Graphics 객체에 8x8 도트 감정 아이콘을 그린다.
 * 중심 기준 렌더링 (원점 = 아이콘 중앙)
 */
export function drawEmotionIcon(
  graphics: Phaser.GameObjects.Graphics,
  emotion: string
): void {
  graphics.clear()

  const sprite = SPRITES[emotion] ?? SPRITES['neutral']
  const totalSize = GRID_SIZE * PIXEL_SIZE
  const offsetX = -totalSize / 2
  const offsetY = -totalSize / 2

  for (let row = 0; row < GRID_SIZE; row++) {
    for (let col = 0; col < GRID_SIZE; col++) {
      const value = sprite.grid[row][col]
      if (value === 0) continue

      const color = sprite.colors[value] ?? 0xffffff
      graphics.fillStyle(color, 1)
      graphics.fillRect(
        offsetX + col * PIXEL_SIZE,
        offsetY + row * PIXEL_SIZE,
        PIXEL_SIZE,
        PIXEL_SIZE
      )
    }
  }
}
