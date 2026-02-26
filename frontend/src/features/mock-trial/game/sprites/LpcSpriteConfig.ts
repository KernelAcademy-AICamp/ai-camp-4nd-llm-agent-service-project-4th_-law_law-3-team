/** LPC (Liberated Pixel Cup) 스프라이트 시트 설정 상수 */

/** 프레임 크기 (px) */
export const LPC_FRAME_SIZE = 64

/** 시트 열/행 수 */
export const LPC_COLUMNS = 13
export const LPC_ROWS = 21

/** 시트 전체 크기 (px) */
export const LPC_SHEET_WIDTH = LPC_FRAME_SIZE * LPC_COLUMNS // 832
export const LPC_SHEET_HEIGHT = LPC_FRAME_SIZE * LPC_ROWS // 1344

/**
 * LPC 행 레이아웃 (4방향: up=0, left=1, down=2, right=3)
 * - Spellcast: Row 0-3 (7프레임) → 발언 애니메이션
 * - Thrust:    Row 4-7 (8프레임) → react 애니메이션
 * - Walk:      Row 8-11 (9프레임) → 이동/idle
 * - Slash:     Row 12-15 (6프레임)
 * - Shoot:     Row 16-19 (13프레임)
 * - Hurt:      Row 20 (6프레임)
 */
export const LPC_ROW = {
  SPELLCAST_UP: 0,
  SPELLCAST_LEFT: 1,
  SPELLCAST_DOWN: 2,
  SPELLCAST_RIGHT: 3,
  THRUST_UP: 4,
  THRUST_LEFT: 5,
  THRUST_DOWN: 6,
  THRUST_RIGHT: 7,
  WALK_UP: 8,
  WALK_LEFT: 9,
  WALK_DOWN: 10,
  WALK_RIGHT: 11,
  SLASH_UP: 12,
  SLASH_LEFT: 13,
  SLASH_DOWN: 14,
  SLASH_RIGHT: 15,
  SHOOT_UP: 16,
  SHOOT_LEFT: 17,
  SHOOT_DOWN: 18,
  SHOOT_RIGHT: 19,
  HURT: 20,
} as const

/** 각 애니메이션의 유효 프레임 수 */
export const LPC_FRAME_COUNTS: Record<string, number> = {
  spellcast: 7,
  thrust: 8,
  walk: 9,
  slash: 6,
  shoot: 13,
  hurt: 6,
}

/** 게임 상태 → LPC 행+프레임 매핑 (정면=down 방향) */
export interface LpcAnimationDef {
  row: number
  frames: number
  frameRate: number
  repeat: number // -1 = 무한
}

export const STATE_TO_LPC: Record<string, LpcAnimationDef> = {
  idle: {
    row: LPC_ROW.WALK_DOWN,
    frames: 1, // 첫 프레임만 (정지)
    frameRate: 1,
    repeat: 0,
  },
  idle_left: {
    row: LPC_ROW.WALK_LEFT,
    frames: 1,
    frameRate: 1,
    repeat: 0,
  },
  idle_right: {
    row: LPC_ROW.WALK_RIGHT,
    frames: 1,
    frameRate: 1,
    repeat: 0,
  },
  speak: {
    row: LPC_ROW.SPELLCAST_DOWN,
    frames: LPC_FRAME_COUNTS.spellcast,
    frameRate: 6,
    repeat: -1,
  },
  speak_left: {
    row: LPC_ROW.SPELLCAST_LEFT,
    frames: LPC_FRAME_COUNTS.spellcast,
    frameRate: 6,
    repeat: -1,
  },
  speak_right: {
    row: LPC_ROW.SPELLCAST_RIGHT,
    frames: LPC_FRAME_COUNTS.spellcast,
    frameRate: 6,
    repeat: -1,
  },
  react: {
    row: LPC_ROW.THRUST_DOWN,
    frames: LPC_FRAME_COUNTS.thrust,
    frameRate: 8,
    repeat: 0,
  },
  react_left: {
    row: LPC_ROW.THRUST_LEFT,
    frames: LPC_FRAME_COUNTS.thrust,
    frameRate: 8,
    repeat: 0,
  },
  react_right: {
    row: LPC_ROW.THRUST_RIGHT,
    frames: LPC_FRAME_COUNTS.thrust,
    frameRate: 8,
    repeat: 0,
  },
  walk_down: {
    row: LPC_ROW.WALK_DOWN,
    frames: LPC_FRAME_COUNTS.walk,
    frameRate: 10,
    repeat: -1,
  },
  walk_left: {
    row: LPC_ROW.WALK_LEFT,
    frames: LPC_FRAME_COUNTS.walk,
    frameRate: 10,
    repeat: -1,
  },
  walk_right: {
    row: LPC_ROW.WALK_RIGHT,
    frames: LPC_FRAME_COUNTS.walk,
    frameRate: 10,
    repeat: -1,
  },
  walk_up: {
    row: LPC_ROW.WALK_UP,
    frames: LPC_FRAME_COUNTS.walk,
    frameRate: 10,
    repeat: -1,
  },
}

/** 역할 → Phaser spritesheet key */
export const CHARACTER_SPRITE_KEYS: Record<string, string> = {
  judge: 'lpc-judge',
  prosecutor: 'lpc-prosecutor',
  attorney: 'lpc-attorney',
  defendant: 'lpc-defendant',
  clerk: 'lpc-clerk',
  juror: 'lpc-juror',
}

/** 스프라이트 시트 파일 경로 */
export const SPRITE_SHEET_PATHS: Record<string, string> = {
  judge: 'assets/mock-trial/sprites/judge.png',
  prosecutor: 'assets/mock-trial/sprites/prosecutor.png',
  attorney: 'assets/mock-trial/sprites/attorney.png',
  defendant: 'assets/mock-trial/sprites/defendant.png',
  clerk: 'assets/mock-trial/sprites/clerk.png',
  juror: 'assets/mock-trial/sprites/juror.png',
}

/**
 * 애니메이션 키 생성 헬퍼
 * @example animKey('judge', 'speak') → 'lpc-judge-speak'
 */
export function animKey(role: string, state: string): string {
  const spriteKey = CHARACTER_SPRITE_KEYS[role]
  return `${spriteKey}-${state}`
}
