/** Phaser 게임 설정 상수 */

export const GAME_WIDTH = 800
export const GAME_HEIGHT = 480

/** 캐릭터 색상 (Graphics API 플레이스홀더) */
export const CHARACTER_COLORS: Record<string, number> = {
  judge: 0x1a237e, // 남색 (판사)
  prosecutor: 0xb71c1c, // 빨강 (검사)
  attorney: 0x1b5e20, // 초록 (변호사)
  defendant: 0x4e342e, // 갈색 (피고인)
  clerk: 0x37474f, // 진회색 (서기)
}

/** 캐릭터 한글 이름 */
export const CHARACTER_NAMES: Record<string, string> = {
  judge: '판사',
  prosecutor: '검사',
  attorney: '변호사',
  defendant: '피고인',
  clerk: '서기',
}

/** 법정 배경 색상 */
export const COURT_BACKGROUND_COLOR = 0xf5f0e8

/** 법정 내 캐릭터 배치 좌표 */
export const CHARACTER_POSITIONS: Record<string, { x: number; y: number }> = {
  judge: { x: 400, y: 140 },
  prosecutor: { x: 200, y: 260 },
  attorney: { x: 600, y: 260 },
  defendant: { x: 400, y: 340 },
  clerk: { x: 120, y: 140 },
}
