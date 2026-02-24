/** Phaser 게임 설정 상수 */

export const GAME_WIDTH = 800
export const GAME_HEIGHT = 480

/** 픽셀아트 기본 단위 (12x16 그리드 = 48x64px) - PixelCharacterRenderer용 보존 */
export const PIXEL_SIZE = 4

/** LPC 스프라이트 스케일 */
export const CHARACTER_SCALE = 1.0
export const JUROR_SCALE = 0.55

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

/** 법정 내 캐릭터 배치 좌표 (LPC 스프라이트 발 앵커 기준, 타일맵 오버라이드 가능) */
export const CHARACTER_POSITIONS: Record<string, { x: number; y: number }> = {
  judge: { x: 400, y: 172 },
  prosecutor: { x: 200, y: 292 },
  attorney: { x: 560, y: 292 },
  defendant: { x: 400, y: 372 },
  clerk: { x: 120, y: 172 },
}

/** 캐릭터별 말풍선 앵커 오프셋 (LPC 스프라이트 발 앵커 기준, 머리 위) */
export const BUBBLE_OFFSETS: Record<string, { x: number; y: number }> = {
  judge: { x: 0, y: -80 },
  prosecutor: { x: 0, y: -75 },
  attorney: { x: 0, y: -75 },
  defendant: { x: 0, y: -75 },
  clerk: { x: 0, y: -75 },
}

/** 배심원 4명 좌표 (우측 배치) */
export const JURY_POSITIONS: { x: number; y: number }[] = [
  { x: 710, y: 150 },
  { x: 760, y: 150 },
  { x: 710, y: 200 },
  { x: 760, y: 200 },
]

/** 로비 캐릭터 최종 도착 위치 (건물 앞 하단, LPC 발 앵커 기준) */
export const LOBBY_CHARACTER_POSITIONS: Record<string, { x: number; y: number }> = {
  clerk: { x: 160, y: 387 },
  defendant: { x: 300, y: 397 },
  judge: { x: 400, y: 367 },
  prosecutor: { x: 540, y: 397 },
  attorney: { x: 640, y: 387 },
}

/** 로비 입장 시퀀스 설정 */
export interface LobbyEntranceEntry {
  role: string
  startX: number
  startY: number
  delay: number
}

export const LOBBY_ENTRANCE_SEQUENCE: LobbyEntranceEntry[] = [
  { role: 'clerk', startX: -60, startY: 387, delay: 0 },
  { role: 'defendant', startX: -60, startY: 397, delay: 600 },
  { role: 'prosecutor', startX: 860, startY: 397, delay: 1200 },
  { role: 'attorney', startX: -60, startY: 387, delay: 1800 },
  { role: 'judge', startX: 400, startY: -80, delay: 2800 },
]

/** 입장 이동 시간 (ms) */
export const ENTRANCE_WALK_DURATION = 1200

/** squash&stretch 주기 (ms) */
export const ENTRANCE_BOUNCE_PERIOD = 150

/** 로비 국기 애니메이션 배치 (배경 위 오버레이, 브라우저에서 미세 조정) */
export const LOBBY_FLAGS: { id: string; x: number; y: number; scale: number; depth: number }[] = [
  { id: 'korea', x: 274, y: 108, scale: 0.44, depth: 3 },
  { id: 'court', x: 274, y: 132, scale: 0.40, depth: 2 },
]
