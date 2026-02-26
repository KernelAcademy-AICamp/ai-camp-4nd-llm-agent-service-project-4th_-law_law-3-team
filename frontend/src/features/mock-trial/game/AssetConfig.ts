/** 에셋 키, 경로, 스펙 중앙 관리 */

/** 모든 에셋의 Phaser 텍스처/오디오 키 */
export const ASSET_KEYS = {
  // Phase 1: 배경 이미지
  BG_LOBBY: 'bg-lobby',
  BG_COURT: 'bg-court',
  BG_COURT_OVERLAY: 'bg-court-overlay',

  // Phase 2: UI
  UI_SPEECH_BUBBLE: 'ui-speech-bubble',
  UI_SPEECH_TAIL: 'ui-speech-tail',

  // Phase 2: 감정 아이콘 스프라이트시트
  EMOTION_ICONS: 'emotion-icons',

  // Phase 3: 타일맵
  TILEMAP_COURT: 'tilemap-court',
  TILESET_COURT: 'tileset-court',

  // Phase 5: 국기 스프라이트시트
  FLAG_KOREA: 'flag-korea',
  FLAG_COURT: 'flag-court',

  // Phase 4: 오디오
  BGM_LOBBY: 'bgm-lobby',
  BGM_COURT: 'bgm-court',
  SFX_GAVEL: 'sfx-gavel',
  SFX_TYPING: 'sfx-typing',
  SFX_OBJECTION: 'sfx-objection',
  SFX_STAGE_CHANGE: 'sfx-stage-change',
} as const

/** 에셋 키 → 파일 경로 매핑 */
export const ASSET_PATHS: Record<string, string> = {
  [ASSET_KEYS.BG_LOBBY]: 'assets/mock-trial/backgrounds/lobby-courthouse.png',
  [ASSET_KEYS.BG_COURT]: 'assets/mock-trial/backgrounds/court-interior.png',
  [ASSET_KEYS.BG_COURT_OVERLAY]: 'assets/mock-trial/backgrounds/court-interior-overlay.png',

  [ASSET_KEYS.UI_SPEECH_BUBBLE]: 'assets/mock-trial/ui/speech-bubble.png',
  [ASSET_KEYS.UI_SPEECH_TAIL]: 'assets/mock-trial/ui/speech-bubble-tail.png',

  [ASSET_KEYS.EMOTION_ICONS]: 'assets/mock-trial/effects/emotion-icons.png',

  [ASSET_KEYS.TILEMAP_COURT]: 'assets/mock-trial/tilesets/court-map.json',
  [ASSET_KEYS.TILESET_COURT]: 'assets/mock-trial/tilesets/court-tiles.png',

  [ASSET_KEYS.FLAG_KOREA]: 'assets/mock-trial/sprites/flag-korea.png',
  [ASSET_KEYS.FLAG_COURT]: 'assets/mock-trial/sprites/flag-court.png',

  [ASSET_KEYS.BGM_LOBBY]: 'assets/mock-trial/audio/bgm-lobby.mp3',
  [ASSET_KEYS.BGM_COURT]: 'assets/mock-trial/audio/bgm-court.mp3',
  [ASSET_KEYS.SFX_GAVEL]: 'assets/mock-trial/audio/sfx-gavel.mp3',
  [ASSET_KEYS.SFX_TYPING]: 'assets/mock-trial/audio/sfx-typing.mp3',
  [ASSET_KEYS.SFX_OBJECTION]: 'assets/mock-trial/audio/sfx-objection.mp3',
  [ASSET_KEYS.SFX_STAGE_CHANGE]: 'assets/mock-trial/audio/sfx-stage-change.mp3',
}

/** 감정 아이콘 스프라이트시트 프레임 인덱스 */
export const EMOTION_FRAME_INDEX: Record<string, number> = {
  neutral: 0,
  angry: 1,
  thinking: 2,
  sad: 3,
  confident: 4,
  stern: 5,
  recording: 6,
  judging: 7,
}

/** 감정 아이콘 프레임 크기 (px) */
export const EMOTION_ICON_FRAME_SIZE = 24

/** 국기 스프라이트시트 프레임 크기 (2816×1536, 4프레임 → 첫 프레임만 표시 + tween) */
export const FLAG_FRAME_WIDTH = 704
export const FLAG_FRAME_HEIGHT = 1536

/** NineSlice 슬라이스 크기 (px) */
export const SPEECH_BUBBLE_SLICE = 8

/** 타일맵 타일 크기 (px) */
export const TILEMAP_TILE_SIZE = 32

/**
 * 에셋 존재 여부를 체크하는 헬퍼.
 * PreloadScene에서 로드 실패한 에셋은 textures에 등록되지 않으므로
 * 런타임에 fallback 렌더링을 결정할 수 있다.
 */
export function hasTexture(scene: Phaser.Scene, key: string): boolean {
  return scene.textures.exists(key)
}

/** 오디오가 로드되었는지 확인 */
export function hasAudio(scene: Phaser.Scene, key: string): boolean {
  return scene.cache.audio.exists(key)
}
