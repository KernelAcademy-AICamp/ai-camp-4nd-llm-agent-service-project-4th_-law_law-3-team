/** 스프라이트 시트 프리로드 + 애니메이션 등록 + 에셋 로드 씬 */

import Phaser from 'phaser'
import {
  LPC_FRAME_SIZE,
  LPC_COLUMNS,
  CHARACTER_SPRITE_KEYS,
  SPRITE_SHEET_PATHS,
  STATE_TO_LPC,
  animKey,
} from './sprites/LpcSpriteConfig'
import {
  ASSET_KEYS,
  ASSET_PATHS,
  EMOTION_ICON_FRAME_SIZE,
  FLAG_FRAME_WIDTH,
  FLAG_FRAME_HEIGHT,
  FLAG_ANIM_FRAMES,
  FLAG_FRAME_RATE,
} from './AssetConfig'

export class PreloadScene extends Phaser.Scene {
  constructor() {
    super({ key: 'PreloadScene' })
  }

  preload(): void {
    this.createLoadingUI()
    this.loadCharacterSprites()
    this.loadBackgrounds()
    this.loadUIAssets()
    this.loadEffects()
    this.loadFlags()
    this.loadTilemap()
    this.loadAudio()
  }

  create(): void {
    this.registerAnimations()
    this.scene.start('LobbyScene')
  }

  /** 로딩 프로그레스 바 */
  private createLoadingUI(): void {
    const { width, height } = this.cameras.main

    this.cameras.main.setBackgroundColor(0x1a237e)

    const titleText = this.add
      .text(width / 2, height / 2 - 40, '모의 법정 로딩 중...', {
        fontSize: '16px',
        color: '#ffffff',
        fontFamily: 'sans-serif',
      })
      .setOrigin(0.5)

    const progressBox = this.add.graphics()
    progressBox.fillStyle(0x222222, 0.8)
    progressBox.fillRoundedRect(width / 2 - 160, height / 2 - 10, 320, 40, 4)

    const progressBar = this.add.graphics()

    const countText = this.add
      .text(width / 2, height / 2 + 45, '', {
        fontSize: '11px',
        color: '#aaaaaa',
        fontFamily: 'sans-serif',
      })
      .setOrigin(0.5)

    this.load.on('progress', (value: number) => {
      progressBar.clear()
      progressBar.fillStyle(0xffd700, 1)
      progressBar.fillRoundedRect(
        width / 2 - 150,
        height / 2,
        300 * value,
        20,
        3
      )
    })

    this.load.on('fileprogress', (file: Phaser.Loader.File) => {
      countText.setText(file.key)
    })

    this.load.on('complete', () => {
      progressBar.destroy()
      progressBox.destroy()
      titleText.destroy()
      countText.destroy()
    })
  }

  /** 6개 캐릭터 스프라이트 시트 (기존) */
  private loadCharacterSprites(): void {
    for (const [role, path] of Object.entries(SPRITE_SHEET_PATHS)) {
      const key = CHARACTER_SPRITE_KEYS[role]
      this.load.spritesheet(key, path, {
        frameWidth: LPC_FRAME_SIZE,
        frameHeight: LPC_FRAME_SIZE,
      })
    }
  }

  /** Phase 1: 배경 이미지 */
  private loadBackgrounds(): void {
    this.load.image(ASSET_KEYS.BG_LOBBY, ASSET_PATHS[ASSET_KEYS.BG_LOBBY])
    this.load.image(ASSET_KEYS.BG_COURT, ASSET_PATHS[ASSET_KEYS.BG_COURT])
    this.load.image(
      ASSET_KEYS.BG_COURT_OVERLAY,
      ASSET_PATHS[ASSET_KEYS.BG_COURT_OVERLAY]
    )
  }

  /** Phase 2: UI 에셋 */
  private loadUIAssets(): void {
    this.load.image(
      ASSET_KEYS.UI_SPEECH_BUBBLE,
      ASSET_PATHS[ASSET_KEYS.UI_SPEECH_BUBBLE]
    )
    this.load.image(
      ASSET_KEYS.UI_SPEECH_TAIL,
      ASSET_PATHS[ASSET_KEYS.UI_SPEECH_TAIL]
    )
  }

  /** Phase 2: 감정 아이콘 스프라이트시트 */
  private loadEffects(): void {
    this.load.spritesheet(
      ASSET_KEYS.EMOTION_ICONS,
      ASSET_PATHS[ASSET_KEYS.EMOTION_ICONS],
      {
        frameWidth: EMOTION_ICON_FRAME_SIZE,
        frameHeight: EMOTION_ICON_FRAME_SIZE,
      }
    )
  }

  /** Phase 5: 국기 스프라이트시트 */
  private loadFlags(): void {
    this.load.spritesheet(ASSET_KEYS.FLAG_KOREA, ASSET_PATHS[ASSET_KEYS.FLAG_KOREA], {
      frameWidth: FLAG_FRAME_WIDTH,
      frameHeight: FLAG_FRAME_HEIGHT,
    })
    this.load.spritesheet(ASSET_KEYS.FLAG_COURT, ASSET_PATHS[ASSET_KEYS.FLAG_COURT], {
      frameWidth: FLAG_FRAME_WIDTH,
      frameHeight: FLAG_FRAME_HEIGHT,
    })
  }

  /** Phase 3: 타일맵 */
  private loadTilemap(): void {
    this.load.tilemapTiledJSON(
      ASSET_KEYS.TILEMAP_COURT,
      ASSET_PATHS[ASSET_KEYS.TILEMAP_COURT]
    )
    this.load.image(
      ASSET_KEYS.TILESET_COURT,
      ASSET_PATHS[ASSET_KEYS.TILESET_COURT]
    )
  }

  /** Phase 4: 오디오 */
  private loadAudio(): void {
    const audioKeys = [
      ASSET_KEYS.BGM_LOBBY,
      ASSET_KEYS.BGM_COURT,
      ASSET_KEYS.SFX_GAVEL,
      ASSET_KEYS.SFX_TYPING,
      ASSET_KEYS.SFX_OBJECTION,
      ASSET_KEYS.SFX_STAGE_CHANGE,
    ]
    for (const key of audioKeys) {
      this.load.audio(key, ASSET_PATHS[key])
    }
  }

  /** 모든 (캐릭터 x 상태) 조합의 Phaser 애니메이션 등록 */
  private registerAnimations(): void {
    const roles = Object.keys(CHARACTER_SPRITE_KEYS)
    const states = Object.keys(STATE_TO_LPC)

    for (const role of roles) {
      const spriteKey = CHARACTER_SPRITE_KEYS[role]

      for (const state of states) {
        const def = STATE_TO_LPC[state]
        const key = animKey(role, state)

        if (this.anims.exists(key)) continue

        const startFrame = def.row * LPC_COLUMNS
        const frames = this.anims.generateFrameNumbers(spriteKey, {
          start: startFrame,
          end: startFrame + def.frames - 1,
        })

        this.anims.create({
          key,
          frames,
          frameRate: def.frameRate,
          repeat: def.repeat,
        })
      }
    }

    // 국기 펄럭임 애니메이션
    const flagKeys = [ASSET_KEYS.FLAG_KOREA, ASSET_KEYS.FLAG_COURT]
    for (const flagKey of flagKeys) {
      const key = `${flagKey}-wave`
      if (this.anims.exists(key)) continue
      this.anims.create({
        key,
        frames: this.anims.generateFrameNumbers(flagKey, {
          start: 0,
          end: FLAG_ANIM_FRAMES - 1,
        }),
        frameRate: FLAG_FRAME_RATE,
        repeat: -1,
        yoyo: true,
      })
    }
  }
}
