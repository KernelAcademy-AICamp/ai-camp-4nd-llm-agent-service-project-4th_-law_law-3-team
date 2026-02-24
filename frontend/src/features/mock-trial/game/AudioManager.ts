/** 오디오 매니저 - BGM/SFX 재생, 음소거 토글 */

import Phaser from 'phaser'
import { ASSET_KEYS, hasAudio } from './AssetConfig'
import { eventBus } from './EventBus'

/** 전역 음소거 상태 (씬 간 공유) */
let globalMuted = false

export class AudioManager {
  private scene: Phaser.Scene
  private bgm: Phaser.Sound.BaseSound | null = null
  private unsubscribe: (() => void) | null = null

  constructor(scene: Phaser.Scene) {
    this.scene = scene
    this.unsubscribe = eventBus.on('audio:toggle', () => {
      globalMuted = !globalMuted
      this.scene.sound.mute = globalMuted
    })
  }

  /** BGM 재생 (이전 BGM 정지 후 교체) */
  playBGM(key: string, volume = 0.3): void {
    if (!hasAudio(this.scene, key)) return
    this.stopBGM()
    this.bgm = this.scene.sound.add(key, { loop: true, volume })
    if (!globalMuted) {
      this.bgm.play()
    }
  }

  /** BGM 정지 */
  stopBGM(): void {
    if (this.bgm) {
      this.bgm.stop()
      this.bgm.destroy()
      this.bgm = null
    }
  }

  /** SFX 재생 (1회) */
  playSFX(key: string, volume = 0.5): void {
    if (!hasAudio(this.scene, key) || globalMuted) return
    this.scene.sound.play(key, { volume })
  }

  /** 판사봉 효과음 */
  playGavel(): void {
    this.playSFX(ASSET_KEYS.SFX_GAVEL, 0.6)
  }

  /** 타이핑 효과음 */
  playTyping(): void {
    this.playSFX(ASSET_KEYS.SFX_TYPING, 0.2)
  }

  /** 이의제기 효과음 */
  playObjection(): void {
    this.playSFX(ASSET_KEYS.SFX_OBJECTION, 0.7)
  }

  /** 단계 전환 효과음 */
  playStageChange(): void {
    this.playSFX(ASSET_KEYS.SFX_STAGE_CHANGE, 0.4)
  }

  /** 현재 음소거 상태 */
  isMuted(): boolean {
    return globalMuted
  }

  destroy(): void {
    this.stopBGM()
    if (this.unsubscribe) {
      this.unsubscribe()
      this.unsubscribe = null
    }
  }
}
