/** 로비/설정 씬 (법정 외관, 면책 고지) */

import Phaser from 'phaser'
import { GAME_WIDTH, GAME_HEIGHT, COURT_BACKGROUND_COLOR } from './config'
import { eventBus } from './EventBus'

export class LobbyScene extends Phaser.Scene {
  constructor() {
    super({ key: 'LobbyScene' })
  }

  create(): void {
    this.cameras.main.setBackgroundColor(COURT_BACKGROUND_COLOR)

    // 법정 건물 그래픽 (단순 형태)
    const graphics = this.add.graphics()

    // 건물 본체
    graphics.fillStyle(0xd7ccc8, 1)
    graphics.fillRect(250, 100, 300, 200)

    // 기둥 (4개)
    graphics.fillStyle(0xbcaaa4, 1)
    const pillarPositions = [270, 340, 410, 480]
    pillarPositions.forEach((positionX) => {
      graphics.fillRect(positionX, 120, 20, 180)
    })

    // 지붕 (삼각형)
    graphics.fillStyle(0x8d6e63, 1)
    graphics.fillTriangle(230, 100, 570, 100, 400, 40)

    // 문
    graphics.fillStyle(0x5d4037, 1)
    graphics.fillRect(370, 220, 60, 80)

    // 타이틀
    this.add
      .text(GAME_WIDTH / 2, 330, '모의 법정', {
        fontSize: '28px',
        color: '#1a237e',
        fontFamily: 'sans-serif',
        fontStyle: 'bold',
      })
      .setOrigin(0.5, 0)

    // 면책 고지
    this.add
      .text(GAME_WIDTH / 2, 380, '본 서비스는 교육 목적의 모의재판이며,\n실제 법률 자문이 아닙니다.', {
        fontSize: '12px',
        color: '#b71c1c',
        fontFamily: 'sans-serif',
        align: 'center',
        lineSpacing: 4,
      })
      .setOrigin(0.5, 0)

    // 안내 텍스트
    this.add
      .text(GAME_WIDTH / 2, 440, '아래 설정을 완료하면 재판이 시작됩니다', {
        fontSize: '13px',
        color: '#666666',
        fontFamily: 'sans-serif',
      })
      .setOrigin(0.5, 0)

    // EventBus: setup:complete 수신 시 CourtScene 전환
    const unsubscribe = eventBus.on('setup:complete', (data) => {
      this.scene.start('CourtScene', {
        caseType: data.caseType,
        userRole: data.userRole,
        caseSummary: data.caseSummary,
      })
      unsubscribe()
    })

    eventBus.emit('game:ready', {} as Record<string, never>)
  }
}
