'use client'

import { useEffect, useRef, useState } from 'react'
import type { Game as PhaserGame } from 'phaser'

export function MockTrialGame() {
  const gameRef = useRef<PhaserGame | null>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [isLoaded, setIsLoaded] = useState(false)

  useEffect(() => {
    let isMounted = true

    const initGame = async (): Promise<void> => {
      const Phaser = (await import('phaser')).default
      const { PreloadScene } = await import('../game/PreloadScene')
      const { CourtScene } = await import('../game/CourtScene')
      const { LobbyScene } = await import('../game/LobbyScene')

      if (!isMounted || !containerRef.current || gameRef.current) return

      gameRef.current = new Phaser.Game({
        type: Phaser.AUTO,
        parent: containerRef.current,
        width: 800,
        height: 480,
        pixelArt: true,
        roundPixels: true,
        scene: [PreloadScene, LobbyScene, CourtScene],
        scale: {
          mode: Phaser.Scale.FIT,
          autoCenter: Phaser.Scale.CENTER_BOTH,
        },
        backgroundColor: '#f5f0e8',
      })
      setIsLoaded(true)
    }

    initGame()

    return () => {
      isMounted = false
      gameRef.current?.destroy(true)
      gameRef.current = null
    }
  }, [])

  return (
    <div className="relative w-full" style={{ maxWidth: 800 }}>
      <div ref={containerRef} className="w-full" style={{ aspectRatio: '800 / 480' }} />
      {!isLoaded && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-100 rounded-lg">
          <p className="text-gray-500">법정 로딩 중...</p>
        </div>
      )}
    </div>
  )
}
