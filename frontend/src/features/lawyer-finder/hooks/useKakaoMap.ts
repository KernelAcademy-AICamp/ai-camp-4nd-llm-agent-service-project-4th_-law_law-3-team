'use client'

import { useEffect, useRef, useState, useCallback } from 'react'

interface UseKakaoMapOptions {
  center: { lat: number; lng: number }
  level?: number
  onBoundsChange?: (bounds: {
    sw: { lat: number; lng: number }
    ne: { lat: number; lng: number }
  }) => void
  onZoomChange?: (level: number) => void
}

export function useKakaoMap({
  center,
  level = 5,
  onBoundsChange,
  onZoomChange,
}: UseKakaoMapOptions) {
  const mapRef = useRef<HTMLDivElement>(null)
  const [map, setMap] = useState<kakao.maps.Map | null>(null)
  const [isLoaded, setIsLoaded] = useState(false)

  // 지도 초기화
  useEffect(() => {
    if (!mapRef.current || typeof window === 'undefined' || !window.kakao?.maps) {
      return
    }

    window.kakao.maps.load(() => {
      if (!mapRef.current) return

      const options: kakao.maps.MapOptions = {
        center: new window.kakao.maps.LatLng(center.lat, center.lng),
        level,
      }

      const kakaoMap = new window.kakao.maps.Map(mapRef.current, options)
      setMap(kakaoMap)
      setIsLoaded(true)

      // 영역 변경 이벤트
      window.kakao.maps.event.addListener(kakaoMap, 'bounds_changed', () => {
        const bounds = kakaoMap.getBounds()
        const sw = bounds.getSouthWest()
        const ne = bounds.getNorthEast()

        onBoundsChange?.({
          sw: { lat: sw.getLat(), lng: sw.getLng() },
          ne: { lat: ne.getLat(), lng: ne.getLng() },
        })
      })

      // 줌 변경 이벤트
      window.kakao.maps.event.addListener(kakaoMap, 'zoom_changed', () => {
        onZoomChange?.(kakaoMap.getLevel())
      })
    })
  }, [])

  // 중심 좌표 변경
  const setCenter = useCallback(
    (lat: number, lng: number) => {
      if (map) {
        map.panTo(new window.kakao.maps.LatLng(lat, lng))
      }
    },
    [map]
  )

  // 줌 레벨 변경
  const setLevel = useCallback(
    (newLevel: number) => {
      if (map) {
        map.setLevel(newLevel)
      }
    },
    [map]
  )

  // 현재 줌 레벨 가져오기
  const getLevel = useCallback(() => {
    return map?.getLevel() ?? level
  }, [map, level])

  // 현재 바운드 가져오기
  const getBounds = useCallback(() => {
    if (!map) return null

    const bounds = map.getBounds()
    const sw = bounds.getSouthWest()
    const ne = bounds.getNorthEast()

    return {
      sw: { lat: sw.getLat(), lng: sw.getLng() },
      ne: { lat: ne.getLat(), lng: ne.getLng() },
    }
  }, [map])

  return {
    mapRef,
    map,
    isLoaded,
    setCenter,
    setLevel,
    getLevel,
    getBounds,
  }
}
