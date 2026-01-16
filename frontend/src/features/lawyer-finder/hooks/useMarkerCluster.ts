'use client'

import { useEffect, useRef, useCallback } from 'react'
import type { Lawyer } from '../types'

interface UseMarkerClusterOptions {
  map: kakao.maps.Map | null
  lawyers: Lawyer[]
  onMarkerClick?: (lawyer: Lawyer) => void
}

export function useMarkerCluster({
  map,
  lawyers,
  onMarkerClick,
}: UseMarkerClusterOptions) {
  const clustererRef = useRef<kakao.maps.MarkerClusterer | null>(null)
  const markersRef = useRef<kakao.maps.Marker[]>([])

  // 마커 클러스터 생성/업데이트
  useEffect(() => {
    if (!map || !window.kakao?.maps) return

    // 기존 클러스터러 제거
    if (clustererRef.current) {
      clustererRef.current.clear()
    }

    // 기존 마커 제거
    markersRef.current.forEach((marker) => marker.setMap(null))
    markersRef.current = []

    // 좌표가 있는 변호사만 필터링
    const lawyersWithCoords = lawyers.filter(
      (l) => l.latitude != null && l.longitude != null
    )

    if (lawyersWithCoords.length === 0) return

    // 새 마커 생성
    const markers = lawyersWithCoords.map((lawyer) => {
      const marker = new window.kakao.maps.Marker({
        position: new window.kakao.maps.LatLng(lawyer.latitude!, lawyer.longitude!),
        title: lawyer.name,
      })

      // 마커 클릭 이벤트
      window.kakao.maps.event.addListener(marker, 'click', () => {
        onMarkerClick?.(lawyer)
      })

      return marker
    })

    markersRef.current = markers

    // 클러스터러 생성
    clustererRef.current = new window.kakao.maps.MarkerClusterer({
      map,
      markers,
      gridSize: 60,
      averageCenter: true,
      minLevel: 5, // 줌 레벨 5 이상에서 클러스터링
      minClusterSize: 2,
      styles: [
        {
          width: '50px',
          height: '50px',
          background: 'rgba(59, 130, 246, 0.85)',
          borderRadius: '50%',
          color: '#fff',
          textAlign: 'center',
          lineHeight: '50px',
          fontSize: '14px',
          fontWeight: 'bold',
        },
        {
          width: '60px',
          height: '60px',
          background: 'rgba(37, 99, 235, 0.85)',
          borderRadius: '50%',
          color: '#fff',
          textAlign: 'center',
          lineHeight: '60px',
          fontSize: '16px',
          fontWeight: 'bold',
        },
        {
          width: '70px',
          height: '70px',
          background: 'rgba(29, 78, 216, 0.85)',
          borderRadius: '50%',
          color: '#fff',
          textAlign: 'center',
          lineHeight: '70px',
          fontSize: '18px',
          fontWeight: 'bold',
        },
      ],
    })

    // 클린업
    return () => {
      if (clustererRef.current) {
        clustererRef.current.clear()
      }
      markersRef.current.forEach((marker) => marker.setMap(null))
    }
  }, [map, lawyers, onMarkerClick])

  // 클러스터 다시 그리기
  const redraw = useCallback(() => {
    clustererRef.current?.redraw()
  }, [])

  return {
    clusterer: clustererRef.current,
    markers: markersRef.current,
    redraw,
  }
}
