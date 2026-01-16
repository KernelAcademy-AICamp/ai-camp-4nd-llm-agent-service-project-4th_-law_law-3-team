'use client'

import { useEffect, useRef, useCallback } from 'react'
import type { Lawyer } from '../types'

interface KakaoMapProps {
  center: { lat: number; lng: number }
  lawyers: Lawyer[]
  selectedLawyer: Lawyer | null
  radius: number
  onMapReady?: (map: kakao.maps.Map) => void
  onLawyerClick?: (lawyer: Lawyer) => void
  showRadius?: boolean
}

export function KakaoMap({
  center,
  lawyers,
  selectedLawyer,
  radius,
  onMapReady,
  onLawyerClick,
  showRadius = true,
}: KakaoMapProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<kakao.maps.Map | null>(null)
  const clustererRef = useRef<kakao.maps.MarkerClusterer | null>(null)
  const markersRef = useRef<kakao.maps.Marker[]>([])
  const circleRef = useRef<kakao.maps.Circle | null>(null)
  const centerMarkerRef = useRef<kakao.maps.Marker | null>(null)

  // 지도 초기화
  useEffect(() => {
    if (!containerRef.current || typeof window === 'undefined' || !window.kakao?.maps) {
      return
    }

    window.kakao.maps.load(() => {
      if (!containerRef.current) return

      const options: kakao.maps.MapOptions = {
        center: new window.kakao.maps.LatLng(center.lat, center.lng),
        level: 5,
      }

      const map = new window.kakao.maps.Map(containerRef.current, options)
      mapRef.current = map
      onMapReady?.(map)
    })
  }, [])

  // 중심 좌표 및 반경 원 업데이트
  useEffect(() => {
    if (!mapRef.current || !window.kakao?.maps) return

    const map = mapRef.current
    const position = new window.kakao.maps.LatLng(center.lat, center.lng)

    // 중심 마커 업데이트
    if (centerMarkerRef.current) {
      centerMarkerRef.current.setPosition(position)
    } else {
      centerMarkerRef.current = new window.kakao.maps.Marker({
        position,
        map,
        title: '현재 위치',
      })
    }

    // 반경 원 업데이트
    if (showRadius) {
      if (circleRef.current) {
        circleRef.current.setPosition(position)
        circleRef.current.setRadius(radius)
      } else {
        circleRef.current = new window.kakao.maps.Circle({
          center: position,
          radius,
          strokeWeight: 2,
          strokeColor: '#3B82F6',
          strokeOpacity: 0.8,
          fillColor: '#3B82F6',
          fillOpacity: 0.1,
          map,
        })
      }
    }

    // 지도 중심 이동
    map.panTo(position)
  }, [center, radius, showRadius])

  // 마커 및 클러스터 업데이트
  useEffect(() => {
    if (!mapRef.current || !window.kakao?.maps) return

    const map = mapRef.current

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
        onLawyerClick?.(lawyer)
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
      minLevel: 5,
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

    return () => {
      if (clustererRef.current) {
        clustererRef.current.clear()
      }
      markersRef.current.forEach((marker) => marker.setMap(null))
    }
  }, [lawyers, onLawyerClick])

  // 선택된 변호사로 지도 이동
  useEffect(() => {
    if (!mapRef.current || !selectedLawyer?.latitude || !selectedLawyer?.longitude) {
      return
    }

    const position = new window.kakao.maps.LatLng(
      selectedLawyer.latitude,
      selectedLawyer.longitude
    )
    mapRef.current.panTo(position)
    mapRef.current.setLevel(3)
  }, [selectedLawyer])

  return (
    <div ref={containerRef} className="w-full h-full" />
  )
}
