'use client'

import { useEffect, useRef } from 'react'
import type { Lawyer } from '../types'

interface KakaoMapProps {
  center: { lat: number; lng: number }
  userLocation?: { lat: number; lng: number } | null  // 실제 GPS 위치 (파란 마커용)
  lawyers: Lawyer[]
  selectedLawyer: Lawyer | null
  radius: number
  onMapReady?: (map: kakao.maps.Map) => void
  onLawyerClick?: (lawyer: Lawyer) => void
  onMyLocationClick?: () => void
  onCenterChange?: (center: { lat: number; lng: number }) => void
  showRadius?: boolean
}

export function KakaoMap({
  center,
  userLocation,
  lawyers,
  selectedLawyer,
  radius,
  onMapReady,
  onLawyerClick,
  onMyLocationClick,
  onCenterChange,
  showRadius = true,
}: KakaoMapProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<kakao.maps.Map | null>(null)
  const clustererRef = useRef<kakao.maps.MarkerClusterer | null>(null)
  const markersRef = useRef<kakao.maps.Marker[]>([])
  const circleRef = useRef<kakao.maps.Circle | null>(null)
  const centerOverlayRef = useRef<kakao.maps.CustomOverlay | null>(null)
  const onCenterChangeRef = useRef(onCenterChange)

  // 콜백 ref 업데이트
  useEffect(() => {
    onCenterChangeRef.current = onCenterChange
  }, [onCenterChange])

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

      // 드래그 완료 시 중심 좌표 변경 콜백
      window.kakao.maps.event.addListener(map, 'dragend', () => {
        const newCenter = map.getCenter()
        onCenterChangeRef.current?.({
          lat: newCenter.getLat(),
          lng: newCenter.getLng(),
        })
      })

      onMapReady?.(map)
    })
  }, [])

  // 파란 펄스 마커 (userLocation 기준 - 실제 GPS 위치에서만 표시)
  useEffect(() => {
    if (!mapRef.current || !window.kakao?.maps) return

    const map = mapRef.current

    // userLocation이 없으면 마커 제거
    if (!userLocation) {
      if (centerOverlayRef.current) {
        centerOverlayRef.current.setMap(null)
        centerOverlayRef.current = null
      }
      return
    }

    const position = new window.kakao.maps.LatLng(userLocation.lat, userLocation.lng)

    // 커스텀 오버레이로 현재 위치 마커 업데이트
    if (centerOverlayRef.current) {
      centerOverlayRef.current.setPosition(position)
      centerOverlayRef.current.setMap(map)
    } else {
      const content = document.createElement('div')
      content.className = 'current-location-marker'
      content.innerHTML = '<div class="pulse"></div><div class="dot"></div>'

      centerOverlayRef.current = new window.kakao.maps.CustomOverlay({
        position,
        content,
        zIndex: 1,
        map,
      })
    }
  }, [userLocation])

  // 반경 원 및 지도 중심 업데이트 (center 기준 - 검색 위치)
  useEffect(() => {
    if (!mapRef.current || !window.kakao?.maps) return

    const map = mapRef.current
    const position = new window.kakao.maps.LatLng(center.lat, center.lng)

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
    <div className="relative w-full h-full">
      <div ref={containerRef} className="w-full h-full" />

      {/* 내 위치 버튼 */}
      {onMyLocationClick && (
        <button
          onClick={onMyLocationClick}
          className="absolute top-4 left-4 z-10 w-10 h-10 bg-white rounded-lg shadow-md
                     flex items-center justify-center hover:bg-gray-50 transition-colors
                     border border-gray-200"
          title="내 위치"
        >
          <svg
            className="w-5 h-5 text-gray-700"
            fill="none"
            stroke="currentColor"
            strokeWidth={2}
            viewBox="0 0 24 24"
          >
            <circle cx="12" cy="12" r="3" />
            <path strokeLinecap="round" d="M12 2v4m0 12v4m-10-10h4m12 0h4" />
          </svg>
        </button>
      )}
    </div>
  )
}
