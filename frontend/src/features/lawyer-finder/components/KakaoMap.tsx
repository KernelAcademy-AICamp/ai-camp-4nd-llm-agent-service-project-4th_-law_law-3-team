'use client'

import { useEffect, useRef, memo } from 'react'
import type { Lawyer, Office, ClusterData } from '../types'

// HTML 이스케이프 함수 (XSS 방지)
const HTML_ESCAPE_MAP: Record<string, string> = {
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&#39;',
}

function escapeHtml(text: string): string {
  return text.replace(/[&<>"']/g, (char) => HTML_ESCAPE_MAP[char])
}

// 마커 SVG 생성 함수 (모듈 레벨)
function createMarkerSvg(color: string): string {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="32" height="40" viewBox="0 0 32 40">
    <defs>
      <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
        <feDropShadow dx="0" dy="1" stdDeviation="2" flood-opacity="0.3"/>
      </filter>
    </defs>
    <path fill="${color}" filter="url(%23shadow)" d="M16 0C7.163 0 0 7.163 0 16c0 8.837 16 24 16 24s16-15.163 16-24C32 7.163 24.837 0 16 0z"/>
    <circle fill="#fff" cx="16" cy="14" r="6"/>
  </svg>`
}

/** 캐시된 마커 SVG 데이터 URL (성능 최적화) */
const MARKER_SVG_CACHE = {
  blue: `data:image/svg+xml;charset=utf-8,${encodeURIComponent(createMarkerSvg('#3B82F6'))}`,
  orange: `data:image/svg+xml;charset=utf-8,${encodeURIComponent(createMarkerSvg('#F97316'))}`,
} as const

// 변호사를 사무소별로 그룹화하는 함수
function groupLawyersByOffice(lawyers: Lawyer[]): Office[] {
  const officeMap = new Map<string, Office>()

  lawyers.forEach((lawyer) => {
    if (lawyer.latitude == null || lawyer.longitude == null) return

    // 사무소명이 없으면 개인 사무소로 처리
    const key = lawyer.office_name || `개인_${lawyer.name}_${lawyer.id}`

    if (!officeMap.has(key)) {
      officeMap.set(key, {
        name: lawyer.office_name || `${lawyer.name} 변호사`,
        address: lawyer.address,
        lat: lawyer.latitude,
        lng: lawyer.longitude,
        lawyers: [],
      })
    }
    officeMap.get(key)!.lawyers.push(lawyer)
  })

  return Array.from(officeMap.values())
}

// 팝업 HTML 생성 함수
function createPopupContent(office: Office, onClick?: () => void): HTMLElement {
  const container = document.createElement('div')
  container.className = 'office-popup'
  container.style.cursor = 'pointer'

  const lawyerListHtml = office.lawyers
    .slice(0, 3)  // 최대 3명만 표시
    .map(
      (lawyer) => `
      <div class="lawyer-item">
        <span class="lawyer-name">${escapeHtml(lawyer.name)}</span>
      </div>
    `
    )
    .join('')

  const moreCount = office.lawyers.length > 3 ? office.lawyers.length - 3 : 0

  container.innerHTML = `
    <div class="office-header">
      <div class="office-name">${escapeHtml(office.name)}</div>
      ${office.address ? `<div class="office-address">${escapeHtml(office.address)}</div>` : ''}
      <div class="lawyer-count">${office.lawyers.length}명의 변호사</div>
    </div>
    <div class="lawyer-list">
      ${lawyerListHtml}
      ${moreCount > 0 ? `<div class="lawyer-more">외 ${moreCount}명 더보기 →</div>` : ''}
    </div>
  `

  if (onClick) {
    container.addEventListener('click', (e) => {
      e.stopPropagation()
      onClick()
    })
  }

  return container
}

interface KakaoMapProps {
  center: { lat: number; lng: number }
  userLocation?: { lat: number; lng: number } | null  // 실제 GPS 위치 (파란 마커용)
  lawyers: Lawyer[]
  selectedLawyer: Lawyer | null
  selectionTrigger?: number  // 선택 시점 timestamp (지도 이동 트리거)
  radius: number
  clusters?: ClusterData[]  // 줌아웃 시 서버사이드 클러스터 데이터
  onMapReady?: (map: kakao.maps.Map) => void
  onLawyerClick?: (lawyer: Lawyer) => void
  onOfficeClick?: (office: Office) => void
  onMyLocationClick?: () => void
  onCenterChange?: (center: { lat: number; lng: number }) => void
  onZoomChange?: (zoom: number) => void
  onBoundsChange?: (bounds: { min_lat: number; max_lat: number; min_lng: number; max_lng: number }) => void
  showRadius?: boolean
}

export function KakaoMap({
  center,
  userLocation,
  lawyers,
  selectedLawyer,
  selectionTrigger,
  radius,
  clusters,
  onMapReady,
  onLawyerClick,
  onOfficeClick,
  onMyLocationClick,
  onCenterChange,
  onZoomChange,
  onBoundsChange,
  showRadius = true,
}: KakaoMapProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<kakao.maps.Map | null>(null)
  const clustererRef = useRef<kakao.maps.MarkerClusterer | null>(null)
  const markersRef = useRef<kakao.maps.Marker[]>([])
  const clusterOverlaysRef = useRef<kakao.maps.CustomOverlay[]>([])
  const activeOverlayRef = useRef<kakao.maps.CustomOverlay | null>(null)
  const circleRef = useRef<kakao.maps.Circle | null>(null)
  const centerOverlayRef = useRef<kakao.maps.CustomOverlay | null>(null)
  const onCenterChangeRef = useRef(onCenterChange)
  const onZoomChangeRef = useRef(onZoomChange)
  const onBoundsChangeRef = useRef(onBoundsChange)
  const prevCenterRef = useRef<{ lat: number; lng: number } | null>(null)
  const skipCenterPanRef = useRef(false)
  const hideTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  // 이벤트 핸들러 refs (클린업용)
  const eventHandlersRef = useRef<{ dragend?: () => void; idle?: () => void }>({})

  // 콜백 ref 업데이트
  useEffect(() => {
    onCenterChangeRef.current = onCenterChange
  }, [onCenterChange])

  useEffect(() => {
    onZoomChangeRef.current = onZoomChange
  }, [onZoomChange])

  useEffect(() => {
    onBoundsChangeRef.current = onBoundsChange
  }, [onBoundsChange])

  // 지도 초기화
  useEffect(() => {
    if (!containerRef.current || typeof window === 'undefined') return

    const loadMap = () => {
      // 카카오 맵 스크립트가 로드되었는지 확인
      if (!window.kakao || !window.kakao.maps) {
        // 아직 로드되지 않았으면 100ms 후 재시도
        const timerId = setTimeout(loadMap, 100)
        return () => clearTimeout(timerId)
      }

      window.kakao.maps.load(() => {
        if (!containerRef.current) return

        const options: kakao.maps.MapOptions = {
          center: new window.kakao.maps.LatLng(center.lat, center.lng),
          level: 5,
        }

        const map = new window.kakao.maps.Map(containerRef.current, options)
        mapRef.current = map

        // 이벤트 핸들러 생성 및 refs에 저장 (클린업용)
        eventHandlersRef.current.dragend = () => {
          const newCenter = map.getCenter()
          onCenterChangeRef.current?.({
            lat: newCenter.getLat(),
            lng: newCenter.getLng(),
          })
        }

        eventHandlersRef.current.idle = () => {
          onZoomChangeRef.current?.(map.getLevel())
          const bounds = map.getBounds()
          if (bounds) {
            onBoundsChangeRef.current?.({
              min_lat: bounds.getSouthWest().getLat(),
              max_lat: bounds.getNorthEast().getLat(),
              min_lng: bounds.getSouthWest().getLng(),
              max_lng: bounds.getNorthEast().getLng(),
            })
          }
        }

        // 드래그 완료 시 중심 좌표 변경 콜백
        window.kakao.maps.event.addListener(map, 'dragend', eventHandlersRef.current.dragend)

        // idle 이벤트: 줌/바운드 변경 알림
        window.kakao.maps.event.addListener(map, 'idle', eventHandlersRef.current.idle)

        onMapReady?.(map)
      })
    }

    const cleanup = loadMap()
    // 클린업용 참조 복사 (lint 경고 해결)
    const handlers = eventHandlersRef.current
    const map = mapRef.current

    return () => {
      if (typeof cleanup === 'function') cleanup()
      // 이벤트 리스너 클린업
      if (map && window.kakao?.maps) {
        const { dragend, idle } = handlers
        if (dragend) {
          window.kakao.maps.event.removeListener(map, 'dragend', dragend)
        }
        if (idle) {
          window.kakao.maps.event.removeListener(map, 'idle', idle)
        }
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // 컨테이너 크기 변경 감지 및 map.relayout() 호출
  useEffect(() => {
    if (!containerRef.current) return

    const observer = new ResizeObserver(() => {
      if (mapRef.current) {
        mapRef.current.relayout()
        // 리레이아웃 후 중심 유지
        if (prevCenterRef.current) {
          mapRef.current.setCenter(
            new window.kakao.maps.LatLng(prevCenterRef.current.lat, prevCenterRef.current.lng)
          )
        }
      }
    })

    observer.observe(containerRef.current)

    return () => {
      observer.disconnect()
    }
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

    // 변호사 선택으로 인한 이동이면 스킵
    if (skipCenterPanRef.current) {
      skipCenterPanRef.current = false
      return
    }

    // center가 실제로 변경되었을 때만 지도 이동
    const prev = prevCenterRef.current
    if (!prev || Math.abs(prev.lat - center.lat) > 0.0001 || Math.abs(prev.lng - center.lng) > 0.0001) {
      map.panTo(position)
      prevCenterRef.current = { lat: center.lat, lng: center.lng }
    }
  }, [center, radius, showRadius])

  // 활성 오버레이 정리 헬퍼
  const clearActiveOverlay = () => {
    if (activeOverlayRef.current) {
      activeOverlayRef.current.setMap(null)
      activeOverlayRef.current = null
    }
    if (hideTimeoutRef.current) {
      clearTimeout(hideTimeoutRef.current)
      hideTimeoutRef.current = null
    }
  }

  // 마커 및 클러스터 업데이트 (사무소 기반 + 서버사이드 클러스터)
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

    // 기존 클러스터 오버레이 제거
    clusterOverlaysRef.current.forEach((overlay) => overlay.setMap(null))
    clusterOverlaysRef.current = []

    // 활성 오버레이 정리
    clearActiveOverlay()

    // 서버사이드 클러스터 모드
    if (clusters && clusters.length > 0) {
      const overlays: kakao.maps.CustomOverlay[] = []

      clusters.forEach((c) => {
        const position = new window.kakao.maps.LatLng(c.latitude, c.longitude)
        const size = Math.min(24 + Math.log2(c.count + 1) * 12, 80)

        const content = document.createElement('div')
        content.style.cssText = `
          width: ${size}px; height: ${size}px;
          background: rgba(59, 130, 246, 0.85);
          border-radius: 50%; color: #fff;
          display: flex; align-items: center; justify-content: center;
          font-size: ${Math.max(11, size / 4.5)}px; font-weight: bold;
          border: 2px solid rgba(255,255,255,0.8);
          box-shadow: 0 2px 6px rgba(0,0,0,0.3);
          cursor: default;
        `
        content.textContent = c.count.toLocaleString()

        const overlay = new window.kakao.maps.CustomOverlay({
          position,
          content,
          yAnchor: 0.5,
          zIndex: 5,
          map,
        })

        overlays.push(overlay)
      })

      clusterOverlaysRef.current = overlays

      return () => {
        overlays.forEach((o) => o.setMap(null))
      }
    }

    // 개별 마커 모드
    const offices = groupLawyersByOffice(lawyers)

    if (offices.length === 0) return

    const markers: kakao.maps.Marker[] = []

    // 캐시된 마커 이미지 사용 (성능 최적화)
    const markerImageSize = new window.kakao.maps.Size(32, 40)
    const markerImageOption = { offset: new window.kakao.maps.Point(16, 40) }

    // 기본 마커 (파란색) - 캐시 사용
    const defaultMarkerImage = new window.kakao.maps.MarkerImage(MARKER_SVG_CACHE.blue, markerImageSize, markerImageOption)

    // 선택된 마커 (주황색) - 캐시 사용
    const selectedMarkerImage = new window.kakao.maps.MarkerImage(MARKER_SVG_CACHE.orange, markerImageSize, markerImageOption)

    offices.forEach((office) => {
      const position = new window.kakao.maps.LatLng(office.lat, office.lng)

      // 선택된 변호사가 이 사무소에 속하는지 확인
      const isSelected = selectedLawyer && office.lawyers.some(l => l.id === selectedLawyer.id)

      // 커스텀 마커 생성
      const marker = new window.kakao.maps.Marker({
        position,
        title: office.name,
        image: isSelected ? selectedMarkerImage : defaultMarkerImage,
      })

      // 마우스 상태 관리 (lazy overlay: hover 시에만 1개 생성)
      let isOverMarker = false
      let isOverPopup = false

      const showOverlay = () => {
        // 기존 활성 오버레이 제거
        clearActiveOverlay()

        const popupContent = createPopupContent(office, () => onOfficeClick?.(office))

        // 팝업에 마우스 이벤트 추가
        popupContent.addEventListener('mouseenter', () => {
          isOverPopup = true
          if (hideTimeoutRef.current) {
            clearTimeout(hideTimeoutRef.current)
            hideTimeoutRef.current = null
          }
        })

        popupContent.addEventListener('mouseleave', () => {
          isOverPopup = false
          scheduleHide()
        })

        const overlay = new window.kakao.maps.CustomOverlay({
          content: popupContent,
          position,
          yAnchor: 1.3,
          zIndex: 10,
          map,
        })

        activeOverlayRef.current = overlay
      }

      const scheduleHide = () => {
        if (hideTimeoutRef.current) clearTimeout(hideTimeoutRef.current)
        hideTimeoutRef.current = setTimeout(() => {
          if (!isOverMarker && !isOverPopup) {
            if (activeOverlayRef.current) {
              activeOverlayRef.current.setMap(null)
              activeOverlayRef.current = null
            }
          }
        }, 100)
      }

      // 마우스 오버 시 팝업 표시
      window.kakao.maps.event.addListener(marker, 'mouseover', () => {
        isOverMarker = true
        showOverlay()
      })

      // 마우스 아웃 시 팝업 숨김 (딜레이 적용)
      window.kakao.maps.event.addListener(marker, 'mouseout', () => {
        isOverMarker = false
        scheduleHide()
      })

      // 마커 클릭 시 첫 번째 변호사 선택
      window.kakao.maps.event.addListener(marker, 'click', () => {
        if (office.lawyers.length > 0) {
          onLawyerClick?.(office.lawyers[0])
        }
      })

      markers.push(marker)
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
      clearActiveOverlay()
      if (clustererRef.current) {
        clustererRef.current.clear()
      }
      markersRef.current.forEach((marker) => marker.setMap(null))
    }
  }, [lawyers, clusters, selectedLawyer, onLawyerClick, onOfficeClick])

  // 선택된 변호사로 지도 이동 (selectionTrigger 변경 시 트리거)
  useEffect(() => {
    if (!selectionTrigger) return  // 초기값 0이면 무시
    if (!mapRef.current) return
    if (!selectedLawyer) return
    if (!selectedLawyer.latitude || !selectedLawyer.longitude) return

    // center useEffect의 panTo를 스킵하도록 플래그 설정
    skipCenterPanRef.current = true

    const position = new window.kakao.maps.LatLng(
      selectedLawyer.latitude,
      selectedLawyer.longitude
    )
    mapRef.current.panTo(position)
    mapRef.current.setLevel(4)
  }, [selectionTrigger, selectedLawyer])

  return (
    <div className="relative w-full h-full">
      <div ref={containerRef} className="w-full h-full" />

      {/* 내 위치 버튼 */}
      {onMyLocationClick && (
        <button
          type="button"
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
            aria-hidden="true"
          >
            <circle cx="12" cy="12" r="3" />
            <path strokeLinecap="round" d="M12 2v4m0 12v4m-10-10h4m12 0h4" />
          </svg>
        </button>
      )}
    </div>
  )
}

// React.memo로 불필요한 리렌더링 방지
export const MemoizedKakaoMap = memo(KakaoMap)
