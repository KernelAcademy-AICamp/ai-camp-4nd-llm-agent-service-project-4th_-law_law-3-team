'use client'

import { useState, useCallback } from 'react'

interface GeolocationState {
  latitude: number | null
  longitude: number | null
  error: string | null
  loading: boolean
}

// 서울 시청 좌표 (기본값)
const DEFAULT_CENTER = {
  lat: 37.5665,
  lng: 126.978,
}

export function useGeolocation() {
  const [state, setState] = useState<GeolocationState>({
    latitude: null,
    longitude: null,
    error: null,
    loading: false,
  })

  const getCurrentPosition = useCallback(() => {
    if (typeof window === 'undefined') return

    if (!navigator.geolocation) {
      setState((prev) => ({
        ...prev,
        error: '이 브라우저는 위치 서비스를 지원하지 않습니다',
        loading: false,
      }))
      return
    }

    setState((prev) => ({ ...prev, loading: true, error: null }))

    navigator.geolocation.getCurrentPosition(
      (position) => {
        setState({
          latitude: position.coords.latitude,
          longitude: position.coords.longitude,
          error: null,
          loading: false,
        })
      },
      (error) => {
        let errorMessage = '위치를 가져올 수 없습니다'

        switch (error.code) {
          case error.PERMISSION_DENIED:
            errorMessage = '위치 권한이 거부되었습니다'
            break
          case error.POSITION_UNAVAILABLE:
            errorMessage = '위치 정보를 사용할 수 없습니다'
            break
          case error.TIMEOUT:
            errorMessage = '위치 요청 시간이 초과되었습니다'
            break
        }

        setState({
          latitude: null,
          longitude: null,
          error: errorMessage,
          loading: false,
        })
      },
      {
        enableHighAccuracy: true,
        timeout: 10000,
        maximumAge: 300000, // 5분 캐시
      }
    )
  }, [])

  // 위치가 없으면 기본값 사용
  const getEffectiveLocation = useCallback(() => {
    if (state.latitude && state.longitude) {
      return { lat: state.latitude, lng: state.longitude }
    }
    return DEFAULT_CENTER
  }, [state.latitude, state.longitude])

  return {
    ...state,
    getCurrentPosition,
    getEffectiveLocation,
    hasLocation: state.latitude !== null && state.longitude !== null,
    DEFAULT_CENTER,
  }
}
