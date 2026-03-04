import { useSearchParams, useRouter, usePathname } from 'next/navigation'
import { useMemo, useCallback } from 'react'

/** URL searchParams에서 파싱한 검색 상태 */
export interface UrlSearchState {
  lat: number | null
  lng: number | null
  radius: number
  province: string
  sigungu: string
  category: string
  specialty: string
  zoom: number | undefined
  searchAll: boolean
}

export const SEARCH_DEFAULTS = {
  radius: 3000,
  province: '서울',
} as const

function safeFloat(value: string | null): number | null {
  if (!value) return null
  const parsed = parseFloat(value)
  return isNaN(parsed) ? null : parsed
}

function safeInt(value: string | null, fallback: number): number {
  if (!value) return fallback
  const parsed = parseInt(value, 10)
  return isNaN(parsed) ? fallback : parsed
}

/**
 * URL searchParams 단일 파싱 + 업데이트 훅
 *
 * - useMemo로 단일 파싱 포인트 (이중 읽기 제거)
 * - updateUrl로 shallow replace (공유/북마크 지원)
 */
export function useLawyerSearchParams() {
  const searchParams = useSearchParams()
  const router = useRouter()
  const pathname = usePathname()

  const state = useMemo<UrlSearchState>(() => ({
    lat: safeFloat(searchParams.get('lat')),
    lng: safeFloat(searchParams.get('lng')),
    radius: safeInt(searchParams.get('radius'), SEARCH_DEFAULTS.radius),
    province: searchParams.get('province') || SEARCH_DEFAULTS.province,
    sigungu: searchParams.get('sigungu') || '',
    category: searchParams.get('category') || '',
    specialty: searchParams.get('specialty') || '',
    zoom: searchParams.get('zoom')
      ? safeInt(searchParams.get('zoom'), 5)
      : undefined,
    searchAll: searchParams.get('searchAll') === 'true',
  }), [searchParams])

  /** 필터 상태 → URL 반영 (shallow replace, 스크롤 유지) */
  const updateUrl = useCallback(
    (updates: Record<string, string | number | null>) => {
      const next = new URLSearchParams(searchParams.toString())

      for (const [key, value] of Object.entries(updates)) {
        if (value === null || value === undefined || value === '') {
          next.delete(key)
        } else {
          next.set(key, String(value))
        }
      }

      // 기본값은 URL에서 제거 (깔끔한 URL 유지)
      if (next.get('province') === SEARCH_DEFAULTS.province) next.delete('province')
      if (next.get('radius') === String(SEARCH_DEFAULTS.radius)) next.delete('radius')

      const queryString = next.toString()
      router.replace(
        queryString ? `${pathname}?${queryString}` : pathname,
        { scroll: false },
      )
    },
    [searchParams, router, pathname],
  )

  return { state, updateUrl }
}
