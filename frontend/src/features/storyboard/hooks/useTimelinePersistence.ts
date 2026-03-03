'use client'

import { useEffect, useRef } from 'react'
import type { TimelineItem } from '../types'

const STORAGE_KEY = 'storyboard_timeline'
const DEBOUNCE_MS = 500

export interface PersistedTimelineState {
  items: TimelineItem[]
  title: string
  originalText?: string
  summary?: string
  savedAt: number
}

/** sessionStorage에서 타임라인 상태 읽기 (SSR 가드 포함) */
export function loadPersistedState(): PersistedTimelineState | null {
  if (typeof window === 'undefined') return null
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY)
    if (!raw) return null
    return JSON.parse(raw) as PersistedTimelineState
  } catch {
    return null
  }
}

/** sessionStorage에 타임라인 상태 쓰기 (QuotaExceededError 방어) */
export function savePersistedState(state: PersistedTimelineState): void {
  try {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(state))
  } catch {
    // QuotaExceededError 방어: imageUrl/imagePrompt 제거 후 재시도
    const lightweight: PersistedTimelineState = {
      ...state,
      items: state.items.map(({ imageUrl, imagePrompt, ...rest }) => rest as TimelineItem),
    }
    try {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(lightweight))
    } catch {
      // 재시도도 실패하면 무시
    }
  }
}

/** sessionStorage에서 타임라인 상태 삭제 */
export function clearPersistedState(): void {
  if (typeof window === 'undefined') return
  try {
    sessionStorage.removeItem(STORAGE_KEY)
  } catch {
    // 무시
  }
}

/**
 * 상태 변경 시 500ms 디바운스로 sessionStorage에 자동 저장하는 훅.
 * isActive가 false이면 저장하지 않는다.
 */
export function useTimelinePersistence(
  state: PersistedTimelineState,
  isActive: boolean
): void {
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    if (!isActive) return

    if (timerRef.current) {
      clearTimeout(timerRef.current)
    }

    timerRef.current = setTimeout(() => {
      savePersistedState({ ...state, savedAt: Date.now() })
    }, DEBOUNCE_MS)

    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current)
      }
    }
  }, [state, isActive])
}
