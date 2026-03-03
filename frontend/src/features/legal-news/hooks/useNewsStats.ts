'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import type { NewsCategoryStats, NewsStatsDaily, RagContributionStats } from '../types'
import { fetchNewsCategoryStats, fetchNewsStatsDaily, fetchRagContributionStats } from '../services'

interface UseNewsStatsReturn {
  dailyStats: NewsStatsDaily | null
  categoryStats: NewsCategoryStats | null
  loading: boolean
  error: string | null
  periodDays: number
  setPeriodDays: (days: number) => void
  categoryPeriodDays: number
  setCategoryPeriodDays: (days: number) => void
  categoryLoading: boolean
  ragStats: RagContributionStats | null
  ragLoading: boolean
}

export function useNewsStats(): UseNewsStatsReturn {
  const [dailyStats, setDailyStats] = useState<NewsStatsDaily | null>(null)
  const [categoryStats, setCategoryStats] = useState<NewsCategoryStats | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [periodDays, setPeriodDays] = useState(7)
  const [categoryPeriodDays, setCategoryPeriodDays] = useState(7)
  const [categoryLoading, setCategoryLoading] = useState(false)
  const [ragStats, setRagStats] = useState<RagContributionStats | null>(null)
  const [ragLoading, setRagLoading] = useState(false)
  const dailyAbortRef = useRef<AbortController | null>(null)
  const categoryAbortRef = useRef<AbortController | null>(null)
  const ragAbortRef = useRef<AbortController | null>(null)

  // 일별 통계: periodDays 변경 시 재조회
  const loadDailyStats = useCallback(async (days: number) => {
    dailyAbortRef.current?.abort()
    const controller = new AbortController()
    dailyAbortRef.current = controller

    setLoading(true)
    setError(null)

    try {
      const daily = await fetchNewsStatsDaily(days, controller.signal)
      if (!controller.signal.aborted) {
        setDailyStats(daily)
      }
    } catch (err: unknown) {
      if (err instanceof Error && err.name === 'CanceledError') return
      if (!controller.signal.aborted) {
        setError(err instanceof Error ? err.message : '통계 로딩 실패')
      }
    } finally {
      if (!controller.signal.aborted) {
        setLoading(false)
      }
    }
  }, [])

  // 카테고리 통계: categoryPeriodDays 변경 시 재조회
  const loadCategoryStats = useCallback(async (days: number) => {
    categoryAbortRef.current?.abort()
    const controller = new AbortController()
    categoryAbortRef.current = controller

    setCategoryLoading(true)

    try {
      const category = await fetchNewsCategoryStats(days, controller.signal)
      if (!controller.signal.aborted) {
        setCategoryStats(category)
      }
    } catch (err: unknown) {
      if (err instanceof Error && err.name === 'CanceledError') return
      if (!controller.signal.aborted) {
        setError(err instanceof Error ? err.message : '카테고리 통계 로딩 실패')
      }
    } finally {
      if (!controller.signal.aborted) {
        setCategoryLoading(false)
      }
    }
  }, [])

  // RAG 기여도 통계: 마운트 시 1회
  const loadRagStats = useCallback(async () => {
    ragAbortRef.current?.abort()
    const controller = new AbortController()
    ragAbortRef.current = controller

    setRagLoading(true)

    try {
      const rag = await fetchRagContributionStats(controller.signal)
      if (!controller.signal.aborted) {
        setRagStats(rag)
      }
    } catch (err: unknown) {
      if (err instanceof Error && err.name === 'CanceledError') return
      // RAG 실패는 error에 설정하지 않음 (Graceful Degradation)
    } finally {
      if (!controller.signal.aborted) {
        setRagLoading(false)
      }
    }
  }, [])

  // 카테고리 통계: categoryPeriodDays 변경 시
  useEffect(() => {
    loadCategoryStats(categoryPeriodDays)
  }, [loadCategoryStats, categoryPeriodDays])

  // 일별 통계: periodDays 변경 시
  useEffect(() => {
    loadDailyStats(periodDays)
  }, [loadDailyStats, periodDays])

  // RAG 기여도: 마운트 시 1회
  useEffect(() => {
    loadRagStats()
  }, [loadRagStats])

  useEffect(() => {
    return () => {
      dailyAbortRef.current?.abort()
      categoryAbortRef.current?.abort()
      ragAbortRef.current?.abort()
    }
  }, [])

  return {
    dailyStats, categoryStats, loading, error,
    periodDays, setPeriodDays,
    categoryPeriodDays, setCategoryPeriodDays, categoryLoading,
    ragStats, ragLoading,
  }
}
