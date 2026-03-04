'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
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
  const [periodDays, setPeriodDays] = useState(7)
  const [categoryPeriodDays, setCategoryPeriodDays] = useState(7)

  const dailyQuery = useQuery({
    queryKey: ['news-stats-daily', periodDays],
    queryFn: () => fetchNewsStatsDaily(periodDays),
  })

  const categoryQuery = useQuery({
    queryKey: ['news-stats-category', categoryPeriodDays],
    queryFn: () => fetchNewsCategoryStats(categoryPeriodDays),
  })

  const ragQuery = useQuery({
    queryKey: ['news-stats-rag'],
    queryFn: () => fetchRagContributionStats(),
  })

  const error = dailyQuery.error
    ? (dailyQuery.error instanceof Error ? dailyQuery.error.message : '통계 로딩 실패')
    : categoryQuery.error
      ? (categoryQuery.error instanceof Error ? categoryQuery.error.message : '카테고리 통계 로딩 실패')
      : null

  return {
    dailyStats: dailyQuery.data ?? null,
    categoryStats: categoryQuery.data ?? null,
    loading: dailyQuery.isLoading,
    error,
    periodDays,
    setPeriodDays,
    categoryPeriodDays,
    setCategoryPeriodDays,
    categoryLoading: categoryQuery.isLoading,
    ragStats: ragQuery.data ?? null,
    ragLoading: ragQuery.isLoading,
  }
}
