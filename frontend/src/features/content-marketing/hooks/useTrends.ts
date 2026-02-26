'use client'

import { useCallback, useState } from 'react'
import type { TrendDetailResponse, TrendFilters, TrendIssue, TrendResponse } from '../types'
import { fetchTrendDetail, fetchTrends } from '../services'

export function useTrends() {
  const [trends, setTrends] = useState<TrendIssue[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [cacheHit, setCacheHit] = useState(false)

  const loadTrends = useCallback(async (filters: TrendFilters) => {
    setLoading(true)
    setError(null)
    try {
      const response: TrendResponse = await fetchTrends({
        time_range: filters.time_range,
        category: filters.category,
        limit: 10,
        query: null,
      })
      setTrends(response.trends)
      setCacheHit(response.cache_hit)
    } catch (err) {
      setError(err instanceof Error ? err.message : '트렌드 조회에 실패했습니다.')
    } finally {
      setLoading(false)
    }
  }, [])

  const loadDetail = useCallback(async (trendId: string): Promise<TrendDetailResponse | null> => {
    try {
      return await fetchTrendDetail(trendId)
    } catch {
      return null
    }
  }, [])

  return { trends, loading, error, cacheHit, loadTrends, loadDetail }
}
