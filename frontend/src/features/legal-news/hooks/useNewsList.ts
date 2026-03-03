'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import type {
  NewsArticleSummary,
  NewsListFilters,
  NewsListResponse,
} from '../types'
import { fetchNewsList } from '../services'

const DEFAULT_FILTERS: NewsListFilters = {
  source: null,
  published_date: null,
  page: 1,
  page_size: 20,
}

export function useNewsList() {
  const [items, setItems] = useState<NewsArticleSummary[]>([])
  const [total, setTotal] = useState(0)
  const [hasNext, setHasNext] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [filters, setFiltersState] = useState<NewsListFilters>(DEFAULT_FILTERS)
  const abortControllerRef = useRef<AbortController | null>(null)

  const loadList = useCallback(async (overrideFilters?: Partial<NewsListFilters>) => {
    // 이전 요청 취소
    abortControllerRef.current?.abort()
    const controller = new AbortController()
    abortControllerRef.current = controller

    // 필터 병합: 현재 상태 기반으로 override 적용 (DEFAULT_FILTERS 아닌 현재 filters)
    if (overrideFilters) {
      setFiltersState((prev) => ({ ...prev, ...overrideFilters }))
    }

    setLoading(true)
    setError(null)

    try {
      // override가 있으면 현재 상태에 merge, 없으면 현재 상태 그대로
      const effectiveFilters = overrideFilters
        ? { ...filters, ...overrideFilters }
        : filters

      const response: NewsListResponse = await fetchNewsList(
        {
          source: effectiveFilters.source,
          published_date: effectiveFilters.published_date,
          page: effectiveFilters.page,
          page_size: effectiveFilters.page_size,
        },
        controller.signal,
      )

      if (!controller.signal.aborted) {
        setItems(response.items)
        setTotal(response.total)
        setHasNext(response.has_next)
      }
    } catch (err) {
      if (!controller.signal.aborted) {
        setError(err instanceof Error ? err.message : '뉴스 목록 조회에 실패했습니다.')
      }
    } finally {
      if (!controller.signal.aborted) {
        setLoading(false)
      }
    }
  }, [filters])

  const setFilters = useCallback((newFilters: Partial<NewsListFilters>) => {
    setFiltersState((prev) => ({ ...prev, ...newFilters }))
  }, [])

  // 컴포넌트 언마운트 시 정리
  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort()
    }
  }, [])

  return {
    items,
    total,
    hasNext,
    loading,
    error,
    filters,
    setFilters,
    loadList,
  }
}
