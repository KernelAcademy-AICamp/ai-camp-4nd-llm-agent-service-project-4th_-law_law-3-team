'use client'

import { useCallback, useState } from 'react'
import { keepPreviousData, useQuery } from '@tanstack/react-query'
import type { NewsListFilters } from '../types'
import { fetchNewsList } from '../services'

const DEFAULT_FILTERS: NewsListFilters = {
  source: null,
  published_date: null,
  page: 1,
  page_size: 20,
}

export function useNewsList() {
  const [filters, setFiltersState] = useState<NewsListFilters>(DEFAULT_FILTERS)

  const { data, isLoading, error } = useQuery({
    queryKey: ['news-list', filters.source, filters.published_date, filters.page, filters.page_size],
    queryFn: () =>
      fetchNewsList({
        source: filters.source,
        published_date: filters.published_date,
        page: filters.page,
        page_size: filters.page_size,
      }),
    placeholderData: keepPreviousData,
  })

  const setFilters = useCallback((newFilters: Partial<NewsListFilters>) => {
    setFiltersState((prev) => ({ ...prev, ...newFilters }))
  }, [])

  return {
    items: data?.items ?? [],
    total: data?.total ?? 0,
    hasNext: data?.has_next ?? false,
    loading: isLoading,
    error: error ? (error instanceof Error ? error.message : '뉴스 목록 조회에 실패했습니다.') : null,
    filters,
    setFilters,
  }
}
