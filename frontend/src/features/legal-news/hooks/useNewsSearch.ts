'use client'

import { useCallback, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import type { NewsSource, NewsSearchRequest } from '../types'
import { searchNews } from '../services'

export function useNewsSearch() {
  const [query, setQuery] = useState('')
  const [sourceFilter, setSourceFilter] = useState<NewsSource | null>(null)
  const [limit, setLimit] = useState(10)

  // 검색 실행 시점의 파라미터를 별도 상태로 관리 (submit 시에만 갱신)
  const [submittedParams, setSubmittedParams] = useState<NewsSearchRequest | null>(null)
  const [validationError, setValidationError] = useState<string | null>(null)

  const { data, isLoading, error: fetchError } = useQuery({
    queryKey: ['news-search', submittedParams?.query, submittedParams?.source, submittedParams?.limit],
    queryFn: () => searchNews(submittedParams!),
    enabled: !!submittedParams,
  })

  const search = useCallback((searchQuery?: string) => {
    const currentQuery = (searchQuery ?? query).trim()
    if (!currentQuery || currentQuery.length < 2) {
      setValidationError('검색어를 2자 이상 입력해주세요.')
      return
    }
    setValidationError(null)
    setSubmittedParams({ query: currentQuery, limit, source: sourceFilter })
  }, [query, limit, sourceFilter])

  const clearResults = useCallback(() => {
    setSubmittedParams(null)
    setValidationError(null)
  }, [])

  const searchError = validationError
    ?? (fetchError ? (fetchError instanceof Error ? fetchError.message : '검색에 실패했습니다.') : null)

  return {
    query,
    setQuery,
    sourceFilter,
    setSourceFilter,
    limit,
    setLimit,
    results: data?.results ?? [],
    totalResults: data?.total ?? 0,
    searching: isLoading,
    searchError,
    search,
    clearResults,
  }
}
