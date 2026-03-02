'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import type { NewsSource, NewsSearchResult, NewsSearchResponse } from '../types'
import { searchNews } from '../services'

export function useNewsSearch() {
  const [query, setQuery] = useState('')
  const [sourceFilter, setSourceFilter] = useState<NewsSource | null>(null)
  const [limit, setLimit] = useState(10)
  const [results, setResults] = useState<NewsSearchResult[]>([])
  const [totalResults, setTotalResults] = useState(0)
  const [searching, setSearching] = useState(false)
  const [searchError, setSearchError] = useState<string | null>(null)
  const abortControllerRef = useRef<AbortController | null>(null)

  const search = useCallback(async (searchQuery?: string) => {
    const currentQuery = searchQuery ?? query
    if (!currentQuery.trim() || currentQuery.trim().length < 2) {
      setSearchError('검색어를 2자 이상 입력해주세요.')
      return
    }

    // 이전 요청 취소
    abortControllerRef.current?.abort()
    const controller = new AbortController()
    abortControllerRef.current = controller

    setSearching(true)
    setSearchError(null)

    try {
      const response: NewsSearchResponse = await searchNews(
        {
          query: currentQuery.trim(),
          limit,
          source: sourceFilter,
        },
        controller.signal,
      )

      if (!controller.signal.aborted) {
        setResults(response.results)
        setTotalResults(response.total)
      }
    } catch (err) {
      if (!controller.signal.aborted) {
        setSearchError(err instanceof Error ? err.message : '검색에 실패했습니다.')
      }
    } finally {
      if (!controller.signal.aborted) {
        setSearching(false)
      }
    }
  }, [query, limit, sourceFilter])

  const clearResults = useCallback(() => {
    abortControllerRef.current?.abort()
    setResults([])
    setTotalResults(0)
    setSearchError(null)
  }, [])

  // 컴포넌트 언마운트 시 정리
  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort()
    }
  }, [])

  return {
    query,
    setQuery,
    sourceFilter,
    setSourceFilter,
    limit,
    setLimit,
    results,
    totalResults,
    searching,
    searchError,
    search,
    clearResults,
  }
}
