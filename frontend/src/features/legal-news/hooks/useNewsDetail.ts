'use client'

import { useCallback, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import type { NewsArticleResponse } from '../types'
import { fetchNewsDetail } from '../services'

export function useNewsDetail() {
  const [articleId, setArticleId] = useState<string | null>(null)

  const { data, isLoading, error } = useQuery<NewsArticleResponse>({
    queryKey: ['news-detail', articleId],
    queryFn: () => fetchNewsDetail(articleId!),
    enabled: !!articleId,
  })

  const selectArticle = useCallback((id: string) => {
    setArticleId(id)
  }, [])

  const clearSelection = useCallback(() => {
    setArticleId(null)
  }, [])

  return {
    selectedArticle: articleId ? (data ?? null) : null,
    detailLoading: isLoading && !!articleId,
    detailError: error ? (error instanceof Error ? error.message : '기사 상세 조회에 실패했습니다.') : null,
    selectArticle,
    clearSelection,
  }
}
