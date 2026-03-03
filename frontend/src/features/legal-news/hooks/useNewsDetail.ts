'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import type { NewsArticleResponse } from '../types'
import { fetchNewsDetail } from '../services'

export function useNewsDetail() {
  const [selectedArticle, setSelectedArticle] = useState<NewsArticleResponse | null>(null)
  const [detailLoading, setDetailLoading] = useState(false)
  const [detailError, setDetailError] = useState<string | null>(null)
  const abortControllerRef = useRef<AbortController | null>(null)

  const selectArticle = useCallback(async (articleId: string) => {
    // 이전 요청 취소
    abortControllerRef.current?.abort()
    const controller = new AbortController()
    abortControllerRef.current = controller

    setSelectedArticle(null)
    setDetailLoading(true)
    setDetailError(null)

    try {
      const detail = await fetchNewsDetail(articleId, controller.signal)
      if (!controller.signal.aborted) {
        setSelectedArticle(detail)
      }
    } catch (err) {
      if (!controller.signal.aborted) {
        setDetailError(err instanceof Error ? err.message : '기사 상세 조회에 실패했습니다.')
      }
    } finally {
      if (!controller.signal.aborted) {
        setDetailLoading(false)
      }
    }
  }, [])

  const clearSelection = useCallback(() => {
    abortControllerRef.current?.abort()
    setSelectedArticle(null)
    setDetailError(null)
  }, [])

  // 컴포넌트 언마운트 시 정리
  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort()
    }
  }, [])

  return {
    selectedArticle,
    detailLoading,
    detailError,
    selectArticle,
    clearSelection,
  }
}
