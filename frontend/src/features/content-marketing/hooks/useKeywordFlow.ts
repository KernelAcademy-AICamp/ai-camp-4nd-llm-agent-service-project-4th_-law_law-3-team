'use client'

import { useCallback, useRef, useState } from 'react'
import type {
  KeywordItem,
  KeywordNewsResponse,
  KeywordStreamEvent,
} from '../types'
import { searchKeywordNews, streamKeywordCollect } from '../services'

export type KeywordFlowStep = 'idle' | 'collecting' | 'keywords' | 'searching' | 'news'

export interface StreamProgress {
  progress: number
  message: string
}

export function useKeywordFlow() {
  const [step, setStep] = useState<KeywordFlowStep>('idle')
  const [keywords, setKeywords] = useState<KeywordItem[]>([])
  const [selectedKeyword, setSelectedKeyword] = useState<KeywordItem | null>(null)
  const [newsResponse, setNewsResponse] = useState<KeywordNewsResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [cacheHit, setCacheHit] = useState(false)
  const [streamProgress, setStreamProgress] = useState<StreamProgress | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  const handleCollect = useCallback(() => {
    setStep('collecting')
    setError(null)
    setKeywords([])
    setSelectedKeyword(null)
    setNewsResponse(null)
    setStreamProgress({ progress: 0, message: '준비 중...' })

    // 이전 스트림 취소
    if (abortRef.current) {
      abortRef.current.abort()
    }

    abortRef.current = streamKeywordCollect(
      (event: KeywordStreamEvent) => {
        setStreamProgress({
          progress: event.progress,
          message: event.message,
        })

        if (event.step === 'done' && event.data) {
          setKeywords(event.data.keywords)
          setCacheHit(event.data.cache_hit)
          setStreamProgress(null)
          setStep('keywords')
        } else if (event.step === 'error') {
          setError(event.error || '키워드 수집에 실패했습니다.')
          setStreamProgress(null)
          setStep('idle')
        }
      },
      (errorMessage: string) => {
        setError(errorMessage)
        setStreamProgress(null)
        setStep('idle')
      },
      10,
    )
  }, [])

  const handleSearchNews = useCallback(async (keyword: KeywordItem) => {
    setSelectedKeyword(keyword)
    setStep('searching')
    setError(null)
    setNewsResponse(null)

    try {
      const response = await searchKeywordNews(keyword.id, {
        max_results: 10,
      })
      setNewsResponse(response)
      setStep('news')
    } catch (err) {
      const message = err instanceof Error ? err.message : '뉴스 검색에 실패했습니다.'
      setError(message)
      setStep('keywords')
    }
  }, [])

  const handleBack = useCallback(() => {
    setSelectedKeyword(null)
    setNewsResponse(null)
    setStep('keywords')
  }, [])

  const handleReset = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort()
      abortRef.current = null
    }
    setStep('idle')
    setKeywords([])
    setSelectedKeyword(null)
    setNewsResponse(null)
    setError(null)
    setCacheHit(false)
    setStreamProgress(null)
  }, [])

  return {
    step,
    keywords,
    selectedKeyword,
    newsResponse,
    error,
    cacheHit,
    streamProgress,
    handleCollect,
    handleSearchNews,
    handleBack,
    handleReset,
  }
}
