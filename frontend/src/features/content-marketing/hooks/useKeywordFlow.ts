'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import type {
  KeywordItem,
  KeywordNewsResponse,
  KeywordStreamEvent,
  LawyerPersona,
  NewsArticle,
  SourceFailInfo,
  TimeRange,
  TrendCategory,
} from '../types'
import { clearKeywordCache, searchKeywordNews, streamKeywordCollect } from '../services'

export type KeywordFlowStep = 'idle' | 'collecting' | 'keywords' | 'searching' | 'news'

export interface StreamProgress {
  progress: number
  message: string
}

const TIME_RANGE_STORAGE_KEY = 'keyword-time-range'
const CATEGORY_STORAGE_KEY = 'keyword-category'

function getInitialTimeRange(): TimeRange {
  if (typeof window === 'undefined') return '48h'
  return (localStorage.getItem(TIME_RANGE_STORAGE_KEY) as TimeRange) || '48h'
}

function getInitialCategory(): TrendCategory {
  if (typeof window === 'undefined') return 'all'
  return (localStorage.getItem(CATEGORY_STORAGE_KEY) as TrendCategory) || 'all'
}

export function useKeywordFlow(
  personaId: string | null = null,
  persona?: LawyerPersona | null,
) {
  // 페르소나에서 첫 번째 비-'all' 전문분야 추출
  const personaCategory = persona?.specialty_areas?.find((a) => a !== 'all') ?? null

  const [step, setStep] = useState<KeywordFlowStep>('idle')
  const [keywords, setKeywords] = useState<KeywordItem[]>([])
  const [selectedKeyword, setSelectedKeyword] = useState<KeywordItem | null>(null)
  const [newsResponse, setNewsResponse] = useState<KeywordNewsResponse | null>(null)
  const [selectedArticles, setSelectedArticles] = useState<NewsArticle[]>([])
  const [error, setError] = useState<string | null>(null)
  const [cacheHit, setCacheHit] = useState(false)
  const [sourcesUsed, setSourcesUsed] = useState<string[]>([])
  const [sourcesFailed, setSourcesFailed] = useState<SourceFailInfo[]>([])
  const [streamProgress, setStreamProgress] = useState<StreamProgress | null>(null)
  const [collectedAt, setCollectedAt] = useState<string | null>(null)
  const [timeRange, setTimeRangeState] = useState<TimeRange>(getInitialTimeRange)
  const [category, setCategoryState] = useState<TrendCategory>(() => {
    if (personaCategory) return personaCategory
    return getInitialCategory()
  })
  const abortRef = useRef<AbortController | null>(null)

  // 페르소나 변경 시 카테고리 동기화
  const prevPersonaCategoryRef = useRef(personaCategory)
  useEffect(() => {
    if (personaCategory && personaCategory !== prevPersonaCategoryRef.current) {
      setCategoryState(personaCategory)
      localStorage.setItem(CATEGORY_STORAGE_KEY, personaCategory)
    }
    prevPersonaCategoryRef.current = personaCategory
  }, [personaCategory])

  const setTimeRange = useCallback((value: TimeRange) => {
    setTimeRangeState(value)
    localStorage.setItem(TIME_RANGE_STORAGE_KEY, value)
  }, [])

  const setCategory = useCallback((value: TrendCategory) => {
    setCategoryState(value)
    localStorage.setItem(CATEGORY_STORAGE_KEY, value)
  }, [])

  const handleCollect = useCallback((forceRefresh: boolean = false) => {
    setStep('collecting')
    setError(null)
    setKeywords([])
    setSelectedKeyword(null)
    setNewsResponse(null)
    setCacheHit(false)
    setSourcesUsed([])
    setSourcesFailed([])
    setCollectedAt(null)
    setStreamProgress({ progress: 0, message: '준비 중...' })

    // 이전 스트림 취소
    if (abortRef.current) {
      abortRef.current.abort()
    }

    abortRef.current = streamKeywordCollect(
      (event: KeywordStreamEvent) => {
        // 캐시 히트 시 프로그레스 바 없이 즉시 결과 표시
        if (event.step === 'cache_hit') {
          setStreamProgress(null)
          return
        }

        if (event.step === 'done' && event.data) {
          setKeywords(event.data.keywords)
          setCacheHit(event.data.cache_hit)
          setSourcesUsed(event.data.sources_used ?? [])
          setSourcesFailed(event.data.sources_failed ?? [])
          setCollectedAt(event.data.collected_at)
          setStreamProgress(null)
          setStep('keywords')
        } else if (event.step === 'error') {
          setError(event.error || '키워드 수집에 실패했습니다.')
          setStreamProgress(null)
          setStep('idle')
        } else {
          setStreamProgress({
            progress: event.progress,
            message: event.message,
          })
        }
      },
      (errorMessage: string) => {
        setError(errorMessage)
        setStreamProgress(null)
        setStep('idle')
      },
      10,
      timeRange,
      forceRefresh,
      category,
      personaId,
    )
  }, [timeRange, category, personaId])

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

  const handleToggleArticle = useCallback((article: NewsArticle) => {
    setSelectedArticles((prev) => {
      const exists = prev.some((a) => a.url === article.url)
      if (exists) {
        return prev.filter((a) => a.url !== article.url)
      }
      return [...prev, article]
    })
  }, [])

  const handleBack = useCallback(() => {
    setSelectedKeyword(null)
    setNewsResponse(null)
    setSelectedArticles([])
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
    setSelectedArticles([])
    setError(null)
    setCacheHit(false)
    setSourcesUsed([])
    setSourcesFailed([])
    setStreamProgress(null)
    setCollectedAt(null)
  }, [])

  const [isClearingCache, setIsClearingCache] = useState(false)

  const handleClearCache = useCallback(async () => {
    setIsClearingCache(true)
    setError(null)
    try {
      await clearKeywordCache()
      handleReset()
      // 캐시 삭제 후 강제 재수집으로 새 결과를 즉시 표시
      handleCollect(true)
    } catch (err) {
      const message = err instanceof Error ? err.message : '캐시 초기화에 실패했습니다.'
      setError(message)
    } finally {
      setIsClearingCache(false)
    }
  }, [handleReset, handleCollect])

  return {
    step,
    keywords,
    selectedKeyword,
    newsResponse,
    selectedArticles,
    error,
    cacheHit,
    sourcesUsed,
    sourcesFailed,
    streamProgress,
    collectedAt,
    timeRange,
    setTimeRange,
    category,
    setCategory,
    handleCollect,
    handleSearchNews,
    handleToggleArticle,
    handleBack,
    handleReset,
    handleClearCache,
    isClearingCache,
  }
}
