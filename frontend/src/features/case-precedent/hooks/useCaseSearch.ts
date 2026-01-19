'use client'

import { useState, useCallback } from 'react'
import { casePrecedentService } from '../services'
import type {
  PrecedentItem,
  PrecedentDetail,
  SearchFilters,
  AIQuestionResponse,
} from '../types'

interface UseCaseSearchReturn {
  // Search state
  searchResults: PrecedentItem[]
  totalResults: number
  isSearching: boolean
  searchError: string | null

  // Detail state
  selectedCase: PrecedentDetail | null
  isLoadingDetail: boolean
  detailError: string | null

  // AI Q&A state
  aiResponse: AIQuestionResponse | null
  isAskingAI: boolean
  aiError: string | null

  // Filters
  filters: SearchFilters
  setFilters: (filters: Partial<SearchFilters>) => void

  // Actions
  search: () => Promise<void>
  selectCase: (id: string) => Promise<void>
  clearSelection: () => void
  askAI: (question: string) => Promise<void>
  clearAIResponse: () => void
}

const DEFAULT_FILTERS: SearchFilters = {
  keyword: '',
  docType: '',
  court: '',
  limit: 20,
}

export function useCaseSearch(): UseCaseSearchReturn {
  // Search state
  const [searchResults, setSearchResults] = useState<PrecedentItem[]>([])
  const [totalResults, setTotalResults] = useState(0)
  const [isSearching, setIsSearching] = useState(false)
  const [searchError, setSearchError] = useState<string | null>(null)

  // Detail state
  const [selectedCase, setSelectedCase] = useState<PrecedentDetail | null>(null)
  const [isLoadingDetail, setIsLoadingDetail] = useState(false)
  const [detailError, setDetailError] = useState<string | null>(null)

  // AI Q&A state
  const [aiResponse, setAiResponse] = useState<AIQuestionResponse | null>(null)
  const [isAskingAI, setIsAskingAI] = useState(false)
  const [aiError, setAiError] = useState<string | null>(null)

  // Filters
  const [filters, setFiltersState] = useState<SearchFilters>(DEFAULT_FILTERS)

  const setFilters = useCallback((newFilters: Partial<SearchFilters>) => {
    setFiltersState((prev) => ({ ...prev, ...newFilters }))
  }, [])

  const search = useCallback(async () => {
    if (!filters.keyword.trim()) {
      setSearchError('검색어를 입력해주세요')
      return
    }

    setIsSearching(true)
    setSearchError(null)

    try {
      const response = await casePrecedentService.searchPrecedents(filters)
      setSearchResults(response.precedents)
      setTotalResults(response.total)
    } catch (error) {
      setSearchError('검색 중 오류가 발생했습니다')
      console.error('Search error:', error)
    } finally {
      setIsSearching(false)
    }
  }, [filters])

  const selectCase = useCallback(async (id: string) => {
    setIsLoadingDetail(true)
    setDetailError(null)
    setAiResponse(null)

    try {
      const detail = await casePrecedentService.getPrecedentDetail(id)
      setSelectedCase(detail)
    } catch (error) {
      setDetailError('판례 상세 정보를 불러오는 데 실패했습니다')
      console.error('Detail load error:', error)
    } finally {
      setIsLoadingDetail(false)
    }
  }, [])

  const clearSelection = useCallback(() => {
    setSelectedCase(null)
    setAiResponse(null)
  }, [])

  const askAI = useCallback(async (question: string) => {
    if (!selectedCase) {
      setAiError('먼저 판례를 선택해주세요')
      return
    }

    if (!question.trim()) {
      setAiError('질문을 입력해주세요')
      return
    }

    setIsAskingAI(true)
    setAiError(null)

    try {
      const response = await casePrecedentService.askAboutPrecedent(selectedCase.id, question)
      setAiResponse(response)
    } catch (error) {
      setAiError('AI 응답을 받는 데 실패했습니다')
      console.error('AI ask error:', error)
    } finally {
      setIsAskingAI(false)
    }
  }, [selectedCase])

  const clearAIResponse = useCallback(() => {
    setAiResponse(null)
    setAiError(null)
  }, [])

  return {
    searchResults,
    totalResults,
    isSearching,
    searchError,
    selectedCase,
    isLoadingDetail,
    detailError,
    aiResponse,
    isAskingAI,
    aiError,
    filters,
    setFilters,
    search,
    selectCase,
    clearSelection,
    askAI,
    clearAIResponse,
  }
}
