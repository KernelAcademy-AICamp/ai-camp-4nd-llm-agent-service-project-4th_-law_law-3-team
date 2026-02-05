'use client'

import { useState, useCallback, useEffect } from 'react'
import { casePrecedentService } from '../services'
import { useChat } from '@/context/ChatContext'
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
  const { sessionData } = useChat()

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

  // Handle AI Generated Cases from Chat
  useEffect(() => {
    const rawRefs = sessionData.aiReferences as PrecedentDetail[] | undefined
    if (rawRefs && Array.isArray(rawRefs) && rawRefs.length > 0) {
      // 중복 제거: case_number(판례) 또는 law_name(법령) 기준
      const seen = new Set<string>()
      const refs = rawRefs.filter((ref) => {
        const key = ref.doc_type === 'law'
          ? (ref as Record<string, unknown>).law_name as string
          : ref.case_number
        if (!key || seen.has(key)) return false
        seen.add(key)
        return true
      })
      // aiReferences에서 모든 판례를 검색 결과 목록에 추가
      const newResults: PrecedentItem[] = refs.map((ref, idx) => ({
        id: ref.id || `ai-ref-${idx}-${Date.now()}`,
        case_name: ref.case_name || '',
        case_number: ref.case_number || '',
        doc_type: ref.doc_type || 'precedent',
        court: ref.court || ref.court_name || '',
        date: ref.date || ref.decision_date || '',
        summary: ref.summary || '',
        similarity: 100 - idx,
      }))
      setSearchResults(newResults)
      setTotalResults(newResults.length)
      // 첫 번째 판례를 자동 선택
      const firstCase = refs[0] as PrecedentDetail
      setSelectedCase({
        ...firstCase,
        id: firstCase.id || newResults[0].id,
      })
    } else if (sessionData.aiGeneratedCase) {
      // aiReferences가 없으면 기존 방식으로 단일 판례 처리
      const aiCase = sessionData.aiGeneratedCase as PrecedentDetail
      setSelectedCase(aiCase)
      setSearchResults((prev) => {
        if (prev.some((item) => item.id === aiCase.id)) return prev
        return [
          {
            id: aiCase.id,
            case_name: aiCase.case_name,
            case_number: aiCase.case_number,
            doc_type: aiCase.doc_type,
            court: aiCase.court,
            date: aiCase.date,
            summary: aiCase.summary,
            similarity: 100,
          },
          ...prev,
        ]
      })
    }
  }, [sessionData.aiReferences, sessionData.aiGeneratedCase])

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

    // aiReferences에서 먼저 검색 (채팅에서 전달받은 판례)
    const refs = sessionData.aiReferences as PrecedentDetail[] | undefined
    if (refs && Array.isArray(refs)) {
      // searchResults에서 해당 ID의 case_number를 찾아 aiReferences와 매칭
      const matchedResult = searchResults.find((r) => r.id === id)
      if (matchedResult) {
        const found = refs.find(
          (ref) => ref.case_number === matchedResult.case_number
        )
        if (found) {
          setSelectedCase({ ...found, id } as PrecedentDetail)
          setIsLoadingDetail(false)
          return
        }
      }
    }

    try {
      const detail = await casePrecedentService.getPrecedentDetail(id)
      setSelectedCase(detail)
    } catch (error) {
      setDetailError('판례 상세 정보를 불러오는 데 실패했습니다')
      console.error('Detail load error:', error)
    } finally {
      setIsLoadingDetail(false)
    }
  }, [sessionData.aiReferences, searchResults])

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
