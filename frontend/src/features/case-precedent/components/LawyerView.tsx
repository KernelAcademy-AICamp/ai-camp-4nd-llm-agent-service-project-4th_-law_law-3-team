'use client'

import { useState, useEffect, useRef, useMemo, useCallback } from 'react'
import { useChat } from '@/context/ChatContext'
import { CaseDetailPanel } from './CaseDetailPanel'
import { CaseCard } from './CaseCard'
import { FilterPanel } from './FilterPanel'
import { FilteredResultList } from './FilteredResultList'
import { useCaseSearch } from '../hooks/useCaseSearch'
import { usePrecedentFilter } from '../hooks/usePrecedentFilter'
import type { ChatSource, DatePreset, SortOrder, PrecedentItem } from '../types'

function toYear(dateString: string | undefined | null): number | null {
  if (!dateString) return null
  const year = parseInt(dateString.slice(0, 4), 10)
  return isNaN(year) ? null : year
}

interface LawyerViewProps {
  initialCaseId?: string
}

export function LawyerView({ initialCaseId }: LawyerViewProps) {
  const { highlightedCaseNumber, setHighlightedCaseNumber, sessionData } = useChat()

  // ── 데이터 소스 1: 채팅 참조 (aiReferences) ──
  const chatSearch = useCaseSearch(initialCaseId)

  // ── 데이터 소스 2: 서버사이드 필터 검색 ──
  const serverSearch = usePrecedentFilter()

  // 모드: 채팅 참조가 있으면 client, 없으면 server
  const hasChatReferences = chatSearch.searchResults.length > 0

  // ── 클라이언트 필터 상태 (채팅 참조 모드용) ──
  const [clientKeyword, setClientKeyword] = useState('')
  const [clientCaseType, setClientCaseType] = useState('')
  const [clientDatePreset, setClientDatePreset] = useState<DatePreset>('all')
  const [clientDateFrom, setClientDateFrom] = useState('')
  const [clientDateTo, setClientDateTo] = useState('')
  const [clientSortOrder, setClientSortOrder] = useState<SortOrder>('relevance')

  const [showFilter, setShowFilter] = useState(true)

  // ── 클라이언트 사이드 필터링 (채팅 참조 모드) ──
  const refMetaMap = useMemo(() => {
    const map = new Map<string, { case_type?: string; decision_date?: string }>()
    const refs = sessionData.aiReferences as ChatSource[] | undefined
    if (refs && Array.isArray(refs)) {
      refs.forEach((ref) => {
        const key = ref.case_number || ref.law_name || ''
        if (key) {
          map.set(key, { case_type: ref.case_type, decision_date: ref.decision_date })
        }
      })
    }
    return map
  }, [sessionData.aiReferences])

  const clientCaseTypes = useMemo(() => {
    const types = new Set<string>()
    refMetaMap.forEach((meta) => {
      if (meta.case_type) types.add(meta.case_type)
    })
    return Array.from(types).sort()
  }, [refMetaMap])

  const clientDateRange = useMemo(() => {
    if (clientDatePreset === 'all') return { from: null, to: null }
    if (clientDatePreset === 'custom') {
      return {
        from: clientDateFrom ? parseInt(clientDateFrom, 10) : null,
        to: clientDateTo ? parseInt(clientDateTo, 10) : null,
      }
    }
    const years = clientDatePreset === '3y' ? 3 : clientDatePreset === '5y' ? 5 : 10
    const currentYear = new Date().getFullYear()
    return { from: currentYear - years, to: currentYear }
  }, [clientDatePreset, clientDateFrom, clientDateTo])

  const filteredChatResults = useMemo(() => {
    let result = [...chatSearch.searchResults]

    if (clientKeyword) {
      const lower = clientKeyword.toLowerCase()
      result = result.filter((r) =>
        r.case_name?.toLowerCase().includes(lower) ||
        r.case_number?.toLowerCase().includes(lower) ||
        r.summary?.toLowerCase().includes(lower)
      )
    }

    if (clientCaseType) {
      result = result.filter((r) => {
        const meta = refMetaMap.get(r.case_number)
        return meta?.case_type === clientCaseType
      })
    }

    if (clientDateRange.from !== null || clientDateRange.to !== null) {
      result = result.filter((r) => {
        const meta = refMetaMap.get(r.case_number)
        const year = toYear(meta?.decision_date || r.date)
        if (year === null) return false
        if (clientDateRange.from !== null && year < clientDateRange.from) return false
        if (clientDateRange.to !== null && year > clientDateRange.to) return false
        return true
      })
    }

    if (clientSortOrder === 'relevance' && clientKeyword) {
      const lower = clientKeyword.toLowerCase()
      result.sort((a, b) => {
        const scoreA =
          (a.case_name?.toLowerCase().includes(lower) ? 3 : 0) +
          (a.case_number?.toLowerCase().includes(lower) ? 2 : 0) +
          (a.summary?.toLowerCase().includes(lower) ? 1 : 0)
        const scoreB =
          (b.case_name?.toLowerCase().includes(lower) ? 3 : 0) +
          (b.case_number?.toLowerCase().includes(lower) ? 2 : 0) +
          (b.summary?.toLowerCase().includes(lower) ? 1 : 0)
        return scoreB - scoreA
      })
    } else if (clientSortOrder === 'latest') {
      result.sort((a, b) => {
        const metaA = refMetaMap.get(a.case_number)
        const metaB = refMetaMap.get(b.case_number)
        const dateA = metaA?.decision_date || a.date || ''
        const dateB = metaB?.decision_date || b.date || ''
        return dateB.localeCompare(dateA)
      })
    }

    return result
  }, [chatSearch.searchResults, clientKeyword, clientCaseType, clientDateRange, clientSortOrder, refMetaMap])

  // ── 서버 사이드 결과를 PrecedentItem으로 정규화 ──
  const normalizedServerResults: PrecedentItem[] = useMemo(() =>
    serverSearch.precedents.map((p) => ({
      id: p.id,
      case_name: p.case_name || '',
      case_number: p.case_number || '',
      doc_type: 'precedent',
      court: p.court_name || '',
      date: p.decision_date || '',
      summary: p.summary || '',
      similarity: 0,
    }))
  , [serverSearch.precedents])

  // ── 통합 데이터 접근 ──
  const keyword = hasChatReferences ? clientKeyword : serverSearch.keyword
  const setKeyword = hasChatReferences ? setClientKeyword : serverSearch.setKeyword
  const caseType = hasChatReferences ? clientCaseType : serverSearch.caseType
  const setCaseType = hasChatReferences ? setClientCaseType : serverSearch.setCaseType
  const datePreset = hasChatReferences ? clientDatePreset : serverSearch.datePreset
  const setDatePreset = hasChatReferences ? setClientDatePreset : serverSearch.setDatePreset
  const dateFrom = hasChatReferences ? clientDateFrom : serverSearch.dateFrom
  const setDateFrom = hasChatReferences ? setClientDateFrom : serverSearch.setDateFrom
  const dateTo = hasChatReferences ? clientDateTo : serverSearch.dateTo
  const setDateTo = hasChatReferences ? setClientDateTo : serverSearch.setDateTo
  const sortOrder = hasChatReferences ? clientSortOrder : serverSearch.sortOrder
  const setSortOrder = hasChatReferences ? setClientSortOrder : serverSearch.setSortOrder
  const caseTypes = hasChatReferences ? clientCaseTypes : serverSearch.caseTypes

  const results = hasChatReferences ? filteredChatResults : normalizedServerResults
  const total = hasChatReferences ? filteredChatResults.length : serverSearch.total
  const isLoading = hasChatReferences ? chatSearch.isSearching : serverSearch.isLoading

  const handleSearch = hasChatReferences ? () => {} : serverSearch.search

  const chatSelectCase = chatSearch.selectCase
  const serverSelectItem = serverSearch.selectItem
  const handleSelect = useCallback((id: string) => {
    if (hasChatReferences) {
      chatSelectCase(id)
    } else {
      serverSelectItem(id)
    }
  }, [hasChatReferences, chatSelectCase, serverSelectItem])

  const detail = hasChatReferences ? chatSearch.selectedCase : serverSearch.detail
  const isDetailLoading = hasChatReferences ? chatSearch.isLoadingDetail : serverSearch.isDetailLoading
  const detailError = hasChatReferences ? chatSearch.detailError : serverSearch.detailError

  // 서버 모드: 정렬 변경 시 재검색
  const isFirstRender = useRef(true)
  useEffect(() => {
    if (hasChatReferences) return
    if (isFirstRender.current) {
      isFirstRender.current = false
      return
    }
    if (serverSearch.hasSearched) {
      serverSearch.search()
    }
  }, [serverSearch.sortOrder]) // eslint-disable-line react-hooks/exhaustive-deps

  // ── 하이라이트 + 스크롤 ──
  const prevHighlightRef = useRef<string | null>(null)
  const cardRefs = useRef<Map<string, HTMLDivElement>>(new Map())

  useEffect(() => {
    if (!highlightedCaseNumber) return
    if (highlightedCaseNumber === prevHighlightRef.current) return
    prevHighlightRef.current = highlightedCaseNumber

    const matchingResult = chatSearch.searchResults.find(
      (r) => r.case_number && r.case_number.includes(highlightedCaseNumber)
    )
    if (matchingResult) {
      chatSelectCase(matchingResult.id)
      const cardEl = cardRefs.current.get(matchingResult.id)
      if (cardEl) {
        cardEl.scrollIntoView({ behavior: 'smooth', block: 'center' })
      }
    }

    const timer = setTimeout(() => {
      setHighlightedCaseNumber(null)
      prevHighlightRef.current = null
    }, 3000)
    return () => clearTimeout(timer)
  }, [highlightedCaseNumber, chatSearch.searchResults, chatSelectCase, setHighlightedCaseNumber])

  // ── 참조 조문 ──
  const provisions = useMemo(() =>
    detail?.reference_provisions
      ? detail.reference_provisions.split(',').map((s) => s.trim()).filter(Boolean)
      : []
  , [detail?.reference_provisions])

  const [isProvisionsOpen, setIsProvisionsOpen] = useState(false)

  // 빈 상태 메시지
  const emptyMessage = (() => {
    if (hasChatReferences && results.length === 0) {
      return { main: '필터 조건에 맞는 결과가 없습니다', hint: datePreset !== 'all' }
    }
    if (!hasChatReferences && !serverSearch.hasSearched) {
      return { main: '필터를 설정하고 검색하세요', hint: false }
    }
    if (!hasChatReferences && serverSearch.hasSearched && results.length === 0) {
      return { main: '검색 결과가 없습니다', hint: datePreset !== 'all' }
    }
    return null
  })()

  return (
    <div className="h-full flex overflow-hidden">
      {/* Left Panel */}
      <div className="w-80 bg-white border-r border-gray-200 flex flex-col h-full">
        {/* 헤더: 결과 수 + 필터 토글 */}
        <div className="px-4 py-2 bg-gray-50 border-b flex items-center justify-between">
          <span className="text-sm text-gray-600">
            {isLoading ? '검색 중...' : hasChatReferences ? `관련 문서 (${total}건)` : `판례 검색`}
          </span>
          <button
            onClick={() => setShowFilter(!showFilter)}
            className={`px-3 py-1 text-xs rounded-lg border transition-colors ${
              showFilter
                ? 'bg-blue-50 border-blue-300 text-blue-600'
                : 'bg-white border-gray-300 text-gray-600 hover:border-gray-400'
            }`}
          >
            {showFilter ? '필터 닫기' : '필터'}
          </button>
        </div>

        {/* 필터 패널 */}
        {showFilter && (
          <FilterPanel
            keyword={keyword}
            onKeywordChange={setKeyword}
            caseType={caseType}
            onCaseTypeChange={setCaseType}
            datePreset={datePreset}
            onDatePresetChange={setDatePreset}
            dateFrom={dateFrom}
            onDateFromChange={setDateFrom}
            dateTo={dateTo}
            onDateToChange={setDateTo}
            caseTypes={caseTypes}
            onSearch={handleSearch}
            isLoading={isLoading}
          />
        )}

        {hasChatReferences ? (
          <>
            {/* 정렬 (채팅 참조 모드) */}
            <div className="flex items-center justify-between px-4 py-2 border-b border-gray-100">
              <span className="text-xs text-gray-500">총 {total.toLocaleString()}건</span>
              <div className="flex gap-1 text-xs">
                <button
                  onClick={() => setSortOrder('relevance')}
                  className={`px-2 py-0.5 rounded ${
                    sortOrder === 'relevance'
                      ? 'text-blue-600 font-medium'
                      : 'text-gray-400 hover:text-gray-600'
                  }`}
                >
                  정확도순
                </button>
                <span className="text-gray-300">|</span>
                <button
                  onClick={() => setSortOrder('latest')}
                  className={`px-2 py-0.5 rounded ${
                    sortOrder === 'latest'
                      ? 'text-blue-600 font-medium'
                      : 'text-gray-400 hover:text-gray-600'
                  }`}
                >
                  최신순
                </button>
              </div>
            </div>

            {/* 결과 목록 (채팅 참조 모드) */}
            <div className="flex-1 overflow-y-auto p-3 space-y-2">
              {isLoading ? (
                <div className="flex items-center justify-center h-32">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600" />
                </div>
              ) : filteredChatResults.length > 0 ? (
                filteredChatResults.map((case_) => {
                  const isHighlighted = !!(
                    highlightedCaseNumber &&
                    case_.case_number &&
                    case_.case_number.includes(highlightedCaseNumber)
                  )
                  return (
                    <div
                      key={case_.id}
                      ref={(el) => {
                        if (el) cardRefs.current.set(case_.id, el)
                      }}
                      className={isHighlighted ? 'ring-2 ring-yellow-400 rounded-lg animate-pulse' : ''}
                    >
                      <CaseCard
                        case_={case_}
                        selected={detail?.id === case_.id}
                        onSelect={handleSelect}
                      />
                    </div>
                  )
                })
              ) : (
                <div className="p-6 text-center text-gray-400">
                  <svg
                    className="w-12 h-12 mx-auto mb-3 text-gray-300"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={1.5}
                      d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                    />
                  </svg>
                  <p className="text-sm">필터 조건에 맞는 결과가 없습니다</p>
                  {datePreset !== 'all' && (
                    <p className="text-xs text-gray-400 mt-2">
                      기간을 &apos;전체&apos;로 변경하면 더 많은 결과를 볼 수 있습니다
                    </p>
                  )}
                </div>
              )}
            </div>
          </>
        ) : (
          /* 서버 검색 모드: FilterablePrecedentView와 동일한 상세 리스트 */
          <FilteredResultList
            precedents={serverSearch.precedents}
            total={serverSearch.total}
            selectedId={serverSearch.selectedId}
            onSelect={serverSelectItem}
            hasMore={serverSearch.hasMore}
            onLoadMore={serverSearch.loadMore}
            isLoading={serverSearch.isLoading}
            error={serverSearch.error}
            hasSearched={serverSearch.hasSearched}
            hasDateFilter={serverSearch.datePreset !== 'all'}
            highlightKeyword={serverSearch.keyword}
            sortOrder={serverSearch.sortOrder}
            onSortChange={serverSearch.setSortOrder}
          />
        )}

        {/* 참조 조문 */}
        {provisions.length > 0 && (
          <div className="border-t border-gray-200">
            <button
              onClick={() => setIsProvisionsOpen(!isProvisionsOpen)}
              className="w-full flex items-center gap-2 px-4 py-3 hover:bg-gray-50 transition-colors"
            >
              <svg className="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25" />
              </svg>
              <span className="text-sm font-medium text-gray-700">참조 조문</span>
              <span className="text-xs text-gray-400">{provisions.length}</span>
              <svg
                className={`w-4 h-4 text-gray-400 ml-auto transition-transform ${isProvisionsOpen ? 'rotate-180' : ''}`}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
              </svg>
            </button>
            {isProvisionsOpen && (
              <ul className="px-4 pb-3 space-y-1 max-h-48 overflow-y-auto">
                {provisions.map((provision, idx) => (
                  <li
                    key={idx}
                    className="text-sm text-gray-700 py-1.5 px-3 rounded hover:bg-gray-50 cursor-default"
                  >
                    {provision}
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>

      {/* Right Panel - Detail */}
      <CaseDetailPanel
        case_={detail}
        isLoading={isDetailLoading}
        error={detailError}
      />
    </div>
  )
}
