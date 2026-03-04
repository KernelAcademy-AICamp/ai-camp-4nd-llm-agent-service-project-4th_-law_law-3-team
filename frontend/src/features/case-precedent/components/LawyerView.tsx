'use client'

import { useEffect, useRef } from 'react'
import { useChat } from '@/context/ChatContext'
import { SearchPanel } from '@/features/case-precedent/components/SearchPanel'
import { CaseDetailPanel } from '@/features/case-precedent/components/CaseDetailPanel'
import { useCaseSearch } from '@/features/case-precedent/hooks/useCaseSearch'

interface LawyerViewProps {
  initialCaseId?: string
}

export function LawyerView({ initialCaseId }: LawyerViewProps) {
  const { highlightedCaseNumber, setHighlightedCaseNumber } = useChat()
  const {
    searchResults,
    totalResults,
    isSearching,
    selectedCase,
    isLoadingDetail,
    detailError,
    selectCase,
  } = useCaseSearch(initialCaseId)

  const prevHighlightRef = useRef<string | null>(null)

  // 채팅 답변 내 판례번호 클릭 → 왼쪽 패널에서 해당 문서 선택
  useEffect(() => {
    if (!highlightedCaseNumber) return
    if (highlightedCaseNumber === prevHighlightRef.current) return
    prevHighlightRef.current = highlightedCaseNumber

    const matchingResult = searchResults.find(
      (r) => r.case_number && r.case_number.includes(highlightedCaseNumber)
    )
    if (matchingResult) {
      selectCase(matchingResult.id)
    }

    // 3초 후 하이라이트 해제
    const timer = setTimeout(() => {
      setHighlightedCaseNumber(null)
      prevHighlightRef.current = null
    }, 3000)
    return () => clearTimeout(timer)
  }, [highlightedCaseNumber, searchResults, selectCase, setHighlightedCaseNumber])

  return (
    <div className="h-full flex overflow-hidden">
      {/* Left Panel - Search Results */}
      <SearchPanel
        results={searchResults}
        totalResults={totalResults}
        isSearching={isSearching}
        error={null}
        selectedCaseId={selectedCase?.id || null}
        selectedCase={selectedCase}
        onCaseSelect={selectCase}
        highlightedCaseNumber={highlightedCaseNumber}
      />

      {/* Right Panel - Detail */}
      <CaseDetailPanel
        case_={selectedCase}
        isLoading={isLoadingDetail}
        error={detailError}
      />
    </div>
  )
}
