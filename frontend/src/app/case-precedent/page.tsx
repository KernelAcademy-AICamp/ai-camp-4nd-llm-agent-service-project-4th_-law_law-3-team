'use client'

import { SearchPanel, CaseDetailPanel } from '@/features/case-precedent/components'
import { useCaseSearch } from '@/features/case-precedent/hooks/useCaseSearch'

export default function CasePrecedentPage() {
  const {
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
    askAI,
    clearAIResponse,
  } = useCaseSearch()

  return (
    <div className="h-screen flex flex-col bg-gray-100">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="flex items-center gap-3">
          <span className="text-2xl">📚</span>
          <div>
            <h1 className="text-xl font-bold text-gray-900">판례 검색</h1>
            <p className="text-sm text-gray-500">RAG 기반 상황 분석 및 관련 판례 제공</p>
          </div>
        </div>
      </header>

      {/* Main Content - Split View */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Panel - Search */}
        <SearchPanel
          results={searchResults}
          totalResults={totalResults}
          isSearching={isSearching}
          error={searchError}
          filters={filters}
          selectedCaseId={selectedCase?.id || null}
          onFilterChange={setFilters}
          onSearch={search}
          onCaseSelect={selectCase}
        />

        {/* Right Panel - Detail */}
        <CaseDetailPanel
          case_={selectedCase}
          isLoading={isLoadingDetail}
          error={detailError}
          aiResponse={aiResponse}
          isAskingAI={isAskingAI}
          aiError={aiError}
          onAsk={askAI}
          onClearAI={clearAIResponse}
        />
      </div>
    </div>
  )
}
