'use client'

import { SearchPanel, CaseDetailPanel } from '@/features/case-precedent/components'
import { useCaseSearch } from '@/features/case-precedent/hooks/useCaseSearch'

export function LawyerView() {
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
    <div className="h-full flex overflow-hidden">
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
  )
}
