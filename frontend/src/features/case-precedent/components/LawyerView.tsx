'use client'

import { SearchPanel } from '@/features/case-precedent/components/SearchPanel'
import { CaseDetailPanel } from '@/features/case-precedent/components/CaseDetailPanel'
import { useCaseSearch } from '@/features/case-precedent/hooks/useCaseSearch'

interface LawyerViewProps {
  initialCaseId?: string
}

export function LawyerView({ initialCaseId }: LawyerViewProps) {
  const {
    searchResults,
    totalResults,
    isSearching,
    searchError,
    selectedCase,
    isLoadingDetail,
    detailError,
    filters,
    setFilters,
    search,
    selectCase,
  } = useCaseSearch(initialCaseId)

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
        selectedCase={selectedCase}
        onFilterChange={setFilters}
        onSearch={search}
        onCaseSelect={selectCase}
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
