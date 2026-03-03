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
    selectedCase,
    isLoadingDetail,
    detailError,
    selectCase,
  } = useCaseSearch(initialCaseId)

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
