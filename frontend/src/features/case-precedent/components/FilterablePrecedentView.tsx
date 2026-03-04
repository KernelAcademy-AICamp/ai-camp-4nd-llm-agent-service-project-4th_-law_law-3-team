'use client'

import { useEffect, useRef } from 'react'
import { usePrecedentFilter } from '../hooks/usePrecedentFilter'
import { FilterPanel } from './FilterPanel'
import { FilteredResultList } from './FilteredResultList'
import { CaseDetailPanel } from './CaseDetailPanel'

export function FilterablePrecedentView() {
  const {
    keyword, setKeyword,
    caseType, setCaseType,
    datePreset, setDatePreset,
    dateFrom, setDateFrom,
    dateTo, setDateTo,
    sortOrder, setSortOrder,
    caseTypes,
    precedents, total, isLoading, error, hasSearched, hasMore,
    search, loadMore,
    selectedId, selectItem,
    detail, isDetailLoading, detailError,
  } = usePrecedentFilter()

  // 정렬 변경 시 자동 재검색 (검색 결과가 있을 때만)
  const isFirstRender = useRef(true)
  useEffect(() => {
    if (isFirstRender.current) {
      isFirstRender.current = false
      return
    }
    if (hasSearched) {
      search()
    }
  }, [sortOrder]) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="flex h-full">
      {/* 좌측: 필터 + 결과 목록 */}
      <div className="w-96 flex flex-col bg-white border-r border-gray-200">
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
          onSearch={search}
          isLoading={isLoading}
        />
        <FilteredResultList
          precedents={precedents}
          total={total}
          selectedId={selectedId}
          onSelect={selectItem}
          hasMore={hasMore}
          onLoadMore={loadMore}
          isLoading={isLoading}
          error={error}
          hasSearched={hasSearched}
          hasDateFilter={datePreset !== 'all'}
          highlightKeyword={keyword}
          sortOrder={sortOrder}
          onSortChange={setSortOrder}
        />
      </div>

      {/* 우측: 상세 보기 */}
      <CaseDetailPanel
        case_={detail}
        isLoading={isDetailLoading}
        error={detailError}
      />
    </div>
  )
}
