'use client'

import { useEffect, useRef } from 'react'
import { useLawFilter } from '../hooks/useLawFilter'
import { LawFilterPanel } from './LawFilterPanel'
import { FilteredLawResultList } from './FilteredLawResultList'
import type { LawFullText } from '../types'

function LawDetailPanel({
  law,
  isLoading,
  error,
}: {
  law: LawFullText | null
  isLoading: boolean
  error: string | null
}) {
  if (isLoading) {
    return (
      <div className="flex-1 flex items-center justify-center text-gray-400">
        <div className="text-center">
          <div className="animate-spin w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full mx-auto mb-3" />
          <p className="text-sm">법령 전문을 불러오는 중...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex-1 flex items-center justify-center text-red-500">
        <p className="text-sm">{error}</p>
      </div>
    )
  }

  if (!law) {
    return (
      <div className="flex-1 flex items-center justify-center text-gray-400">
        <div className="text-center">
          <svg className="w-16 h-16 mx-auto mb-4 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          <p className="text-sm">좌측에서 법령을 선택하면</p>
          <p className="text-sm">전문을 확인할 수 있습니다</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-y-auto">
      {/* 헤더 */}
      <div className="sticky top-0 bg-white border-b border-gray-200 px-6 py-4 z-10">
        <div className="flex items-center gap-2 mb-1">
          {law.law_type && (
            <span className="px-2 py-0.5 text-xs rounded-full bg-blue-100 text-blue-700">
              {law.law_type}
            </span>
          )}
          {law.ministry && (
            <span className="text-xs text-gray-500">{law.ministry}</span>
          )}
        </div>
        <h2 className="text-lg font-bold text-gray-900">{law.law_name}</h2>
        <div className="flex gap-4 mt-1 text-xs text-gray-500">
          {law.enforcement_date && <span>시행일: {law.enforcement_date}</span>}
          {law.promulgation_date && <span>공포일: {law.promulgation_date}</span>}
          {law.promulgation_no && <span>공포번호: {law.promulgation_no}</span>}
          <span>총 {law.total_articles}개 조문</span>
        </div>
      </div>

      {/* AI 요약 */}
      {law.ai_summary && (
        <div className="mx-6 mt-4 p-4 bg-blue-50 rounded-lg border border-blue-100">
          <h3 className="text-sm font-medium text-blue-800 mb-1">AI 요약</h3>
          <p className="text-sm text-blue-700 whitespace-pre-wrap">{law.ai_summary}</p>
        </div>
      )}

      {/* 조문 목록 */}
      <div className="px-6 py-4 space-y-4">
        {law.articles.map((article) => (
          <div key={article.article_number} className="border-b border-gray-100 pb-4 last:border-0">
            <h4 className="text-sm font-medium text-gray-800 mb-1">
              {article.article_number}
              {article.article_title && (
                <span className="text-gray-600"> ({article.article_title})</span>
              )}
            </h4>
            <p className="text-sm text-gray-600 whitespace-pre-wrap leading-relaxed">
              {article.article_content}
            </p>
          </div>
        ))}
      </div>

      {/* 부칙 */}
      {law.supplementary && (
        <div className="px-6 pb-6">
          <h3 className="text-sm font-medium text-gray-800 mb-2 border-t border-gray-200 pt-4">부칙</h3>
          <p className="text-sm text-gray-600 whitespace-pre-wrap">{law.supplementary}</p>
        </div>
      )}
    </div>
  )
}

export function FilterableLawView() {
  const {
    keyword, setKeyword,
    lawType, setLawType,
    ministry, setMinistry,
    promulgationPreset, setPromulgationPreset,
    promulgationFrom, setPromulgationFrom,
    promulgationTo, setPromulgationTo,
    enforcementPreset, setEnforcementPreset,
    enforcementFrom, setEnforcementFrom,
    enforcementTo, setEnforcementTo,
    sortOrder, setSortOrder,
    lawTypes, ministries,
    laws, total, isLoading, error, hasSearched, hasMore,
    search, loadMore,
    selectedId, selectItem,
    detail, isDetailLoading, detailError,
  } = useLawFilter()

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
        <LawFilterPanel
          keyword={keyword}
          onKeywordChange={setKeyword}
          lawType={lawType}
          onLawTypeChange={setLawType}
          ministry={ministry}
          onMinistryChange={setMinistry}
          promulgationPreset={promulgationPreset}
          onPromulgationPresetChange={setPromulgationPreset}
          promulgationFrom={promulgationFrom}
          onPromulgationFromChange={setPromulgationFrom}
          promulgationTo={promulgationTo}
          onPromulgationToChange={setPromulgationTo}
          enforcementPreset={enforcementPreset}
          onEnforcementPresetChange={setEnforcementPreset}
          enforcementFrom={enforcementFrom}
          onEnforcementFromChange={setEnforcementFrom}
          enforcementTo={enforcementTo}
          onEnforcementToChange={setEnforcementTo}
          lawTypes={lawTypes}
          ministries={ministries}
          onSearch={search}
          isLoading={isLoading}
        />
        <FilteredLawResultList
          laws={laws}
          total={total}
          selectedId={selectedId}
          onSelect={selectItem}
          hasMore={hasMore}
          onLoadMore={loadMore}
          isLoading={isLoading}
          error={error}
          hasSearched={hasSearched}
          highlightKeyword={keyword}
          sortOrder={sortOrder}
          onSortChange={setSortOrder}
        />
      </div>

      {/* 우측: 법령 전문 상세 */}
      <LawDetailPanel
        law={detail}
        isLoading={isDetailLoading}
        error={detailError}
      />
    </div>
  )
}
