'use client'

import { useState, useCallback, useEffect, useRef, useMemo } from 'react'
import ReactMarkdown from 'react-markdown'
import { useLawFilter } from '../hooks/useLawFilter'
import { LawFilterPanel } from './LawFilterPanel'
import { FilteredLawResultList } from './FilteredLawResultList'
import type { LawFullText } from '../types'
import { buildArticleTree, hasTreeStructure } from '../utils/articleTreeParser'
import type { ArticleTreeNode } from '../utils/articleTreeParser'

/** ai_summary 평문/마크다운 텍스트를 ReactMarkdown용으로 정규화 */
function normalizeSummaryMarkdown(text: string): string {
  let result = text

  if (!text.includes('### ')) {
    result = result.replace(/(\d{1,2})\.\s+([가-힣])/g, '\n\n### $1. $2')
  }

  result = result
    .replace(/([^\n])\s*(###\s)/g, '$1\n\n$2')
    .replace(/([^\n])\s*(- )/g, '$1\n$2')
    .replace(/([^\n])(※)/g, '$1\n\n$2')
    .trim()

  const firstHeading = result.indexOf('\n\n###')
  if (firstHeading > 0) {
    const title = result.slice(0, firstHeading).trim()
    if (title && !title.startsWith('#') && !title.startsWith('**')) {
      result = `**${title}**${result.slice(firstHeading)}`
    }
  }

  return result
}

// 세션 내 LLM 요약 완료 캐시 (법령 이동 후 복귀 시 유지)
const summarizedDocIds = new Set<string>()

function LawDetailPanel({
  law,
  isLoading,
  error,
}: {
  law: LawFullText | null
  isLoading: boolean
  error: string | null
}) {
  // LLM 요약 상태
  const [isSummarizing, setIsSummarizing] = useState(false)
  const [showSummary, setShowSummary] = useState(false)

  // 트리 접기/펼치기 상태
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set())

  // 조문 트리 구조
  const articleTree = useMemo(
    () => law ? buildArticleTree(law.articles) : [],
    [law]
  )
  const isTree = hasTreeStructure(articleTree)

  // 법령 변경 시 상태 초기화
  const prevLawId = useRef<string | undefined>(undefined)
  useEffect(() => {
    const lawId = law?.law_id
    if (lawId === prevLawId.current) return
    prevLawId.current = lawId

    setIsSummarizing(false)
    setExpandedSections(new Set())

    if (lawId && summarizedDocIds.has(lawId)) {
      setShowSummary(true)
    } else {
      setShowSummary(false)
    }
  }, [law?.law_id])

  const toggleSection = useCallback((label: string) => {
    setExpandedSections((prev) => {
      const next = new Set(prev)
      if (next.has(label)) next.delete(label)
      else next.add(label)
      return next
    })
  }, [])

  // LLM 요약: 5초 로딩 후 한 번에 표시 + 캐시 저장
  const handleSummarize = useCallback(() => {
    if (isSummarizing || showSummary) return
    setIsSummarizing(true)
    setTimeout(() => {
      setIsSummarizing(false)
      setShowSummary(true)
      if (law?.law_id) summarizedDocIds.add(law.law_id)
    }, 5000)
  }, [isSummarizing, showSummary, law?.law_id])

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
    <div className="flex-1 overflow-y-auto bg-white">
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
        <div className="flex items-center gap-2">
          <h2 className="text-lg font-bold text-gray-900">{law.law_name}</h2>
          {law.ai_summary && !showSummary && !isSummarizing && (
            <button
              onClick={handleSummarize}
              className="inline-flex items-center gap-1 px-2 py-0.5 text-xs bg-blue-100 text-blue-700 rounded-full hover:bg-blue-200 transition-colors shrink-0"
            >
              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z" />
              </svg>
              LLM 요약
            </button>
          )}
          {isSummarizing && (
            <span className="inline-flex items-center gap-1 px-2 py-0.5 text-xs bg-blue-100 text-blue-700 rounded-full shrink-0">
              <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-blue-500" />
              요약 생성 중...
            </span>
          )}
          {showSummary && (
            <span className="inline-flex items-center gap-1 px-2 py-0.5 text-xs bg-blue-100 text-blue-700 rounded-full shrink-0">
              LLM 요약 완료
            </span>
          )}
        </div>
        <div className="flex gap-4 mt-1 text-xs text-gray-500">
          {law.enforcement_date && <span>시행일: {law.enforcement_date}</span>}
          {law.promulgation_date && <span>공포일: {law.promulgation_date}</span>}
          {law.promulgation_no && <span>공포번호: {law.promulgation_no}</span>}
          <span>총 {law.total_articles}개 조문</span>
        </div>
      </div>

      {/* LLM 요약 로딩 중 */}
      {isSummarizing && (
        <div className="mx-6 mt-4 bg-blue-50 rounded-lg border border-blue-200 p-4 flex flex-col items-center gap-3">
          <div className="flex items-center gap-2">
            <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-500" />
            <span className="text-sm font-medium text-blue-700">AI가 법령을 분석하고 있습니다...</span>
          </div>
          <div className="w-full bg-blue-100 rounded-full h-1.5 overflow-hidden">
            <div className="bg-blue-500 h-full rounded-full animate-[progress_5s_ease-in-out_forwards]" />
          </div>
        </div>
      )}

      {/* LLM 요약 결과 */}
      {showSummary && law.ai_summary && (
        <div className="mx-6 mt-4 bg-blue-50 rounded-lg border border-blue-200 p-4 animate-[fadeIn_0.5s_ease-in]">
          <div className="flex items-center gap-2 mb-2">
            <svg className="w-4 h-4 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z" />
            </svg>
            <h4 className="text-sm font-semibold text-blue-700">LLM 요약</h4>
          </div>
          <div className="text-sm text-gray-800 leading-relaxed prose prose-sm max-w-none prose-headings:text-blue-800 prose-headings:text-sm prose-headings:mt-3 prose-headings:mb-1 prose-h2:text-base prose-h2:mt-0 prose-h2:mb-2 prose-h2:font-bold prose-h3:font-semibold prose-ul:my-1 prose-li:my-0.5 prose-p:my-1">
            <ReactMarkdown>{normalizeSummaryMarkdown(law.ai_summary)}</ReactMarkdown>
          </div>
        </div>
      )}

      {/* 전문 */}
      <div className="mt-4 mx-6">
        <div className="border border-gray-200 rounded-xl overflow-hidden">
          <div className="px-5 py-2.5 bg-white border-b border-gray-200">
            <span className="text-sm font-medium text-gray-700">전문</span>
            <span className="text-xs text-gray-400 ml-2">
              {law.total_articles}개 조문
            </span>
          </div>

          {law.articles.length > 0 ? (
            <div className="max-h-[600px] overflow-y-auto">
              {isTree ? (
                <FilterArticleTreeView
                  nodes={articleTree}
                  expandedSections={expandedSections}
                  onToggle={toggleSection}
                />
              ) : (
                <div className="divide-y divide-gray-100">
                  {law.articles.map((article) => (
                    <div key={article.article_number} className="px-5 py-4">
                      <h4 className="text-sm font-bold mb-1 text-gray-800">
                        제{article.article_number}
                        {article.article_title && (
                          <span className="font-normal text-gray-500 ml-1">
                            ({article.article_title})
                          </span>
                        )}
                      </h4>
                      <div className="text-base text-gray-800 leading-loose prose prose-base max-w-none">
                        <ReactMarkdown>{article.article_content}</ReactMarkdown>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* 부칙 */}
              {law.supplementary && (
                <div className="px-5 py-4 border-t border-gray-200 bg-white">
                  <h4 className="text-sm font-bold text-gray-600 mb-2">부칙</h4>
                  <div className="text-sm text-gray-600 leading-relaxed prose prose-sm max-w-none">
                    <ReactMarkdown>{law.supplementary}</ReactMarkdown>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="p-4 text-sm text-gray-500 text-center">
              조문 데이터가 없습니다.
            </div>
          )}
        </div>
      </div>

      {/* 프로그레스 바 + fadeIn 애니메이션용 keyframes */}
      <style jsx>{`
        @keyframes progress {
          from { width: 0%; }
          to { width: 100%; }
        }
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(-8px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  )
}

/** 장/절/조 트리뷰 (FilterableLawView 전용) */
function FilterArticleTreeView({
  nodes,
  expandedSections,
  onToggle,
}: {
  nodes: ArticleTreeNode[]
  expandedSections: Set<string>
  onToggle: (label: string) => void
}) {
  return (
    <div className="divide-y divide-gray-100">
      {nodes.map((node) => {
        if (node.type === 'article' && node.article) {
          return (
            <div key={node.article.article_number} className="px-5 py-4">
              <h4 className="text-sm font-bold mb-1 text-gray-800">
                제{node.article.article_number}
                {node.article.article_title && (
                  <span className="font-normal text-gray-500 ml-1">
                    ({node.article.article_title})
                  </span>
                )}
              </h4>
              <div className="text-base text-gray-800 leading-loose prose prose-base max-w-none">
                <ReactMarkdown>{node.article.article_content}</ReactMarkdown>
              </div>
            </div>
          )
        }

        const isExpanded = expandedSections.has(node.label)
        const isChapter = node.type === 'chapter'
        return (
          <div key={node.label}>
            <button
              onClick={() => onToggle(node.label)}
              className={`w-full text-left px-5 py-3 flex items-center gap-2 hover:bg-gray-50 transition-colors ${
                isChapter ? 'bg-white font-bold text-gray-800' : 'bg-white font-medium text-gray-700'
              }`}
            >
              <svg
                className={`w-3.5 h-3.5 text-gray-400 transition-transform ${isExpanded ? 'rotate-90' : ''}`}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
              <span className="text-sm">{node.label}</span>
              <span className="text-xs text-gray-400 ml-auto">
                {node.children.length}개
              </span>
            </button>
            {isExpanded && (
              <div className={isChapter ? 'ml-2' : 'ml-4'}>
                <FilterArticleTreeView
                  nodes={node.children}
                  expandedSections={expandedSections}
                  onToggle={onToggle}
                />
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

export function FilterableLawView() {
  const {
    keyword, setKeyword,
    lawType, setLawType,
    promulgationPreset, setPromulgationPreset,
    promulgationFrom, setPromulgationFrom,
    promulgationTo, setPromulgationTo,
    enforcementPreset, setEnforcementPreset,
    enforcementFrom, setEnforcementFrom,
    enforcementTo, setEnforcementTo,
    sortOrder, setSortOrder,
    lawTypes,
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
      <div className="w-80 flex flex-col bg-white border-r border-gray-200">
        <LawFilterPanel
          keyword={keyword}
          onKeywordChange={setKeyword}
          lawType={lawType}
          onLawTypeChange={setLawType}
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
