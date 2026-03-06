'use client'

import { useState, useCallback, useEffect, useMemo, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import type { ChatSource, LawFullText, StatuteHierarchyResponse } from '../types'
import { casePrecedentService } from '../services'
import { formatIsoDate, formatPromulgationDate } from '../utils/dateUtils'
import { buildArticleTree, hasTreeStructure, findExpandedLabels } from '../utils/articleTreeParser'
import type { ArticleTreeNode } from '../utils/articleTreeParser'
import { normalizeSummaryMarkdown } from '../utils/normalizeSummaryMarkdown'

interface LawDetailUserProps {
  source: ChatSource
}

// 세션 내 LLM 요약 완료 캐시 (법령 이동 후 복귀 시 유지)
const summarizedDocIds = new Set<string>()

export function LawDetailUser({ source }: LawDetailUserProps) {
  const hasArticle = !!source.article_number
  const [isFullTextOpen, setIsFullTextOpen] = useState(false)
  const [fullText, setFullText] = useState<LawFullText | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [hierarchy, setHierarchy] = useState<StatuteHierarchyResponse | null>(null)

  // LLM 요약 상태
  const [isSummarizing, setIsSummarizing] = useState(false)
  const [showSummary, setShowSummary] = useState(false)

  // source 변경 시 상태 초기화 + 데이터 로딩
  useEffect(() => {
    setFullText(null)
    setError(null)
    setIsFullTextOpen(false)
    setIsSummarizing(false)
    setHierarchy(null)

    // 이전에 요약 완료한 법령이면 요약 상태 복원
    if (source.doc_id && summarizedDocIds.has(source.doc_id)) {
      setShowSummary(true)
    } else {
      setShowSummary(false)
    }

    if (!source.doc_id) return
    let cancelled = false

    // 전문 로딩
    setIsLoading(true)
    casePrecedentService.getLawFullText(source.doc_id)
      .then((data) => { if (!cancelled) setFullText(data) })
      .catch(() => {})
      .finally(() => { if (!cancelled) setIsLoading(false) })

    // 법령 계층 로딩
    casePrecedentService.getStatuteHierarchy(source.doc_id)
      .then((data) => { if (!cancelled) setHierarchy(data) })
      .catch(() => {})

    return () => { cancelled = true }
  }, [source.doc_id])

  // 조문 트리 구조
  const articleTree = useMemo(
    () => fullText ? buildArticleTree(fullText.articles) : [],
    [fullText]
  )
  const isTree = hasTreeStructure(articleTree)

  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set())

  // 현재 조문이 포함된 장/절 자동 펼침
  useEffect(() => {
    if (!isTree || !source.article_number) return
    const labels = findExpandedLabels(articleTree, source.article_number)
    if (labels.size > 0) setExpandedSections(labels)
  }, [isTree, articleTree, source.article_number])

  const toggleSection = useCallback((label: string) => {
    setExpandedSections((prev) => {
      const next = new Set(prev)
      if (next.has(label)) next.delete(label)
      else next.add(label)
      return next
    })
  }, [])

  const currentArticleRef = useRef<HTMLDivElement | null>(null)

  const handleToggleFullText = useCallback(() => {
    if (!fullText && !isLoading) {
      if (!source.doc_id) {
        setError('법령 ID가 없어 전문을 불러올 수 없습니다.')
      }
    }
    setIsFullTextOpen((prev) => {
      const willOpen = !prev
      if (willOpen && source.article_number) {
        // 아코디언 열릴 때 검색된 조문으로 자동 스크롤
        setTimeout(() => {
          currentArticleRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' })
        }, 200)
      }
      return willOpen
    })
  }, [fullText, isLoading, source.doc_id, source.article_number])

  // LLM 요약: 5초 로딩 후 한 번에 표시 + 캐시 저장
  const handleSummarize = useCallback(() => {
    if (isSummarizing || showSummary) return
    setIsSummarizing(true)
    setTimeout(() => {
      setIsSummarizing(false)
      setShowSummary(true)
      if (source.doc_id) summarizedDocIds.add(source.doc_id)
    }, 5000)
  }, [isSummarizing, showSummary, source.doc_id])

  return (
    <div className="space-y-4">
      {/* 법령 계층 (상위 > 현재 > 하위) — 법령명 바로 아래 */}
      {hierarchy && (hierarchy.upper.length > 0 || hierarchy.lower.length > 0) && (
        <div className="bg-gray-50 rounded-xl border border-gray-200 p-4">
          <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">법령 단계 구조</h4>
          <div className="flex items-center gap-2 flex-wrap text-sm">
            {hierarchy.upper.map((node) => (
              <a
                key={node.id}
                href={`/statute-hierarchy?id=${node.id}&name=${encodeURIComponent(node.name)}&type=${encodeURIComponent(node.type)}`}
                className="text-blue-600 hover:underline"
              >
                {node.name}
              </a>
            ))}
            {hierarchy.upper.length > 0 && <span className="text-gray-400">&rsaquo;</span>}
            <span className="font-semibold text-gray-900 bg-gray-200 px-2 py-0.5 rounded">
              {source.law_name}
            </span>
            {hierarchy.lower.length > 0 && <span className="text-gray-400">&rsaquo;</span>}
            {hierarchy.lower.map((node) => (
              <a
                key={node.id}
                href={`/statute-hierarchy?id=${node.id}&name=${encodeURIComponent(node.name)}&type=${encodeURIComponent(node.type)}`}
                className="text-blue-600 hover:underline"
              >
                {node.name}
              </a>
            ))}
          </div>
        </div>
      )}

      {/* 법령 정보 (접기/펼치기) — 변호사 뷰와 동일한 형식 */}
      <details open className="group rounded-xl border border-gray-200 overflow-hidden">
        <summary className="px-4 py-2.5 bg-gray-50 text-sm font-medium text-gray-700 cursor-pointer hover:bg-gray-100 transition-colors flex items-center justify-between list-none [&::-webkit-details-marker]:hidden">
          <span className="flex items-center gap-2">
            법령 정보
            {fullText?.ai_summary && !showSummary && !isSummarizing && (
              <button
                onClick={(e) => { e.preventDefault(); handleSummarize() }}
                className="inline-flex items-center gap-1 px-2 py-0.5 text-xs bg-blue-100 text-blue-700 rounded-full hover:bg-blue-200 transition-colors"
              >
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z" />
                </svg>
                LLM 요약
              </button>
            )}
            {isSummarizing && (
              <span className="inline-flex items-center gap-1 px-2 py-0.5 text-xs bg-blue-100 text-blue-700 rounded-full">
                <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-blue-500" />
                요약 생성 중...
              </span>
            )}
            {showSummary && (
              <span className="inline-flex items-center gap-1 px-2 py-0.5 text-xs bg-blue-100 text-blue-700 rounded-full">
                LLM 요약 완료
              </span>
            )}
          </span>
          <svg className="w-4 h-4 text-gray-400 transition-transform group-open:rotate-180" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </summary>
        <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm p-4">
          <dt className="text-gray-500 font-medium">법령명</dt>
          <dd className="text-gray-800">{source.law_name || '-'}</dd>
          {source.law_type && (
            <>
              <dt className="text-gray-500 font-medium">법령종류</dt>
              <dd className="text-gray-800">{source.law_type}</dd>
            </>
          )}
          {source.ministry && (
            <>
              <dt className="text-gray-500 font-medium">소관부처</dt>
              <dd className="text-gray-800">{source.ministry}</dd>
            </>
          )}
          {source.article_title && (
            <>
              <dt className="text-gray-500 font-medium">조문제목</dt>
              <dd className="text-gray-800">{source.article_title}</dd>
            </>
          )}
          {fullText?.enforcement_date && (
            <>
              <dt className="text-gray-500 font-medium">시행일</dt>
              <dd className="text-gray-800">{formatIsoDate(fullText.enforcement_date)}</dd>
            </>
          )}
          {fullText?.promulgation_date && (
            <>
              <dt className="text-gray-500 font-medium">공포일자</dt>
              <dd className="text-gray-800">{formatPromulgationDate(fullText.promulgation_date)}</dd>
            </>
          )}
          {fullText?.promulgation_no && (
            <>
              <dt className="text-gray-500 font-medium">공포번호</dt>
              <dd className="text-gray-800">제{fullText.promulgation_no}호</dd>
            </>
          )}
        </dl>

        {/* LLM 요약 로딩 중 */}
        {isSummarizing && (
          <div className="mx-4 mb-4 bg-blue-50 rounded-lg border border-blue-200 p-4 flex flex-col items-center gap-3">
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
        {showSummary && fullText?.ai_summary && (
          <div className="mx-4 mb-4 bg-blue-50 rounded-lg border border-blue-200 p-4 animate-[fadeIn_0.5s_ease-in]">
            <div className="flex items-center gap-2 mb-2">
              <svg className="w-4 h-4 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z" />
              </svg>
              <h4 className="text-sm font-semibold text-blue-700">LLM 요약</h4>
            </div>
            <div className="text-sm text-gray-800 leading-relaxed prose prose-sm max-w-none prose-headings:text-blue-800 prose-headings:text-sm prose-headings:mt-3 prose-headings:mb-1 prose-h2:text-base prose-h2:mt-0 prose-h2:mb-2 prose-h2:font-bold prose-h3:font-semibold prose-ul:my-1 prose-li:my-0.5 prose-p:my-1">
              <ReactMarkdown>{normalizeSummaryMarkdown(fullText.ai_summary)}</ReactMarkdown>
            </div>
          </div>
        )}
      </details>

      {hasArticle ? (
        <>
          {/* 조문 헤더 */}
          <div className="bg-gray-50 px-5 py-3 rounded-xl border border-gray-200 border-l-4 border-l-blue-500">
            <h3 className="text-sm font-semibold text-gray-900">
              <span className="text-gray-500 font-medium">법령 내 질문 관련 조항</span>
              <span className="text-gray-300 mx-2">|</span>
              제{source.article_number}
              {source.article_title && (
                <span className="font-normal text-gray-500 ml-2">
                  ({source.article_title})
                </span>
              )}
            </h3>
          </div>

          {/* 조문 본문 */}
          <div className="prose prose-lg max-w-none text-gray-700">
            <div className="bg-gray-50 p-6 rounded-xl border border-gray-200 leading-relaxed">
              <ReactMarkdown>
                {source.content || '조문 내용을 불러올 수 없습니다.'}
              </ReactMarkdown>
            </div>
          </div>
        </>
      ) : (
        <div className="bg-gray-50 p-5 rounded-xl border border-gray-200">
          <h3 className="font-semibold text-gray-700 mb-2 flex items-center gap-2">
            <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            조문 정보 없음
          </h3>
          <p className="text-sm text-gray-600">
            해당 법령의 조문 내용을 불러올 수 없습니다.
            챗봇에게 구체적인 조문 번호를 포함하여 질문해보세요.
          </p>
        </div>
      )}

      {/* 법령 전문 (아코디언) */}
      {source.doc_id && (
        <div className="border border-gray-200 rounded-xl overflow-hidden">
          <button
            onClick={handleToggleFullText}
            disabled={isLoading}
            className="w-full flex items-center gap-2 px-5 py-3 bg-gray-50 hover:bg-gray-100 transition-colors text-left"
          >
            <span className="text-sm font-medium text-gray-700 flex-1">
              {source.law_name || '법령'} 전문 보기
              {fullText && (
                <span className="text-gray-400 font-normal ml-1">
                  ({fullText.total_articles}개 조문)
                </span>
              )}
            </span>
            {isLoading ? (
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500" />
            ) : (
              <svg
                className={`w-4 h-4 text-gray-400 transition-transform ${isFullTextOpen ? 'rotate-180' : ''}`}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            )}
          </button>

          {isFullTextOpen && (
            <div className="border-t border-gray-200">
              {error ? (
                <div className="p-4 text-sm text-red-600 bg-red-50">{error}</div>
              ) : fullText && fullText.articles.length > 0 ? (
                <div className="max-h-[600px] overflow-y-auto">
                  {isTree ? (
                    <ArticleTreeView
                      nodes={articleTree}
                      currentArticleNumber={source.article_number}
                      expandedSections={expandedSections}
                      onToggle={toggleSection}
                      currentArticleRef={currentArticleRef}
                    />
                  ) : (
                    <div className="divide-y divide-gray-100">
                      {fullText.articles.map((article) => {
                        const isCurrentArticle = source.article_number === article.article_number
                        return (
                          <div
                            key={article.article_number}
                            ref={isCurrentArticle ? currentArticleRef : undefined}
                            className={`px-5 py-4 ${isCurrentArticle ? 'bg-blue-50 border-l-4 border-blue-400' : ''}`}
                          >
                            <h4 className={`text-sm font-bold mb-1 ${isCurrentArticle ? 'text-blue-800' : 'text-gray-800'}`}>
                              제{article.article_number}
                              {article.article_title && (
                                <span className="font-normal text-gray-500 ml-1">
                                  ({article.article_title})
                                </span>
                              )}
                              {isCurrentArticle && (
                                <span className="ml-2 text-xs bg-blue-200 text-blue-800 px-1.5 py-0.5 rounded">
                                  검색된 조문
                                </span>
                              )}
                            </h4>
                            <div className="text-sm text-gray-600 leading-relaxed prose prose-sm max-w-none">
                              <ReactMarkdown>{article.article_content}</ReactMarkdown>
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  )}

                  {/* 부칙 */}
                  {fullText.supplementary && (
                    <div className="px-5 py-4 border-t border-gray-200 bg-gray-50">
                      <h4 className="text-sm font-bold text-gray-600 mb-2">부칙</h4>
                      <div className="text-sm text-gray-600 leading-relaxed prose prose-sm max-w-none">
                        <ReactMarkdown>{fullText.supplementary}</ReactMarkdown>
                      </div>
                    </div>
                  )}
                </div>
              ) : fullText ? (
                <div className="p-4 text-sm text-gray-500 text-center">
                  조문 데이터가 없습니다.
                </div>
              ) : null}
            </div>
          )}
        </div>
      )}

      {/* 법령 활용 안내 */}
      <p className="text-xs text-gray-400 mt-2 text-center">
        이 법령의 적용 여부는 구체적인 사실관계에 따라 달라질 수 있습니다. 정확한 법률 해석이 필요한 경우 전문가 상담을 권장합니다.
      </p>

      {/* 그래프 보강 정보 */}
      {(source.cited_statutes?.length || source.similar_cases?.length) ? (
        <div className="mt-6 pt-4 border-t border-gray-200">
          <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">관련 정보</h3>

          {source.cited_statutes && source.cited_statutes.length > 0 && (
            <p className="text-sm text-gray-600 mb-2">
              <span className="font-medium">관련 법령:</span>{' '}
              {source.cited_statutes.join(', ')}
            </p>
          )}
          {source.similar_cases && source.similar_cases.length > 0 && (
            <p className="text-sm text-gray-600">
              <span className="font-medium">관련 판례:</span>{' '}
              {source.similar_cases.join(', ')}
            </p>
          )}
        </div>
      ) : null}

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

/** 장/절/조 트리 뷰 컴포넌트 */
function ArticleTreeView({
  nodes,
  currentArticleNumber,
  expandedSections,
  onToggle,
  currentArticleRef,
}: {
  nodes: ArticleTreeNode[]
  currentArticleNumber?: string
  expandedSections: Set<string>
  onToggle: (label: string) => void
  currentArticleRef?: React.Ref<HTMLDivElement>
}) {
  return (
    <div className="divide-y divide-gray-100">
      {nodes.map((node) => {
        if (node.type === 'article' && node.article) {
          const isCurrentArticle = currentArticleNumber === node.article.article_number
          return (
            <div
              key={node.article.article_number}
              ref={isCurrentArticle ? currentArticleRef : undefined}
              className={`px-5 py-4 ${isCurrentArticle ? 'bg-blue-50 border-l-4 border-blue-400' : ''}`}
            >
              <h4 className={`text-sm font-bold mb-1 ${isCurrentArticle ? 'text-blue-800' : 'text-gray-800'}`}>
                제{node.article.article_number}
                {node.article.article_title && (
                  <span className="font-normal text-gray-500 ml-1">
                    ({node.article.article_title})
                  </span>
                )}
                {isCurrentArticle && (
                  <span className="ml-2 text-xs bg-blue-200 text-blue-800 px-1.5 py-0.5 rounded">
                    검색된 조문
                  </span>
                )}
              </h4>
              <div className="text-sm text-gray-600 leading-relaxed prose prose-sm max-w-none">
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
                isChapter ? 'bg-gray-100 font-bold text-gray-800' : 'bg-gray-50 font-medium text-gray-700'
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
                <ArticleTreeView
                  nodes={node.children}
                  currentArticleNumber={currentArticleNumber}
                  expandedSections={expandedSections}
                  onToggle={onToggle}
                  currentArticleRef={currentArticleRef}
                />
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}
