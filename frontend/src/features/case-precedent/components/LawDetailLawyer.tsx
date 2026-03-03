'use client'

import { useState, useCallback, useEffect, useMemo } from 'react'
import ReactMarkdown from 'react-markdown'
import type { ChatSource, PrecedentDetail, LawFullText, StatuteHierarchyResponse } from '../types'
import { casePrecedentService } from '../services'
import { formatIsoDate, formatPromulgationDate } from '../utils/dateUtils'
import { buildArticleTree, hasTreeStructure, findExpandedLabels } from '../utils/articleTreeParser'
import type { ArticleTreeNode } from '../utils/articleTreeParser'

interface LawDetailLawyerProps {
  source: ChatSource | PrecedentDetail
}

export function LawDetailLawyer({ source }: LawDetailLawyerProps) {
  const hasArticle = !!source.article_number
  const [isFullTextOpen, setIsFullTextOpen] = useState(false)
  const [isSummaryOpen, setIsSummaryOpen] = useState(false)
  const [fullText, setFullText] = useState<LawFullText | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const docId = 'doc_id' in source ? source.doc_id : undefined
  const lawName = source.law_name || ''

  const [hierarchy, setHierarchy] = useState<StatuteHierarchyResponse | null>(null)

  // source 변경 시 상태 초기화 + 데이터 로딩
  useEffect(() => {
    setFullText(null)
    setError(null)
    setIsFullTextOpen(false)
    setIsSummaryOpen(false)
    setHierarchy(null)

    if (!docId) return
    let cancelled = false

    // 전문 로딩
    setIsLoading(true)
    casePrecedentService.getLawFullText(docId)
      .then((data) => { if (!cancelled) setFullText(data) })
      .catch(() => {})
      .finally(() => { if (!cancelled) setIsLoading(false) })

    // 법령 계층 로딩
    casePrecedentService.getStatuteHierarchy(docId)
      .then((data) => { if (!cancelled) setHierarchy(data) })
      .catch(() => {})

    return () => { cancelled = true }
  }, [docId])

  // 조문 트리 구조
  const articleTree = useMemo(
    () => fullText ? buildArticleTree(fullText.articles) : [],
    [fullText]
  )
  const isTree = hasTreeStructure(articleTree)

  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set())

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

  const handleToggleFullText = useCallback(() => {
    if (!fullText && !isLoading) {
      if (!docId) {
        setError('법령 ID가 없어 전문을 불러올 수 없습니다.')
      }
    }
    setIsFullTextOpen((prev) => !prev)
  }, [fullText, isLoading, docId])

  return (
    <div className="space-y-4">
      {/* 법령 메타정보 테이블 */}
      <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm bg-gray-50 p-4 rounded-xl border border-gray-200">
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
        {source.article_number && (
          <>
            <dt className="text-gray-500 font-medium">조문번호</dt>
            <dd className="text-gray-800">제{source.article_number}</dd>
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

      {/* 법령 계층 (상위 > 현재 > 하위) */}
      {hierarchy && (hierarchy.upper.length > 0 || hierarchy.lower.length > 0) && (
        <div className="bg-blue-50 rounded-xl border border-blue-100 p-4">
          <h4 className="text-xs font-semibold text-blue-600 mb-2">법령 단계 구조</h4>
          <div className="flex items-center gap-2 flex-wrap text-sm">
            {hierarchy.upper.map((node) => (
              <a
                key={node.id}
                href={`/statute-hierarchy?id=${node.id}&name=${encodeURIComponent(node.name)}&type=${encodeURIComponent(node.type)}`}
                className="text-blue-700 hover:underline"
              >
                {node.name}
              </a>
            ))}
            {hierarchy.upper.length > 0 && <span className="text-gray-400">&rsaquo;</span>}
            <span className="font-bold text-blue-900 bg-blue-100 px-2 py-0.5 rounded">
              {lawName}
            </span>
            {hierarchy.lower.length > 0 && <span className="text-gray-400">&rsaquo;</span>}
            {hierarchy.lower.map((node) => (
              <a
                key={node.id}
                href={`/statute-hierarchy?id=${node.id}&name=${encodeURIComponent(node.name)}&type=${encodeURIComponent(node.type)}`}
                className="text-blue-700 hover:underline"
              >
                {node.name}
              </a>
            ))}
          </div>
        </div>
      )}

      {hasArticle ? (
        /* 조문 원문 (법원 스타일) */
        <div className="border border-gray-200 rounded-xl overflow-hidden">
          <div className="bg-gray-800 px-4 py-2">
            <span className="text-white text-sm font-medium">
              제{source.article_number}
              {source.article_title && ` (${source.article_title})`}
            </span>
          </div>
          <div className="bg-white p-5 text-gray-800 leading-relaxed">
            <ReactMarkdown>
              {source.content || '조문 내용을 불러올 수 없습니다.'}
            </ReactMarkdown>
          </div>
        </div>
      ) : (
        <div className="bg-amber-50 p-5 rounded-xl border border-amber-200">
          <h3 className="font-bold text-amber-800 mb-2">조문 정보 없음</h3>
          <p className="text-sm text-amber-700">
            해당 법령의 조문 내용을 불러올 수 없습니다.
            챗봇에게 구체적인 조문 번호를 포함하여 질문해보세요.
          </p>
        </div>
      )}

      {/* 법령 전문 (아코디언) */}
      {docId && (
        <div className="border border-gray-200 rounded-xl overflow-hidden">
          <button
            onClick={handleToggleFullText}
            disabled={isLoading}
            className="w-full flex items-center gap-2 px-5 py-3 bg-gray-50 hover:bg-gray-100 transition-colors text-left"
          >
            <svg className="w-5 h-5 text-indigo-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
            </svg>
            <span className="text-sm font-medium text-gray-700 flex-1">
              {lawName || '법령'} 전문 보기
              {fullText && (
                <span className="text-gray-400 font-normal ml-1">
                  ({fullText.total_articles}개 조문)
                </span>
              )}
            </span>
            {isLoading ? (
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-indigo-500" />
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
                    <LawyerArticleTreeView
                      nodes={articleTree}
                      currentArticleNumber={source.article_number}
                      expandedSections={expandedSections}
                      onToggle={toggleSection}
                    />
                  ) : (
                    <div className="divide-y divide-gray-100">
                      {fullText.articles.map((article) => {
                        const isCurrentArticle = source.article_number === article.article_number
                        return (
                          <div
                            key={article.article_number}
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

      {/* LLM 요약 보기 (아코디언) */}
      {fullText?.ai_summary && (
        <div className="border border-gray-200 rounded-xl overflow-hidden">
          <button
            onClick={() => setIsSummaryOpen(!isSummaryOpen)}
            className="w-full flex items-center gap-2 px-5 py-3 bg-gray-50 hover:bg-gray-100 transition-colors text-left"
          >
            <svg className="w-5 h-5 text-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.455 2.456L21.75 6l-1.036.259a3.375 3.375 0 00-2.455 2.456zM16.894 20.567L16.5 21.75l-.394-1.183a2.25 2.25 0 00-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 001.423-1.423l.394-1.183.394 1.183a2.25 2.25 0 001.423 1.423l1.183.394-1.183.394a2.25 2.25 0 00-1.423 1.423z" />
            </svg>
            <span className="text-sm font-medium text-gray-700 flex-1">
              LLM 요약 보기
            </span>
            <svg
              className={`w-4 h-4 text-gray-400 transition-transform ${isSummaryOpen ? 'rotate-180' : ''}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>

          {isSummaryOpen && (
            <div className="border-t border-gray-200 p-4 bg-indigo-50">
              <p className="text-sm text-indigo-900 leading-relaxed whitespace-pre-wrap">
                {fullText.ai_summary}
              </p>
            </div>
          )}
        </div>
      )}

      {/* 그래프 보강 정보 (관련 법령) */}
      {(source.cited_statutes?.length || source.similar_cases?.length) ? (
        <div className="mt-6 pt-4 border-t border-gray-200">
          <h3 className="font-bold text-gray-700 mb-3">관련 정보</h3>
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
    </div>
  )
}

function LawyerArticleTreeView({
  nodes,
  currentArticleNumber,
  expandedSections,
  onToggle,
}: {
  nodes: ArticleTreeNode[]
  currentArticleNumber?: string
  expandedSections: Set<string>
  onToggle: (label: string) => void
}) {
  return (
    <div className="divide-y divide-gray-100">
      {nodes.map((node) => {
        if (node.type === 'article' && node.article) {
          const isCurrentArticle = currentArticleNumber === node.article.article_number
          return (
            <div
              key={node.article.article_number}
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
                <LawyerArticleTreeView
                  nodes={node.children}
                  currentArticleNumber={currentArticleNumber}
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
