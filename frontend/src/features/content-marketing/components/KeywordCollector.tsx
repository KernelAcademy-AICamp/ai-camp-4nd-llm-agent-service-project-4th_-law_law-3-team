'use client'

import type { TrendIssue } from '../types'
import { useKeywordFlow } from '../hooks/useKeywordFlow'
import { KeywordCard } from './KeywordCard'
import { KeywordNewsList } from './KeywordNewsList'

interface KeywordCollectorProps {
  onGenerateScript: (issue: TrendIssue) => void
}

export function KeywordCollector({ onGenerateScript }: KeywordCollectorProps) {
  const {
    step,
    keywords,
    selectedKeyword,
    newsResponse,
    error,
    cacheHit,
    streamProgress,
    handleCollect,
    handleSearchNews,
    handleBack,
    handleReset,
  } = useKeywordFlow()

  // 뉴스 → 대본 생성 연결
  const handleGenerateFromNews = (keyword: string) => {
    const pseudoIssue: TrendIssue = {
      id: `keyword-${keyword}`,
      title: keyword,
      summary: `"${keyword}" 관련 뉴스 분석 기반 콘텐츠`,
      key_points: [],
      score: 0,
      mention_score: 0,
      legal_relevance_score: 0,
      category: 'all',
      score_detail: null,
      fitness_label: null,
      sources: [],
      source_articles: [],
      related_laws: [],
      related_cases: [],
      collected_at: new Date().toISOString(),
    }
    onGenerateScript(pseudoIssue)
  }

  // 뉴스 리스트 뷰
  if (step === 'news' && newsResponse && selectedKeyword) {
    return (
      <KeywordNewsList
        keyword={selectedKeyword.keyword}
        articles={newsResponse.articles}
        relatedLaws={newsResponse.related_laws ?? []}
        sourcesUsed={newsResponse.sources_used ?? []}
        totalCount={newsResponse.total_count}
        onBack={handleBack}
        onGenerateScript={handleGenerateFromNews}
      />
    )
  }

  return (
    <div className="space-y-5">
      {/* 상단 액션 바 */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-gray-900">키워드 탐색</h2>
          <p className="text-sm text-gray-500 mt-0.5">
            커뮤니티 트렌드에서 법률 콘텐츠 키워드를 발견하세요
          </p>
        </div>
        <div className="flex items-center gap-2">
          {keywords.length > 0 && (
            <button
              onClick={handleReset}
              className="px-3 py-2 text-sm text-gray-600 hover:text-gray-900 transition-colors"
            >
              초기화
            </button>
          )}
          <button
            onClick={handleCollect}
            disabled={step === 'collecting'}
            className="px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {step === 'collecting' ? '수집 중...' : '키워드 수집'}
          </button>
        </div>
      </div>

      {/* 캐시 힌트 */}
      {cacheHit && (
        <div className="text-xs text-amber-600 bg-amber-50 rounded-lg px-3 py-2">
          캐시된 결과입니다. 새로운 키워드를 수집하려면 잠시 후 다시 시도하세요.
        </div>
      )}

      {/* 에러 */}
      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 text-sm rounded-lg px-4 py-3">
          {error}
        </div>
      )}

      {/* SSE 프로그레스 바 (§7.4) */}
      {step === 'collecting' && streamProgress && (
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <div className="flex flex-col items-center gap-4">
            <div className="w-full max-w-md">
              <div className="flex justify-between text-xs text-gray-500 mb-1.5">
                <span>{streamProgress.message}</span>
                <span>{streamProgress.progress}%</span>
              </div>
              <div className="w-full h-2.5 bg-gray-100 rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue-500 rounded-full transition-all duration-500 ease-out"
                  style={{ width: `${streamProgress.progress}%` }}
                />
              </div>
            </div>
            <span className="text-xs text-gray-400">
              Tavily + Naver 병렬 수집 + LLM 스코어링
            </span>
          </div>
        </div>
      )}

      {/* 뉴스 검색 중 */}
      {step === 'searching' && selectedKeyword && (
        <div className="flex items-center justify-center py-16">
          <div className="flex flex-col items-center gap-3 text-gray-500">
            <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
            <span className="text-sm">
              &ldquo;{selectedKeyword.keyword}&rdquo; 관련 뉴스를 검색하고 있습니다...
            </span>
          </div>
        </div>
      )}

      {/* 키워드 카드 그리드 */}
      {step === 'keywords' && keywords.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {keywords.map((keyword) => (
            <KeywordCard
              key={keyword.id}
              keyword={keyword}
              onSearchNews={handleSearchNews}
            />
          ))}
        </div>
      )}

      {/* 초기 안내 */}
      {step === 'idle' && keywords.length === 0 && !error && (
        <div className="text-center py-16 text-gray-400">
          <p className="text-sm">&ldquo;키워드 수집&rdquo; 버튼을 눌러 커뮤니티 트렌드를 분석하세요.</p>
          <p className="text-xs mt-1">DC갤러리, 에펨코리아, 더쿠, 보배드림, 인벤에서 수집합니다.</p>
        </div>
      )}
    </div>
  )
}
