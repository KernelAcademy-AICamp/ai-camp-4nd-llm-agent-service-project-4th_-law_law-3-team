'use client'

import type { LawyerPersona, NewsArticleForScript, SourceFailInfo, TimeRange, TrendCategory, TrendIssue } from '../types'
import { useKeywordFlow } from '../hooks/useKeywordFlow'
import { KeywordCard } from './KeywordCard'
import { KeywordNewsList } from './KeywordNewsList'

const TIME_RANGE_OPTIONS: { value: TimeRange; label: string; hint: string }[] = [
  { value: '48h', label: '48시간 이내', hint: '빠름' },
  { value: '7d', label: '일주일 이내', hint: '보통' },
  { value: '14d', label: '2주일 이내', hint: '넓음' },
  { value: '30d', label: '한달 이내', hint: '폭넓음' },
]

const CATEGORY_OPTIONS: { value: TrendCategory; label: string }[] = [
  { value: 'all', label: '전체' },
  { value: 'criminal', label: '형사' },
  { value: 'civil', label: '민사' },
  { value: 'labor', label: '노동' },
  { value: 'family', label: '가사' },
  { value: 'administrative', label: '행정' },
  { value: 'corporate', label: '기업' },
  { value: 'ip', label: '지식재산' },
]

const ERROR_TYPE_LABELS: Record<string, string> = {
  timeout: '시간 초과',
  auth: '인증 오류',
  rate_limit: '할당량 초과',
  network: '네트워크 오류',
  parse: '응답 파싱 오류',
  unknown: '알 수 없는 오류',
}

interface KeywordCollectorProps {
  onGenerateScript: (issue: TrendIssue) => void
  onGenerateScriptWithNews?: (keyword: string, articles: NewsArticleForScript[]) => void
  personaId?: string | null
  persona?: LawyerPersona | null
}

export function KeywordCollector({ onGenerateScript, onGenerateScriptWithNews, personaId, persona }: KeywordCollectorProps) {
  const {
    step,
    keywords,
    selectedKeyword,
    newsResponse,
    selectedArticles,
    error,
    cacheHit,
    sourcesUsed,
    sourcesFailed,
    streamProgress,
    collectedAt,
    timeRange,
    setTimeRange,
    category,
    setCategory,
    handleCollect,
    handleSearchNews,
    handleToggleArticle,
    handleBack,
    handleReset,
    handleClearCache,
    isClearingCache,
  } = useKeywordFlow(personaId ?? null, persona)

  // 뉴스 → 대본 생성 연결 (기존 단건)
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

  // 선택한 여러 뉴스로 대본 생성
  const handleGenerateFromSelectedNews = () => {
    if (selectedArticles.length === 0 || !selectedKeyword) return

    const articlesForScript: NewsArticleForScript[] = selectedArticles.map((a) => ({
      title: a.title,
      snippet: a.snippet,
      source: a.source,
      published_at: a.published_at,
    }))

    if (onGenerateScriptWithNews) {
      onGenerateScriptWithNews(selectedKeyword.keyword, articlesForScript)
    }
  }

  // tavily, naver만 표시
  const VISIBLE_SOURCES = new Set(['tavily', 'naver'])

  // 뉴스 리스트 뷰
  if (step === 'news' && newsResponse && selectedKeyword) {
    return (
      <KeywordNewsList
        keyword={selectedKeyword.keyword}
        articles={newsResponse.articles}
        relatedLaws={newsResponse.related_laws ?? []}
        sourcesUsed={(newsResponse.sources_used ?? []).filter((s) => VISIBLE_SOURCES.has(s))}
        sourcesFailed={(newsResponse.sources_failed ?? []).filter((s) => VISIBLE_SOURCES.has(s.source_name))}
        totalCount={newsResponse.total_count}
        selectedArticles={selectedArticles}
        onToggleArticle={handleToggleArticle}
        onBack={handleBack}
        onGenerateScript={handleGenerateFromNews}
        onGenerateWithSelected={handleGenerateFromSelectedNews}
      />
    )
  }
  const filteredSourcesUsed = sourcesUsed.filter((s) => VISIBLE_SOURCES.has(s))
  const filteredSourcesFailed = sourcesFailed.filter((s) => VISIBLE_SOURCES.has(s.source_name))

  const hasResults = keywords.length > 0 || cacheHit
  const showSourceStatus = step === 'keywords' && (filteredSourcesUsed.length > 0 || filteredSourcesFailed.length > 0)

  return (
    <div className="space-y-5">
      {/* 상단 액션 바 */}
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-gray-900">키워드 탐색</h2>
          <p className="text-sm text-gray-500 mt-0.5">
            커뮤니티 트렌드에서 법률 콘텐츠 키워드를 발견하세요
          </p>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <select
            value={category}
            onChange={(e) => setCategory(e.target.value as TrendCategory)}
            disabled={step === 'collecting'}
            className="px-3 py-2 text-sm border border-gray-300 rounded-lg bg-white text-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {CATEGORY_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
          <select
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value as TimeRange)}
            disabled={step === 'collecting'}
            className="px-3 py-2 text-sm border border-gray-300 rounded-lg bg-white text-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {TIME_RANGE_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label} ({option.hint})
              </option>
            ))}
          </select>
          <button
            onClick={() => handleCollect()}
            disabled={step === 'collecting'}
            className="px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {step === 'collecting' ? '수집 중...' : '키워드 수집'}
          </button>
        </div>
      </div>

      {/* 캐시 안내 배너 */}
      {cacheHit && (
        <div className="flex items-center justify-between bg-amber-50 border border-amber-200 rounded-lg px-4 py-2.5">
          <div className="flex items-center gap-2">
            <svg className="w-4 h-4 text-amber-500 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span className="text-xs text-amber-700">
              캐시된 결과입니다. 새로운 데이터를 가져오려면 &ldquo;새로 수집&rdquo;을 클릭하세요.
            </span>
          </div>
          <button
            onClick={() => handleCollect(true)}
            disabled={step === 'collecting'}
            className="px-3 py-1 text-xs font-medium text-amber-700 bg-white border border-amber-300 rounded-md hover:bg-amber-50 disabled:opacity-50 transition-colors ml-3 shrink-0"
          >
            새로 수집
          </button>
        </div>
      )}

      {/* 에러 */}
      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 text-sm rounded-lg px-4 py-3">
          {error}
        </div>
      )}

      {/* SSE 프로그레스 바 */}
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
              멀티소스 병렬 수집 + LLM 스코어링
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

      {/* 소스 상태 패널 */}
      {showSourceStatus && (
        <SourceStatusPanel
          sourcesUsed={filteredSourcesUsed}
          sourcesFailed={filteredSourcesFailed}
          hasResults={hasResults}
          onRefresh={handleReset}
        />
      )}

      {/* Early Signal 배너 (v3: is_early_signal 키워드 존재 시) */}
      {step === 'keywords' && keywords.some((k) => k.is_early_signal) && (
        <div className="flex items-center gap-2 bg-amber-50 border border-amber-200 rounded-lg px-4 py-2.5">
          <span className="px-1.5 py-0.5 text-[10px] font-semibold bg-amber-100 text-amber-700 border border-amber-300 rounded shrink-0">
            Early Signal
          </span>
          <span className="text-xs text-amber-700">
            아직 주류 뉴스에 노출되지 않았지만 잠재력이 높은 키워드가 {keywords.filter((k) => k.is_early_signal).length}건 감지되었습니다.
          </span>
        </div>
      )}

      {/* 결과 메타 정보 + 캐시 초기화 */}
      {step === 'keywords' && keywords.length > 0 && (
        <div className="flex items-center justify-between text-xs text-gray-500">
          <div className="flex items-center gap-3">
            {collectedAt && (
              <span className="inline-flex items-center gap-1">
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <CollectedAtLabel collectedAt={collectedAt} />
              </span>
            )}
            {filteredSourcesUsed.length > 0 && (
              <span>
                소스 {filteredSourcesUsed.length}/{filteredSourcesUsed.length + filteredSourcesFailed.length} 성공
              </span>
            )}
            <button
              onClick={() => handleCollect(true)}
              className="text-blue-600 hover:text-blue-800 font-medium transition-colors"
            >
              새로 수집
            </button>
          </div>
          <button
            onClick={handleClearCache}
            disabled={isClearingCache}
            className="px-2.5 py-1 text-xs font-medium text-red-500 hover:text-red-700 border border-red-200 rounded-md hover:bg-red-50 disabled:opacity-50 transition-colors"
          >
            {isClearingCache ? '초기화 중...' : '캐시 초기화'}
          </button>
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

// ── 소스 상태 패널 컴포넌트 ──

interface SourceStatusPanelProps {
  sourcesUsed: string[]
  sourcesFailed: SourceFailInfo[]
  hasResults: boolean
  onRefresh: () => void
}

function SourceStatusPanel({
  sourcesUsed,
  sourcesFailed,
  hasResults,
  onRefresh,
}: SourceStatusPanelProps) {
  if (sourcesUsed.length === 0 && sourcesFailed.length === 0) return null

  const allFailed = sourcesUsed.length === 0 && sourcesFailed.length > 0
  const partialFailed = sourcesUsed.length > 0 && sourcesFailed.length > 0

  return (
    <div className={`border rounded-lg px-4 py-3 ${
      allFailed
        ? 'bg-red-50 border-red-200'
        : partialFailed
          ? 'bg-amber-50 border-amber-200'
          : 'bg-gray-50 border-gray-200'
    }`}>
      <div className="flex items-center justify-between mb-2">
        <p className={`text-xs font-medium ${
          allFailed ? 'text-red-700' : partialFailed ? 'text-amber-700' : 'text-gray-600'
        }`}>
          {allFailed
            ? '모든 소스에서 수집에 실패했습니다'
            : partialFailed
              ? `데이터 소스 상태 — ${sourcesFailed.length}개 소스 실패`
              : `데이터 소스 상태 — ${sourcesUsed.length}개 소스 성공`}
        </p>
        {!hasResults && (
          <button
            onClick={onRefresh}
            className="text-xs text-blue-600 hover:text-blue-800 transition-colors"
          >
            다시 시도
          </button>
        )}
      </div>
      <div className="flex flex-wrap gap-1.5">
        {sourcesUsed.map((source) => (
          <span
            key={source}
            className="inline-flex items-center gap-1 text-xs bg-green-50 text-green-700 border border-green-200 px-2 py-0.5 rounded-full"
          >
            <span className="w-1.5 h-1.5 bg-green-500 rounded-full" />
            {source}
          </span>
        ))}
        {sourcesFailed.map((fail) => (
          <FailedSourceBadge key={fail.source_name} fail={fail} />
        ))}
      </div>
    </div>
  )
}

// ── 실패 소스 배지 (툴팁 포함) ──

function CollectedAtLabel({ collectedAt }: { collectedAt: string }) {
  const diffMinutes = Math.floor(
    (Date.now() - new Date(collectedAt).getTime()) / 60000,
  )
  if (diffMinutes < 1) return <span>방금 수집</span>
  if (diffMinutes < 60) return <span>{diffMinutes}분 전 수집</span>
  const diffHours = Math.floor(diffMinutes / 60)
  return <span>{diffHours}시간 전 수집</span>
}

function FailedSourceBadge({ fail }: { fail: SourceFailInfo }) {
  const errorLabel = ERROR_TYPE_LABELS[fail.error_type] ?? fail.error_type
  const tooltipText = fail.error_message
    ? `${errorLabel}: ${fail.error_message}`
    : errorLabel

  return (
    <span
      className="inline-flex items-center gap-1 text-xs bg-red-50 text-red-600 border border-red-200 px-2 py-0.5 rounded-full cursor-help"
      title={tooltipText}
    >
      <span className="w-1.5 h-1.5 bg-red-400 rounded-full" />
      {fail.source_name}
      <span className="text-red-400">({errorLabel})</span>
    </span>
  )
}
