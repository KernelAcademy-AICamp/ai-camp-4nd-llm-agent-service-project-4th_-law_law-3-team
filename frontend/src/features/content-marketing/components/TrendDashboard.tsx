'use client'

import { useCallback, useState } from 'react'
import type { LawyerPersona, NewsArticleForScript, TrendFilters as TrendFiltersType, TrendIssue } from '../types'
import { useTrends } from '../hooks/useTrends'
import { TrendCard } from './TrendCard'
import { TrendDetailView } from './TrendDetailView'
import { TrendFilters } from './TrendFilters'
import { KeywordCollector } from './KeywordCollector'

type DashboardTab = 'keywords' | 'trends'

interface TrendDashboardProps {
  onGenerateScript: (issue: TrendIssue) => void
  onGenerateScriptWithNews?: (keyword: string, articles: NewsArticleForScript[]) => void
  personaId?: string | null
  persona?: LawyerPersona | null
}

export function TrendDashboard({ onGenerateScript, onGenerateScriptWithNews, personaId, persona }: TrendDashboardProps) {
  const [activeTab, setActiveTab] = useState<DashboardTab>('keywords')
  const { trends, loading, error, cacheHit, loadTrends, loadDetail } = useTrends()
  const [filters, setFilters] = useState<TrendFiltersType>({
    time_range: '48h',
    category: 'all',
  })
  const [selectedIssue, setSelectedIssue] = useState<TrendIssue | null>(null)
  const [hasStarted, setHasStarted] = useState(false)

  const handleStart = useCallback(() => {
    setHasStarted(true)
    loadTrends(filters)
  }, [loadTrends, filters])

  const handleFilterChange = useCallback(
    (newFilters: TrendFiltersType) => {
      setFilters(newFilters)
      loadTrends(newFilters)
    },
    [loadTrends],
  )

  const handleRefresh = useCallback(() => {
    loadTrends(filters)
  }, [loadTrends, filters])

  return (
    <div className="space-y-5">
      {/* 탭 네비게이션 */}
      <div className="flex border-b border-gray-200">
        <button
          onClick={() => setActiveTab('keywords')}
          className={`px-4 py-2.5 text-sm font-medium border-b-2 transition-colors ${
            activeTab === 'keywords'
              ? 'text-blue-600 border-blue-600'
              : 'text-gray-500 border-transparent hover:text-gray-700'
          }`}
        >
          키워드 탐색
        </button>
        <button
          onClick={() => setActiveTab('trends')}
          className={`px-4 py-2.5 text-sm font-medium border-b-2 transition-colors ${
            activeTab === 'trends'
              ? 'text-blue-600 border-blue-600'
              : 'text-gray-500 border-transparent hover:text-gray-700'
          }`}
        >
          트렌드 분석
        </button>
      </div>

      {/* 키워드 탐색 탭 */}
      {activeTab === 'keywords' && (
        <KeywordCollector onGenerateScript={onGenerateScript} onGenerateScriptWithNews={onGenerateScriptWithNews} personaId={personaId} persona={persona} />
      )}

      {/* 트렌드 분석 탭 */}
      {activeTab === 'trends' && (
        <>
          {!hasStarted ? (
            /* 시작 전 안내 화면 */
            <div className="flex flex-col items-center justify-center py-20 space-y-4">
              <p className="text-gray-500 text-sm">
                실시간 법률 트렌드를 수집하고 분석합니다.
              </p>
              <button
                onClick={handleStart}
                className="px-6 py-2.5 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-700 transition-colors"
              >
                트렌드 분석 시작
              </button>
            </div>
          ) : (
            <>
              {/* 필터 바 */}
              <TrendFilters
                filters={filters}
                onFilterChange={handleFilterChange}
                onRefresh={handleRefresh}
                loading={loading}
                cacheHit={cacheHit}
              />

              {/* 에러 */}
              {error && (
                <div className="bg-red-50 border border-red-200 text-red-700 text-sm rounded-lg px-4 py-3">
                  {error}
                </div>
              )}

              {/* 로딩 */}
              {loading && (
                <div className="flex items-center justify-center py-16">
                  <div className="flex items-center gap-3 text-gray-500">
                    <div className="w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
                    트렌드를 수집하고 분석하고 있습니다...
                  </div>
                </div>
              )}

              {/* 카드 그리드 */}
              {!loading && trends.length > 0 && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {trends.map((issue, index) => (
                    <TrendCard
                      key={issue.id}
                      issue={issue}
                      rank={index + 1}
                      onSelect={setSelectedIssue}
                      onGenerateScript={onGenerateScript}
                    />
                  ))}
                </div>
              )}

              {/* 빈 상태 */}
              {!loading && !error && trends.length === 0 && (
                <div className="text-center py-16 text-gray-400">
                  트렌드가 없습니다. 필터를 변경하거나 새로고침해 보세요.
                </div>
              )}
            </>
          )}

          {/* 상세 뷰 모달 */}
          {selectedIssue && (
            <TrendDetailView
              issue={selectedIssue}
              onClose={() => setSelectedIssue(null)}
              onGenerateScript={(issue) => {
                setSelectedIssue(null)
                onGenerateScript(issue)
              }}
              loadDetail={loadDetail}
            />
          )}
        </>
      )}
    </div>
  )
}
