'use client'

import { useState } from 'react'
import { BackButton } from '@/components/ui/BackButton'
import { DisclaimerBanner } from '@/features/legal-news/components/DisclaimerBanner'
import { NewsBarChart } from '@/features/legal-news/components/NewsBarChart'
import { NewsDonutChart } from '@/features/legal-news/components/NewsDonutChart'
import { NewsListPanel } from '@/features/legal-news/components/NewsListPanel'
import { RagContributionCard } from '@/features/legal-news/components/RagContributionCard'
import { SearchPanel } from '@/features/legal-news/components/SearchPanel'
import { useNewsStats } from '@/features/legal-news/hooks/useNewsStats'
import type { NewsTab } from '@/features/legal-news/types'

export default function LegalNewsPage() {
  const [activeTab, setActiveTab] = useState<NewsTab>('list')
  const {
    dailyStats, categoryStats, loading: statsLoading, error: statsError,
    periodDays, setPeriodDays,
    categoryPeriodDays, setCategoryPeriodDays, categoryLoading,
    ragStats, ragLoading,
  } = useNewsStats()

  return (
    <div className="min-h-screen bg-gray-50">
      {/* 면책 고지 */}
      <DisclaimerBanner />

      {/* 헤더 */}
      <header className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center gap-3">
          <BackButton />
          <div>
            <h1 className="text-xl font-bold text-gray-900">법률 뉴스</h1>
            <p className="text-sm text-gray-500">
              법률 뉴스 수집·요약 및 하이브리드 검색
            </p>
          </div>
        </div>
      </header>

      {/* 탭 */}
      <div className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-6 flex gap-1">
          <button
            onClick={() => setActiveTab('list')}
            className={`px-5 py-3 text-sm font-medium border-b-2 transition-colors ${
              activeTab === 'list'
                ? 'border-blue-600 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            뉴스 목록
          </button>
          <button
            onClick={() => setActiveTab('search')}
            className={`px-5 py-3 text-sm font-medium border-b-2 transition-colors ${
              activeTab === 'search'
                ? 'border-blue-600 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            뉴스 검색
          </button>
        </div>
      </div>

      {/* 탭 콘텐츠 */}
      <main className="max-w-7xl mx-auto px-6 py-6">
        {activeTab === 'list' ? (
          <div className="flex flex-col lg:flex-row gap-6">
            {/* 좌측: 뉴스 목록 */}
            <div className="flex-1 min-w-0">
              <NewsListPanel />
            </div>

            {/* 우측: 통계 대시보드 */}
            <aside className="w-full lg:w-[380px] shrink-0 order-first lg:order-last space-y-4">
              {statsLoading && (
                <div className="flex items-center justify-center h-32">
                  <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600" />
                </div>
              )}
              {statsError && (
                <div className="text-sm text-red-600 bg-red-50 rounded-lg p-3">
                  {statsError}
                </div>
              )}
              {!statsLoading && dailyStats && (
                <NewsBarChart
                  items={dailyStats.items}
                  periodDays={periodDays}
                  onPeriodChange={setPeriodDays}
                />
              )}
              {!statsLoading && categoryStats && (
                <NewsDonutChart
                  items={categoryStats.items}
                  total={categoryStats.total}
                  periodDays={categoryPeriodDays}
                  onPeriodChange={setCategoryPeriodDays}
                  loading={categoryLoading}
                />
              )}
              <RagContributionCard
                stats={ragStats}
                loading={ragLoading}
                error={statsError}
              />
            </aside>
          </div>
        ) : (
          <SearchPanel />
        )}
      </main>
    </div>
  )
}
