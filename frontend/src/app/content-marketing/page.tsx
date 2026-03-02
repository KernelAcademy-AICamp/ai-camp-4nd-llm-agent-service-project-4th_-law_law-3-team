'use client'

import { useCallback, useState } from 'react'
import { BackButton } from '@/components/ui/BackButton'
import { DisclaimerBanner } from '@/features/content-marketing/components/DisclaimerBanner'
import { PersonaBanner } from '@/features/content-marketing/components/PersonaBanner'
import { PersonaGate } from '@/features/content-marketing/components/PersonaGate'
import { ScriptGenerator } from '@/features/content-marketing/components/ScriptGenerator'
import { TrendDashboard } from '@/features/content-marketing/components/TrendDashboard'
import { submitPersonaFeedback } from '@/features/content-marketing/services'
import type { NewsArticleForScript, TrendIssue } from '@/features/content-marketing/types'

type Tab = 'trends' | 'script'

export default function ContentMarketingPage() {
  const [activeTab, setActiveTab] = useState<Tab>('trends')
  const [scriptTopic, setScriptTopic] = useState('')
  const [scriptTrend, setScriptTrend] = useState<TrendIssue | null>(null)
  const [scriptNewsArticles, setScriptNewsArticles] = useState<NewsArticleForScript[] | null>(null)

  const handleGenerateFromTrend = useCallback((issue: TrendIssue) => {
    setScriptTopic(issue.title)
    setScriptTrend(issue)
    setScriptNewsArticles(null)
    setActiveTab('script')
  }, [])

  const handleGenerateScriptWithNews = useCallback((keyword: string, articles: NewsArticleForScript[]) => {
    setScriptTopic(keyword)
    setScriptTrend(null)
    setScriptNewsArticles(articles)
    setActiveTab('script')
  }, [])

  return (
    <div className="min-h-screen bg-gray-50">
      {/* 면책 고지 */}
      <DisclaimerBanner />

      {/* 헤더 */}
      <header className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="max-w-6xl mx-auto flex items-center gap-3">
          <BackButton />
          <div>
            <h1 className="text-xl font-bold text-gray-900">콘텐츠 마케팅 자동화</h1>
            <p className="text-sm text-gray-500">
              법률 트렌드 분석 및 AI 유튜브 대본 생성
            </p>
          </div>
        </div>
      </header>

      {/* PersonaGate: persona 준비 완료 후 탭+콘텐츠 렌더 */}
      <PersonaGate>
        {({ persona, personaId, onEditPersona, onQuickUpdatePersona }) => (
          <>
            {/* 페르소나 배너 */}
            <PersonaBanner persona={persona} onEdit={onEditPersona} onQuickUpdate={onQuickUpdatePersona} />

            {/* 탭 */}
            <div className="bg-white border-b border-gray-200">
              <div className="max-w-6xl mx-auto px-6 flex gap-1">
                <button
                  onClick={() => setActiveTab('trends')}
                  className={`px-5 py-3 text-sm font-medium border-b-2 transition-colors ${
                    activeTab === 'trends'
                      ? 'border-blue-600 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700'
                  }`}
                >
                  트렌드 분석
                </button>
                <button
                  onClick={() => setActiveTab('script')}
                  className={`px-5 py-3 text-sm font-medium border-b-2 transition-colors ${
                    activeTab === 'script'
                      ? 'border-blue-600 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700'
                  }`}
                >
                  대본 생성
                </button>
              </div>
            </div>

            {/* 탭 콘텐츠 */}
            <main className="max-w-6xl mx-auto px-6 py-6">
              {activeTab === 'trends' ? (
                <TrendDashboard onGenerateScript={handleGenerateFromTrend} onGenerateScriptWithNews={handleGenerateScriptWithNews} personaId={personaId} persona={persona} />
              ) : (
                <ScriptGenerator
                  key={`${scriptTrend?.id ?? 'manual'}-${scriptNewsArticles?.length ?? 0}`}
                  initialTopic={scriptTopic}
                  initialTrend={scriptTrend}
                  initialNewsArticles={scriptNewsArticles}
                  personaId={personaId}
                  savedPersona={persona}
                  onFeedback={submitPersonaFeedback}
                />
              )}
            </main>
          </>
        )}
      </PersonaGate>
    </div>
  )
}
