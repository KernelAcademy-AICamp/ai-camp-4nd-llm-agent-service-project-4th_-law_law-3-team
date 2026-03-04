'use client'

import { Suspense } from 'react'
import dynamic from 'next/dynamic'
import { useSearchParams } from 'next/navigation'
import { useChat } from '@/context/ChatContext'
import { useUI } from '@/context/UIContext'
import { BackButton } from '@/components/ui/BackButton'

const LawyerView = dynamic(
  () => import('@/features/case-precedent/components/LawyerView').then((m) => m.LawyerView),
  { ssr: false }
)

const UserView = dynamic(
  () => import('@/features/case-precedent/components/UserView').then((m) => m.UserView),
  { ssr: false }
)

const FilterablePrecedentView = dynamic(
  () => import('@/features/case-precedent/components/FilterablePrecedentView').then((m) => m.FilterablePrecedentView),
  { ssr: false }
)

function ViewSkeleton() {
  return (
    <div className="flex h-full">
      <div className="w-96 bg-white border-r border-navy-100 p-4">
        <div className="h-10 bg-navy-100 rounded-lg animate-pulse mb-4" />
        <div className="space-y-3">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-24 bg-navy-50 rounded-lg animate-pulse" />
          ))}
        </div>
      </div>
      <div className="flex-1 p-6">
        <div className="h-8 w-48 bg-navy-100 rounded animate-pulse mb-4" />
        <div className="h-64 bg-navy-50 rounded-lg animate-pulse" />
      </div>
    </div>
  )
}

function CasePrecedentContent() {
  const { userRole, sessionData } = useChat()
  const { isChatOpen } = useUI()
  const searchParams = useSearchParams()
  const agentType = searchParams.get('agent')
  const initialCaseId = searchParams.get('id')

  const isLawSearch = agentType === 'law_search'

  // 진입 모드 판단: 채팅에서 aiReferences로 진입했는지 여부
  const aiReferences = sessionData.aiReferences as unknown[] | undefined
  const hasChatReferences = Array.isArray(aiReferences) && aiReferences.length > 0
  const isFilterMode = !hasChatReferences && agentType === 'case_search' && !initialCaseId

  const pageTitle = isFilterMode ? '판례 검색' : isLawSearch ? '법령 검색' : '판례 검색'
  const pageIcon = isLawSearch ? '📖' : '📚'
  const pageDescription = isFilterMode
    ? '사건종류, 기간으로 판례를 검색하세요'
    : userRole === 'lawyer'
      ? (isLawSearch ? '전문가용 법령 검색 및 분석 시스템' : '전문가용 판례 검색 및 분석 시스템')
      : (isLawSearch ? 'AI 기반 쉬운 법령 열람' : 'AI 기반 쉬운 판례/법령 열람')

  return (
    <div
      className={`h-screen flex flex-col bg-gray-100 transition-all duration-500 ease-in-out ${
        isChatOpen ? 'w-1/2 border-r border-gray-200' : 'w-full'
      }`}
    >
      <header className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="flex items-center gap-3">
          <BackButton />
          <span className="text-2xl">{pageIcon}</span>
          <div>
            <h1 className="text-xl font-bold text-gray-900">{pageTitle}</h1>
            <p className="text-sm text-gray-500">{pageDescription}</p>
          </div>
        </div>
      </header>

      <div className="flex-1 overflow-hidden">
        <Suspense fallback={<ViewSkeleton />}>
          {initialCaseId ? (
            <LawyerView initialCaseId={initialCaseId} />
          ) : isFilterMode ? (
            <FilterablePrecedentView />
          ) : userRole === 'lawyer' ? (
            <LawyerView />
          ) : (
            <UserView />
          )}
        </Suspense>
      </div>
    </div>
  )
}

export default function CasePrecedentPage() {
  return (
    <Suspense fallback={<ViewSkeleton />}>
      <CasePrecedentContent />
    </Suspense>
  )
}
