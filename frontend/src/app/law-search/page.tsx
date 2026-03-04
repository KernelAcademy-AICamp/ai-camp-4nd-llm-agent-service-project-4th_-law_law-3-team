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

const FilterableLawView = dynamic(
  () => import('@/features/case-precedent/components/FilterableLawView').then((m) => m.FilterableLawView),
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

function LawSearchContent() {
  const { userRole, sessionData } = useChat()
  const { isChatOpen, chatMode } = useUI()
  const searchParams = useSearchParams()
  const initialCaseId = searchParams.get('id')

  const aiReferences = sessionData.aiReferences as unknown[] | undefined
  const hasChatReferences = Array.isArray(aiReferences) && aiReferences.length > 0
  const isFilterMode = !hasChatReferences && !initialCaseId

  const pageTitle = '법령 검색'

  return (
    <div
      className={`h-screen flex flex-col bg-gray-100 transition-all duration-500 ease-in-out ${
        isChatOpen && chatMode === 'split' ? 'w-1/2 border-r border-gray-200' : 'w-full'
      }`}
    >
      <header className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="flex items-center gap-3">
          <BackButton />
          <h1 className="text-xl font-bold text-gray-900">{pageTitle}</h1>
        </div>
      </header>

      <div className="flex-1 overflow-hidden">
        <Suspense fallback={<ViewSkeleton />}>
          {initialCaseId ? (
            <LawyerView initialCaseId={initialCaseId} pageType="law" />
          ) : isFilterMode ? (
            <FilterableLawView />
          ) : userRole === 'lawyer' ? (
            <LawyerView pageType="law" />
          ) : (
            <UserView pageType="law" />
          )}
        </Suspense>
      </div>
    </div>
  )
}

export default function LawSearchPage() {
  return (
    <Suspense fallback={<ViewSkeleton />}>
      <LawSearchContent />
    </Suspense>
  )
}
