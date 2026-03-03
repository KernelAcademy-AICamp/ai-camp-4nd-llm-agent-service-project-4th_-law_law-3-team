'use client'

import { Suspense } from 'react'
import { LawyerFinderMain } from '@/features/lawyer-finder/components/LawyerFinderMain'

export default function LawyerFinderPageWrapper() {
  return (
    <Suspense fallback={<LawyerFinderLoading />}>
      <LawyerFinderPage />
    </Suspense>
  )
}

function LawyerFinderLoading() {
  return (
    <div className="h-screen flex items-center justify-center bg-gray-50">
      <div className="text-center">
        <div className="w-8 h-8 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
        <p className="text-gray-600">변호사 찾기 페이지 로딩 중...</p>
      </div>
    </div>
  )
}

function LawyerFinderPage() {
  return <LawyerFinderMain />
}
