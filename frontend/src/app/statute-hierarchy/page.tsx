'use client'

import { Suspense } from 'react'
import dynamic from 'next/dynamic'

const StatuteHierarchyView = dynamic(
  () => import('@/features/case-precedent/components/StatuteHierarchyView').then((m) => m.StatuteHierarchyView),
  { ssr: false }
)

function ViewSkeleton() {
  return (
    <div className="flex h-full w-full items-center justify-center bg-slate-900">
      <div className="text-center">
        <div className="w-16 h-16 bg-slate-700 rounded-full animate-pulse mx-auto mb-4" />
        <div className="h-4 w-48 bg-slate-700 rounded animate-pulse mx-auto" />
      </div>
    </div>
  )
}

export default function StatuteHierarchyPage() {
  return (
    <div className="fixed inset-0 bg-slate-900">
      <Suspense fallback={<ViewSkeleton />}>
        <StatuteHierarchyView />
      </Suspense>
    </div>
  )
}
