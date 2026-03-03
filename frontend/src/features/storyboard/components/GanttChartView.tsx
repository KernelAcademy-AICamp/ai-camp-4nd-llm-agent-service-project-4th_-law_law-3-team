'use client'

import dynamic from 'next/dynamic'
import type { TimelineItem, EvidenceFile, MergeConflict } from '../types'

interface GanttChartViewProps {
  items: TimelineItem[]
  evidenceFiles: EvidenceFile[]
  onItemSelect: (item: TimelineItem) => void
  onEvidenceClick: (evidenceId: string) => void
  conflicts?: MergeConflict[]
}

const GanttChartViewInner = dynamic(() => import('./GanttChartViewInner'), {
  ssr: false,
  loading: () => (
    <div className="w-full rounded-xl border border-gray-200 bg-gray-50 animate-pulse" style={{ minHeight: 320 }} />
  ),
})

export function GanttChartView(props: GanttChartViewProps) {
  return <GanttChartViewInner {...props} />
}
