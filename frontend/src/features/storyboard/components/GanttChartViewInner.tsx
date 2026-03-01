'use client'

import { useEffect, useRef } from 'react'
import { Timeline, DataSet } from 'vis-timeline/standalone'
import type { DataItem, DataGroup, TimelineOptions } from 'vis-timeline'
import type { TimelineItem, EvidenceFile, MergeConflict } from '../types'
import {
  timelineItemsToVisItems,
  topicsToVisGroups,
  evidenceToVisMarkers,
  type VisItem,
  type VisGroup,
} from '../hooks/useGanttChart'
import '@/styles/gantt.css'

interface GanttChartViewInnerProps {
  items: TimelineItem[]
  evidenceFiles: EvidenceFile[]
  onItemSelect: (item: TimelineItem) => void
  onEvidenceClick: (evidenceId: string) => void
  conflicts?: MergeConflict[]
}

const TIMELINE_OPTIONS: TimelineOptions = {
  orientation: { axis: 'top', item: 'top' },
  stack: true,
  showCurrentTime: false,
  zoomMin: 1000 * 60 * 60 * 24 * 7,         // 최소 1주
  zoomMax: 1000 * 60 * 60 * 24 * 365 * 5,   // 최대 5년
  tooltip: { followMouse: true, overflowMethod: 'cap' },
  groupOrder: 'content',
  margin: { item: 10, axis: 40 },
  locale: 'ko',
  selectable: true,
  multiselect: false,
}

// VisItem → DataItem 변환 (vis-timeline 내부 타입 호환)
function toDataItems(visItems: VisItem[]): DataItem[] {
  return visItems.map((v) => ({
    id: v.id,
    group: v.group,
    content: v.content,
    start: v.start,
    end: v.end,
    type: v.type,
    className: v.className,
    title: v.title,
  }))
}

// VisGroup → DataGroup 변환
function toDataGroups(visGroups: VisGroup[]): DataGroup[] {
  return visGroups.map((g) => ({
    id: g.id,
    content: g.content,
    className: g.className,
  }))
}

export default function GanttChartViewInner({
  items,
  evidenceFiles,
  onItemSelect,
  onEvidenceClick,
  conflicts = [],
}: GanttChartViewInnerProps) {
  // 충돌 항목 ID 집합 (빠른 조회)
  const conflictItemIds = new Set<string>(
    conflicts.flatMap((c) => [c.existingItemId, c.newItemId]),
  )
  const containerRef = useRef<HTMLDivElement>(null)
  const timelineRef = useRef<Timeline | null>(null)
  const itemsDataSetRef = useRef<DataSet<DataItem> | null>(null)
  const groupsDataSetRef = useRef<DataSet<DataGroup> | null>(null)

  function buildAllItems(
    currentItems: TimelineItem[],
    currentEvidence: EvidenceFile[],
  ): DataItem[] {
    const visItems = timelineItemsToVisItems(currentItems).map((v) => ({
      ...v,
      className: conflictItemIds.has(v.id)
        ? `${v.className} gantt-conflict`
        : v.className,
    }))
    const markers = evidenceToVisMarkers(currentEvidence, currentItems)
    return toDataItems([...visItems, ...markers])
  }

  function buildAllGroups(
    currentItems: TimelineItem[],
    currentEvidence: EvidenceFile[],
  ): DataGroup[] {
    const groups = topicsToVisGroups(currentItems)
    const hasMarkers = evidenceToVisMarkers(currentEvidence, currentItems).length > 0
    if (hasMarkers) {
      groups.push({ id: 'evidence-marker', content: '증거', className: 'evidence-group' })
    }
    return toDataGroups(groups)
  }

  // 초기화
  useEffect(() => {
    if (!containerRef.current) return

    const allItems = buildAllItems(items, evidenceFiles)
    const allGroups = buildAllGroups(items, evidenceFiles)

    const itemsDataSet = new DataSet<DataItem>(allItems)
    const groupsDataSet = new DataSet<DataGroup>(allGroups)

    itemsDataSetRef.current = itemsDataSet
    groupsDataSetRef.current = groupsDataSet

    const timeline = new Timeline(
      containerRef.current,
      itemsDataSet,
      groupsDataSet,
      TIMELINE_OPTIONS,
    )
    timelineRef.current = timeline

    // 항목 선택 이벤트
    timeline.on('select', (properties: { items: (string | number)[] }) => {
      const selectedId = properties.items[0]
      if (selectedId === undefined || selectedId === null) return

      const idStr = String(selectedId)
      if (idStr.startsWith('evidence-')) {
        const evidenceId = idStr.replace('evidence-', '')
        onEvidenceClick(evidenceId)
        return
      }

      const found = items.find((item) => item.id === idStr)
      if (found) onItemSelect(found)
    })

    if (items.length > 0) {
      timeline.fit({ animation: false })
    }

    return () => {
      timeline.destroy()
      timelineRef.current = null
      itemsDataSetRef.current = null
      groupsDataSetRef.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // items/evidenceFiles/conflicts 변경 시 DataSet 업데이트
  useEffect(() => {
    const itemsDs = itemsDataSetRef.current
    const groupsDs = groupsDataSetRef.current
    if (!timelineRef.current || !itemsDs || !groupsDs) return

    const allItems = buildAllItems(items, evidenceFiles)
    const allGroups = buildAllGroups(items, evidenceFiles)

    itemsDs.clear()
    itemsDs.add(allItems)
    groupsDs.clear()
    groupsDs.add(allGroups)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [items, evidenceFiles, conflicts])

  return (
    <div className="w-full rounded-xl border border-gray-200 overflow-hidden bg-white">
      <div ref={containerRef} style={{ minHeight: 320 }} />
    </div>
  )
}
