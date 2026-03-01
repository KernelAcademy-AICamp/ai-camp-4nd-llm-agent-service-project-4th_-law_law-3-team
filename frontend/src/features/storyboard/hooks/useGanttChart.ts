'use client'

import { useState, useMemo } from 'react'
import type { TimelineItem, EvidenceFile } from '../types'

// vis-timeline DataSet 아이템 형태
export interface VisItem {
  id: string
  group: string
  content: string
  start: string
  end: string
  type: 'range' | 'point'
  className: string
  title: string
}

// vis-timeline 그룹 형태
export interface VisGroup {
  id: string
  content: string
  className?: string
}

// 신뢰도 레벨 분류
function getConfidenceLevel(confidence: number | undefined): 'high' | 'medium' | 'low' {
  if (confidence === undefined || confidence >= 0.8) return 'high'
  if (confidence >= 0.5) return 'medium'
  return 'low'
}

// TimelineItem[] → VisItem[] 변환
export function timelineItemsToVisItems(items: TimelineItem[]): VisItem[] {
  return items.map((item) => {
    const start = item.dateStart ?? item.date
    const end = item.dateEnd ?? item.dateStart ?? item.date
    const isRange = item.dateStart && item.dateEnd && item.dateStart !== item.dateEnd
    const confidenceLevel = getConfidenceLevel(item.confidence)

    return {
      id: item.id,
      group: item.topic ?? 'uncategorized',
      content: item.title,
      start,
      end,
      type: isRange ? 'range' : 'point',
      className: `gantt-item confidence-${confidenceLevel}`,
      title: item.descriptionShort ?? item.description,
    }
  })
}

// topics → VisGroup[] 변환 (Y축 그룹)
export function topicsToVisGroups(items: TimelineItem[]): VisGroup[] {
  const topicSet = new Set<string>()
  for (const item of items) {
    if (item.topic) topicSet.add(item.topic)
  }

  const topicGroups: VisGroup[] = Array.from(topicSet).map((topic) => ({
    id: topic,
    content: topic,
  }))

  if (items.some((item) => !item.topic)) {
    topicGroups.push({ id: 'uncategorized', content: '미분류' })
  }

  return topicGroups
}

// 증거 파일 → 마커 VisItem[] 변환
export function evidenceToVisMarkers(
  evidenceFiles: EvidenceFile[],
  items: TimelineItem[],
): VisItem[] {
  const EVIDENCE_ICONS: Record<string, string> = {
    kakao_txt: '📱',
    messenger_screenshot: '💬',
    voice_recording: '🎤',
    document: '📄',
    photo: '📷',
    text_input: '✏️',
    other: '📎',
  }

  // evidenceId → 연결된 TimelineItem 날짜 매핑
  const evidenceDateMap = new Map<string, string>()
  for (const item of items) {
    for (const evidenceId of item.evidenceIds ?? []) {
      if (!evidenceDateMap.has(evidenceId)) {
        evidenceDateMap.set(evidenceId, item.dateStart ?? item.date)
      }
    }
  }

  return evidenceFiles
    .filter((file) => evidenceDateMap.has(file.evidenceId))
    .map((file) => ({
      id: `evidence-${file.evidenceId}`,
      group: 'evidence-marker',
      content: EVIDENCE_ICONS[file.evidenceType] ?? '📎',
      start: evidenceDateMap.get(file.evidenceId) ?? '',
      end: evidenceDateMap.get(file.evidenceId) ?? '',
      type: 'point' as const,
      className: 'evidence-marker-item',
      title: file.filename,
    }))
}

interface UseGanttChartReturn {
  visItems: VisItem[]
  visGroups: VisGroup[]
  visMarkers: VisItem[]
  selectedItem: TimelineItem | null
  setSelectedItem: (item: TimelineItem | null) => void
  visibleTopics: Set<string>
  setVisibleTopics: (topics: Set<string>) => void
}

export function useGanttChart(
  items: TimelineItem[],
  evidenceFiles: EvidenceFile[],
): UseGanttChartReturn {
  const [selectedItem, setSelectedItem] = useState<TimelineItem | null>(null)
  const [visibleTopics, setVisibleTopics] = useState<Set<string>>(new Set())

  const visItems = useMemo(() => timelineItemsToVisItems(items), [items])
  const visGroups = useMemo(() => topicsToVisGroups(items), [items])
  const visMarkers = useMemo(
    () => evidenceToVisMarkers(evidenceFiles, items),
    [evidenceFiles, items],
  )

  return {
    visItems,
    visGroups,
    visMarkers,
    selectedItem,
    setSelectedItem,
    visibleTopics,
    setVisibleTopics,
  }
}
