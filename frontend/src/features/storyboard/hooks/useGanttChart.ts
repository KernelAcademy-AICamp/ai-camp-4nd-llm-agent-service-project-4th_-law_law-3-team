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
  style?: string
}

// vis-timeline 그룹 형태
export interface VisGroup {
  id: string
  content: string
  className?: string
}

// HTML 이스케이프 (vis-timeline content는 HTML 렌더링 → XSS 방어)
function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
}

// topic별 색상 팔레트 (최대 8개 topic, HSL 값)
const TOPIC_COLORS = [
  'hsl(220, 70%, 50%)',   // blue
  'hsl(160, 60%, 42%)',   // emerald
  'hsl(38, 80%, 50%)',    // amber
  'hsl(350, 65%, 52%)',   // rose
  'hsl(270, 55%, 52%)',   // violet
  'hsl(190, 65%, 45%)',   // cyan
  'hsl(25, 75%, 50%)',    // orange
  'hsl(85, 55%, 45%)',    // lime
] as const

const UNCATEGORIZED_COLOR = 'hsl(220, 10%, 60%)' // 회색

// topic → 배경색 (inline style용)
function getTopicBgColor(topic: string | undefined, topicIndex: Map<string, number>): string {
  if (!topic) return UNCATEGORIZED_COLOR
  const idx = topicIndex.get(topic) ?? 0
  return TOPIC_COLORS[idx % TOPIC_COLORS.length]
}

// 신뢰도 레벨 분류
function getConfidenceLevel(confidence: number | undefined): 'high' | 'medium' | 'low' {
  if (confidence === undefined || confidence >= 0.8) return 'high'
  if (confidence >= 0.5) return 'medium'
  return 'low'
}

// TimelineItem[] → VisItem[] 변환
export function timelineItemsToVisItems(items: TimelineItem[]): VisItem[] {
  // topic 등장 순서 인덱스 (색상 안정 할당)
  const topicIndex = new Map<string, number>()
  let idx = 0
  for (const item of items) {
    if (item.topic && !topicIndex.has(item.topic)) {
      topicIndex.set(item.topic, idx++)
    }
  }

  return items.map((item) => {
    const start = item.dateStart ?? item.date
    const end = item.dateEnd ?? item.dateStart ?? item.date
    const isRange = item.dateStart && item.dateEnd && item.dateStart !== item.dateEnd
    const confidenceLevel = getConfidenceLevel(item.confidence)
    const bgColor = getTopicBgColor(item.topic, topicIndex)

    return {
      id: item.id,
      group: item.topic ?? 'uncategorized',
      content: escapeHtml(item.title),
      start,
      end,
      type: isRange ? 'range' : 'point',
      className: `gantt-item confidence-${confidenceLevel}`,
      title: escapeHtml(item.descriptionShort ?? item.description),
      style: `background-color: ${bgColor}; color: #fff; border-radius: 6px; border: none;`,
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
      title: escapeHtml(file.filename),
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
