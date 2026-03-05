'use client'

import { useState } from 'react'
import type { WebtoonPanel } from '../types'
import type { StoryboardPhase, StoryboardProgress } from '../hooks/useStoryboardStream'
import { StoryboardPanelCard } from './StoryboardPanelCard'

interface StoryboardPanelProps {
  panels: WebtoonPanel[]
  phase: StoryboardPhase
  progress: StoryboardProgress
  activeSection: string | null
  onRegeneratePanel: (panelNumber: number) => void
}

const SECTION_ORDER = ['hooking', 'analysis', 'advice_cta'] as const
const SECTION_LABELS: Record<string, string> = {
  hooking: '도입 (Hooking)',
  analysis: '본론 (Analysis)',
  advice_cta: '결론 (Advice & CTA)',
}

export function StoryboardPanel({
  panels,
  phase,
  progress,
  activeSection,
  onRegeneratePanel,
}: StoryboardPanelProps) {
  const [expandedImage, setExpandedImage] = useState<string | null>(null)

  // 섹션별 패널 그룹핑
  const groupedPanels = SECTION_ORDER.reduce(
    (acc, section) => {
      acc[section] = panels.filter((p) => p.section === section)
      return acc
    },
    {} as Record<string, WebtoonPanel[]>,
  )

  if (phase === 'idle') {
    return (
      <div className="flex items-center justify-center h-full text-gray-400 text-sm">
        <p>대본 생성 완료 후 스토리보드가 표시됩니다</p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* 진행률 */}
      {(phase === 'scene_split' || phase === 'generating') && (
        <div className="space-y-1.5">
          <div className="flex items-center justify-between text-xs">
            <span className="text-gray-600 font-medium">{progress.label}</span>
            <span className="text-gray-500 tabular-nums">{progress.percent}%</span>
          </div>
          <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
            <div
              className="h-full rounded-full bg-indigo-500 transition-all duration-500 ease-out"
              style={{ width: `${progress.percent}%` }}
            />
          </div>
        </div>
      )}

      {/* 장면 분할 중 */}
      {phase === 'scene_split' && (
        <div className="flex items-center gap-2 text-xs text-indigo-600 px-2 py-3 bg-indigo-50 rounded-lg">
          <div className="w-4 h-4 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin" />
          {progress.label || 'AI가 대본을 장면으로 분할하고 있습니다...'}
        </div>
      )}

      {/* 패널 그리드 (섹션별) */}
      {panels.length > 0 && (
        <div className="space-y-4">
          {SECTION_ORDER.map((section) => {
            const sectionPanels = groupedPanels[section]
            if (!sectionPanels || sectionPanels.length === 0) return null

            return (
              <div key={section}>
                <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2 px-1">
                  {SECTION_LABELS[section]}
                </h4>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  {sectionPanels.map((panel) => (
                    <StoryboardPanelCard
                      key={panel.panel_number}
                      panel={panel}
                      isHighlighted={activeSection === section}
                      onRegenerate={() => onRegeneratePanel(panel.panel_number)}
                      onImageClick={() => {
                        if (panel.image_url) setExpandedImage(panel.image_url)
                      }}
                    />
                  ))}
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* 완료 상태 */}
      {phase === 'completed' && (
        <div className="text-center py-2 text-xs text-green-600 font-medium">
          스토리보드 생성 완료 ({panels.length}패널)
        </div>
      )}

      {/* 에러 상태 */}
      {phase === 'error' && (
        <div className="text-center py-3 text-xs text-red-500">
          스토리보드 생성 중 오류가 발생했습니다
        </div>
      )}

      {/* 이미지 확대 모달 */}
      {expandedImage && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4"
          onClick={() => setExpandedImage(null)}
          role="dialog"
          aria-modal="true"
          aria-label="이미지 확대 보기"
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={expandedImage}
            alt="확대된 웹툰 패널"
            className="max-w-full max-h-full rounded-lg shadow-2xl"
          />
        </div>
      )}
    </div>
  )
}
