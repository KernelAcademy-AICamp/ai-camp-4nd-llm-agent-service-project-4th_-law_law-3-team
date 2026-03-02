'use client'

import type { WebtoonPanel } from '../types'

interface StoryboardPanelCardProps {
  panel: WebtoonPanel
  isHighlighted: boolean
  onRegenerate: () => void
  onImageClick: () => void
}

const SECTION_LABELS: Record<string, string> = {
  hooking: '도입',
  analysis: '본론',
  advice_cta: '결론',
}

export function StoryboardPanelCard({
  panel,
  isHighlighted,
  onRegenerate,
  onImageClick,
}: StoryboardPanelCardProps) {
  const sectionLabel = SECTION_LABELS[panel.section] || panel.section

  return (
    <div
      id={`panel-${panel.panel_number}`}
      className={`rounded-lg border transition-all ${
        isHighlighted
          ? 'border-blue-400 ring-2 ring-blue-200 shadow-md'
          : 'border-gray-200 shadow-sm'
      }`}
    >
      {/* 헤더 */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-gray-100 bg-gray-50 rounded-t-lg">
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold text-gray-500">#{panel.panel_number}</span>
          <span className="text-xs px-1.5 py-0.5 rounded bg-blue-100 text-blue-700 font-medium">
            {sectionLabel}
          </span>
        </div>
        {panel.image_status === 'completed' && (
          <button
            onClick={onRegenerate}
            className="text-xs text-gray-400 hover:text-blue-600 transition-colors"
            title="이미지 재생성"
          >
            재생성
          </button>
        )}
        {panel.image_status === 'error' && (
          <button
            onClick={onRegenerate}
            className="text-xs text-red-500 hover:text-red-700 transition-colors font-medium"
          >
            다시 시도
          </button>
        )}
      </div>

      {/* 이미지 영역 */}
      <div className="relative aspect-video bg-gray-100">
        {panel.image_status === 'pending' && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center">
              <div className="w-8 h-8 mx-auto mb-2 rounded bg-gray-200" />
              <span className="text-xs text-gray-400">대기 중</span>
            </div>
          </div>
        )}

        {panel.image_status === 'generating' && (
          <div className="absolute inset-0 flex items-center justify-center animate-pulse bg-gray-200">
            <div className="text-center">
              <div className="w-6 h-6 mx-auto mb-2 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
              <span className="text-xs text-blue-600">생성 중...</span>
            </div>
          </div>
        )}

        {panel.image_status === 'retrying' && (
          <div className="absolute inset-0 flex items-center justify-center animate-pulse bg-yellow-50">
            <div className="text-center">
              <div className="w-6 h-6 mx-auto mb-2 border-2 border-yellow-400 border-t-transparent rounded-full animate-spin" />
              <span className="text-xs text-yellow-600">재시도 중...</span>
            </div>
          </div>
        )}

        {panel.image_status === 'completed' && panel.image_url && (
          <button
            onClick={onImageClick}
            className="w-full h-full cursor-pointer focus:outline-none"
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={panel.image_url}
              alt={`패널 ${panel.panel_number}: ${panel.scene_description}`}
              className="w-full h-full object-cover rounded-b-lg"
              loading="lazy"
            />
          </button>
        )}

        {panel.image_status === 'error' && (
          <div className="absolute inset-0 flex items-center justify-center bg-red-50">
            <div className="text-center px-4">
              <span className="text-2xl block mb-1">!</span>
              <span className="text-xs text-red-500">
                {panel.error_message || '이미지 생성 실패'}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* 캡션 */}
      {panel.script_excerpt && (
        <div className="px-3 py-2">
          <p className="text-xs text-gray-600 line-clamp-2">{panel.script_excerpt}</p>
        </div>
      )}
    </div>
  )
}
