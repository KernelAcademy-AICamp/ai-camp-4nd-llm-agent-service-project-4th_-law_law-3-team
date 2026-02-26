'use client'

import type { StageInfo } from '../hooks/useScript'
import type { SectionType } from '../types'

const SECTION_TITLES: Record<SectionType, string> = {
  hooking: '1. 도입 (Hooking)',
  analysis: '2. 본론 (Legal Analysis)',
  advice_cta: '3. 결론 (Advice & CTA)',
}

interface ScriptPreviewProps {
  sections: Record<SectionType, string>
  currentSection: SectionType | null
  isGenerating: boolean
  stageInfo: StageInfo | null
}

export function ScriptPreview({
  sections,
  currentSection,
  isGenerating,
  stageInfo,
}: ScriptPreviewProps) {
  const sectionOrder: SectionType[] = ['hooking', 'analysis', 'advice_cta']
  const hasContent = sectionOrder.some((s) => sections[s].length > 0)

  if (!hasContent && !isGenerating) {
    return (
      <div className="bg-gray-50 rounded-xl border border-gray-200 p-8 text-center text-gray-400">
        대본을 생성하면 여기에 표시됩니다.
      </div>
    )
  }

  return (
    <div className="bg-white rounded-xl border border-gray-200 divide-y divide-gray-100">
      {/* RAG 검색 진행 상태 (stage_update) */}
      {stageInfo && isGenerating && (
        <div className="px-5 py-3 flex items-center gap-3 bg-blue-50 text-blue-700">
          <svg
            className="w-4 h-4 animate-spin"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
            />
          </svg>
          <span className="text-sm font-medium">{stageInfo.detail}</span>
          {stageInfo.status === 'completed' && (
            <span className="text-xs text-blue-500 ml-auto">완료</span>
          )}
        </div>
      )}

      {sectionOrder.map((sectionKey) => {
        const content = sections[sectionKey]
        const isActive = currentSection === sectionKey && isGenerating
        if (!content && !isActive) return null

        return (
          <div key={sectionKey} className="p-5">
            <h3 className="text-sm font-bold text-gray-800 mb-3 flex items-center gap-2">
              {SECTION_TITLES[sectionKey]}
              {isActive && (
                <span className="inline-block w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
              )}
            </h3>
            <div className="text-sm text-gray-700 whitespace-pre-wrap leading-relaxed">
              {content}
              {isActive && (
                <span className="inline-block w-1 h-4 bg-blue-500 animate-pulse ml-0.5 align-middle" />
              )}
            </div>
          </div>
        )
      })}
    </div>
  )
}
