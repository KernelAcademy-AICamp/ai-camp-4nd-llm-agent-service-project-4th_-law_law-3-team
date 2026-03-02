'use client'

import { useEffect, useRef, useState } from 'react'
import type { SectionType, WebtoonPanel } from '../types'
import type { StageInfo } from '../hooks/useScript'
import type { StoryboardPhase, StoryboardProgress } from '../hooks/useStoryboardStream'
import { ScriptPreview } from './ScriptPreview'
import { StoryboardPanel } from './StoryboardPanel'

interface ScriptStoryboardSplitViewProps {
  sections: Record<SectionType, string>
  currentSection: SectionType | null
  isGenerating: boolean
  stageInfo: StageInfo | null
  panels: WebtoonPanel[]
  storyboardPhase: StoryboardPhase
  storyboardProgress: StoryboardProgress
  onRegeneratePanel: (panelNumber: number) => void
}

const SECTION_KEYS: SectionType[] = ['hooking', 'analysis', 'advice_cta']

type ViewTab = 'script' | 'storyboard'

export function ScriptStoryboardSplitView({
  sections,
  currentSection,
  isGenerating,
  stageInfo,
  panels,
  storyboardPhase,
  storyboardProgress,
  onRegeneratePanel,
}: ScriptStoryboardSplitViewProps) {
  const [activeSection, setActiveSection] = useState<SectionType | null>(null)
  const [activeTab, setActiveTab] = useState<ViewTab>('script')
  const sectionRefs = useRef<Record<SectionType, HTMLDivElement | null>>({
    hooking: null,
    analysis: null,
    advice_cta: null,
  })

  const hasContent = Object.values(sections).some((s) => s.length > 0)
  const hasStoryboard = storyboardPhase !== 'idle'

  // IntersectionObserver: 대본 섹션 가시성 → 스토리보드 하이라이트
  useEffect(() => {
    if (panels.length === 0) return

    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            const section = entry.target.getAttribute('data-section') as SectionType
            setActiveSection(section)
            const firstPanel = panels.find((p) => p.section === section)
            if (firstPanel) {
              document
                .getElementById(`panel-${firstPanel.panel_number}`)
                ?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
            }
          }
        }
      },
      { threshold: 0.3 },
    )

    for (const key of SECTION_KEYS) {
      const ref = sectionRefs.current[key]
      if (ref) observer.observe(ref)
    }

    return () => observer.disconnect()
  }, [panels])

  // 스토리보드 시작 시 탭 자동 전환 (태블릿 이하)
  useEffect(() => {
    if (storyboardPhase === 'scene_split' || storyboardPhase === 'generating') {
      setActiveTab('storyboard')
    }
  }, [storyboardPhase])

  // 대본만 있고 스토리보드 없는 경우 기존 뷰
  if (!hasStoryboard) {
    return (
      <ScriptPreview
        sections={sections}
        currentSection={currentSection}
        isGenerating={isGenerating}
        stageInfo={stageInfo}
      />
    )
  }

  return (
    <div className="space-y-3">
      {/* 태블릿/모바일: 탭 전환 */}
      <div className="flex md:hidden border-b border-gray-200">
        <button
          onClick={() => setActiveTab('script')}
          className={`flex-1 py-2 text-sm font-medium transition-colors ${
            activeTab === 'script'
              ? 'text-blue-600 border-b-2 border-blue-600'
              : 'text-gray-500 hover:text-gray-700'
          }`}
        >
          대본
        </button>
        <button
          onClick={() => setActiveTab('storyboard')}
          className={`flex-1 py-2 text-sm font-medium transition-colors ${
            activeTab === 'storyboard'
              ? 'text-indigo-600 border-b-2 border-indigo-600'
              : 'text-gray-500 hover:text-gray-700'
          }`}
        >
          스토리보드
          {storyboardPhase === 'generating' && (
            <span className="ml-1.5 inline-flex h-2 w-2 rounded-full bg-indigo-400 animate-pulse" />
          )}
        </button>
      </div>

      {/* 데스크톱: 분할 뷰 / 태블릿·모바일: 탭 뷰 */}
      <div className="flex gap-4">
        {/* 좌측: 대본 (45%) */}
        <div
          className={`md:w-[45%] md:block ${activeTab === 'script' ? 'block w-full' : 'hidden'}`}
        >
          <div className="space-y-4 max-h-[70vh] overflow-y-auto pr-2">
            {SECTION_KEYS.map((section) => (
              <div
                key={section}
                ref={(el) => {
                  sectionRefs.current[section] = el
                }}
                data-section={section}
              >
                <ScriptSectionBlock
                  section={section}
                  content={sections[section]}
                  isActive={currentSection === section}
                />
              </div>
            ))}
          </div>
        </div>

        {/* 우측: 스토리보드 (55%) */}
        <div
          className={`md:w-[55%] md:block ${activeTab === 'storyboard' ? 'block w-full' : 'hidden'}`}
        >
          <div className="max-h-[70vh] overflow-y-auto pl-2">
            <StoryboardPanel
              panels={panels}
              phase={storyboardPhase}
              progress={storyboardProgress}
              activeSection={activeSection}
              onRegeneratePanel={onRegeneratePanel}
            />
          </div>
        </div>
      </div>
    </div>
  )
}

// 대본 섹션 블록 (내부 컴포넌트)
const SECTION_NAMES: Record<SectionType, string> = {
  hooking: '도입 (Hooking)',
  analysis: '본론 (Analysis)',
  advice_cta: '결론 (Advice & CTA)',
}

function ScriptSectionBlock({
  section,
  content,
  isActive,
}: {
  section: SectionType
  content: string
  isActive: boolean
}) {
  if (!content) return null

  return (
    <div
      className={`rounded-lg border p-4 transition-all ${
        isActive ? 'border-blue-300 bg-blue-50/50' : 'border-gray-200 bg-white'
      }`}
    >
      <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
        {SECTION_NAMES[section]}
      </h4>
      <div className="text-sm text-gray-800 whitespace-pre-wrap leading-relaxed">{content}</div>
    </div>
  )
}
