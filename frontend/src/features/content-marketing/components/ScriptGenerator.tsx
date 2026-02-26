'use client'

import { useCallback, useState } from 'react'
import type { PersonaFeedbackRequest, PersonaTone, PersonaType, ScriptDuration, TrendIssue } from '../types'
import { useScript } from '../hooks/useScript'
import { ExportButton } from './ExportButton'
import { FeedbackPanel } from './FeedbackPanel'
import { MetadataPanel } from './MetadataPanel'
import { PersonaSelector } from './PersonaSelector'
import { RAGPreview } from './RAGPreview'
import { ScriptPreview } from './ScriptPreview'
import { StageProgress } from './StageProgress'

interface ScriptGeneratorProps {
  initialTopic?: string
  initialTrend?: TrendIssue | null
  personaId?: string | null
  onFeedback?: (request: PersonaFeedbackRequest) => Promise<void>
}

export function ScriptGenerator({
  initialTopic,
  initialTrend,
  personaId,
  onFeedback,
}: ScriptGeneratorProps) {
  const [topic, setTopic] = useState(initialTopic || '')
  const [persona, setPersona] = useState<PersonaType>('professional')
  const [tone, setTone] = useState<PersonaTone>('professional')
  const [duration, setDuration] = useState<ScriptDuration>(10)

  const {
    sections,
    metadata,
    isGenerating,
    currentSection,
    stageInfo,
    progress,
    error,
    generate,
    stopGeneration,
    refreshMetadata,
    reset,
  } = useScript()

  const isDone = !isGenerating && Object.values(sections).some((s) => s.length > 0)

  const handleGenerate = useCallback(() => {
    if (!topic.trim() || topic.trim().length < 5) return

    generate({
      topic: topic.trim(),
      trend_id: initialTrend?.id ?? null,
      persona,
      persona_id: personaId ?? undefined,
      duration,
      related_laws: initialTrend?.related_laws.map((l) => l.law_id) ?? [],
      related_cases: initialTrend?.related_cases.map((c) => c.case_id) ?? [],
    })
  }, [topic, persona, personaId, duration, initialTrend, generate])

  const handleReset = useCallback(() => {
    reset()
    setTopic(initialTopic || '')
  }, [reset, initialTopic])

  return (
    <div className="space-y-5">
      {/* 주제 입력 */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1.5">주제</label>
        <textarea
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
          placeholder="대본 주제를 입력하세요 (최소 5자)"
          disabled={isGenerating}
          rows={2}
          className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg resize-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-100"
        />
        {initialTrend && (
          <p className="mt-1 text-xs text-gray-400">
            트렌드에서 선택됨: {initialTrend.title}
          </p>
        )}
      </div>

      {/* RAG 프리뷰 (트렌드의 관련 법령/판례) */}
      {initialTrend && (
        <RAGPreview
          relatedLaws={initialTrend.related_laws}
          relatedCases={initialTrend.related_cases}
        />
      )}

      {/* 설정 */}
      <PersonaSelector
        persona={persona}
        duration={duration}
        tone={tone}
        onPersonaChange={setPersona}
        onDurationChange={setDuration}
        onToneChange={setTone}
        disabled={isGenerating}
      />

      {/* 생성/중단 버튼 */}
      <div className="flex items-center gap-3">
        {isGenerating ? (
          <button
            onClick={stopGeneration}
            className="px-6 py-2.5 text-sm font-medium text-white bg-red-500 rounded-lg hover:bg-red-600 transition-colors"
          >
            생성 중단
          </button>
        ) : (
          <button
            onClick={handleGenerate}
            disabled={!topic.trim() || topic.trim().length < 5}
            className="px-6 py-2.5 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            대본 생성하기
          </button>
        )}

        {isDone && (
          <button
            onClick={handleReset}
            className="px-4 py-2.5 text-sm font-medium text-gray-600 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
          >
            초기화
          </button>
        )}
      </div>

      {/* 진행도 바 */}
      {(isGenerating || progress.percent === 100) && (
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-600 font-medium">{progress.label}</span>
            <span className="text-gray-500 tabular-nums">{progress.percent}%</span>
          </div>
          <div className="w-full h-2.5 bg-gray-200 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-500 ease-out ${
                progress.percent === 100
                  ? 'bg-green-500'
                  : 'bg-blue-500'
              }`}
              style={{ width: `${progress.percent}%` }}
            />
          </div>
          {isGenerating && progress.percent < 100 && (
            <div className="flex items-center gap-2 text-xs text-gray-400">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-blue-500" />
              </span>
              AI가 대본을 작성하고 있습니다
            </div>
          )}
        </div>
      )}

      {/* 스테이지 진행 (독립 컴포넌트) */}
      <StageProgress stageInfo={stageInfo} isGenerating={isGenerating} />

      {/* 에러 */}
      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 text-sm rounded-lg px-4 py-3">
          {error}
        </div>
      )}

      {/* 대본 미리보기 */}
      <ScriptPreview
        sections={sections}
        currentSection={currentSection}
        isGenerating={isGenerating}
        stageInfo={stageInfo}
      />

      {/* 메타데이터 */}
      {metadata && (
        <MetadataPanel
          metadata={metadata}
          onRefresh={() => refreshMetadata(topic, persona)}
        />
      )}

      {/* 피드백 패널 (생성 완료 + personaId 존재 시) */}
      {isDone && personaId && onFeedback && (
        <FeedbackPanel
          personaId={personaId}
          scriptId={null}
          onSubmit={onFeedback}
        />
      )}

      {/* 내보내기 */}
      <ExportButton sections={sections} metadata={metadata} topic={topic} />
    </div>
  )
}
