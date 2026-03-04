'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import type { LawyerPersona, NewsArticleForScript, PersonaFeedbackRequest, PersonaTone, PersonaType, ScriptDuration, TrendIssue } from '../types'
import { useScript } from '../hooks/useScript'
import { useStoryboardStream } from '../hooks/useStoryboardStream'
import { ExportButton } from './ExportButton'
import { FeedbackPanel } from './FeedbackPanel'
import { MetadataPanel } from './MetadataPanel'
import { PersonaSelector } from './PersonaSelector'
import { RAGPreview } from './RAGPreview'
import { ScriptStoryboardSplitView } from './ScriptStoryboardSplitView'
import { StageProgress } from './StageProgress'

const MIN_TOPIC_LENGTH = 2

interface ScriptGeneratorProps {
  initialTopic?: string
  initialTrend?: TrendIssue | null
  initialNewsArticles?: NewsArticleForScript[] | null
  personaId?: string | null
  savedPersona?: LawyerPersona | null
  onFeedback?: (request: PersonaFeedbackRequest) => Promise<void>
}

export function ScriptGenerator({
  initialTopic,
  initialTrend,
  initialNewsArticles,
  personaId,
  savedPersona,
  onFeedback,
}: ScriptGeneratorProps) {
  const initialTone: PersonaTone = savedPersona?.preferred_tone ?? 'professional'
  const initialPersonaType: PersonaType =
    (initialTone === 'professional' || initialTone === 'educational') ? 'professional' : 'casual'

  const [topic, setTopic] = useState(initialTopic || '')
  const [persona, setPersona] = useState<PersonaType>(initialPersonaType)
  const [tone, setTone] = useState<PersonaTone>(initialTone)
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

  const {
    panels,
    phase: storyboardPhase,
    progress: storyboardProgress,
    error: storyboardError,
    startGeneration: startStoryboard,
    regeneratePanel,
    reset: resetStoryboard,
  } = useStoryboardStream()

  const isDone = !isGenerating && Object.values(sections).some((s) => s.length > 0)
  const prevIsDoneRef = useRef(false)

  // 대본 생성 완료 상태 추적 (자동 스토리보드 트리거 제거 — 수동 버튼으로 시작)
  useEffect(() => {
    prevIsDoneRef.current = isDone
  }, [isDone])

  const handleGenerate = useCallback(() => {
    if (!topic.trim() || topic.trim().length < MIN_TOPIC_LENGTH) return

    generate({
      topic: topic.trim(),
      trend_id: initialTrend?.id ?? null,
      persona,
      persona_id: personaId ?? undefined,
      duration,
      related_laws: initialTrend?.related_laws.map((l) => l.law_id) ?? [],
      related_cases: initialTrend?.related_cases.map((c) => c.case_id) ?? [],
      news_articles: initialNewsArticles ?? null,
    })
  }, [topic, persona, personaId, duration, initialTrend, initialNewsArticles, generate])

  const handleStartStoryboard = useCallback(() => {
    if (!isDone) return
    startStoryboard({
      topic: topic.trim(),
      sections: { ...sections },
      persona,
      persona_id: personaId ?? undefined,
    })
  }, [isDone, topic, sections, persona, personaId, startStoryboard])

  const handleReset = useCallback(() => {
    reset()
    resetStoryboard()
    setTopic(initialTopic || '')
    prevIsDoneRef.current = false
  }, [reset, resetStoryboard, initialTopic])

  return (
    <div className="space-y-5">
      {/* 주제 입력 */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1.5">주제</label>
        <textarea
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
          placeholder="대본 주제를 입력하세요 (최소 2자)"
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

      {/* 선택된 뉴스 프리뷰 */}
      {initialNewsArticles && initialNewsArticles.length > 0 && (
        <div className="bg-blue-50 border border-blue-100 rounded-lg px-4 py-3">
          <p className="text-xs font-medium text-blue-700 mb-2">
            선택된 뉴스 기사 ({initialNewsArticles.length}건)
          </p>
          <div className="space-y-1">
            {initialNewsArticles.slice(0, 3).map((article, index) => (
              <p key={`script-${index}`} className="text-xs text-blue-600 truncate">
                {article.title}
              </p>
            ))}
            {initialNewsArticles.length > 3 && (
              <p className="text-xs text-blue-400">
                외 {initialNewsArticles.length - 3}개
              </p>
            )}
          </div>
        </div>
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
            disabled={!topic.trim() || topic.trim().length < MIN_TOPIC_LENGTH}
            className="px-6 py-2.5 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            대본 생성하기
          </button>
        )}

        {/* 스토리보드 수동 생성 버튼 */}
        {isDone && storyboardPhase === 'idle' && (
          <button
            onClick={handleStartStoryboard}
            className="px-5 py-2.5 text-sm font-medium text-white bg-indigo-600 rounded-lg hover:bg-indigo-700 transition-colors"
          >
            스토리보드 생성
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
      {storyboardError && (
        <div className="bg-red-50 border border-red-200 text-red-700 text-sm rounded-lg px-4 py-3">
          스토리보드 오류: {storyboardError}
        </div>
      )}

      {/* 대본 + 스토리보드 분할 뷰 */}
      <ScriptStoryboardSplitView
        sections={sections}
        currentSection={currentSection}
        isGenerating={isGenerating}
        stageInfo={stageInfo}
        panels={panels}
        storyboardPhase={storyboardPhase}
        storyboardProgress={storyboardProgress}
        onRegeneratePanel={regeneratePanel}
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
