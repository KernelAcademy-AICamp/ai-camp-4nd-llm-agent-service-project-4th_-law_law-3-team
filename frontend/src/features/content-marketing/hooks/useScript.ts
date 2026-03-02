'use client'

import { useCallback, useRef, useState } from 'react'
import type {
  NewsArticleForScript,
  PersonaType,
  ScriptDuration,
  ScriptMetadata,
  ScriptStreamEvent,
  SectionType,
} from '../types'
import { regenerateMetadata, streamScript } from '../services'

export interface StageInfo {
  stage: string
  status: string
  detail: string
}

export interface GenerationProgress {
  percent: number
  label: string
}

const SECTION_PROGRESS_RANGES: Record<string, { start: number; end: number; label: string }> = {
  hooking: { start: 20, end: 35, label: '도입부 생성 중...' },
  analysis: { start: 35, end: 80, label: '본론 생성 중...' },
  advice_cta: { start: 80, end: 95, label: '결론 생성 중...' },
}

const INITIAL_SECTIONS: Record<SectionType, string> = {
  hooking: '',
  analysis: '',
  advice_cta: '',
}

export function useScript() {
  const [sections, setSections] = useState<Record<SectionType, string>>(INITIAL_SECTIONS)
  const [metadata, setMetadata] = useState<ScriptMetadata | null>(null)
  const [isGenerating, setIsGenerating] = useState(false)
  const [currentSection, setCurrentSection] = useState<SectionType | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [stageInfo, setStageInfo] = useState<StageInfo | null>(null)
  const [progress, setProgress] = useState<GenerationProgress>({ percent: 0, label: '' })
  const controllerRef = useRef<AbortController | null>(null)
  const chunkCountRef = useRef<number>(0)

  const generate = useCallback(
    (params: {
      topic: string
      trend_id: string | null
      persona: PersonaType
      persona_id?: string
      duration: ScriptDuration
      related_laws: string[]
      related_cases: string[]
      news_articles?: NewsArticleForScript[] | null
    }) => {
      // 이전 스트리밍 중단
      controllerRef.current?.abort()

      setSections(INITIAL_SECTIONS)
      setMetadata(null)
      setIsGenerating(true)
      setError(null)
      setStageInfo(null)
      setProgress({ percent: 2, label: '대본 생성 준비 중...' })
      chunkCountRef.current = 0

      const controller = streamScript(
        params,
        (event: ScriptStreamEvent) => {
          if (event.event === 'stage_update' && event.stage && event.status) {
            setStageInfo({
              stage: event.stage,
              status: event.status,
              detail: event.detail ?? '',
            })
            if (event.status === 'started') {
              setProgress({ percent: 5, label: '법령/판례 검색 중...' })
            } else if (event.status === 'completed') {
              setProgress({ percent: 15, label: '검색 완료, 대본 생성 시작...' })
            }
          } else if (event.event === 'section_start' && event.section) {
            setStageInfo(null)
            setCurrentSection(event.section)
            chunkCountRef.current = 0
            const range = SECTION_PROGRESS_RANGES[event.section]
            if (range) {
              setProgress({ percent: range.start, label: range.label })
            }
          } else if (event.event === 'content' && event.section) {
            setSections((prev) => ({
              ...prev,
              [event.section!]: prev[event.section!] + event.content,
            }))
            // 청크 수 기반 섹션 내 진행도 보간
            chunkCountRef.current += 1
            const range = SECTION_PROGRESS_RANGES[event.section]
            if (range) {
              const chunkProgress = Math.min(chunkCountRef.current / 60, 0.95)
              const sectionPercent = range.start + (range.end - range.start) * chunkProgress
              setProgress({ percent: Math.round(sectionPercent), label: range.label })
            }
          } else if (event.event === 'metadata' && event.metadata) {
            setMetadata(event.metadata)
            setProgress({ percent: 98, label: '메타데이터 생성 완료' })
          }
        },
        (errorMessage: string) => {
          setError(errorMessage)
          setIsGenerating(false)
          setCurrentSection(null)
          setProgress({ percent: 0, label: '' })
        },
        () => {
          setIsGenerating(false)
          setCurrentSection(null)
          setProgress({ percent: 100, label: '대본 생성 완료!' })
        },
      )

      controllerRef.current = controller
    },
    [],
  )

  const stopGeneration = useCallback(() => {
    controllerRef.current?.abort()
    setIsGenerating(false)
    setCurrentSection(null)
  }, [])

  const updateSection = useCallback((section: SectionType, content: string) => {
    setSections((prev) => ({ ...prev, [section]: content }))
  }, [])

  const refreshMetadata = useCallback(
    async (topic: string, persona: PersonaType) => {
      const fullContent = Object.values(sections).join('\n\n')
      if (fullContent.length < 100) return
      try {
        const newMetadata = await regenerateMetadata({
          script_content: fullContent,
          topic,
          persona,
        })
        setMetadata(newMetadata)
      } catch {
        // 실패 시 기존 메타데이터 유지
      }
    },
    [sections],
  )

  const reset = useCallback(() => {
    controllerRef.current?.abort()
    setSections(INITIAL_SECTIONS)
    setMetadata(null)
    setIsGenerating(false)
    setCurrentSection(null)
    setError(null)
    setStageInfo(null)
    setProgress({ percent: 0, label: '' })
    chunkCountRef.current = 0
  }, [])

  return {
    sections,
    metadata,
    isGenerating,
    currentSection,
    stageInfo,
    progress,
    error,
    generate,
    stopGeneration,
    updateSection,
    refreshMetadata,
    reset,
  }
}
