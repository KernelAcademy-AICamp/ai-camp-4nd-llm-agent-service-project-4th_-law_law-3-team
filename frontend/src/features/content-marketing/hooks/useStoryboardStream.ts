'use client'

import { useCallback, useRef, useState } from 'react'
import type {
  WebtoonGenerateRequest,
  WebtoonPanel,
  WebtoonStreamEvent,
} from '../types'
import {
  createWebtoonJob,
  getWebtoonJobStatus,
  regenerateWebtoonPanel,
  streamWebtoonProgress,
} from '../services'

export type StoryboardPhase =
  | 'idle'
  | 'scene_split'
  | 'generating'
  | 'completed'
  | 'error'

export interface StoryboardProgress {
  percent: number
  label: string
  currentPanel: number
  totalPanels: number
}

const INITIAL_PROGRESS: StoryboardProgress = {
  percent: 0,
  label: '',
  currentPanel: 0,
  totalPanels: 0,
}

export function useStoryboardStream() {
  const [panels, setPanels] = useState<WebtoonPanel[]>([])
  const [phase, setPhase] = useState<StoryboardPhase>('idle')
  const [progress, setProgress] = useState<StoryboardProgress>(INITIAL_PROGRESS)
  const [error, setError] = useState<string | null>(null)
  const [jobId, setJobId] = useState<string | null>(null)

  const controllerRef = useRef<AbortController | null>(null)

  const startGeneration = useCallback(async (request: WebtoonGenerateRequest) => {
    controllerRef.current?.abort()
    setPanels([])
    setPhase('scene_split')
    setError(null)
    setProgress({ percent: 5, label: '장면 분할 중...', currentPanel: 0, totalPanels: 0 })

    try {
      const job = await createWebtoonJob(request)
      setJobId(job.job_id)

      const controller = streamWebtoonProgress(
        job.job_id,
        (event: WebtoonStreamEvent) => {
          if (event.event === 'scene_split_done') {
            setPhase('generating')
            setProgress((prev) => ({
              ...prev,
              percent: 15,
              label: '이미지 생성 시작...',
              totalPanels: event.total_panels || 0,
            }))
          } else if (event.event === 'panel_start' && event.panel_number != null) {
            setPanels((prev) => {
              const exists = prev.some((p) => p.panel_number === event.panel_number)
              if (exists) {
                return prev.map((p) =>
                  p.panel_number === event.panel_number
                    ? { ...p, image_status: 'generating' as const }
                    : p,
                )
              }
              return [
                ...prev,
                {
                  panel_number: event.panel_number!,
                  section: (event.section || 'analysis') as WebtoonPanel['section'],
                  scene_type: 'legal_explanation',
                  script_excerpt: event.caption || '',
                  scene_description: event.scene_description || '',
                  location: '',
                  time_of_day: '',
                  characters: [],
                  emotion: '',
                  visual_focus: '',
                  camera_angle: '',
                  legal_keyword: '',
                  image_prompt: null,
                  image_url: null,
                  image_status: 'generating',
                  error_message: null,
                  model_version: null,
                  prompt_version: null,
                  generation_cost_ms: null,
                  safety_flags: [],
                },
              ]
            })
          } else if (event.event === 'panel_complete' && event.panel_number != null) {
            setPanels((prev) =>
              prev.map((p) =>
                p.panel_number === event.panel_number
                  ? { ...p, image_url: event.image_url || null, image_status: 'completed' as const }
                  : p,
              ),
            )
            const total = event.total_panels || 1
            const pct = 15 + (85 * (event.panel_number! / total))
            setProgress({
              percent: Math.round(pct),
              label: `이미지 생성 중... (${event.panel_number}/${total})`,
              currentPanel: event.panel_number!,
              totalPanels: total,
            })
          } else if (event.event === 'panel_failed' && event.panel_number != null) {
            setPanels((prev) =>
              prev.map((p) =>
                p.panel_number === event.panel_number
                  ? {
                      ...p,
                      image_status: 'error' as const,
                      error_message: event.error || '생성 실패',
                      image_url: event.image_url || null,
                    }
                  : p,
              ),
            )
          }
        },
        (errorMessage: string) => {
          setError(errorMessage)
          setPhase('error')
        },
        () => {
          setPhase('completed')
          setProgress((prev) => ({ ...prev, percent: 100, label: '스토리보드 완료!' }))
          // 최종 상태 폴링으로 전체 패널 데이터 동기화
          if (job.job_id) {
            getWebtoonJobStatus(job.job_id)
              .then((status) => {
                if (status.panels.length > 0) {
                  setPanels(status.panels)
                }
              })
              .catch(() => {
                /* 폴링 실패는 무시 — SSE 데이터 유지 */
              })
          }
        },
      )
      controllerRef.current = controller
    } catch (err) {
      setError(err instanceof Error ? err.message : '스토리보드 생성 실패')
      setPhase('error')
    }
  }, [])

  const regeneratePanel = useCallback(
    async (panelNumber: number) => {
      if (!jobId) return
      setPanels((prev) =>
        prev.map((p) =>
          p.panel_number === panelNumber
            ? { ...p, image_status: 'generating' as const, error_message: null }
            : p,
        ),
      )
      try {
        const result = await regenerateWebtoonPanel(jobId, panelNumber)
        setPanels((prev) =>
          prev.map((p) =>
            p.panel_number === panelNumber
              ? { ...p, image_url: result.image_url, image_status: 'completed' as const }
              : p,
          ),
        )
      } catch {
        setPanels((prev) =>
          prev.map((p) =>
            p.panel_number === panelNumber
              ? { ...p, image_status: 'error' as const, error_message: '재생성 실패' }
              : p,
          ),
        )
      }
    },
    [jobId],
  )

  const stopGeneration = useCallback(() => {
    controllerRef.current?.abort()
    setPhase('idle')
  }, [])

  const reset = useCallback(() => {
    controllerRef.current?.abort()
    setPanels([])
    setPhase('idle')
    setError(null)
    setJobId(null)
    setProgress(INITIAL_PROGRESS)
  }, [])

  return {
    panels,
    phase,
    progress,
    error,
    jobId,
    startGeneration,
    regeneratePanel,
    stopGeneration,
    reset,
  }
}
