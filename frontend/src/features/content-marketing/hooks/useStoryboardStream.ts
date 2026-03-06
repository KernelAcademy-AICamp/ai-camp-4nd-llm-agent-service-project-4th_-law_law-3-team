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

// 시뮬레이션 프로그레스: ~180초에 걸쳐 0→95%까지 랜덤 증가
const SIMULATED_DURATION_MS = 240_000
const SIMULATED_MAX_PERCENT = 95
const TICK_INTERVAL_MS = 1500

// 시간대별 기본 라벨 (실제 이벤트가 없을 때 사용)
const SIMULATED_LABELS: { after: number; label: string }[] = [
  { after: 0, label: '대본 분석을 준비하고 있습니다...' },
  { after: 8_000, label: '대본 내용을 검토하고 있습니다...' },
  { after: 20_000, label: '법률 쟁점을 파악하고 있습니다...' },
  { after: 40_000, label: '장면 구성안을 작성하고 있습니다...' },
  { after: 65_000, label: '장면별 연출을 설계하고 있습니다...' },
  { after: 95_000, label: '시각 자료 생성을 준비하고 있습니다...' },
  { after: 130_000, label: '스토리보드 이미지를 생성하고 있습니다...' },
  { after: 170_000, label: '생성된 결과물을 검수하고 있습니다...' },
  { after: 210_000, label: '최종 마무리 작업 중입니다...' },
]

export function useStoryboardStream() {
  const [panels, setPanels] = useState<WebtoonPanel[]>([])
  const [phase, setPhase] = useState<StoryboardPhase>('idle')
  const [progress, setProgress] = useState<StoryboardProgress>(INITIAL_PROGRESS)
  const [error, setError] = useState<string | null>(null)
  const [jobId, setJobId] = useState<string | null>(null)

  const controllerRef = useRef<AbortController | null>(null)
  const tickTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const stopSimulatedProgress = useCallback(() => {
    if (tickTimerRef.current) {
      clearInterval(tickTimerRef.current)
      tickTimerRef.current = null
    }
  }, [])

  const realLabelRef = useRef(false)

  const startSimulatedProgress = useCallback(() => {
    stopSimulatedProgress()
    realLabelRef.current = false
    const startTime = Date.now()
    tickTimerRef.current = setInterval(() => {
      const elapsed = Date.now() - startTime
      const ratio = Math.min(elapsed / SIMULATED_DURATION_MS, 1)
      // ease-out: 처음 빠르게, 나중에 느리게
      const eased = 1 - (1 - ratio) ** 2
      const target = Math.round(eased * SIMULATED_MAX_PERCENT)
      // 랜덤 jitter: ±2%
      const jitter = Math.floor(Math.random() * 5) - 2
      const percent = Math.min(Math.max(target + jitter, 1), SIMULATED_MAX_PERCENT)

      // 실제 이벤트로 라벨이 업데이트되지 않았으면 시간대별 라벨 사용
      let fallbackLabel: string | undefined
      if (!realLabelRef.current) {
        for (let i = SIMULATED_LABELS.length - 1; i >= 0; i--) {
          if (elapsed >= SIMULATED_LABELS[i].after) {
            fallbackLabel = SIMULATED_LABELS[i].label
            break
          }
        }
      }

      setProgress((prev) => ({
        ...prev,
        percent: Math.max(prev.percent, percent),
        ...(fallbackLabel ? { label: fallbackLabel } : {}),
      }))
    }, TICK_INTERVAL_MS)
  }, [stopSimulatedProgress])

  const startGeneration = useCallback(async (request: WebtoonGenerateRequest) => {
    controllerRef.current?.abort()
    stopSimulatedProgress()
    setPanels([])
    setPhase('scene_split')
    setError(null)
    setProgress({ percent: 2, label: '대본 분석 준비 중...', currentPanel: 0, totalPanels: 0 })
    startSimulatedProgress()

    try {
      const job = await createWebtoonJob(request)
      setJobId(job.job_id)

      const controller = streamWebtoonProgress(
        job.job_id,
        (event: WebtoonStreamEvent) => {
          // 라벨만 실제 이벤트로 업데이트 (퍼센트는 시뮬레이션이 담당)
          if (event.event === 'scene_split_progress' && event.message) {
            realLabelRef.current = true
            setProgress((prev) => ({ ...prev, label: event.message! }))
          } else if (event.event === 'scene_split_done') {
            realLabelRef.current = true
            setPhase('generating')
            setProgress((prev) => ({
              ...prev,
              label: `장면 분할 완료 (${event.total_panels ?? 0}패널), 이미지 생성 시작...`,
              totalPanels: event.total_panels || 0,
            }))
          } else if (event.event === 'panel_start' && event.panel_number != null) {
            realLabelRef.current = true
            setProgress((prev) => ({
              ...prev,
              label: `이미지 생성 중... (${event.panel_number}/${prev.totalPanels})`,
              currentPanel: event.panel_number!,
            }))
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
            realLabelRef.current = true
            setPanels((prev) =>
              prev.map((p) =>
                p.panel_number === event.panel_number
                  ? { ...p, image_url: event.image_url || null, image_status: 'completed' as const }
                  : p,
              ),
            )
            setProgress((prev) => ({
              ...prev,
              label: `이미지 생성 완료 (${event.panel_number}/${event.total_panels ?? prev.totalPanels})`,
              currentPanel: event.panel_number!,
              totalPanels: event.total_panels ?? prev.totalPanels,
            }))
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
          stopSimulatedProgress()
          setError(errorMessage)
          setPhase('error')
        },
        () => {
          stopSimulatedProgress()
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
      stopSimulatedProgress()
      setError(err instanceof Error ? err.message : '스토리보드 생성 실패')
      setPhase('error')
    }
  }, [startSimulatedProgress, stopSimulatedProgress])

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
    stopSimulatedProgress()
    setPhase('idle')
  }, [stopSimulatedProgress])

  const reset = useCallback(() => {
    controllerRef.current?.abort()
    stopSimulatedProgress()
    setPanels([])
    setPhase('idle')
    setError(null)
    setJobId(null)
    setProgress(INITIAL_PROGRESS)
  }, [stopSimulatedProgress])

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
