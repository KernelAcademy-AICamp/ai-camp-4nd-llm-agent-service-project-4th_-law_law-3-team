'use client'

import { useEffect, useState } from 'react'

interface AnalysisStage {
  label: string
  doneLabel: string
  startAt: number
}

const STAGES: AnalysisStage[] = [
  { label: '대화 이력 수집 중...', doneLabel: '대화 이력 수집 완료', startAt: 0 },
  { label: '전문분야 패턴 분석 중...', doneLabel: '전문분야 패턴 분석 완료', startAt: 3000 },
  { label: '페르소나 구성 중...', doneLabel: '페르소나 구성 완료', startAt: 8000 },
]

const TOTAL_DURATION = 12000

interface PersonaAnalysisProgressProps {
  onCancel: () => void
}

export function PersonaAnalysisProgress({ onCancel }: PersonaAnalysisProgressProps) {
  const [currentStage, setCurrentStage] = useState(0)
  const [progress, setProgress] = useState(0)

  useEffect(() => {
    const timers: ReturnType<typeof setTimeout>[] = []

    // 단계 전환 타이머
    STAGES.forEach((stage, index) => {
      if (index === 0) return
      timers.push(
        setTimeout(() => {
          setCurrentStage(index)
        }, stage.startAt),
      )
    })

    // 프로그레스 바 애니메이션 (100ms 간격)
    const interval = setInterval(() => {
      setProgress((prev) => {
        const next = prev + (100 / (TOTAL_DURATION / 100))
        return next >= 95 ? 95 : next
      })
    }, 100)

    return () => {
      timers.forEach(clearTimeout)
      clearInterval(interval)
    }
  }, [])

  return (
    <div className="max-w-md mx-auto py-16 px-4 text-center">
      {/* 헤드라인 */}
      <h3 className="text-lg font-bold text-slate-800 mb-8">
        대화 이력을 분석하고 있습니다
      </h3>

      {/* 단계 표시 */}
      <div className="space-y-3 mb-8 text-left max-w-xs mx-auto">
        {STAGES.map((stage, index) => {
          const isDone = index < currentStage
          const isActive = index === currentStage
          const isPending = index > currentStage

          return (
            <div
              key={stage.label}
              className={`flex items-center gap-3 text-sm transition-colors duration-300 ${
                isDone
                  ? 'text-green-600'
                  : isActive
                    ? 'text-blue-600 font-medium'
                    : 'text-gray-400'
              }`}
            >
              {/* 상태 아이콘 */}
              <span className="flex-shrink-0 w-5 h-5 flex items-center justify-center">
                {isDone && (
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                  </svg>
                )}
                {isActive && (
                  <span className="w-3 h-3 rounded-full bg-blue-600 animate-pulse" />
                )}
                {isPending && (
                  <span className="w-3 h-3 rounded-full border-2 border-gray-300" />
                )}
              </span>

              {/* 라벨 */}
              <span>{isDone ? stage.doneLabel : stage.label}</span>
            </div>
          )
        })}
      </div>

      {/* 프로그레스 바 */}
      <div className="w-full max-w-xs mx-auto mb-4">
        <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
          <div
            className="h-full bg-blue-600 rounded-full transition-all duration-200 ease-linear"
            style={{ width: `${progress}%` }}
          />
        </div>
        <p className="text-xs text-gray-400 mt-2 text-right">{Math.round(progress)}%</p>
      </div>

      {/* 예상 시간 */}
      <p className="text-xs text-gray-400 mb-6">
        예상 소요 시간: 약 10~15초
      </p>

      {/* 취소 버튼 */}
      <button
        onClick={onCancel}
        className="text-sm text-gray-500 hover:text-gray-700 transition-colors px-4 py-2 rounded-lg hover:bg-gray-100"
      >
        취소
      </button>
    </div>
  )
}
