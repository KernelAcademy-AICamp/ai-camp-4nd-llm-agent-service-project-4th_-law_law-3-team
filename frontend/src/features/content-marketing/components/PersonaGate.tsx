'use client'

import { type ReactNode, useCallback, useEffect, useState } from 'react'
import type { LawyerPersona, OnboardingState } from '../types'
import { usePersona } from '../hooks/usePersona'
import { PersonaConfirmation } from './PersonaConfirmation'
import { PersonaEditor } from './PersonaEditor'
import { PersonaOnboarding } from './PersonaOnboarding'

type GateScreen =
  | 'loading'
  | 'track_select'
  | 'track1_analyzing'
  | 'track1_confirm'
  | 'track2_onboarding'
  | 'ready'

interface PersonaGateProps {
  children: (props: {
    persona: LawyerPersona
    personaId: string
    onEditPersona: () => void
    onQuickUpdatePersona: (update: import('../types').PersonaUpdateRequest) => void
  }) => ReactNode
}

export function PersonaGate({ children }: PersonaGateProps) {
  const {
    persona,
    loadState,
    isAnalyzing,
    isSaving,
    error,
    loadCurrentPersona,
    runAnalysis,
    completeOnboarding,
    editPersona,
  } = usePersona()

  const [screen, setScreen] = useState<GateScreen>('loading')
  const [showEditor, setShowEditor] = useState(false)

  // 초기 로드
  useEffect(() => {
    loadCurrentPersona().then((result) => {
      setScreen(result ? 'ready' : 'track_select')
    })
  }, [loadCurrentPersona])

  // Track 1: 자동 분석
  const handleTrack1 = useCallback(async () => {
    setScreen('track1_analyzing')
    const result = await runAnalysis()
    if (result) {
      setScreen('track1_confirm')
    } else {
      setScreen('track_select')
    }
  }, [runAnalysis])

  // Track 1: 분석 결과 승인
  const handleApprove = useCallback(() => {
    setScreen('ready')
  }, [])

  // Track 2: 온보딩 완료
  const handleOnboardingComplete = useCallback(
    async (state: OnboardingState) => {
      if (!state.target_audience || !state.preferred_tone) return
      const result = await completeOnboarding({
        specialty_areas: state.specialty_areas,
        target_audience: state.target_audience,
        preferred_tone: state.preferred_tone,
        channel_style: state.channel_style,
        focus_topics: state.focus_topics,
      })
      if (result) {
        setScreen('ready')
      }
    },
    [completeOnboarding],
  )

  // 에디터 저장
  const handleEditorSave = useCallback(
    async (request: Parameters<typeof editPersona>[0]) => {
      const result = await editPersona(request)
      if (result) {
        setShowEditor(false)
        setScreen('ready')
      }
    },
    [editPersona],
  )

  // 로딩 화면
  if (screen === 'loading' || loadState === 'loading') {
    return (
      <div className="flex flex-col items-center justify-center py-20 text-gray-500">
        <svg
          className="w-8 h-8 animate-spin mb-3"
          xmlns="http://www.w3.org/2000/svg"
          fill="none"
          viewBox="0 0 24 24"
        >
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
        <span className="text-sm">페르소나 확인 중...</span>
      </div>
    )
  }

  // Track 선택 화면
  if (screen === 'track_select') {
    return (
      <div className="max-w-lg mx-auto py-10">
        <div className="text-center mb-8">
          <h2 className="text-xl font-bold text-gray-900">페르소나 설정</h2>
          <p className="text-sm text-gray-500 mt-2">
            콘텐츠 맞춤 생성을 위해 페르소나를 설정하세요.
          </p>
        </div>

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 text-sm rounded-lg px-4 py-3 mb-6">
            {error}
          </div>
        )}

        <div className="grid grid-cols-1 gap-4">
          {/* Track 1 */}
          <button
            onClick={handleTrack1}
            className="flex flex-col items-start p-5 bg-white border border-gray-200 rounded-xl hover:border-blue-300 hover:shadow-sm transition-all text-left"
          >
            <span className="text-sm font-bold text-gray-900 mb-1">
              AI 자동 분석
            </span>
            <span className="text-xs text-gray-500">
              기존 대화 이력을 기반으로 AI가 자동 분석합니다.
            </span>
          </button>

          {/* Track 2 */}
          <button
            onClick={() => setScreen('track2_onboarding')}
            className="flex flex-col items-start p-5 bg-white border border-gray-200 rounded-xl hover:border-blue-300 hover:shadow-sm transition-all text-left"
          >
            <span className="text-sm font-bold text-gray-900 mb-1">
              직접 설정
            </span>
            <span className="text-xs text-gray-500">
              전문분야, 톤, 타겟 독자를 직접 선택합니다.
            </span>
          </button>
        </div>
      </div>
    )
  }

  // Track 1: 분석 중
  if (screen === 'track1_analyzing') {
    return (
      <div className="flex flex-col items-center justify-center py-20 text-gray-500">
        <svg
          className="w-8 h-8 animate-spin mb-3"
          xmlns="http://www.w3.org/2000/svg"
          fill="none"
          viewBox="0 0 24 24"
        >
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
        <span className="text-sm">대화 이력을 분석하고 있습니다...</span>
        {isAnalyzing && (
          <span className="text-xs text-gray-400 mt-1">잠시만 기다려주세요</span>
        )}
      </div>
    )
  }

  // Track 1: 확인 화면
  if (screen === 'track1_confirm' && persona) {
    return (
      <div className="py-10">
        <PersonaConfirmation
          persona={persona}
          onApprove={handleApprove}
          onEdit={() => setShowEditor(true)}
          onSwitchTrack={() => setScreen('track2_onboarding')}
        />
        {showEditor && (
          <PersonaEditor
            persona={persona}
            onSave={handleEditorSave}
            onClose={() => setShowEditor(false)}
            isSaving={isSaving}
          />
        )}
      </div>
    )
  }

  // Track 2: 온보딩
  if (screen === 'track2_onboarding') {
    return (
      <div className="py-10">
        <PersonaOnboarding
          onComplete={handleOnboardingComplete}
          onBack={() => setScreen('track_select')}
          isLoading={isSaving}
        />
      </div>
    )
  }

  // Ready: 페르소나 확정, 하위 트리 렌더
  if (screen === 'ready' && persona) {
    return (
      <>
        {children({
          persona,
          personaId: persona.id,
          onEditPersona: () => setShowEditor(true),
          onQuickUpdatePersona: editPersona,
        })}
        {showEditor && (
          <PersonaEditor
            persona={persona}
            onSave={handleEditorSave}
            onClose={() => setShowEditor(false)}
            isSaving={isSaving}
          />
        )}
      </>
    )
  }

  // Fallback
  return null
}
