'use client'

import { type ReactNode, useState } from 'react'
import type { LawyerPersona, PersonaUpdateRequest } from '../types'
import { usePersonaGate } from '../hooks/usePersonaGate'
import { PersonaAnalysisProgress } from './PersonaAnalysisProgress'
import { PersonaEditor } from './PersonaEditor'
import { PersonaHybridReview } from './PersonaHybridReview'
import { PersonaOnboarding } from './PersonaOnboarding'
import { PersonaWelcome } from './PersonaWelcome'

interface PersonaGateProps {
  children: (props: {
    persona: LawyerPersona
    personaId: string
    onEditPersona: () => void
    onQuickUpdatePersona: (update: PersonaUpdateRequest) => void
  }) => ReactNode
}

export function PersonaGate({ children }: PersonaGateProps) {
  const {
    state,
    handleStartAnalysis,
    handleCancelAnalysis,
    handleSwitchToManual,
    handleOnboardingBack,
    handleHybridConfirm,
    handleOnboardingComplete,
    handleSkip,
    handleEditorSave,
    handleEditPersona,
  } = usePersonaGate()

  const [showEditor, setShowEditor] = useState(false)

  switch (state.screen) {
    case 'loading':
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

    case 'welcome':
      return (
        <PersonaWelcome
          chatHistoryCount={state.chatHistoryCount}
          error={state.error}
          onStartAnalysis={handleStartAnalysis}
          onStartOnboarding={() => handleSwitchToManual()}
          onSkip={handleSkip}
        />
      )

    case 'track1_analyzing':
      return <PersonaAnalysisProgress onCancel={handleCancelAnalysis} />

    case 'track1_review':
      return state.persona && state.analysisInsights ? (
        <PersonaHybridReview
          persona={state.persona}
          insights={state.analysisInsights}
          onConfirm={handleHybridConfirm}
          onSwitchToManual={handleSwitchToManual}
          isSaving={state.isSaving}
        />
      ) : null

    case 'track2_onboarding':
      return (
        <div className="py-10">
          <PersonaOnboarding
            onComplete={handleOnboardingComplete}
            onBack={handleOnboardingBack}
            isLoading={state.isSaving}
            error={state.error}
            initialState={state.draftOnboarding}
          />
        </div>
      )

    case 'ready':
      if (!state.persona) return null
      return (
        <>
          {children({
            persona: state.persona,
            personaId: state.persona.id,
            onEditPersona: () => setShowEditor(true),
            onQuickUpdatePersona: handleEditorSave,
          })}
          {showEditor && (
            <PersonaEditor
              persona={state.persona}
              onSave={async (request) => {
                await handleEditorSave(request)
                setShowEditor(false)
              }}
              onClose={() => setShowEditor(false)}
              isSaving={state.isSaving}
            />
          )}
        </>
      )

    default:
      return null
  }
}
