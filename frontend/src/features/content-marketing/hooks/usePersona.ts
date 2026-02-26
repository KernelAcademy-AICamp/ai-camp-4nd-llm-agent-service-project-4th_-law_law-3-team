'use client'

import { useCallback, useState } from 'react'
import type {
  LawyerPersona,
  PersonaAnalysisRequest,
  PersonaFeedbackRequest,
  PersonaOnboardingRequest,
  PersonaUpdateRequest,
} from '../types'
import {
  analyzePersona,
  createPersonaFromOnboarding,
  getCurrentPersona,
  submitPersonaFeedback,
  updatePersona,
} from '../services'

type LoadState = 'idle' | 'loading' | 'loaded' | 'error'

export function usePersona() {
  const [persona, setPersona] = useState<LawyerPersona | null>(null)
  const [loadState, setLoadState] = useState<LoadState>('idle')
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [isSaving, setIsSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const loadCurrentPersona = useCallback(async () => {
    setLoadState('loading')
    setError(null)
    try {
      const result = await getCurrentPersona()
      setPersona(result)
      setLoadState('loaded')
      return result
    } catch (err) {
      const message = err instanceof Error ? err.message : '페르소나 조회 실패'
      setError(message)
      setLoadState('error')
      return null
    }
  }, [])

  const runAnalysis = useCallback(async (request?: PersonaAnalysisRequest) => {
    setIsAnalyzing(true)
    setError(null)
    try {
      const result = await analyzePersona(request ?? {})
      setPersona(result)
      setLoadState('loaded')
      return result
    } catch (err) {
      const message = err instanceof Error ? err.message : '페르소나 분석 실패'
      setError(message)
      return null
    } finally {
      setIsAnalyzing(false)
    }
  }, [])

  const completeOnboarding = useCallback(async (request: PersonaOnboardingRequest) => {
    setIsSaving(true)
    setError(null)
    try {
      const result = await createPersonaFromOnboarding(request)
      setPersona(result)
      setLoadState('loaded')
      return result
    } catch (err) {
      const message = err instanceof Error ? err.message : '온보딩 저장 실패'
      setError(message)
      return null
    } finally {
      setIsSaving(false)
    }
  }, [])

  const editPersona = useCallback(async (request: PersonaUpdateRequest) => {
    setIsSaving(true)
    setError(null)
    try {
      const result = await updatePersona(request)
      setPersona(result)
      return result
    } catch (err) {
      const message = err instanceof Error ? err.message : '페르소나 수정 실패'
      setError(message)
      return null
    } finally {
      setIsSaving(false)
    }
  }, [])

  const sendFeedback = useCallback(async (request: PersonaFeedbackRequest) => {
    try {
      await submitPersonaFeedback(request)
    } catch {
      // 피드백 실패는 무시 (UX 차단하지 않음)
    }
  }, [])

  return {
    persona,
    loadState,
    isAnalyzing,
    isSaving,
    error,
    loadCurrentPersona,
    runAnalysis,
    completeOnboarding,
    editPersona,
    sendFeedback,
  }
}
