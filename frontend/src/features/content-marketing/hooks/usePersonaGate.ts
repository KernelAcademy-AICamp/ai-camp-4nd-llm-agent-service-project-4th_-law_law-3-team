'use client'

import { useCallback, useEffect, useReducer, useRef } from 'react'
import type {
  AnalysisInsights,
  GateAction,
  GateScreen,
  LawyerPersona,
  OnboardingState,
  PersonaGateState,
  PersonaOnboardingRequest,
  PersonaUpdateRequest,
} from '../types'
import {
  analyzePersona,
  createPersonaFromOnboarding,
  fetchChatHistoryCount,
  getCurrentPersona,
  updatePersona,
} from '../services'

// ── SessionStorage 키 ──
const DRAFT_STORAGE_KEY = 'persona_gate_draft'

// ── 상태 전이 매트릭스 (v1.1 — Red Team + Consultant 피드백) ──
const VALID_TRANSITIONS: Record<GateScreen, GateAction['type'][]> = {
  loading: ['LOAD_START', 'LOAD_SUCCESS', 'LOAD_ERROR'],
  welcome: ['START_ANALYSIS', 'SET_SCREEN', 'SKIP_SETUP', 'CLEAR_ERROR'],
  track1_analyzing: ['ANALYSIS_SUCCESS', 'ANALYSIS_FAIL', 'SET_SCREEN'],
  track1_review: ['SAVE_START', 'SAVE_SUCCESS', 'SAVE_ERROR', 'SET_SCREEN', 'CLEAR_ERROR'],
  track2_onboarding: ['SAVE_START', 'SAVE_SUCCESS', 'SAVE_ERROR', 'SET_SCREEN', 'SET_DRAFT', 'CLEAR_ERROR'],
  ready: ['SET_SCREEN', 'SAVE_START', 'SAVE_SUCCESS', 'SAVE_ERROR'],
}

// ── 초기 상태 ──
const initialState: PersonaGateState = {
  screen: 'loading',
  persona: null,
  analysisInsights: null,
  draftOnboarding: null,
  chatHistoryCount: null,
  error: null,
  isAnalyzing: false,
  isSaving: false,
}

// ── SessionStorage 유틸 (v1.1 — 유효성 검사 포함) ──
function loadDraftFromStorage(): Partial<OnboardingState> | null {
  try {
    const raw = sessionStorage.getItem(DRAFT_STORAGE_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw) as Partial<OnboardingState>
    if (typeof parsed !== 'object' || parsed === null) return null
    if (parsed.specialty_areas && !Array.isArray(parsed.specialty_areas)) return null
    return parsed
  } catch {
    sessionStorage.removeItem(DRAFT_STORAGE_KEY)
    return null
  }
}

function saveDraftToStorage(draft: Partial<OnboardingState> | null): void {
  if (draft) {
    sessionStorage.setItem(DRAFT_STORAGE_KEY, JSON.stringify(draft))
  } else {
    sessionStorage.removeItem(DRAFT_STORAGE_KEY)
  }
}

// ── Reducer ──
function gateReducer(state: PersonaGateState, action: GateAction): PersonaGateState {
  const allowed = VALID_TRANSITIONS[state.screen]
  if (!allowed?.includes(action.type)) {
    if (process.env.NODE_ENV === 'development') {
      console.warn(`[PersonaGate] Invalid transition: ${state.screen} + ${action.type}`)
    }
    return state
  }

  switch (action.type) {
    case 'LOAD_START':
      return { ...state, screen: 'loading', error: null }

    case 'LOAD_SUCCESS':
      return {
        ...state,
        screen: 'welcome',
        persona: action.persona,
        chatHistoryCount: action.chatHistoryCount,
        draftOnboarding: loadDraftFromStorage(),
      }

    case 'LOAD_ERROR':
      return { ...state, screen: 'welcome', error: action.error }

    case 'START_ANALYSIS':
      return { ...state, screen: 'track1_analyzing', isAnalyzing: true, error: null }

    case 'ANALYSIS_SUCCESS':
      return {
        ...state,
        screen: 'track1_review',
        persona: action.persona,
        analysisInsights: action.insights,
        isAnalyzing: false,
      }

    case 'ANALYSIS_FAIL':
      return {
        ...state,
        screen: action.redirectScreen ?? 'welcome',
        isAnalyzing: false,
        error: action.error,
      }

    case 'SAVE_START':
      return { ...state, isSaving: true, error: null }

    case 'SAVE_SUCCESS':
      saveDraftToStorage(null)
      return {
        ...state,
        screen: 'ready',
        persona: action.persona,
        isSaving: false,
        draftOnboarding: null,
        analysisInsights: null,
      }

    case 'SAVE_ERROR':
      return { ...state, isSaving: false, error: action.error }

    case 'SET_SCREEN':
      return { ...state, screen: action.screen, error: null }

    case 'SET_DRAFT':
      saveDraftToStorage(action.draft)
      return { ...state, draftOnboarding: action.draft }

    case 'CLEAR_ERROR':
      return { ...state, error: null }

    case 'SKIP_SETUP':
      saveDraftToStorage(null)
      return { ...state, screen: 'ready' }

    default:
      return state
  }
}

// ── Hook ──
export function usePersonaGate() {
  const [state, dispatch] = useReducer(gateReducer, initialState)
  const analysisAbortRef = useRef<AbortController | null>(null)

  // 초기 로드: 페르소나 + 대화 이력 건수 병렬 조회
  useEffect(() => {
    dispatch({ type: 'LOAD_START' })
    Promise.all([
      getCurrentPersona(),
      fetchChatHistoryCount().catch(() => ({
        count: 0,
        has_sufficient_history: false,
        oldest_date: null,
      })),
    ])
      .then(([persona, history]) => {
        dispatch({
          type: 'LOAD_SUCCESS',
          persona,
          chatHistoryCount: history.count,
        })
      })
      .catch((err) => {
        dispatch({
          type: 'LOAD_ERROR',
          error: err instanceof Error ? err.message : '페르소나 로드 실패',
        })
      })
  }, [])

  // Track 1: AI 자동 분석
  const handleStartAnalysis = useCallback(async () => {
    dispatch({ type: 'START_ANALYSIS' })
    const controller = new AbortController()
    analysisAbortRef.current = controller

    try {
      const result = await analyzePersona({})
      if (!controller.signal.aborted) {
        dispatch({
          type: 'ANALYSIS_SUCCESS',
          persona: result.persona,
          insights: result.analysis_insights,
        })
      }
    } catch (err: unknown) {
      if (!controller.signal.aborted) {
        // axios 에러에서 백엔드 detail 추출
        const axiosError = err as { response?: { status?: number; data?: { detail?: string } } }
        const status = axiosError.response?.status
        const detail = axiosError.response?.data?.detail

        if (status === 422) {
          // 대화 이력 부족 → 직접 설정으로 안내
          dispatch({
            type: 'ANALYSIS_FAIL',
            error: '대화 이력이 충분하지 않아 AI 자동 분석을 할 수 없습니다. 아래에서 직접 설정해주세요.',
            redirectScreen: 'track2_onboarding',
          })
        } else {
          const message = detail ?? (err instanceof Error ? err.message : '페르소나 분석 실패')
          dispatch({ type: 'ANALYSIS_FAIL', error: message })
        }
      }
    } finally {
      analysisAbortRef.current = null
    }
  }, [])

  // 분석 취소
  const handleCancelAnalysis = useCallback(() => {
    analysisAbortRef.current?.abort()
    dispatch({ type: 'SET_SCREEN', screen: 'welcome' })
  }, [])

  // Track 1 → Track 2 전환 (AI 결과를 초기값으로)
  const handleSwitchToManual = useCallback(() => {
    if (state.persona && state.screen === 'track1_review') {
      const draft: Partial<OnboardingState> = {
        specialty_areas: state.persona.specialty_areas.filter((a) => a !== 'all'),
        target_audience: state.persona.target_audience,
        preferred_tone: state.persona.preferred_tone,
        channel_style: state.persona.channel_style,
        focus_topics: state.persona.focus_topics,
      }
      dispatch({ type: 'SET_DRAFT', draft })
    }
    dispatch({ type: 'SET_SCREEN', screen: 'track2_onboarding' })
  }, [state.persona, state.screen])

  // Track 2 → Welcome 복귀
  const handleOnboardingBack = useCallback(() => {
    dispatch({ type: 'SET_SCREEN', screen: 'welcome' })
  }, [])

  // Track 1: HybridReview 확인 → 저장
  const handleHybridConfirm = useCallback(
    async (modified: PersonaUpdateRequest) => {
      dispatch({ type: 'SAVE_START' })
      try {
        const result = await updatePersona(modified)
        dispatch({ type: 'SAVE_SUCCESS', persona: result })
      } catch (err) {
        dispatch({
          type: 'SAVE_ERROR',
          error: err instanceof Error ? err.message : '페르소나 저장 실패',
        })
      }
    },
    [],
  )

  // Track 2: 온보딩 완료
  const handleOnboardingComplete = useCallback(
    async (onboardingState: OnboardingState) => {
      if (!onboardingState.target_audience || !onboardingState.preferred_tone) return
      dispatch({ type: 'SAVE_START' })
      try {
        const request: PersonaOnboardingRequest = {
          specialty_areas: onboardingState.specialty_areas,
          target_audience: onboardingState.target_audience,
          preferred_tone: onboardingState.preferred_tone,
          channel_style: onboardingState.channel_style,
          focus_topics: onboardingState.focus_topics,
        }
        const result = await createPersonaFromOnboarding(request)
        dispatch({ type: 'SAVE_SUCCESS', persona: result })
      } catch (err) {
        dispatch({
          type: 'SAVE_ERROR',
          error: err instanceof Error ? err.message : '온보딩 저장 실패',
        })
      }
    },
    [],
  )

  // Skip: 기본 페르소나로 진행
  const handleSkip = useCallback(() => {
    dispatch({ type: 'SKIP_SETUP' })
  }, [])

  // 에디터 저장 (ready 상태에서)
  const handleEditorSave = useCallback(
    async (request: PersonaUpdateRequest) => {
      dispatch({ type: 'SAVE_START' })
      try {
        const result = await updatePersona(request)
        dispatch({ type: 'SAVE_SUCCESS', persona: result })
      } catch (err) {
        dispatch({
          type: 'SAVE_ERROR',
          error: err instanceof Error ? err.message : '페르소나 수정 실패',
        })
      }
    },
    [],
  )

  // 에러 클리어
  const clearError = useCallback(() => {
    dispatch({ type: 'CLEAR_ERROR' })
  }, [])

  // 페르소나 편집 시작
  const handleEditPersona = useCallback(() => {
    dispatch({ type: 'SET_SCREEN', screen: 'welcome' })
  }, [])

  return {
    state,
    dispatch,
    handleStartAnalysis,
    handleCancelAnalysis,
    handleSwitchToManual,
    handleOnboardingBack,
    handleHybridConfirm,
    handleOnboardingComplete,
    handleSkip,
    handleEditorSave,
    handleEditPersona,
    clearError,
  }
}
