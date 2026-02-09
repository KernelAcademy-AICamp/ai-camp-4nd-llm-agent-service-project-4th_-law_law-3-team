'use client'

import { useEffect, useCallback, useState } from 'react'

// 챗봇과 소액소송 UI 간 공유 상태 키
export const WIZARD_STORAGE_KEY = 'small_claims_wizard_state'

export interface SharedWizardState {
  currentStep: string
  disputeType: string | null
  caseInfo: Record<string, unknown>
  checkedEvidence: string[]
  // 챗봇에서 업데이트한 상태
  chatUpdated?: boolean
  chatStep?: string
  chatDisputeType?: string
  chatClaimAmount?: number
}

/**
 * 챗봇과 소액소송 UI 간 양방향 동기화 훅
 * 
 * sessionStorage를 통해 상태를 공유하고,
 * storage 이벤트를 통해 다른 탭/컴포넌트의 변경을 감지합니다.
 */
export function useSmallClaimsSync() {
  const [sharedState, setSharedState] = useState<SharedWizardState | null>(null)

  // sessionStorage에서 상태 읽기
  const readState = useCallback((): SharedWizardState | null => {
    if (typeof window === 'undefined') return null
    try {
      const saved = sessionStorage.getItem(WIZARD_STORAGE_KEY)
      return saved ? JSON.parse(saved) : null
    } catch {
      return null
    }
  }, [])

  // sessionStorage에 상태 쓰기 (병합 모드)
  const writeState = useCallback((updates: Partial<SharedWizardState>) => {
    if (typeof window === 'undefined') return
    try {
      const current = readState() || {
        currentStep: 'dispute_type',
        disputeType: null,
        caseInfo: {},
        checkedEvidence: [],
      }
      const newState = { ...current, ...updates }
      sessionStorage.setItem(WIZARD_STORAGE_KEY, JSON.stringify(newState))
      setSharedState(newState)
      
      // 같은 탭 내 다른 컴포넌트에 알림 (storage 이벤트는 다른 탭에만 발생)
      window.dispatchEvent(new CustomEvent('wizardStateChange', { detail: newState }))
    } catch (e) {
      console.error('Failed to write wizard state:', e)
    }
  }, [readState])

  // 챗봇에서 호출: 분쟁 유형 설정 시
  const setChatDisputeType = useCallback((disputeType: string) => {
    writeState({
      disputeType,
      chatUpdated: true,
      chatDisputeType: disputeType,
      currentStep: 'case_info', // 다음 단계로 자동 이동
    })
  }, [writeState])

  // 챗봇에서 호출: 청구 금액 설정 시
  const setChatClaimAmount = useCallback((amount: number) => {
    writeState({
      chatUpdated: true,
      chatClaimAmount: amount,
    })
  }, [writeState])

  // 챗봇에서 호출: 현재 단계 설정
  const setChatStep = useCallback((step: string) => {
    writeState({
      chatUpdated: true,
      chatStep: step,
    })
  }, [writeState])

  // 초기 로드 및 변경 감지
  useEffect(() => {
    // 초기 상태 로드
    setSharedState(readState())

    // 다른 탭에서의 변경 감지
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === WIZARD_STORAGE_KEY) {
        setSharedState(e.newValue ? JSON.parse(e.newValue) : null)
      }
    }

    // 같은 탭 내 변경 감지 (커스텀 이벤트)
    const handleCustomEvent = (e: CustomEvent<SharedWizardState>) => {
      setSharedState(e.detail)
    }

    window.addEventListener('storage', handleStorageChange)
    window.addEventListener('wizardStateChange', handleCustomEvent as EventListener)

    return () => {
      window.removeEventListener('storage', handleStorageChange)
      window.removeEventListener('wizardStateChange', handleCustomEvent as EventListener)
    }
  }, [readState])

  return {
    sharedState,
    readState,
    writeState,
    setChatDisputeType,
    setChatClaimAmount,
    setChatStep,
  }
}
