'use client'

import { useState, useCallback, useEffect, useMemo } from 'react'
import { smallClaimsService } from '../services'
import type {
  WizardStep,
  DisputeType,
  CaseInfo,
  DocumentResponse,
  EvidenceItem,
  RelatedCaseItem,
  DocumentType,
  WIZARD_STEPS,
} from '../types'

const STORAGE_KEY = 'small_claims_wizard_state'

interface UseWizardStateReturn {
  // Step management
  currentStep: WizardStep
  goToStep: (step: WizardStep) => void
  goToNextStep: () => void
  goToPreviousStep: () => void
  canGoNext: boolean
  canGoPrevious: boolean

  // Dispute type
  disputeType: DisputeType | null
  setDisputeType: (type: DisputeType) => void

  // Case info
  caseInfo: Partial<CaseInfo>
  updateCaseInfo: (info: Partial<CaseInfo>) => void

  // Evidence
  evidenceItems: EvidenceItem[]
  checkedEvidence: Set<string>
  toggleEvidence: (id: string) => void
  isLoadingEvidence: boolean

  // Document
  generatedDocument: DocumentResponse | null
  isGenerating: boolean
  generateError: string | null
  generateDocument: (documentType: DocumentType) => Promise<void>

  // Related cases
  relatedCases: RelatedCaseItem[]
  isLoadingRelatedCases: boolean

  // Reset
  resetWizard: () => void
}

const STEP_ORDER: WizardStep[] = ['dispute_type', 'case_info', 'evidence', 'document']

export function useWizardState(): UseWizardStateReturn {
  const [currentStep, setCurrentStep] = useState<WizardStep>('dispute_type')
  const [disputeType, setDisputeTypeState] = useState<DisputeType | null>(null)
  const [caseInfo, setCaseInfo] = useState<Partial<CaseInfo>>({})
  const [checkedEvidence, setCheckedEvidence] = useState<Set<string>>(new Set())
  const [generatedDocument, setGeneratedDocument] = useState<DocumentResponse | null>(null)

  // Evidence state
  const [evidenceItems, setEvidenceItems] = useState<EvidenceItem[]>([])
  const [isLoadingEvidence, setIsLoadingEvidence] = useState(false)

  // Document state
  const [isGenerating, setIsGenerating] = useState(false)
  const [generateError, setGenerateError] = useState<string | null>(null)

  // Related cases state
  const [relatedCases, setRelatedCases] = useState<RelatedCaseItem[]>([])
  const [isLoadingRelatedCases, setIsLoadingRelatedCases] = useState(false)

  // Load saved state from sessionStorage
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const saved = sessionStorage.getItem(STORAGE_KEY)
      if (saved) {
        try {
          const parsed = JSON.parse(saved)
          setCurrentStep(parsed.currentStep || 'dispute_type')
          setDisputeTypeState(parsed.disputeType || null)
          setCaseInfo(parsed.caseInfo || {})
          setCheckedEvidence(new Set(parsed.checkedEvidence || []))
        } catch (e) {
          console.error('Failed to parse saved wizard state:', e)
        }
      }
    }
  }, [])

  // Save state to sessionStorage
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const state = {
        currentStep,
        disputeType,
        caseInfo,
        checkedEvidence: Array.from(checkedEvidence),
      }
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(state))
    }
  }, [currentStep, disputeType, caseInfo, checkedEvidence])

  // Load evidence checklist and related cases in parallel when dispute type changes
  useEffect(() => {
    if (disputeType) {
      setIsLoadingEvidence(true)
      setIsLoadingRelatedCases(true)

      // 병렬로 두 API 호출 실행
      Promise.all([
        smallClaimsService.getEvidenceChecklist(disputeType),
        smallClaimsService.getRelatedCases(disputeType),
      ])
        .then(([evidenceResponse, casesResponse]) => {
          setEvidenceItems(evidenceResponse.items)
          setRelatedCases(casesResponse.cases)
        })
        .catch((error) => {
          console.error('Failed to load data:', error)
        })
        .finally(() => {
          setIsLoadingEvidence(false)
          setIsLoadingRelatedCases(false)
        })
    }
  }, [disputeType])

  const goToStep = useCallback((step: WizardStep) => {
    setCurrentStep(step)
  }, [])

  // useMemo로 파생 값 최적화
  const { currentStepIndex, canGoNext, canGoPrevious } = useMemo(() => {
    const index = STEP_ORDER.indexOf(currentStep)
    return {
      currentStepIndex: index,
      canGoNext: index < STEP_ORDER.length - 1,
      canGoPrevious: index > 0,
    }
  }, [currentStep])

  const goToNextStep = useCallback(() => {
    if (canGoNext) {
      setCurrentStep(STEP_ORDER[currentStepIndex + 1])
    }
  }, [currentStepIndex, canGoNext])

  const goToPreviousStep = useCallback(() => {
    if (canGoPrevious) {
      setCurrentStep(STEP_ORDER[currentStepIndex - 1])
    }
  }, [currentStepIndex, canGoPrevious])

  const setDisputeType = useCallback((type: DisputeType) => {
    setDisputeTypeState(type)
    setCheckedEvidence(new Set())
    setGeneratedDocument(null)
  }, [])

  const updateCaseInfo = useCallback((info: Partial<CaseInfo>) => {
    setCaseInfo((prev) => ({ ...prev, ...info }))
  }, [])

  const toggleEvidence = useCallback((id: string) => {
    setCheckedEvidence((prev) => {
      const newSet = new Set(prev)
      if (newSet.has(id)) {
        newSet.delete(id)
      } else {
        newSet.add(id)
      }
      return newSet
    })
  }, [])

  const generateDocument = useCallback(
    async (documentType: DocumentType) => {
      if (!disputeType) {
        setGenerateError('분쟁 유형을 선택해주세요')
        return
      }

      const requiredFields = ['plaintiff_name', 'plaintiff_address', 'defendant_name', 'amount', 'description']
      const missingFields = requiredFields.filter((field) => !caseInfo[field as keyof CaseInfo])

      if (missingFields.length > 0) {
        setGenerateError('필수 정보를 모두 입력해주세요')
        return
      }

      setIsGenerating(true)
      setGenerateError(null)

      try {
        const fullCaseInfo: CaseInfo = {
          dispute_type: disputeType,
          plaintiff_name: caseInfo.plaintiff_name!,
          plaintiff_address: caseInfo.plaintiff_address!,
          plaintiff_phone: caseInfo.plaintiff_phone,
          defendant_name: caseInfo.defendant_name!,
          defendant_address: caseInfo.defendant_address,
          defendant_phone: caseInfo.defendant_phone,
          amount: caseInfo.amount!,
          description: caseInfo.description!,
          incident_date: caseInfo.incident_date,
        }

        const document = await smallClaimsService.generateDocument(documentType, fullCaseInfo)
        setGeneratedDocument(document)
      } catch (error) {
        console.error('Failed to generate document:', error)
        setGenerateError('서류 생성에 실패했습니다. 다시 시도해주세요.')
      } finally {
        setIsGenerating(false)
      }
    },
    [disputeType, caseInfo]
  )

  const resetWizard = useCallback(() => {
    setCurrentStep('dispute_type')
    setDisputeTypeState(null)
    setCaseInfo({})
    setCheckedEvidence(new Set())
    setGeneratedDocument(null)
    setEvidenceItems([])
    setRelatedCases([])
    if (typeof window !== 'undefined') {
      sessionStorage.removeItem(STORAGE_KEY)
    }
  }, [])

  return {
    currentStep,
    goToStep,
    goToNextStep,
    goToPreviousStep,
    canGoNext,
    canGoPrevious,
    disputeType,
    setDisputeType,
    caseInfo,
    updateCaseInfo,
    evidenceItems,
    checkedEvidence,
    toggleEvidence,
    isLoadingEvidence,
    generatedDocument,
    isGenerating,
    generateError,
    generateDocument,
    relatedCases,
    isLoadingRelatedCases,
    resetWizard,
  }
}
