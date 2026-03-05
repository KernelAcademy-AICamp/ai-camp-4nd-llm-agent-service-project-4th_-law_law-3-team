'use client'

import { useState, useCallback, useEffect, useMemo, useRef } from 'react'
import { smallClaimsService } from '../services'
import type {
  WizardStep,
  DisputeType,
  CaseInfo,
  DocumentResponse,
  EvidenceItem,
  RelatedCaseItem,
  DocumentType,
  UploadedFile,
} from '../types'
import { MAX_FILE_SIZE_BYTES, REQUIRED_CASE_FIELDS } from '../types'

const STORAGE_KEY = 'small_claims_wizard_state'

const VALID_DISPUTE_TYPES = new Set<string>([
  'product_payment',
  'fraud',
  'deposit',
  'service_payment',
  'wage',
])

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
  evidenceError: string | null
  uploadedFiles: Map<string, UploadedFile[]>
  handleFileUpload: (evidenceItemId: string, files: File[]) => Promise<void>
  removeUploadedFile: (evidenceItemId: string, fileId: string) => void

  // Document
  generatedDocument: DocumentResponse | null
  isGenerating: boolean
  generateError: string | null
  generateDocument: (documentType: DocumentType) => Promise<void>

  // Related cases
  relatedCases: RelatedCaseItem[]
  isLoadingRelatedCases: boolean
  relatedCasesError: string | null

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
  const [evidenceError, setEvidenceError] = useState<string | null>(null)
  const [uploadedFiles, setUploadedFiles] = useState<Map<string, UploadedFile[]>>(new Map())

  // Document state
  const [isGenerating, setIsGenerating] = useState(false)
  const [generateError, setGenerateError] = useState<string | null>(null)

  // Related cases state
  const [relatedCases, setRelatedCases] = useState<RelatedCaseItem[]>([])
  const [isLoadingRelatedCases, setIsLoadingRelatedCases] = useState(false)
  const [relatedCasesError, setRelatedCasesError] = useState<string | null>(null)

  // Refs for event handler (avoid stale closures)
  const currentStepRef = useRef(currentStep)
  const disputeTypeRef = useRef(disputeType)
  const caseInfoRef = useRef(caseInfo)
  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Keep refs in sync
  currentStepRef.current = currentStep
  disputeTypeRef.current = disputeType
  caseInfoRef.current = caseInfo

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

  // 챗봇에서 상태 변경 시 UI 동기화
  useEffect(() => {
    if (typeof window === 'undefined') return

    const handleWizardStateChange = (e: CustomEvent) => {
      const newState = e.detail
      if (newState?.chatUpdated) {
        // 챗봇에서 분쟁 유형 설정 시 UI 업데이트 (매핑 필요)
        if (newState.chatDisputeType) {
          const disputeTypeMapping: Record<string, DisputeType> = {
            '물품대금': 'product_payment',
            '중고거래': 'fraud',
            '임대차': 'deposit',
            '용역대금': 'service_payment',
            '임금체불': 'wage',
          }
          const targetDisputeType =
            disputeTypeMapping[newState.chatDisputeType] || newState.chatDisputeType

          if (
            targetDisputeType &&
            VALID_DISPUTE_TYPES.has(targetDisputeType) &&
            targetDisputeType !== disputeTypeRef.current
          ) {
            setDisputeTypeState(targetDisputeType as DisputeType)
            setCheckedEvidence(new Set())
            setGeneratedDocument(null)
          }
        }

        // 챗봇에서 단계 변경 시 UI 업데이트 (매핑 필요)
        if (newState.chatStep) {
          const stepMapping: Record<string, WizardStep> = {
            init: 'dispute_type',
            gather_info: 'case_info',
            evidence: 'evidence',
            demand_letter: 'document',
            court: 'document',
            complete: 'document',
          }
          const targetStep = stepMapping[newState.chatStep]
          if (targetStep && targetStep !== currentStepRef.current) {
            setCurrentStep(targetStep)
          }
        }

        // 챗봇에서 청구 금액 설정 시 caseInfo 업데이트
        if (newState.chatClaimAmount) {
          setCaseInfo((prev) => ({ ...prev, amount: newState.chatClaimAmount }))
        }
      }
    }

    window.addEventListener('wizardStateChange', handleWizardStateChange as EventListener)
    return () => {
      window.removeEventListener('wizardStateChange', handleWizardStateChange as EventListener)
    }
  }, [])

  // Save state to sessionStorage (debounced 500ms)
  useEffect(() => {
    if (typeof window === 'undefined') return

    if (saveTimerRef.current) {
      clearTimeout(saveTimerRef.current)
    }

    saveTimerRef.current = setTimeout(() => {
      const state = {
        currentStep,
        disputeType,
        caseInfo,
        checkedEvidence: Array.from(checkedEvidence),
      }
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(state))
    }, 500)

    return () => {
      if (saveTimerRef.current) {
        clearTimeout(saveTimerRef.current)
      }
    }
  }, [currentStep, disputeType, caseInfo, checkedEvidence])

  // Load evidence checklist and related cases when dispute type changes
  useEffect(() => {
    if (!disputeType) return

    const controller = new AbortController()
    setIsLoadingEvidence(true)
    setIsLoadingRelatedCases(true)
    setEvidenceError(null)
    setRelatedCasesError(null)

    Promise.all([
      smallClaimsService.getEvidenceChecklist(disputeType),
      smallClaimsService.getRelatedCases(disputeType),
    ])
      .then(([evidenceResponse, casesResponse]) => {
        if (controller.signal.aborted) return
        setEvidenceItems(evidenceResponse.items)
        setRelatedCases(casesResponse.cases)
      })
      .catch((error) => {
        if (controller.signal.aborted) return
        console.error('Failed to load data:', error)
        setEvidenceError('증거 체크리스트를 불러오지 못했습니다')
        setRelatedCasesError('유사 판례를 불러오지 못했습니다')
      })
      .finally(() => {
        if (controller.signal.aborted) return
        setIsLoadingEvidence(false)
        setIsLoadingRelatedCases(false)
      })

    return () => {
      controller.abort()
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

  const handleFileUpload = useCallback(
    async (evidenceItemId: string, files: File[]) => {
      const oversized = files.find((f) => f.size > MAX_FILE_SIZE_BYTES)
      if (oversized) {
        throw new Error(`파일 크기가 10MB를 초과합니다: ${oversized.name}`)
      }

      const response = await smallClaimsService.uploadEvidence(files, evidenceItemId)
      setUploadedFiles((prev) => {
        const next = new Map(prev)
        const existing = next.get(evidenceItemId) ?? []
        next.set(evidenceItemId, [...existing, ...response.uploaded_files])
        return next
      })
    },
    []
  )

  const removeUploadedFile = useCallback((evidenceItemId: string, fileId: string) => {
    setUploadedFiles((prev) => {
      const next = new Map(prev)
      const existing = next.get(evidenceItemId) ?? []
      next.set(
        evidenceItemId,
        existing.filter((f) => f.file_id !== fileId)
      )
      return next
    })
  }, [])

  const generateDocument = useCallback(
    async (documentType: DocumentType) => {
      if (!disputeType) {
        setGenerateError('분쟁 유형을 선택해주세요')
        return
      }

      const currentCaseInfo = caseInfoRef.current
      const missingFields = REQUIRED_CASE_FIELDS.filter((field) => !currentCaseInfo[field])

      if (missingFields.length > 0) {
        setGenerateError('필수 정보를 모두 입력해주세요')
        return
      }

      setIsGenerating(true)
      setGenerateError(null)

      try {
        const fullCaseInfo: CaseInfo = {
          dispute_type: disputeType,
          plaintiff_name: currentCaseInfo.plaintiff_name!,
          plaintiff_address: currentCaseInfo.plaintiff_address!,
          plaintiff_phone: currentCaseInfo.plaintiff_phone,
          defendant_name: currentCaseInfo.defendant_name!,
          defendant_address: currentCaseInfo.defendant_address,
          defendant_phone: currentCaseInfo.defendant_phone,
          amount: currentCaseInfo.amount!,
          description: currentCaseInfo.description!,
          incident_date: currentCaseInfo.incident_date,
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
    [disputeType]
  )

  const resetWizard = useCallback(() => {
    setCurrentStep('dispute_type')
    setDisputeTypeState(null)
    setCaseInfo({})
    setCheckedEvidence(new Set())
    setUploadedFiles(new Map())
    setGeneratedDocument(null)
    setEvidenceItems([])
    setRelatedCases([])
    setEvidenceError(null)
    setRelatedCasesError(null)
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
    evidenceError,
    uploadedFiles,
    handleFileUpload,
    removeUploadedFile,
    generatedDocument,
    isGenerating,
    generateError,
    generateDocument,
    relatedCases,
    isLoadingRelatedCases,
    relatedCasesError,
    resetWizard,
  }
}
