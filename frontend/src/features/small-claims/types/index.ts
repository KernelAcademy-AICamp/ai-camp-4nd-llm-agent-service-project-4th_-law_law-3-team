export type DisputeType =
  | 'product_payment'
  | 'fraud'
  | 'deposit'
  | 'service_payment'
  | 'wage'

export type DocumentType = 'demand_letter' | 'payment_order' | 'complaint'

export type WizardStep = 'dispute_type' | 'case_info' | 'evidence' | 'document'

export interface DisputeTypeOption {
  id: DisputeType
  name: string
  description: string
  icon: string
}

export interface EvidenceItem {
  id: string
  label: string
  required: boolean
  description: string
}

export interface EvidenceChecklistResponse {
  dispute_type: string
  description: string
  items: EvidenceItem[]
}

export interface CaseInfo {
  dispute_type: string
  plaintiff_name: string
  plaintiff_address: string
  plaintiff_phone?: string
  defendant_name: string
  defendant_address?: string
  defendant_phone?: string
  amount: number
  description: string
  incident_date?: string
}

export interface DocumentResponse {
  document_type: string
  title: string
  content: string
  template_sections: Record<string, string>
  pdf_url?: string
  docx_url?: string
}

export interface RelatedCaseItem {
  id: string
  case_name: string
  case_number: string
  summary: string
  similarity: number
  relevance: string
}

export interface RelatedCasesResponse {
  dispute_type: string
  cases: RelatedCaseItem[]
}

export interface WizardState {
  currentStep: WizardStep
  disputeType: DisputeType | null
  caseInfo: Partial<CaseInfo>
  checkedEvidence: Set<string>
  generatedDocument: DocumentResponse | null
}

export const DISPUTE_TYPE_OPTIONS: DisputeTypeOption[] = [
  {
    id: 'product_payment',
    name: '물품대금',
    description: '물품을 판매했으나 대금을 받지 못한 경우',
    icon: '📦',
  },
  {
    id: 'fraud',
    name: '중고거래 사기',
    description: '중고거래에서 물건을 받지 못했거나 상품이 설명과 다른 경우',
    icon: '🚨',
  },
  {
    id: 'deposit',
    name: '임대차 보증금',
    description: '전세/월세 보증금을 돌려받지 못한 경우',
    icon: '🏠',
  },
  {
    id: 'service_payment',
    name: '용역대금',
    description: '용역(서비스)을 제공했으나 대금을 받지 못한 경우',
    icon: '💼',
  },
  {
    id: 'wage',
    name: '임금 체불',
    description: '근무했으나 급여/알바비를 받지 못한 경우',
    icon: '💰',
  },
]

export const WIZARD_STEPS: { step: WizardStep; label: string }[] = [
  { step: 'dispute_type', label: '분쟁 유형' },
  { step: 'case_info', label: '사건 정보' },
  { step: 'evidence', label: '증거 체크' },
  { step: 'document', label: '서류 생성' },
]
