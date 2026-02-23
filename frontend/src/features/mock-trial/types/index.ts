/** 모의 법정 TypeScript 타입 + 상수 정의 */

// ── 사건 유형 ──

export type CaseType = 'criminal' | 'civil'

export type CriminalCategory =
  | 'criminal_assault'
  | 'criminal_fraud'
  | 'criminal_theft'
  | 'criminal_embezzlement'
  | 'criminal_other'

export type CivilCategory =
  | 'civil_damages'
  | 'civil_contract'
  | 'civil_property'
  | 'civil_other'

export type CaseCategory = CriminalCategory | CivilCategory

// ── 역할 ──

export type CriminalRole = 'prosecutor' | 'attorney'
/** 민사 역할: Backend API 계약상 prosecutor=원고측, attorney=피고측 매핑 */
export type CivilRole = 'prosecutor' | 'attorney'
export type UserRole = CriminalRole | CivilRole

// ── 재판 단계 ──

export type CriminalStage =
  | 'setup'
  | 'identity'
  | 'opening'
  | 'evidence'
  | 'examination'
  | 'closing'
  | 'verdict'

export type CivilStage =
  | 'setup'
  | 'pretrial'
  | 'claims'
  | 'evidence'
  | 'argument'
  | 'closing'
  | 'verdict'

export type TrialStage = CriminalStage | CivilStage

// ── 감정 이모지 (FR-51) ──

export type EmotionType =
  | 'neutral'
  | 'angry'
  | 'thinking'
  | 'sad'
  | 'confident'
  | 'stern'
  | 'recording'
  | 'judging'

export const EMOTION_EMOJI: Record<EmotionType, string> = {
  neutral: '😐',
  angry: '😤',
  thinking: '🤔',
  sad: '😢',
  confident: '😊',
  stern: '😠',
  recording: '📝',
  judging: '⚖️',
}

export const DEFAULT_ROLE_EMOTION: Record<string, EmotionType> = {
  judge: 'stern',
  prosecutor: 'confident',
  attorney: 'thinking',
  defendant: 'sad',
  clerk: 'recording',
}

// ── 인터페이스 ──

export interface EvidenceItem {
  id: string
  title: string
  summary: string
  relevance_score: number
  source: string
}

export interface ReferenceItem {
  id: string
  type: 'case' | 'law'
  title: string
  summary: string
  relevance_score: number
  source: string
  /** 채팅에서 감지된 원문 텍스트 */
  matched_text: string
}

export interface CourtEvent {
  stage: string
  speaker: string
  content: string
  timestamp: string
  emotion?: EmotionType
}

export interface TrialSetup {
  case_type: CaseType
  case_category: CaseCategory
  user_role: UserRole
  case_summary: string
}

export interface JudgmentResult {
  judgment: string
  feedback: string
  cited_cases: EvidenceItem[]
  cited_articles: EvidenceItem[]
}

export interface StageInfo {
  id: string
  name: string
  order: number
  legal_basis: string
  description: string
  user_action: string
  duration_hint: string
}

// ── 상수: 사건 유형 옵션 ──

export const CASE_TYPE_OPTIONS: { id: CaseType; name: string; icon: string }[] = [
  { id: 'criminal', name: '형사 재판', icon: '⚖️' },
  { id: 'civil', name: '민사 재판', icon: '📜' },
]

// ── 상수: 세부 유형 ──

export const CRIMINAL_CATEGORIES: {
  id: CriminalCategory
  name: string
  description: string
}[] = [
  { id: 'criminal_assault', name: '폭행/상해', description: '폭행죄, 상해죄 등' },
  { id: 'criminal_fraud', name: '사기', description: '사기죄, 횡령죄 등' },
  { id: 'criminal_theft', name: '절도', description: '절도죄, 강도죄 등' },
  { id: 'criminal_embezzlement', name: '횡령/배임', description: '횡령죄, 배임죄 등' },
  { id: 'criminal_other', name: '기타', description: '기타 형사 사건' },
]

export const CIVIL_CATEGORIES: {
  id: CivilCategory
  name: string
  description: string
}[] = [
  { id: 'civil_damages', name: '손해배상', description: '불법행위, 채무불이행 등' },
  { id: 'civil_contract', name: '계약 분쟁', description: '계약 해제, 이행 청구 등' },
  { id: 'civil_property', name: '부동산', description: '임대차, 소유권 분쟁 등' },
  { id: 'civil_other', name: '기타', description: '기타 민사 사건' },
]

// ── 상수: 재판 단계 정보 ──

export const CRIMINAL_STAGES: StageInfo[] = [
  {
    id: 'identity',
    name: '인정신문',
    order: 1,
    legal_basis: '형사소송법 §284',
    description: '피고인 인적사항 확인, 진술거부권 고지',
    user_action: '자동 진행',
    duration_hint: '1-2분',
  },
  {
    id: 'opening',
    name: '모두진술',
    order: 2,
    legal_basis: '형사소송법 §285~§286',
    description: '검사 공소사실, 피고인 의견 진술',
    user_action: '역할에 따라 진술 입력',
    duration_hint: '3-5분',
  },
  {
    id: 'evidence',
    name: '증거조사',
    order: 3,
    legal_basis: '형사소송법 §290~§313',
    description: '판례/법령 검색, 증거 제출',
    user_action: '증거 선택/제출',
    duration_hint: '5-10분',
  },
  {
    id: 'examination',
    name: '피고인신문',
    order: 4,
    legal_basis: '형사소송법 §296-2',
    description: '검사/변호인이 피고인에게 질문',
    user_action: '질문 입력',
    duration_hint: '3-5분',
  },
  {
    id: 'closing',
    name: '최종변론',
    order: 5,
    legal_basis: '형사소송법 §302~§303',
    description: '검사 구형, 변호인 최후변론, 피고인 최후진술',
    user_action: '최후변론 입력',
    duration_hint: '3-5분',
  },
  {
    id: 'verdict',
    name: '판결선고',
    order: 6,
    legal_basis: '형사소송법 §43(판결선고방식), §39(판결선고기일), §323(유죄이유고지)',
    description: 'AI 판사 판결문 낭독',
    user_action: '관전',
    duration_hint: '2-3분',
  },
]

export const CIVIL_STAGES: StageInfo[] = [
  {
    id: 'pretrial',
    name: '변론준비',
    order: 1,
    legal_basis: '민사소송법 §258~§268',
    description: '쟁점 정리, 증거 목록 확인',
    user_action: '자동 진행',
    duration_hint: '1-2분',
  },
  {
    id: 'claims',
    name: '주장/답변',
    order: 2,
    legal_basis: '민사소송법 §256~§257',
    description: '원고 청구원인, 피고 답변',
    user_action: '역할에 따라 입력',
    duration_hint: '3-5분',
  },
  {
    id: 'evidence',
    name: '증거조사',
    order: 3,
    legal_basis: '민사소송법 §288~§344',
    description: '판례/법령 검색, 서증 제출',
    user_action: '증거 선택/제출',
    duration_hint: '5-10분',
  },
  {
    id: 'argument',
    name: '변론',
    order: 4,
    legal_basis: '민사소송법 §134~§148',
    description: '양측 주장/반박 교환',
    user_action: '주장 입력 (2-3 라운드)',
    duration_hint: '5-10분',
  },
  {
    id: 'closing',
    name: '변론종결',
    order: 5,
    legal_basis: '민사소송법 §200',
    description: '양측 최종 주장 정리',
    user_action: '최종 주장 입력',
    duration_hint: '2-3분',
  },
  {
    id: 'verdict',
    name: '판결선고',
    order: 6,
    legal_basis: '민사소송법 §206~§208',
    description: 'AI 판사 판결문 낭독',
    user_action: '관전',
    duration_hint: '2-3분',
  },
]
