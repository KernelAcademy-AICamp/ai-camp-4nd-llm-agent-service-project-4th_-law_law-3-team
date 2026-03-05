import type { ChatAction } from '../ChatActions'
import type { ChatSource } from '@/features/case-precedent/types'

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  actions?: ChatAction[]
  agentUsed?: string
  sources?: ChatSource[]
}

export interface MultiAgentChatResponse {
  response: string
  agent_used: string
  sources: ChatSource[]
  actions: ChatAction[]
  session_data: Record<string, unknown>
}

// pathname → agent 매핑 (페이지 진입 시 자동 선택)
export const PATHNAME_AGENT_MAP: Record<string, string> = {
  '/lawyer-finder': 'lawyer_finder',
  '/storyboard': 'storyboard',
  '/lawyer-stats': 'lawyer_stats',
  '/law-study': 'law_study',
  '/small-claims': 'small_claims',
  '/statute-hierarchy': 'law_search',
  '/case-precedent': 'case_search',
  '/law-search': 'law_search',
}

// agent → 이동할 페이지 매핑 (에이전트 선택 시 자동 이동)
export const AGENT_PAGE_MAP: Record<string, string> = {
  'lawyer_finder': '/lawyer-finder',
  'case_search': '/case-precedent',
  'law_search': '/law-search',
  'legal_search': '/case-precedent',
  'legal_answer': '/case-precedent',
  'storyboard': '/storyboard',
  'lawyer_stats': '/lawyer-stats',
  'law_study': '/law-study',
  'small_claims': '/small-claims',
}

// agent 한글명 (헤더 표시용)
export const AGENT_DISPLAY_NAMES: Record<string, string> = {
  'lawyer_finder': '변호사 찾기',
  'case_search': '판례 검색',
  'law_search': '법령 검색',
  'legal_search': '법률 검색',
  'storyboard': '스토리보드',
  'lawyer_stats': '변호사 통계',
  'law_study': '로스쿨 학습',
  'small_claims': '소액소송',
  'general': '일반 채팅',
}

// 에이전트별 초기 인사 메시지
export const AGENT_GREETINGS: Record<string, string> = {
  'case_search': '안녕하세요! 판례 검색 AI입니다.\n\n**판례에 대해 질문해주세요.**\n- 관련 판례 검색\n- 법률 상담',
  'law_search': '안녕하세요! 법령 검색 AI입니다.\n\n**법령에 대해 질문해주세요.**\n- 관련 법령 조항 검색\n- 법령 해석 및 적용 사례',
  'lawyer_finder': '안녕하세요! 변호사 찾기 AI입니다.\n\n**주변 변호사를 찾아드릴게요.**\n- 위치 기반 변호사 검색\n- 전문분야별 추천',
  'storyboard': '안녕하세요! 스토리보드 AI입니다.\n\n**사건 타임라인을 정리해드릴게요.**\n- 사건 경위 정리\n- 시간순 타임라인 생성',
  'lawyer_stats': '안녕하세요! 변호사 통계 AI입니다.\n\n**변호사 통계 정보를 안내해드릴게요.**\n- 지역별 변호사 현황\n- 전문분야별 분포',
  'law_study': '안녕하세요! 법학 학습 AI입니다.\n\n**법학 공부를 도와드릴게요.**\n- 법령 학습 자료\n- 학습 가이드',
  'small_claims': '안녕하세요! 소액소송 가이드 AI입니다.\n\n**소액소송 절차를 안내해드릴게요.**\n- 내용증명 작성\n- 지급명령 신청\n- 소액심판 절차',
}

// floating 모드 기본 적용 페이지
export const FLOATING_MODE_PATHS = new Set([
  '/',
  '/lawyer-finder',
  '/small-claims',
  '/lawyer-stats',
  '/storyboard',
  '/statute-hierarchy',
  '/workspace',
  '/case-precedent',
  '/law-search',
  '/law-study',
])
