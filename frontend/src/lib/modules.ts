/**
 * 프론트엔드 모듈 설정
 * 기능을 추가/삭제할 때 이 파일을 수정하세요
 */

export interface Module {
  id: string
  name: string
  description: string
  href: string
  icon: string
  enabled: boolean
  roles: ('lawyer' | 'user')[]
  category: string
}

export const modules: Module[] = [
  {
    id: 'lawyer-finder',
    name: '주변 변호사 찾기',
    description: '내 위치 기반으로 가까운 변호사를 찾아보세요',
    href: '/lawyer-finder',
    icon: '📍',
    enabled: true,
    roles: ['user'],
    category: 'problem-solving',
  },
  {
    id: 'lawyer-stats',
    name: '변호사 통계',
    description: '지역별, 전문분야별 변호사 현황 대시보드',
    href: '/lawyer-stats',
    icon: '📊',
    enabled: true,
    roles: ['lawyer'],
    category: 'insight',
  },
  {
    id: 'case-precedent',
    name: '판례 검색',
    description: 'RAG 기반 판례 검색 및 AI 질문',
    href: '/case-precedent?agent=case_search',
    icon: '📚',
    enabled: true,
    roles: ['lawyer', 'user'],
    category: 'research', // User side will treat this as 'information'
  },
  {
    id: 'law-search',
    name: '법령 검색',
    description: 'RAG 기반 법령 검색 및 AI 질문',
    href: '/case-precedent?agent=law_search',
    icon: '📖',
    enabled: true,
    roles: ['lawyer', 'user'],
    category: 'research',
  },
  {
    id: 'storyboard',
    name: '스토리보드',
    description: '사건 타임라인을 시각화합니다',
    href: '/storyboard',
    icon: '🎬',
    enabled: true,
    roles: ['lawyer', 'user'],
    category: 'case-review', // User side: 'case-management'
  },
  {
    id: 'law-study',
    name: '로스쿨 학습',
    description: '법학 공부에 도움되는 자료를 제공합니다',
    href: '/law-study',
    icon: '📖',
    enabled: true,
    roles: ['lawyer'],
    category: 'study',
  },
  {
    id: 'statute-hierarchy',
    name: '법령 체계도',
    description: '법령 간 계급 관계를 시각화합니다',
    href: '/statute-hierarchy',
    icon: '🔗',
    enabled: true,
    roles: ['lawyer'],
    category: 'research',
  },
  {
    id: 'small-claims',
    name: '소액 소송 도우미',
    description: '4단계 위자드로 내용증명, 지급명령, 소액심판 서류 작성',
    href: '/small-claims',
    icon: '⚖️',
    enabled: true,
    roles: ['user'],
    category: 'problem-solving',
  },
  {
    id: 'mock-trial',
    name: '모의 법정',
    description: '픽셀아트 법정에서 AI 에이전트와 함께하는 모의재판 시뮬레이션',
    href: '/mock-trial',
    icon: '🏛️',
    enabled: true,
    roles: ['user', 'lawyer'],
    category: 'case-review', // User side: 'problem-solving'
  },
  {
    id: 'content-marketing',
    name: '콘텐츠 마케팅',
    description: '법률 트렌드 분석 및 AI 유튜브 대본 자동 생성',
    href: '/content-marketing',
    icon: '📹',
    enabled: true,
    roles: ['lawyer'],
    category: 'insight',
  },
  {
    id: 'workspace',
    name: '사건 워크스페이스',
    description: '사건별 대화/태그/타임라인 관리',
    href: '/workspace',
    icon: '💼',
    enabled: true,
    roles: ['lawyer'],
    category: 'case-review',
  },
  {
    id: 'legal-news',
    name: '법률 뉴스',
    description: '법률 뉴스 수집·요약 및 하이브리드 검색',
    href: '/legal-news',
    icon: '📰',
    enabled: true,
    roles: ['lawyer', 'user'],
    category: 'research',
  },
]

export const CATEGORY_NAMES: Record<string, { lawyer: string; user: string }> = {
  'research': { lawyer: '리서치', user: '정보 찾기' },
  'case-review': { lawyer: '사건 검토', user: '문제 해결' }, // 'mock-trial' case
  'insight': { lawyer: '인사이트', user: '인사이트' },
  'study': { lawyer: '학습', user: '학습' },
  'problem-solving': { lawyer: '문제 해결', user: '문제 해결' },
  'case-management': { lawyer: '사건 정리', user: '사건 정리' },
  'information': { lawyer: '정보 찾기', user: '정보 찾기' },
}

// Special overrides for grouping
export const getModuleCategory = (module: Module, role: 'lawyer' | 'user') => {
  if (role === 'user') {
    if (module.id === 'case-precedent' || module.id === 'law-search' || module.id === 'legal-news') return 'information'
    if (module.id === 'storyboard') return 'case-management'
    if (module.id === 'mock-trial') return 'problem-solving'
  }
  return module.category
}

export const getEnabledModules = (role?: 'lawyer' | 'user') =>
  modules.filter((m) => m.enabled && (!role || m.roles.includes(role)))
