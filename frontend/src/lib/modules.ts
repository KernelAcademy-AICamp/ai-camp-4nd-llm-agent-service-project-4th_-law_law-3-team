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
  },
  {
    id: 'case-precedent',
    name: '판례 검색',
    description: '상황에 맞는 관련 판례를 찾아드립니다',
    href: '/case-precedent',
    icon: '📚',
    enabled: true,
    roles: ['lawyer', 'user'],
  },
  {
    id: 'review-price',
    name: '후기/가격 비교',
    description: '다른 사용자들의 상담 후기와 가격을 비교해보세요',
    href: '/review-price',
    icon: '💰',
    enabled: true,
    roles: ['user'],
  },
  {
    id: 'storyboard',
    name: '스토리보드',
    description: '사건 타임라인을 시각화합니다',
    href: '/storyboard',
    icon: '🎬',
    enabled: true,
    roles: ['lawyer', 'user'],
  },
  {
    id: 'law-study',
    name: '로스쿨 학습',
    description: '법학 공부에 도움되는 자료를 제공합니다',
    href: '/law-study',
    icon: '📖',
    enabled: true,
    roles: ['lawyer'],
  },
  {
    id: 'small-claims',
    name: '소액 소송 도우미',
    description: '나홀로 소송을 도와드립니다',
    href: '/small-claims',
    icon: '⚖️',
    enabled: true,
    roles: ['user'],
  },
]

export const getEnabledModules = (role?: 'lawyer' | 'user') => 
  modules.filter((m) => m.enabled && (!role || m.roles.includes(role)))
