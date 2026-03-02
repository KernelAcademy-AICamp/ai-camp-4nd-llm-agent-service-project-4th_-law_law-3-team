/** 법률 뉴스 모듈 공유 상수 */

export const SOURCE_OPTIONS = [
  { value: '', label: '전체 소스' },
  { value: 'lawtimes', label: '법률신문' },
  { value: 'naver', label: '네이버' },
] as const

export const LIMIT_OPTIONS = [
  { value: 5, label: '5건' },
  { value: 10, label: '10건' },
  { value: 20, label: '20건' },
  { value: 50, label: '50건' },
] as const
