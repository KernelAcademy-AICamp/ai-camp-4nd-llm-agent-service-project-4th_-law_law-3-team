/**
 * 타임라인 항목 ID 생성 유틸리티
 * 시간 기반 36진수 + 무작위 문자열 조합
 */
export const generateId = (): string =>
  `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 11)}`
