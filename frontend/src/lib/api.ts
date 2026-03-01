import axios from 'axios'

// Next.js API Route 프록시 사용 (rewrites 버그 우회)
export const api = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 180000, // 180초 타임아웃 (AI 타임라인 추출 등 장시간 요청 대응)
})

// API 모듈별 엔드포인트
export const endpoints = {
  lawyerFinder: '/lawyer-finder',
  lawyerStat: '/lawyer-stats',
  casePrecedent: '/case-precedent',
  storyboard: '/storyboard',
  lawStudy: '/law-study',
  smallClaims: '/small-claims',
  multiAgent: '/multi-agent',
  mockTrial: '/mock-trial',
  contentMarketing: '/content-marketing',
}
