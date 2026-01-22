import axios from 'axios'

// Next.js API Route 프록시 사용 (rewrites 버그 우회)
export const api = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 60000, // 60초 타임아웃 (AI 요청은 시간이 걸릴 수 있음)
})

// API 모듈별 엔드포인트
export const endpoints = {
  lawyerFinder: '/lawyer-finder',
  casePrecedent: '/case-precedent',
  reviewPrice: '/review-price',
  storyboard: '/storyboard',
  lawStudy: '/law-study',
  smallClaims: '/small-claims',
  multiAgent: '/multi-agent',
}
