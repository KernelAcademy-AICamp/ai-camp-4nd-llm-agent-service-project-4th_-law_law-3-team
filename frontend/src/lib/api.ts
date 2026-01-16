import axios from 'axios'

export const api = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
})

// API 모듈별 엔드포인트
export const endpoints = {
  lawyerFinder: '/lawyer-finder',
  casePrecedent: '/case-precedent',
  reviewPrice: '/review-price',
  storyboard: '/storyboard',
  lawStudy: '/law-study',
  smallClaims: '/small-claims',
}
