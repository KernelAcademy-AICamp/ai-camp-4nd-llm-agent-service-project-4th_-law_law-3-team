/** @type {import('next').NextConfig} */
// Vercel 배포 시 BACKEND_URL 미설정 방지 (VERCEL env는 Vercel이 자동 설정)
if (
  process.env.VERCEL === '1' &&
  (!process.env.BACKEND_URL ||
    process.env.BACKEND_URL.includes('127.0.0.1') ||
    process.env.BACKEND_URL.includes('localhost'))
) {
  throw new Error(
    'BACKEND_URL 환경변수가 프로덕션에 설정되지 않았거나 localhost를 가리킵니다. ' +
      'Vercel 대시보드에서 BACKEND_URL=https://api.your-domain.example 을 설정하세요.'
  )
}

const BACKEND_URL = process.env.BACKEND_URL || 'http://127.0.0.1:8000'

const nextConfig = {
  // Enable standalone output for Docker production builds
  output: 'standalone',

  // 프록시 타임아웃 설정 (LLM 응답 대기)
  experimental: {
    proxyTimeout: 180000, // 180초 (axios timeout과 일치)
    optimizePackageImports: ['lucide-react'],
  },

  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          { key: 'X-Frame-Options', value: 'DENY' },
          { key: 'X-Content-Type-Options', value: 'nosniff' },
          { key: 'Referrer-Policy', value: 'strict-origin-when-cross-origin' },
        ],
      },
    ]
  },

  async rewrites() {
    return [
      // storyboard는 API Route로 프록시 (Next.js rewrites 버그 우회)
      // 다른 모듈들은 기존 rewrites 사용
      {
        source: '/api/lawyer-finder/:path*',
        destination: `${BACKEND_URL}/api/lawyer-finder/:path*`,
      },
      {
        source: '/api/lawyer-stats/:path*',
        destination: `${BACKEND_URL}/api/lawyer-stats/:path*`,
      },
      {
        source: '/api/case-precedent/:path*',
        destination: `${BACKEND_URL}/api/case-precedent/:path*`,
      },
      {
        source: '/api/law-study/:path*',
        destination: `${BACKEND_URL}/api/law-study/:path*`,
      },
      {
        source: '/api/small-claims/:path*',
        destination: `${BACKEND_URL}/api/small-claims/:path*`,
      },
      {
        source: '/api/mock-trial/:path*',
        destination: `${BACKEND_URL}/api/mock-trial/:path*`,
      },
      // /api/content-marketing/keywords/collect/stream 은 Next.js API Route에서 SSE 프록시 처리
      // /api/content-marketing/script/generate 도 API Route에서 SSE 프록시 처리
      // /api/content-marketing/script/webtoon/{jobId}/stream 도 API Route에서 SSE 프록시 처리
      // (rewrites는 SSE 스트리밍을 버퍼링하므로 API Route 사용)
      // Next.js는 API Route가 rewrites보다 우선이므로 아래 rewrites는 나머지 경로에만 적용됨
      {
        source: '/api/content-marketing/:path*',
        destination: `${BACKEND_URL}/api/content-marketing/:path*`,
      },
      {
        source: '/api/workspace/:path*',
        destination: `${BACKEND_URL}/api/workspace/:path*`,
      },
      {
        source: '/api/chat/conversations/:path*',
        destination: `${BACKEND_URL}/api/chat/conversations/:path*`,
      },
      {
        source: '/api/chat/conversations',
        destination: `${BACKEND_URL}/api/chat/conversations`,
      },
      {
        source: '/api/legal-news/:path*',
        destination: `${BACKEND_URL}/api/legal-news/:path*`,
      },
      // /api/chat/stream은 Next.js API Route에서 SSE 프록시 처리
      // (rewrites는 SSE 스트리밍을 버퍼링하므로 API Route 사용)
      {
        source: '/api/chat',
        destination: `${BACKEND_URL}/api/chat`,
      },
      {
        source: '/media/:path*',
        destination: `${BACKEND_URL}/media/:path*`,
      },
    ]
  },
}

module.exports = nextConfig
