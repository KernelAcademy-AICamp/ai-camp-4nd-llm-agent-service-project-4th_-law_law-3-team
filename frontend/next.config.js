/** @type {import('next').NextConfig} */
const nextConfig = {
  // Enable standalone output for Docker production builds
  output: 'standalone',

  // 프록시 타임아웃 설정 (LLM 응답 대기)
  experimental: {
    proxyTimeout: 120000, // 120초
  },

  async rewrites() {
    return [
      // storyboard는 API Route로 프록시 (Next.js rewrites 버그 우회)
      // 다른 모듈들은 기존 rewrites 사용
      {
        source: '/api/lawyer-finder/:path*',
        destination: 'http://127.0.0.1:8000/api/lawyer-finder/:path*',
      },
      {
        source: '/api/lawyer-stats/:path*',
        destination: 'http://127.0.0.1:8000/api/lawyer-stats/:path*',
      },
      {
        source: '/api/case-precedent/:path*',
        destination: 'http://127.0.0.1:8000/api/case-precedent/:path*',
      },
      {
        source: '/api/law-study/:path*',
        destination: 'http://127.0.0.1:8000/api/law-study/:path*',
      },
      {
        source: '/api/small-claims/:path*',
        destination: 'http://127.0.0.1:8000/api/small-claims/:path*',
      },
      // /api/chat/stream은 Next.js API Route에서 SSE 프록시 처리
      // (rewrites는 SSE 스트리밍을 버퍼링하므로 API Route 사용)
      {
        source: '/api/chat',
        destination: 'http://127.0.0.1:8000/api/chat',
      },
      {
        source: '/media/:path*',
        destination: 'http://127.0.0.1:8000/media/:path*',
      },
    ]
  },
}

module.exports = nextConfig
