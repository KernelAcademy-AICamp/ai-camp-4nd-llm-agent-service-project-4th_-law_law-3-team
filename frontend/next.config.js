/** @type {import('next').NextConfig} */
const nextConfig = {
  // Enable standalone output for Docker production builds
  output: 'standalone',

  async rewrites() {
    return [
      // storyboard는 API Route로 프록시 (Next.js rewrites 버그 우회)
      // 다른 모듈들은 기존 rewrites 사용
      {
        source: '/api/lawyer-finder/:path*',
        destination: 'http://localhost:8000/api/lawyer-finder/:path*',
      },
      {
        source: '/api/lawyer-stat/:path*',
        destination: 'http://localhost:8000/api/lawyer-stat/:path*',
      },
      {
        source: '/api/case-precedent/:path*',
        destination: 'http://localhost:8000/api/case-precedent/:path*',
      },
      {
        source: '/api/review-price/:path*',
        destination: 'http://localhost:8000/api/review-price/:path*',
      },
      {
        source: '/api/law-study/:path*',
        destination: 'http://localhost:8000/api/law-study/:path*',
      },
      {
        source: '/api/small-claims/:path*',
        destination: 'http://localhost:8000/api/small-claims/:path*',
      },
      {
        source: '/api/multi-agent/:path*',
        destination: 'http://localhost:8000/api/multi-agent/:path*',
      },
      {
        source: '/media/:path*',
        destination: 'http://localhost:8000/media/:path*',
      },
    ]
  },
}

module.exports = nextConfig
