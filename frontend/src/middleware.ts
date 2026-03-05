/**
 * Next.js Middleware
 *
 * 1. 인증 체크: access_token 쿠키 없으면 /login 리다이렉트
 * 2. API 프록시: /api/* 요청에 X-API-Key 헤더 주입
 */

import { NextResponse } from 'next/server'
import type { NextRequest } from 'next/server'

const PUBLIC_PATHS = new Set(['/login', '/register'])

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl

  // 정적 리소스, 공개 경로 제외
  if (
    pathname.startsWith('/_next/') ||
    pathname.startsWith('/assets/') ||
    pathname === '/favicon.ico' ||
    pathname === '/logo.png' ||
    PUBLIC_PATHS.has(pathname)
  ) {
    return NextResponse.next()
  }

  // API 요청: X-API-Key 헤더 주입
  if (pathname.startsWith('/api/')) {
    const apiKey = process.env.API_KEY || ''
    if (apiKey) {
      const headers = new Headers(request.headers)
      headers.set('X-API-Key', apiKey)
      return NextResponse.next({ request: { headers } })
    }
    return NextResponse.next()
  }

  // 인증 체크: access_token 쿠키 없으면 /login으로 리다이렉트
  const accessToken = request.cookies.get('access_token')
  if (!accessToken) {
    const returnUrl = encodeURIComponent(pathname + request.nextUrl.search)
    return NextResponse.redirect(new URL(`/login?returnUrl=${returnUrl}`, request.url))
  }

  return NextResponse.next()
}

export const config = {
  matcher: ['/((?!_next/static|_next/image|favicon.ico).*)'],
}
