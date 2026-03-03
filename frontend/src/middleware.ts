/**
 * Next.js Middleware
 *
 * rewrites로 프록시되는 /api/* 요청에 X-API-Key 헤더를 주입합니다.
 * API Route 핸들러는 middleware 이전에 매칭되므로 별도로 헤더를 추가합니다.
 */

import { NextResponse } from 'next/server'
import type { NextRequest } from 'next/server'

export function middleware(request: NextRequest) {
  const apiKey = process.env.API_KEY || ''

  if (!apiKey) {
    return NextResponse.next()
  }

  const requestHeaders = new Headers(request.headers)
  requestHeaders.set('X-API-Key', apiKey)

  return NextResponse.next({
    request: { headers: requestHeaders },
  })
}

export const config = {
  matcher: '/api/:path*',
}
