import { NextRequest, NextResponse } from 'next/server'

const BACKEND_URL = process.env.BACKEND_URL || 'http://127.0.0.1:8000'

/**
 * 인증 API 프록시 — /api/auth/* 요청을 백엔드로 전달
 *
 * Set-Cookie 헤더를 올바르게 전달하기 위해 rewrites 대신 API Route 사용.
 * (rewrites는 Set-Cookie를 클라이언트로 전달하지 않는 경우가 있음)
 */
async function proxyRequest(request: NextRequest): Promise<NextResponse> {
  const path = request.nextUrl.pathname
  const search = request.nextUrl.search
  const targetUrl = `${BACKEND_URL}${path}${search}`

  const headers = new Headers()
  headers.set('Content-Type', request.headers.get('Content-Type') || 'application/json')

  // 쿠키 전달
  const cookie = request.headers.get('Cookie')
  if (cookie) {
    headers.set('Cookie', cookie)
  }

  // API 키 전달
  const apiKey = request.headers.get('X-API-Key')
  if (apiKey) {
    headers.set('X-API-Key', apiKey)
  }

  // User-Agent 전달
  const userAgent = request.headers.get('User-Agent')
  if (userAgent) {
    headers.set('User-Agent', userAgent)
  }

  // X-Forwarded-For 전달
  const forwardedFor = request.headers.get('X-Forwarded-For') || request.headers.get('x-real-ip')
  if (forwardedFor) {
    headers.set('X-Forwarded-For', forwardedFor)
  }

  const fetchOptions: RequestInit = {
    method: request.method,
    headers,
  }

  if (request.method !== 'GET' && request.method !== 'HEAD') {
    fetchOptions.body = await request.text()
  }

  const backendResponse = await fetch(targetUrl, fetchOptions)

  const responseHeaders = new Headers()

  // Set-Cookie 헤더 전달 (인증 쿠키)
  const setCookieHeaders = backendResponse.headers.getSetCookie()
  for (const setCookie of setCookieHeaders) {
    responseHeaders.append('Set-Cookie', setCookie)
  }

  responseHeaders.set('Content-Type', backendResponse.headers.get('Content-Type') || 'application/json')

  const body = await backendResponse.text()

  return new NextResponse(body, {
    status: backendResponse.status,
    headers: responseHeaders,
  })
}

export async function GET(request: NextRequest): Promise<NextResponse> {
  return proxyRequest(request)
}

export async function POST(request: NextRequest): Promise<NextResponse> {
  return proxyRequest(request)
}

export async function PATCH(request: NextRequest): Promise<NextResponse> {
  return proxyRequest(request)
}

export async function PUT(request: NextRequest): Promise<NextResponse> {
  return proxyRequest(request)
}

export async function DELETE(request: NextRequest): Promise<NextResponse> {
  return proxyRequest(request)
}
