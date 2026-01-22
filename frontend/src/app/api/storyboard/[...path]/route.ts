import { NextRequest, NextResponse } from 'next/server'

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000'

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const { path } = await params
  const targetPath = path.join('/')
  const url = `${BACKEND_URL}/api/storyboard/${targetPath}`

  try {
    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    })

    const data = await response.json()
    return NextResponse.json(data, { status: response.status })
  } catch (error) {
    console.error('Proxy GET error:', error)
    return NextResponse.json(
      { error: 'Backend connection failed' },
      { status: 502 }
    )
  }
}

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const { path } = await params
  const targetPath = path.join('/')
  const url = `${BACKEND_URL}/api/storyboard/${targetPath}`

  try {
    const contentType = request.headers.get('content-type') || ''

    let body: BodyInit | undefined
    let headers: Record<string, string> = {}

    if (contentType.includes('multipart/form-data')) {
      // 파일 업로드 처리
      body = await request.formData()
    } else {
      // JSON 처리
      body = JSON.stringify(await request.json())
      headers['Content-Type'] = 'application/json'
    }

    const response = await fetch(url, {
      method: 'POST',
      headers,
      body,
    })

    const data = await response.json()
    return NextResponse.json(data, { status: response.status })
  } catch (error) {
    console.error('Proxy POST error:', error)
    return NextResponse.json(
      { error: 'Backend connection failed' },
      { status: 502 }
    )
  }
}
