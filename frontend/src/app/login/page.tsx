'use client'

import Link from 'next/link'
import { useRouter, useSearchParams } from 'next/navigation'
import { useState } from 'react'
import { useAuth } from '@/context/AuthContext'
import { getGoogleOAuthUrl, getKakaoOAuthUrl } from '@/features/auth/services'

export default function LoginPage() {
  const { login } = useAuth()
  const router = useRouter()
  const searchParams = useSearchParams()
  const returnUrl = searchParams.get('returnUrl') || '/'

  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setIsSubmitting(true)

    try {
      await login(email, password)
      router.replace(returnUrl)
    } catch {
      setError('이메일 또는 비밀번호가 올바르지 않습니다.')
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleGoogleLogin = async () => {
    try {
      const { url } = await getGoogleOAuthUrl()
      window.location.href = url
    } catch {
      setError('Google 로그인을 시작할 수 없습니다.')
    }
  }

  const handleKakaoLogin = async () => {
    try {
      const { url } = await getKakaoOAuthUrl()
      window.location.href = url
    } catch {
      setError('카카오 로그인을 시작할 수 없습니다.')
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-50 px-4">
      <div className="w-full max-w-md space-y-6">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-gray-900">로그인</h1>
          <p className="mt-2 text-sm text-gray-600">법률 서비스 플랫폼에 로그인하세요</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4 rounded-lg bg-white p-6 shadow">
          {error && (
            <div className="rounded-md bg-red-50 p-3 text-sm text-red-700">{error}</div>
          )}

          <div>
            <label htmlFor="email" className="block text-sm font-medium text-gray-700">
              이메일
            </label>
            <input
              id="email"
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
              placeholder="example@email.com"
            />
          </div>

          <div>
            <label htmlFor="password" className="block text-sm font-medium text-gray-700">
              비밀번호
            </label>
            <input
              id="password"
              type="password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
              placeholder="8자 이상"
            />
          </div>

          <button
            type="submit"
            disabled={isSubmitting}
            className="w-full rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
          >
            {isSubmitting ? '로그인 중...' : '로그인'}
          </button>
        </form>

        <div className="space-y-3">
          <div className="relative">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-gray-300" />
            </div>
            <div className="relative flex justify-center text-sm">
              <span className="bg-gray-50 px-2 text-gray-500">또는</span>
            </div>
          </div>

          <button
            onClick={handleGoogleLogin}
            className="flex w-full items-center justify-center gap-2 rounded-md border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50"
          >
            <svg className="h-5 w-5" viewBox="0 0 24 24">
              <path
                fill="#4285F4"
                d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z"
              />
              <path
                fill="#34A853"
                d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
              />
              <path
                fill="#FBBC05"
                d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
              />
              <path
                fill="#EA4335"
                d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
              />
            </svg>
            Google로 로그인
          </button>

          <button
            onClick={handleKakaoLogin}
            className="flex w-full items-center justify-center gap-2 rounded-md bg-[#FEE500] px-4 py-2 text-sm font-medium text-[#191919] hover:bg-[#FDD800]"
          >
            <svg className="h-5 w-5" viewBox="0 0 24 24">
              <path
                fill="#191919"
                d="M12 3C6.48 3 2 6.36 2 10.5c0 2.67 1.77 5.02 4.44 6.34-.14.52-.92 3.37-.96 3.58 0 0-.02.16.08.22.1.06.22.01.22.01.29-.04 3.37-2.21 3.9-2.58.73.11 1.49.17 2.27.17 5.52 0 10-3.36 10-7.5S17.52 3 12 3"
              />
            </svg>
            카카오로 로그인
          </button>
        </div>

        <p className="text-center text-sm text-gray-600">
          계정이 없으신가요?{' '}
          <Link href="/register" className="font-medium text-blue-600 hover:text-blue-500">
            회원가입
          </Link>
        </p>
      </div>
    </div>
  )
}
