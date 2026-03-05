'use client'

import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { useEffect, useState } from 'react'
import { useAuth } from '@/context/AuthContext'

export default function RegisterPage() {
  const { register, isAuthenticated, isLoading } = useAuth()
  const router = useRouter()

  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [displayName, setDisplayName] = useState('')
  const [role, setRole] = useState<'user' | 'lawyer'>('user')
  const [error, setError] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)

  // 이미 인증된 상태면 홈으로 리다이렉트
  useEffect(() => {
    if (!isLoading && isAuthenticated) {
      router.replace('/')
    }
  }, [isLoading, isAuthenticated, router])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')

    if (password !== confirmPassword) {
      setError('비밀번호가 일치하지 않습니다.')
      return
    }

    if (password.length < 8) {
      setError('비밀번호는 8자 이상이어야 합니다.')
      return
    }

    setIsSubmitting(true)

    try {
      await register(email, password, displayName || undefined, role)
      router.replace('/')
    } catch {
      setError('이미 등록된 이메일이거나 회원가입에 실패했습니다.')
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-50 px-4">
      <div className="w-full max-w-md space-y-6">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-gray-900">회원가입</h1>
          <p className="mt-2 text-sm text-gray-600">법률 서비스 플랫폼에 가입하세요</p>
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
            <label htmlFor="displayName" className="block text-sm font-medium text-gray-700">
              표시 이름 <span className="text-gray-400">(선택)</span>
            </label>
            <input
              id="displayName"
              type="text"
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
              placeholder="닉네임"
              maxLength={100}
            />
          </div>

          <fieldset className="space-y-2">
            <legend className="block text-sm font-medium text-gray-700">역할 선택</legend>
            <div className="grid grid-cols-2 gap-3">
              <button
                type="button"
                onClick={() => setRole('user')}
                className={`rounded-lg border-2 p-3 text-left transition-all ${
                  role === 'user'
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-300 hover:border-gray-400'
                }`}
              >
                <div className="text-sm font-semibold text-gray-900">일반인</div>
                <div className="mt-1 text-xs text-gray-500">법률 상담 및 변호사 찾기</div>
              </button>
              <button
                type="button"
                onClick={() => setRole('lawyer')}
                className={`rounded-lg border-2 p-3 text-left transition-all ${
                  role === 'lawyer'
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-300 hover:border-gray-400'
                }`}
              >
                <div className="text-sm font-semibold text-gray-900">변호사</div>
                <div className="mt-1 text-xs text-gray-500">사건 관리 및 전문가 도구</div>
              </button>
            </div>
          </fieldset>

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
              minLength={8}
            />
          </div>

          <div>
            <label htmlFor="confirmPassword" className="block text-sm font-medium text-gray-700">
              비밀번호 확인
            </label>
            <input
              id="confirmPassword"
              type="password"
              required
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
              placeholder="비밀번호 재입력"
              minLength={8}
            />
          </div>

          <button
            type="submit"
            disabled={isSubmitting}
            className="w-full rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
          >
            {isSubmitting ? '가입 중...' : '회원가입'}
          </button>
        </form>

        <p className="text-center text-sm text-gray-600">
          이미 계정이 있으신가요?{' '}
          <Link href="/login" className="font-medium text-blue-600 hover:text-blue-500">
            로그인
          </Link>
        </p>
      </div>
    </div>
  )
}
