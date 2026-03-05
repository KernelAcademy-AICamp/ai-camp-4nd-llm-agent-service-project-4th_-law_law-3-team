'use client'

import dynamic from 'next/dynamic'
import { usePathname, useRouter } from 'next/navigation'
import { Suspense, useEffect } from 'react'
import Sidebar from '@/components/Sidebar'
import LayoutWrapper from '@/components/LayoutWrapper'
import { useAuth } from '@/context/AuthContext'

const ChatWidget = dynamic(() => import('@/components/ChatWidget'), {
  ssr: false,
})

const PUBLIC_PATHS = new Set(['/login', '/register'])

interface ProtectedRouteProps {
  children: React.ReactNode
}

export function ProtectedRoute({ children }: ProtectedRouteProps) {
  const { isAuthenticated, isLoading } = useAuth()
  const router = useRouter()
  const pathname = usePathname()

  const isPublicPage = PUBLIC_PATHS.has(pathname)

  useEffect(() => {
    if (!isPublicPage && !isLoading && !isAuthenticated) {
      const returnUrl = encodeURIComponent(pathname)
      router.replace(`/login?returnUrl=${returnUrl}`)
    }
  }, [isAuthenticated, isLoading, router, isPublicPage, pathname])

  // 공개 페이지 (로그인, 회원가입): Sidebar 없는 깔끔한 레이아웃
  if (isPublicPage) return <>{children}</>

  if (isLoading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-gray-300 border-t-blue-600" />
      </div>
    )
  }

  if (!isAuthenticated) {
    return null
  }

  // 인증된 페이지: Sidebar + ChatWidget 포함 전체 레이아웃
  return (
    <>
      <div className="flex min-h-screen relative">
        <Suspense fallback={null}>
          <Sidebar />
        </Suspense>
        <Suspense fallback={null}>
          <LayoutWrapper>
            {children}
          </LayoutWrapper>
        </Suspense>
      </div>
      <ChatWidget />
    </>
  )
}
