'use client'

import { usePathname, useRouter } from 'next/navigation'
import { useEffect } from 'react'
import { useAuth } from '@/context/AuthContext'

interface ProtectedRouteProps {
  children: React.ReactNode
}

export function ProtectedRoute({ children }: ProtectedRouteProps) {
  const { isAuthenticated, isLoading } = useAuth()
  const router = useRouter()
  const pathname = usePathname()

  const isPublicPage = pathname === '/login' || pathname === '/register'

  useEffect(() => {
    if (!isPublicPage && !isLoading && !isAuthenticated) {
      const returnUrl = encodeURIComponent(pathname)
      router.replace(`/login?returnUrl=${returnUrl}`)
    }
  }, [isAuthenticated, isLoading, router, isPublicPage, pathname])

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

  return <>{children}</>
}
