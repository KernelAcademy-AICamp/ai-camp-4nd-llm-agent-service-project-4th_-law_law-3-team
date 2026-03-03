'use client'

import React from 'react'
import { usePathname, useSearchParams } from 'next/navigation'
import { cn } from '@/lib/utils'

export default function LayoutWrapper({ children }: { children: React.ReactNode }) {
  const pathname = usePathname()
  const searchParams = useSearchParams()
  const role = searchParams.get('role')

  // 히어로 페이지(역할 선택 전) 여부 확인
  const isHeroPage = pathname === '/' && !role

  return (
    <div className={cn(
      "flex-1 min-w-0 transition-all duration-300 ease-in-out",
      !isHeroPage && "pl-[64px]"
    )}>
      {children}
    </div>
  )
}
