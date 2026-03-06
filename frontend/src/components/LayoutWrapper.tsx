'use client'

import React, { useEffect, useRef } from 'react'
import { usePathname, useSearchParams } from 'next/navigation'
import { cn } from '@/lib/utils'
import { useUI } from '@/context/UIContext'
import InquiryPanel from './chat/InquiryPanel'

export default function LayoutWrapper({ children }: { children: React.ReactNode }) {
  const pathname = usePathname()
  const searchParams = useSearchParams()
  const role = searchParams.get('role')
  const { activePanel, setActivePanel } = useUI()

  // 페이지 이동 시 activePanel 초기화 (stale panel 방지)
  const prevPathnameRef = useRef(pathname)
  useEffect(() => {
    if (prevPathnameRef.current !== pathname) {
      prevPathnameRef.current = pathname
      if (activePanel) {
        setActivePanel(null)
      }
    }
  }, [pathname, activePanel, setActivePanel])

  // 히어로 페이지(역할 선택 전) 여부 확인
  const isHeroPage = pathname === '/' && !role

  return (
    <div className={cn(
      "flex-1 min-w-0 transition-all duration-300 ease-in-out",
      !isHeroPage && "pl-[64px]"
    )}>
      {activePanel ? (
        <InquiryPanel panelId={activePanel} />
      ) : (
        children
      )}
    </div>
  )
}
