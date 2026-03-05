'use client'

import React, { useState, useMemo } from 'react'
import Link from 'next/link'
import Image from 'next/image'
import { usePathname, useRouter, useSearchParams } from 'next/navigation'
import type { LucideIcon } from 'lucide-react'
import {
  Search,
  ClipboardList,
  BarChart,
  BookOpen,
  HelpCircle,
  Settings,
  LogOut,
  ChevronRight,
  ChevronLeft,
  LayoutDashboard,
  Gavel,
  AlertCircle,
  FileText,
  Video,
  Briefcase,
  Layers,
  MapPin,
  Scale
} from 'lucide-react'
import { useChat } from '@/context/ChatContext'
import { useUI } from '@/context/UIContext'
import { modules, getEnabledModules, getModuleCategory, CATEGORY_NAMES } from '@/lib/modules'
import { cn } from '@/lib/utils'

const CATEGORY_ICONS: Record<string, LucideIcon> = {
  'research': Search,
  'case-review': ClipboardList,
  'insight': BarChart,
  'study': BookOpen,
  'problem-solving': HelpCircle,
  'case-management': Layers,
  'information': FileText,
}

const MODULE_ICONS: Record<string, LucideIcon> = {
  'lawyer-finder': MapPin,
  'lawyer-stats': BarChart,
  'case-precedent': Search,
  'law-search': FileText,
  'storyboard': Video,
  'law-study': BookOpen,
  'statute-hierarchy': Layers,
  'small-claims': Scale,
  'mock-trial': Gavel,
  'content-marketing': Video,
  'workspace': Briefcase,
  'legal-news': LayoutDashboard,
}

export default function Sidebar() {
  const pathname = usePathname()
  const router = useRouter()
  const { userRole, setUserRole, sessionData, resetSession } = useChat()
  const { setChatOpen } = useUI()
  const [isExpanded, setIsExpanded] = useState(false)

  const searchParams = useSearchParams()
  // 히어로 페이지에서는 사이드바 숨김
  const isHeroPage = pathname === '/' && !searchParams.get('role')

  const enabledModules = useMemo(() => getEnabledModules(userRole), [userRole])

  const modulesByCategory = useMemo(() => {
    const grouped: Record<string, typeof enabledModules> = {}
    enabledModules.forEach(mod => {
      const cat = getModuleCategory(mod, userRole)
      if (!grouped[cat]) grouped[cat] = []
      grouped[cat].push(mod)
    })
    return grouped
  }, [enabledModules, userRole])

  const categories = Object.keys(modulesByCategory)

  if (isHeroPage) return null

  return (
    <aside
      style={{ width: isExpanded ? 260 : 64 }}
      onMouseEnter={() => setIsExpanded(true)}
      onMouseLeave={() => setIsExpanded(false)}
      className="fixed left-0 top-0 h-screen bg-white/90 backdrop-blur-xl border-r border-[#D2D2D7]/30 z-50 flex flex-col transition-[width] duration-300 ease-in-out"
    >
      {/* Logo Area */}
      <div className="h-16 flex items-center px-4 mb-4 overflow-hidden shrink-0">
        <Link href={`/?role=${userRole}`} className="flex items-center gap-3">
          <Image src="/logo.png" alt="Legal President AI" width={32} height={32} className="shrink-0 object-contain" />
          <span
            className={cn(
              "font-bold text-[#1D1D1F] whitespace-nowrap transition-opacity duration-200",
              isExpanded ? "opacity-100" : "opacity-0"
            )}
          >
            LEGAL PRESIDENT AI
          </span>
        </Link>
      </div>

      {/* Navigation Groups */}
      <div className="flex-1 overflow-y-auto overflow-x-hidden px-2 space-y-6 pt-2">
        {categories.map(cat => (
          <div key={cat} className="space-y-1">
            <div
              className={cn(
                "px-3 transition-all duration-200 overflow-hidden",
                isExpanded ? "opacity-100 h-auto" : "opacity-0 h-0"
              )}
            >
              {isExpanded && (
                <span className="text-[10px] font-bold text-[#86868B] uppercase tracking-wider">
                  {CATEGORY_NAMES[cat]?.[userRole] || cat}
                </span>
              )}
            </div>

            {modulesByCategory[cat].map(mod => {
              const Icon = MODULE_ICONS[mod.id] || Search
              return (
                <NavItem
                  key={mod.id}
                  href={mod.href}
                  icon={Icon}
                  label={mod.name}
                  isActive={pathname === mod.href}
                  isExpanded={isExpanded}
                  onNavigate={pathname !== mod.href ? resetSession : undefined}
                />
              )
            })}
          </div>
        ))}
      </div>

      {/* Bottom Actions */}
      <div className="p-2 border-t border-[#D2D2D7]/30 space-y-1">
        <button
          onClick={() => {
            router.push('/')
            setChatOpen(false)
          }}
          className="w-full flex items-center gap-3 p-3 rounded-xl text-[#86868B] hover:text-[#1D1D1F] hover:bg-black/[0.03] transition-colors"
        >
          <LogOut size={20} className="shrink-0" />
          <span
            className={cn(
              "text-sm font-medium whitespace-nowrap transition-opacity duration-200",
              isExpanded ? "opacity-100" : "opacity-0 w-0 overflow-hidden"
            )}
          >
            나가기
          </span>
        </button>
      </div>
    </aside>
  )
}

function NavItem({
  href,
  icon: Icon,
  label,
  isActive,
  isExpanded,
  onNavigate
}: {
  href: string,
  icon: LucideIcon,
  label: string,
  isActive: boolean,
  isExpanded: boolean,
  onNavigate?: () => void
}) {
  return (
    <Link
      href={href}
      onClick={onNavigate}
      className={cn(
        "group flex items-center gap-3 p-3 rounded-xl transition-all relative",
        isActive
          ? "bg-blue-600/10 text-blue-600 shadow-sm shadow-blue-600/5"
          : "text-[#424245] hover:bg-black/[0.03] hover:text-[#1D1D1F]"
      )}
    >
      <Icon size={20} className={cn("shrink-0 transition-transform group-hover:scale-110", isActive && "text-blue-600")} />

      <span
        className={cn(
          "text-sm font-medium whitespace-nowrap overflow-hidden transition-all duration-200",
          isExpanded ? "opacity-100 max-w-[180px]" : "opacity-0 max-w-0"
        )}
      >
        {label}
      </span>

      {/* Active Indicator */}
      {isActive && (
        <div className="absolute left-0 w-1 h-6 bg-blue-600 rounded-r-full" />
      )}

      {/* Tooltip for collapsed state */}
      {!isExpanded && (
        <div className="absolute left-16 px-2 py-1 bg-[#1D1D1F] text-white text-xs rounded opacity-0 invisible group-hover:visible group-hover:opacity-100 transition-all whitespace-nowrap z-50">
          {label}
        </div>
      )}
    </Link>
  )
}
