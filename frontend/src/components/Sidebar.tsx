'use client'

import React, { useState, useMemo } from 'react'
import Link from 'next/link'
import { usePathname, useRouter, useSearchParams } from 'next/navigation'
import { motion, AnimatePresence } from 'framer-motion'
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

const CATEGORY_ICONS: Record<string, any> = {
  'research': Search,
  'case-review': ClipboardList,
  'insight': BarChart,
  'study': BookOpen,
  'problem-solving': HelpCircle,
  'case-management': Layers,
  'information': FileText,
}

const MODULE_ICONS: Record<string, any> = {
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
  const { userRole, setUserRole, sessionData } = useChat()
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
    <motion.aside
      initial={false}
      animate={{ width: isExpanded ? 260 : 64 }}
      onMouseEnter={() => setIsExpanded(true)}
      onMouseLeave={() => setIsExpanded(false)}
      className="fixed left-0 top-0 h-screen bg-white/90 backdrop-blur-xl border-r border-[#D2D2D7]/30 z-50 flex flex-col transition-all duration-300 ease-in-out"
    >
      {/* Logo Area */}
      <div className="h-16 flex items-center px-4 mb-4 overflow-hidden shrink-0">
        <Link href={`/?role=${userRole}`} className="flex items-center gap-3">
          <img src="/logo.png" alt="Legal President AI" className="w-8 h-8 shrink-0 object-contain" />
          <motion.span
            animate={{ opacity: isExpanded ? 1 : 0 }}
            className="font-bold text-[#1D1D1F] whitespace-nowrap"
          >
            LEGAL PRESIDENT AI
          </motion.span>
        </Link>
      </div>

      {/* Navigation Groups */}
      <div className="flex-1 overflow-y-auto overflow-x-hidden px-2 space-y-6 pt-2">
        {/* Dashboard Link */}
        <NavItem
          href={`/?role=${userRole}`}
          icon={LayoutDashboard}
          label="대시보드"
          isActive={pathname === '/'}
          isExpanded={isExpanded}
        />

        {/* Live Agent Status (New) */}
        {sessionData.active_agent && (
          <div className="px-3 py-2 bg-blue-600/5 rounded-xl border border-blue-600/10 mx-1">
            <div className="flex items-center gap-2 mb-1.5">
              <div className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse" />
              <span className={cn(
                "text-[10px] font-bold text-blue-600 uppercase tracking-wider transition-opacity",
                !isExpanded && "opacity-0 invisible h-0"
              )}>
                Active Agent
              </span>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-white shadow-sm flex items-center justify-center shrink-0 border border-blue-100">
                <span className="text-sm">🤖</span>
              </div>
              {isExpanded && (
                <div className="min-w-0">
                  <p className="text-xs font-bold text-[#1D1D1F] truncate">
                    {modules.find(m => m.id === sessionData.active_agent?.replace('_', '-'))?.name ||
                      sessionData.active_agent?.replace('_', ' ').toUpperCase()}
                  </p>
                  <p className="text-[10px] text-[#86868B]">
                    {sessionData.step || '대기 중...'}
                  </p>
                </div>
              )}
            </div>
          </div>
        )}

        {categories.map(cat => (
          <div key={cat} className="space-y-1">
            <motion.div
              animate={{ opacity: isExpanded ? 1 : 0, height: isExpanded ? 'auto' : 0 }}
              className="px-3"
            >
              {isExpanded && (
                <span className="text-[10px] font-bold text-[#86868B] uppercase tracking-wider">
                  {CATEGORY_NAMES[cat]?.[userRole] || cat}
                </span>
              )}
            </motion.div>

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
          <AnimatePresence>
            {isExpanded && (
              <motion.span
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="text-sm font-medium whitespace-nowrap"
              >
                나가기
              </motion.span>
            )}
          </AnimatePresence>
        </button>
      </div>
    </motion.aside >
  )
}

function NavItem({
  href,
  icon: Icon,
  label,
  isActive,
  isExpanded
}: {
  href: string,
  icon: any,
  label: string,
  isActive: boolean,
  isExpanded: boolean
}) {
  return (
    <Link
      href={href}
      className={cn(
        "group flex items-center gap-3 p-3 rounded-xl transition-all relative",
        isActive
          ? "bg-blue-600/10 text-blue-600 shadow-sm shadow-blue-600/5"
          : "text-[#424245] hover:bg-black/[0.03] hover:text-[#1D1D1F]"
      )}
    >
      <Icon size={20} className={cn("shrink-0 transition-transform group-hover:scale-110", isActive && "text-blue-600")} />

      <AnimatePresence>
        {isExpanded && (
          <motion.span
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -10 }}
            className="text-sm font-medium whitespace-nowrap overflow-hidden text-ellipsis"
          >
            {label}
          </motion.span>
        )}
      </AnimatePresence>

      {/* Active Indicator */}
      {isActive && (
        <motion.div
          layoutId="active-nav"
          className="absolute left-0 w-1 h-6 bg-blue-600 rounded-r-full"
        />
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
