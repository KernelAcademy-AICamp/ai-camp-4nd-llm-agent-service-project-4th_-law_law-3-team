'use client'

import React, { useState, useMemo } from 'react'
import Link from 'next/link'
import { usePathname, useRouter } from 'next/navigation'
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
  Users,
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
import { getEnabledModules, getModuleCategory, CATEGORY_NAMES } from '@/lib/modules'
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
  'mock-trial': Users,
  'content-marketing': Video,
  'workspace': Briefcase,
  'legal-news': LayoutDashboard,
}

export default function Sidebar() {
  const pathname = usePathname()
  const router = useRouter()
  const { userRole, setUserRole } = useChat()
  const { setChatOpen } = useUI()
  const [isExpanded, setIsExpanded] = useState(false)

  // 히어로 페이지에서는 사이드바 숨김
  const isHeroPage = pathname === '/' && !new URLSearchParams(typeof window !== 'undefined' ? window.location.search : '').get('role')

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
      className="fixed left-0 top-0 h-screen bg-[#F5F5F7]/80 backdrop-blur-xl border-r border-[#D2D2D7]/30 z-50 flex flex-col transition-all duration-300 ease-in-out"
    >
      {/* Logo Area */}
      <div className="h-16 flex items-center px-4 mb-4 overflow-hidden shrink-0">
        <Link href={`/?role=${userRole}`} className="flex items-center gap-3">
          <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center shrink-0 shadow-sm">
            <span className="text-white font-bold text-xs">LP</span>
          </div>
          <motion.span
            animate={{ opacity: isExpanded ? 1 : 0 }}
            className="font-bold text-[#1D1D1F] whitespace-nowrap"
          >
            LEGAL AI
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
                  isActive={pathname === mod.href.split('?')[0]}
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
                역할 종료
              </motion.span>
            )}
          </AnimatePresence>
        </button>
      </div>
    </motion.aside>
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
