'use client'

import { useEffect, Suspense, useMemo } from 'react'
import Link from 'next/link'
import { useRouter, useSearchParams } from 'next/navigation'
import { getEnabledModules } from '@/lib/modules'
import { useUI } from '@/context/UIContext'
import { useChat, UserRole } from '@/context/ChatContext'
import {
  MapPin, BarChart3, BookOpen, Book, Clapperboard,
  GraduationCap, Link2, Scale, Landmark,
  Briefcase, Building, User, ArrowLeft, ChevronRight,
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'

const MODULE_ICONS: Record<string, LucideIcon> = {
  'lawyer-finder': MapPin,
  'lawyer-stats': BarChart3,
  'case-precedent': BookOpen,
  'law-search': Book,
  'storyboard': Clapperboard,
  'law-study': GraduationCap,
  'statute-hierarchy': Link2,
  'small-claims': Scale,
  'mock-trial': Landmark,
}

function HomeContent() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const role = searchParams.get('role') as 'lawyer' | 'user' | null

  const { isChatOpen, setChatOpen } = useUI()
  const { setUserRole } = useChat()
  const enabledModules = useMemo(() => getEnabledModules(role || undefined), [role])

  useEffect(() => {
    if (!role) {
      setChatOpen(false)
    } else {
      setChatOpen(true)
      setUserRole(role as UserRole)
    }
  }, [role, setChatOpen, setUserRole])

  const handleRoleSelect = (selectedRole: 'lawyer' | 'user') => {
    setUserRole(selectedRole)
    setChatOpen(true)
    router.push(`/?role=${selectedRole}`)
  }

  const handleResetRole = () => {
    setUserRole('user')
    setChatOpen(false)
    router.push('/')
  }

  if (!role) {
    return (
      <main className="h-screen bg-white flex flex-col items-center justify-end relative overflow-hidden">
        {/* Full-screen Background Image */}
        <img
          src="/assets/hero-background.png"
          alt="법률 서비스 플랫폼"
          className="absolute inset-0 w-full h-full object-contain object-center"
        />

        {/* Role Selection Cards */}
        <div className="relative z-10 w-full max-w-7xl mx-auto px-6 pb-8">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Lawyer Card */}
            <button
              onClick={() => handleRoleSelect('lawyer')}
              className="group relative p-8 rounded-3xl bg-white/70 backdrop-blur-xl shadow-apple hover:shadow-apple-hover hover:bg-white/85 transition-all duration-500 text-left overflow-hidden hover:-translate-y-2 cursor-pointer"
            >
              <div className="absolute top-0 right-0 p-6 opacity-[0.08] group-hover:opacity-[0.12] transition-opacity">
                <span className="text-8xl">⚖️</span>
              </div>
              <div className="relative z-10">
                <div className="text-5xl mb-6">👨‍💼</div>
                <h2 className="text-2xl font-bold text-apple-text mb-4 tracking-tight">법 관련 종사자입니다</h2>
                <p className="text-apple-secondary mb-8 leading-relaxed text-base">의뢰인과 연결되고, 전문성을 발휘하여 업무를 관리하세요.</p>
                <div className="inline-flex items-center text-apple-blue font-bold group-hover:gap-4 gap-2 transition-all text-base">
                  대시보드 입장 <span className="text-xl">→</span>
                </div>
              </div>
            </button>

            {/* Public Institution Card */}
            <button
              className="group relative p-8 rounded-3xl bg-white/50 backdrop-blur-xl shadow-apple-sm transition-all duration-500 text-left overflow-hidden cursor-default"
            >
              <div className="absolute top-0 right-0 p-6 opacity-[0.06]">
                <span className="text-8xl">🏛️</span>
              </div>
              <div className="relative z-10 opacity-60">
                <div className="text-5xl mb-6">🏢</div>
                <h2 className="text-2xl font-bold text-apple-text mb-4 tracking-tight">공공기관입니다</h2>
                <p className="text-apple-secondary mb-8 leading-relaxed text-base">공공 업무 효율화를 위한 맞춤형 법률 AI 솔루션을 활용하세요.</p>
                <div className="inline-flex items-center text-apple-secondary font-bold gap-2 text-base">
                  서비스 준비중 <span className="text-xl">🔒</span>
                </div>
              </div>
            </button>

            {/* User Card */}
            <button
              onClick={() => handleRoleSelect('user')}
              className="group relative p-8 rounded-3xl bg-white/70 backdrop-blur-xl shadow-apple hover:shadow-apple-hover hover:bg-white/85 transition-all duration-500 text-left overflow-hidden hover:-translate-y-2 cursor-pointer"
            >
              <div className="absolute top-0 right-0 p-6 opacity-[0.08] group-hover:opacity-[0.12] transition-opacity">
                <span className="text-8xl">🤝</span>
              </div>
              <div className="relative z-10">
                <div className="text-5xl mb-6">👤</div>
                <h2 className="text-2xl font-bold text-apple-text mb-4 tracking-tight">일반인입니다</h2>
                <p className="text-apple-secondary mb-8 leading-relaxed text-base">나에게 딱 맞는 법률 전문가를 찾고 사건을 해결하세요.</p>
                <div className="inline-flex items-center text-apple-blue font-bold group-hover:gap-4 gap-2 transition-all text-base">
                  도움 받기 <span className="text-xl">→</span>
                </div>
              </div>
            </button>
          </div>
        </div>
      </main>
    )
  }

  return (
    <main className="min-h-screen bg-apple-bg relative transition-all duration-500 ease-in-out">
      <div
        className={`transition-all duration-500 ease-in-out ${
          isChatOpen ? 'w-1/2' : 'w-full'
        }`}
      >
        {/* Header */}
        <header className="bg-white/80 backdrop-blur-xl border-b border-black/[0.06] sticky top-0 z-20">
          <div className={`flex items-center justify-between px-8 py-4 ${isChatOpen ? '' : 'max-w-5xl mx-auto'}`}>
            <div>
              <h1 className="text-[20px] font-semibold text-apple-text tracking-tight">
                법률 서비스
              </h1>
              <p className="text-[13px] text-apple-secondary mt-0.5">
                {role === 'lawyer' ? '변호사님 전용 대시보드' : '사용자 맞춤형 도움 서비스'}
              </p>
            </div>
            <button
              onClick={handleResetRole}
              className="inline-flex items-center gap-1.5 px-4 py-2 text-[13px] font-medium text-apple-blue hover:bg-blue-50 rounded-lg transition-colors duration-200 cursor-pointer"
            >
              <ArrowLeft size={15} />
              역할 변경
            </button>
          </div>
        </header>

        {/* Module Grid */}
        <div className={`p-8 ${isChatOpen ? '' : 'max-w-5xl mx-auto'}`}>
          <div className={`grid gap-4 ${isChatOpen ? 'grid-cols-1 md:grid-cols-2 xl:grid-cols-3' : 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3'}`}>
            {enabledModules.map((module) => {
              const IconComponent = MODULE_ICONS[module.id]
              return (
                <Link
                  key={module.id}
                  href={module.href}
                  className="group block p-6 bg-white rounded-2xl shadow-apple-sm hover:shadow-apple transition-all duration-300 cursor-pointer hover:-translate-y-0.5"
                >
                  <div className="w-10 h-10 rounded-[10px] bg-apple-bg group-hover:bg-blue-50 flex items-center justify-center mb-4 transition-colors duration-300">
                    {IconComponent ? (
                      <IconComponent className="text-apple-secondary group-hover:text-apple-blue transition-colors duration-300" size={20} />
                    ) : (
                      <span className="text-lg">{module.icon}</span>
                    )}
                  </div>
                  <h2 className="text-[15px] font-semibold text-apple-text mb-1 group-hover:text-apple-blue transition-colors duration-200">
                    {module.name}
                  </h2>
                  <p className="text-[13px] text-apple-secondary leading-relaxed">
                    {module.description}
                  </p>
                </Link>
              )
            })}
          </div>
        </div>
      </div>
    </main>
  )
}

export default function Home() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-white" />}>
      <HomeContent />
    </Suspense>
  )
}
