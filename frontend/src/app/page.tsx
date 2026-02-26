'use client'

import { useEffect, Suspense, useMemo } from 'react'
import Link from 'next/link'
import { useRouter, useSearchParams } from 'next/navigation'
import { getEnabledModules } from '@/lib/modules'
import { useUI } from '@/context/UIContext'
import { useChat, UserRole } from '@/context/ChatContext'

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
    <main className="min-h-screen bg-white p-8 relative overflow-hidden transition-all duration-500 ease-in-out">
      <div
        className={`relative z-10 transition-all duration-500 ease-in-out ${
          isChatOpen ? 'w-1/2 pr-8' : 'w-full max-w-6xl mx-auto'
        }`}
      >
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-12 gap-6">
          <div>
            <h1 className="text-3xl font-bold text-[#1D1D1F] mb-2 tracking-tight">
              LEGAL <span className="text-blue-500">PRESIDENT AI</span>
            </h1>
            <p className="text-blue-500/80 font-medium">
              {role === 'lawyer' ? '변호사님 전용 대시보드' : '사용자 맞춤형 도움 서비스'}
            </p>
          </div>
          <button
            onClick={handleResetRole}
            className="px-5 py-2.5 text-sm font-semibold text-[#86868B] hover:text-[#1D1D1F] bg-[#F5F5F7] hover:bg-gray-200 border border-black/[0.06] rounded-xl transition-all duration-300 cursor-pointer"
          >
            ← 역할 변경
          </button>
        </div>

        <div className={`grid gap-5 ${isChatOpen ? 'grid-cols-1 md:grid-cols-2 xl:grid-cols-3' : 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3'}`}>
          {enabledModules.map((module) => (
            <Link
              key={module.id}
              href={module.href}
              className="group relative block p-6 bg-[#F5F5F7] border border-black/[0.06] rounded-2xl hover:bg-blue-50 hover:border-blue-200 transition-all duration-300 overflow-hidden cursor-pointer"
            >
              {/* Subtle card glow on hover */}
              <div className="absolute inset-0 bg-blue-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />

              <div className="relative z-10">
                <div className="text-4xl mb-4 group-hover:scale-110 group-hover:-rotate-3 transition-transform duration-300">
                  {module.icon}
                </div>
                <h2 className="text-xl font-bold text-[#1D1D1F] mb-2 group-hover:text-blue-500 transition-colors">
                  {module.name}
                </h2>
                <p className="text-[#86868B] text-sm leading-relaxed group-hover:text-[#3C3C43] transition-colors">
                  {module.description}
                </p>

                <div className="mt-4 flex items-center text-xs font-bold uppercase tracking-wider text-blue-500/70 group-hover:text-blue-500">
                  Explore <span className="ml-1 group-hover:translate-x-1 transition-transform">→</span>
                </div>
              </div>
            </Link>
          ))}
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
