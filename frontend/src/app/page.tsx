'use client'

import { useEffect, Suspense, useMemo } from 'react'
import Link from 'next/link'
import { useRouter, useSearchParams } from 'next/navigation'
import { getEnabledModules, getModuleCategory, CATEGORY_NAMES } from '@/lib/modules'
import { useUI } from '@/context/UIContext'
import { useChat, UserRole } from '@/context/ChatContext'

function HomeContent() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const role = searchParams.get('role') as 'lawyer' | 'user' | null

  const { isChatOpen, setChatOpen } = useUI()
  const { setUserRole } = useChat()
  const enabledModules = useMemo(() => getEnabledModules(role || undefined), [role])

  const modulesByCategory = useMemo(() => {
    if (!role) return {}
    const grouped: Record<string, typeof enabledModules> = {}
    enabledModules.forEach(mod => {
      const cat = getModuleCategory(mod, role)
      if (!grouped[cat]) grouped[cat] = []
      grouped[cat].push(mod)
    })
    return grouped
  }, [enabledModules, role])

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
                  전문가 모드 시작 <span className="text-xl">→</span>
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
                  일반인 모드 시작 <span className="text-xl">→</span>
                </div>
              </div>
            </button>
          </div>
        </div>
      </main>
    )
  }

  // AI Command Center Layout (when role is selected)
  return (
    <div className="min-h-screen bg-[#F5F5F7] p-6 md:p-12 relative overflow-x-hidden transition-all duration-500 ease-in-out">
      <div
        className={`relative z-10 transition-all duration-500 ease-in-out ${isChatOpen ? 'w-1/2 pr-12' : 'w-full max-w-5xl mx-auto'
          }`}
      >
        {/* Header Section */}
        <header className="mb-16">
          <div className="mb-12">
            <div className="space-y-1">
              <h1 className="text-4xl font-bold text-[#1D1D1F] tracking-tight">
                반갑습니다, <span className="text-blue-600">{role === 'lawyer' ? '변호사님' : '의뢰인님'}</span>
              </h1>
            </div>
          </div>

          <div className="space-y-2">
            <h2 className="text-5xl font-extrabold text-[#1D1D1F] leading-tight max-w-2xl">
              어떤 업무를 <br />도와드릴까요?
            </h2>
          </div>
        </header>

        {/* Quick Recommendations Section */}
        <section>
          <div className="flex items-center mb-8">
            <h3 className="text-sm font-bold text-[#86868B] uppercase tracking-widest flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
              추천 작업
            </h3>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {(() => {
              const lawyerPicks = ['lawyer-stats', 'case-precedent', 'mock-trial', 'storyboard']
              const picks = role === 'lawyer'
                ? lawyerPicks.map(id => enabledModules.find(m => m.id === id)).filter(Boolean)
                : enabledModules.slice(0, 4)
              return picks
            })().map((module) => (
              <Link
                key={module.id}
                href={module.href}
                className="group relative block p-8 bg-white border border-[#D2D2D7]/50 rounded-[2rem] hover:shadow-apple-hover hover:border-blue-200 transition-all duration-300 overflow-hidden"
              >
                {/* Background Decoration */}
                <div className="absolute top-0 right-0 p-8 opacity-[0.03] group-hover:opacity-[0.08] transition-opacity">
                  <span className="text-7xl">{module.icon}</span>
                </div>

                <div className="relative z-10 h-full flex flex-col">
                  <div className="text-3xl mb-6 group-hover:scale-110 transition-transform inline-block">
                    {module.icon}
                  </div>
                  <h4 className="text-xl font-bold text-[#1D1D1F] mb-3 group-hover:text-blue-600 transition-colors">
                    {module.name}
                  </h4>
                  <p className="text-[#86868B] text-sm leading-relaxed mb-6 flex-1">
                    {module.description}
                  </p>

                  <div className="flex items-center text-sm font-bold text-blue-600 opacity-0 group-hover:opacity-100 transform translate-y-2 group-hover:translate-y-0 transition-all">
                    업무 시작 <span className="ml-2">→</span>
                  </div>
                </div>
              </Link>
            ))}
          </div>
        </section>

        {/* Floating Guide */}
        <div className="mt-20 p-8 bg-blue-600 rounded-[2.5rem] text-white shadow-xl shadow-blue-500/20 relative overflow-hidden group">
          <div className="absolute top-0 right-0 p-12 opacity-10 transform rotate-12 group-hover:rotate-0 transition-transform duration-700">
            <span className="text-9xl">💡</span>
          </div>
          <div className="relative z-10 max-w-lg">
            <h4 className="text-2xl font-bold mb-3">AI 인턴 활용 팁</h4>
            <p className="text-blue-50/80 leading-relaxed mb-6">
              "현재 위치 주변에 가사 전문 변호사 찾아줘" 또는 "어제 작업하던 소액소송 서류 다시 열어줘"라고 말해보세요. 우측 채팅창이 당신의 모든 명령을 수행합니다.
            </p>
            <button
              onClick={() => setChatOpen(true)}
              className="px-6 py-3 bg-white text-blue-600 rounded-full font-bold shadow-soft hover:bg-blue-50 transition-colors"
            >
              대화 시작하기
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}


export default function Home() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-white" />}>
      <HomeContent />
    </Suspense>
  )
}
