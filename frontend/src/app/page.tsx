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

  // URL role과 ChatContext 동기화
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
      <main className="min-h-screen bg-slate-950 flex flex-col items-center justify-center p-6 lg:p-12 relative overflow-hidden">
        {/* Background Gradients */}
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-blue-600/20 rounded-full blur-[120px]" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-indigo-600/20 rounded-full blur-[120px]" />

        <div className="relative z-10 w-full max-w-7xl mx-auto flex flex-col items-center text-center">
          <h1 className="text-5xl md:text-7xl font-extrabold text-white mb-6 tracking-tight">
            LEX <span className="text-blue-500">CAPITAL</span>
          </h1>
          <p className="text-gray-400 text-xl md:text-2xl mb-16 max-w-2xl mx-auto">
            당신의 법률 여정을 한 단계 더 높이세요. AI 기반 맞춤형 법률 솔루션을 경험해 보세요.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 w-full max-w-7xl">
            {/* Lawyer Card */}
            <button
              onClick={() => handleRoleSelect('lawyer')}
              className="group relative p-8 rounded-3xl bg-white/5 border border-white/10 backdrop-blur-xl hover:bg-white/10 hover:border-blue-500/50 transition-all duration-500 text-left overflow-hidden hover:-translate-y-2 hover:shadow-2xl hover:shadow-blue-500/20"
            >
              <div className="absolute top-0 right-0 p-6 opacity-10 group-hover:opacity-20 transition-opacity">
                <span className="text-8xl">⚖️</span>
              </div>
              <div className="relative z-10">
                <div className="text-5xl mb-6">👨‍💼</div>
                <h2 className="text-2xl font-bold text-white mb-4 tracking-tight">법 관련 종사자입니다</h2>
                <p className="text-gray-400 mb-8 leading-relaxed text-base">의뢰인과 연결되고, 전문성을 발휘하여 업무를 관리하세요.</p>
                <div className="inline-flex items-center text-blue-400 font-bold group-hover:gap-4 gap-2 transition-all text-base">
                  대시보드 입장 <span className="text-xl">→</span>
                </div>
              </div>
            </button>

            {/* Public Institution Card (UI Only) */}
            <button
              className="group relative p-8 rounded-3xl bg-white/5 border border-white/10 backdrop-blur-xl hover:bg-white/10 hover:border-emerald-500/50 transition-all duration-500 text-left overflow-hidden hover:-translate-y-2 hover:shadow-2xl hover:shadow-emerald-500/20"
            >
              <div className="absolute top-0 right-0 p-6 opacity-10 group-hover:opacity-20 transition-opacity">
                <span className="text-8xl">🏛️</span>
              </div>
              <div className="relative z-10">
                <div className="text-5xl mb-6">🏢</div>
                <h2 className="text-2xl font-bold text-white mb-4 tracking-tight">공공기관입니다</h2>
                <p className="text-gray-400 mb-8 leading-relaxed text-base">공공 업무 효율화를 위한 맞춤형 법률 AI 솔루션을 활용하세요.</p>
                <div className="inline-flex items-center text-emerald-400 font-bold group-hover:gap-4 gap-2 transition-all text-base">
                  서비스 준비중 <span className="text-xl">🔒</span>
                </div>
              </div>
            </button>

            {/* User Card */}
            <button
              onClick={() => handleRoleSelect('user')}
              className="group relative p-8 rounded-3xl bg-white/5 border border-white/10 backdrop-blur-xl hover:bg-white/10 hover:border-indigo-500/50 transition-all duration-500 text-left overflow-hidden hover:-translate-y-2 hover:shadow-2xl hover:shadow-indigo-500/20"
            >
              <div className="absolute top-0 right-0 p-6 opacity-10 group-hover:opacity-20 transition-opacity">
                <span className="text-8xl">🤝</span>
              </div>
              <div className="relative z-10">
                <div className="text-5xl mb-6">👤</div>
                <h2 className="text-2xl font-bold text-white mb-4 tracking-tight">일반인입니다</h2>
                <p className="text-gray-400 mb-8 leading-relaxed text-base">나에게 딱 맞는 법률 전문가를 찾고 사건을 해결하세요.</p>
                <div className="inline-flex items-center text-indigo-400 font-bold group-hover:gap-4 gap-2 transition-all text-base">
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
    <main className="min-h-screen bg-slate-950 p-8 relative overflow-hidden transition-all duration-500 ease-in-out">
      {/* Background Gradients */}
      <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-blue-600/10 rounded-full blur-[120px]" />
      <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-indigo-600/10 rounded-full blur-[120px]" />

      <div 
        className={`relative z-10 transition-all duration-500 ease-in-out ${
          isChatOpen ? 'w-1/2 pr-8' : 'w-full max-w-6xl mx-auto'
        }`}
      >
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-12 gap-6">
          <div>
            <h1 className="text-3xl font-bold text-white mb-2 tracking-tight">
              LEX <span className="text-blue-500">CAPITAL</span>
            </h1>
            <p className="text-blue-400/80 font-medium">
              {role === 'lawyer' ? '변호사님 전용 대시보드' : '사용자 맞춤형 도움 서비스'}
            </p>
          </div>
          <button
            onClick={handleResetRole}
            className="px-5 py-2.5 text-sm font-semibold text-white/70 hover:text-white bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl backdrop-blur-md transition-all duration-300"
          >
            ← 역할 변경
          </button>
        </div>

        <div className={`grid gap-5 ${isChatOpen ? 'grid-cols-1 md:grid-cols-2 xl:grid-cols-3' : 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3'}`}>
          {enabledModules.map((module) => (
            <Link
              key={module.id}
              href={module.href}
              className="group relative block p-6 bg-white/5 border border-white/10 rounded-2xl backdrop-blur-xl hover:bg-white/10 hover:border-blue-500/50 transition-all duration-300 overflow-hidden"
            >
              {/* Subtle card glow on hover */}
              <div className="absolute inset-0 bg-blue-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
              
              <div className="relative z-10">
                <div className="text-4xl mb-4 group-hover:scale-110 group-hover:-rotate-3 transition-transform duration-300">
                  {module.icon}
                </div>
                <h2 className="text-xl font-bold text-white mb-2 group-hover:text-blue-400 transition-colors">
                  {module.name}
                </h2>
                <p className="text-gray-400 text-sm leading-relaxed group-hover:text-gray-300 transition-colors">
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
    <Suspense fallback={<div className="min-h-screen bg-slate-950" />}>
      <HomeContent />
    </Suspense>
  )
}
