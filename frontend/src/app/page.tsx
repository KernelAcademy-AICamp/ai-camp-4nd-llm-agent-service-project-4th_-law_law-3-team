'use client'

import { useState } from 'react'
import Link from 'next/link'
import { getEnabledModules } from '@/lib/modules'

export default function Home() {
  const [role, setRole] = useState<'lawyer' | 'user' | null>(null)
  const enabledModules = getEnabledModules(role || undefined)

  if (!role) {
    return (
      <main className="min-h-screen bg-slate-950 flex flex-col items-center justify-center p-6 relative overflow-hidden">
        {/* Background Gradients */}
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-blue-600/20 rounded-full blur-[120px]" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-indigo-600/20 rounded-full blur-[120px]" />

        <div className="relative z-10 w-full max-w-4xl text-center">
          <h1 className="text-4xl md:text-6xl font-extrabold text-white mb-4 tracking-tight">
            LEX <span className="text-blue-500">CAPITAL</span>
          </h1>
          <p className="text-gray-400 text-lg md:text-xl mb-12">
            당신의 법률 여정을 한 단계 더 높이세요
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {/* Lawyer Card */}
            <button
              onClick={() => setRole('lawyer')}
              className="group relative p-8 rounded-2xl bg-white/5 border border-white/10 backdrop-blur-xl hover:bg-white/10 hover:border-blue-500/50 transition-all duration-300 text-left overflow-hidden"
            >
              <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                <span className="text-8xl">⚖️</span>
              </div>
              <div className="relative z-10">
                <div className="text-4xl mb-4">👨‍💼</div>
                <h2 className="text-2xl font-bold text-white mb-2">변호사입니다</h2>
                <p className="text-gray-400 mb-6">의뢰인과 연결되고, 전문성을 발휘하여 업무를 관리하세요.</p>
                <div className="inline-flex items-center text-blue-400 font-semibold group-hover:translate-x-1 transition-transform">
                  시작하기 <span className="ml-2">→</span>
                </div>
              </div>
            </button>

            {/* User Card */}
            <button
              onClick={() => setRole('user')}
              className="group relative p-8 rounded-2xl bg-white/5 border border-white/10 backdrop-blur-xl hover:bg-white/10 hover:border-indigo-500/50 transition-all duration-300 text-left overflow-hidden"
            >
              <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                <span className="text-8xl">🤝</span>
              </div>
              <div className="relative z-10">
                <div className="text-4xl mb-4">👤</div>
                <h2 className="text-2xl font-bold text-white mb-2">도움이 필요합니다</h2>
                <p className="text-gray-400 mb-6">나에게 딱 맞는 법률 전문가를 찾고 사건을 해결하세요.</p>
                <div className="inline-flex items-center text-indigo-400 font-semibold group-hover:translate-x-1 transition-transform">
                  시작하기 <span className="ml-2">→</span>
                </div>
              </div>
            </button>
          </div>
        </div>
      </main>
    )
  }

  return (
    <main className="min-h-screen bg-slate-950 p-8 relative overflow-hidden">
      {/* Background Gradients */}
      <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-blue-600/10 rounded-full blur-[120px]" />
      <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-indigo-600/10 rounded-full blur-[120px]" />

      <div className="max-w-6xl mx-auto relative z-10">
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
            onClick={() => setRole(null)}
            className="px-5 py-2.5 text-sm font-semibold text-white/70 hover:text-white bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl backdrop-blur-md transition-all duration-300"
          >
            ← 역할 변경
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {enabledModules.map((module) => (
            <Link
              key={module.id}
              href={module.href}
              className="group relative block p-8 bg-white/5 border border-white/10 rounded-2xl backdrop-blur-xl hover:bg-white/10 hover:border-blue-500/50 transition-all duration-300 overflow-hidden"
            >
              {/* Subtle card glow on hover */}
              <div className="absolute inset-0 bg-blue-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
              
              <div className="relative z-10">
                <div className="text-4xl mb-5 group-hover:scale-110 group-hover:-rotate-3 transition-transform duration-300">
                  {module.icon}
                </div>
                <h2 className="text-xl font-bold text-white mb-3 group-hover:text-blue-400 transition-colors">
                  {module.name}
                </h2>
                <p className="text-gray-400 text-sm leading-relaxed group-hover:text-gray-300 transition-colors">
                  {module.description}
                </p>
                
                <div className="mt-6 flex items-center text-xs font-bold uppercase tracking-wider text-blue-500/70 group-hover:text-blue-500">
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
