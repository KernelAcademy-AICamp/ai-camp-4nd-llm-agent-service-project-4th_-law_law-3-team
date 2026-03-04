'use client'

import { useEffect, Suspense, useMemo, useState } from 'react'
import Link from 'next/link'
import Image from 'next/image'
import { useRouter, useSearchParams } from 'next/navigation'
import { getEnabledModules, getModuleCategory, CATEGORY_NAMES } from '@/lib/modules'
import { useUI } from '@/context/UIContext'
import { useChat, UserRole } from '@/context/ChatContext'
import type { LucideIcon } from 'lucide-react'
import {
  Send,
  Search,
  BarChart,
  MapPin,
  Video,
  Scale,
  Gavel
} from 'lucide-react'

const MODULE_ICONS: Record<string, LucideIcon> = {
  'lawyer-finder': MapPin,
  'lawyer-stats': BarChart,
  'case-precedent': Search,
  'storyboard': Video,
  'small-claims': Scale,
  'mock-trial': Gavel,
}

function HomeContent() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const role = searchParams.get('role') as 'lawyer' | 'user' | null

  const { isChatOpen, setChatOpen, setPendingMessage } = useUI()
  const { setUserRole } = useChat()
  const [inputValue, setInputValue] = useState('')
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
      setChatOpen(false) // 대시보드 진입 시 채팅 자동 열림 방지
      setUserRole(role as UserRole)
    }
  }, [role, setChatOpen, setUserRole])

  const handleRoleSelect = (selectedRole: 'lawyer' | 'user') => {
    setUserRole(selectedRole)
    setChatOpen(false) // 역할 선택 시에도 채팅 자동 열림 방지
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
        <Image
          src="/assets/hero-background.png"
          alt="법률 서비스 플랫폼"
          fill
          className="object-contain object-center"
          priority
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
    <div className="min-h-screen bg-[#F5F5F7] pt-12 pb-6 px-6 md:pt-20 md:pb-12 md:px-12 relative transition-all duration-500 ease-in-out">
      <div
        className={`relative z-10 h-full flex flex-col transition-all duration-500 ease-in-out ${isChatOpen ? 'w-1/2 pr-8' : 'w-full max-w-5xl mx-auto'
          }`}
      >
        {/* Header Section */}
        <header className="mb-20 shrink-0">
          <div className="space-y-4">
            <h2 className="text-5xl font-extrabold text-[#1D1D1F] leading-[1.15] tracking-tighter max-w-2xl">
              {role === 'lawyer' ? (
                <><span className="block mb-2 text-[#1D1D1F]">
                  안녕하세요, <span className="relative inline-block text-blue-600">
                    변호사님
                    <span className="absolute -bottom-1 left-0 w-full h-1 bg-blue-600/10 rounded-full" />
                  </span>
                </span>어떤 업무를 도와드릴까요?</>
              ) : (
                <><span className="block mb-2">안녕하세요.</span>어떤 <span className="relative inline-block text-blue-600">
                  법률 도움
                  <span className="absolute -bottom-2 left-0 w-full h-1.5 bg-blue-600/10 rounded-full" />
                </span>이 필요하신가요?</>
              )}
            </h2>
          </div>
        </header>

        {/* Interactive Chat Input Hub */}
        <div className="mb-20 relative group">
          <div className="p-8 bg-white/90 backdrop-blur-2xl border-2 border-[#D2D2D7] shadow-xl rounded-[2.5rem] overflow-hidden relative">
            {/* Background Decoration (Logo) */}
            <div className="absolute bottom-2 right-8 opacity-[0.08] group-hover:opacity-[0.12] transform rotate-12 group-hover:rotate-0 transition-all duration-700 pointer-events-none">
              <Image src="/logo.png" alt="" width={128} height={128} className="object-contain" />
            </div>

            <div className="relative z-10">
              <form
                onSubmit={(e) => {
                  e.preventDefault()
                  if (!inputValue.trim()) return
                  setPendingMessage(inputValue)
                  setChatOpen(true)
                  setInputValue('')
                }}
                className="relative"
              >
                <input
                  type="text"
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  placeholder='"최근 판결 동향 알려줘" 혹은 "내 주변 변호사 찾아줘"'
                  className="w-full px-8 py-6 bg-[#F5F5F7] border-none rounded-2xl text-lg font-medium text-[#1D1D1F] placeholder:text-[#86868B] focus:ring-2 focus:ring-blue-600 transition-all shadow-inner"
                />
                <button
                  type="submit"
                  disabled={!inputValue.trim()}
                  className="absolute right-3 top-3 bottom-3 px-6 bg-blue-600 hover:bg-blue-500 disabled:bg-[#D2D2D7] text-white rounded-xl font-bold shadow-lg shadow-blue-500/20 transition-all active:scale-95 flex items-center gap-2 group/btn"
                >
                  요청하기
                  <Send size={18} className="group-hover/btn:translate-x-1 transition-transform" />
                </button>
              </form>

              <div className="mt-6 flex flex-wrap gap-2">
                {['최근 판례 검색', '변호사 찾기', '소액 소송 절차', '법리 검토 요청'].map((tag) => (
                  <button
                    key={tag}
                    type="button"
                    onClick={() => setInputValue(tag)}
                    className="px-4 py-2 bg-[#F5F5F7] hover:bg-blue-50 text-[#86868B] hover:text-blue-600 text-xs font-bold rounded-full border border-transparent hover:border-blue-200 transition-all"
                  >
                    #{tag}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Quick Recommendations Section */}
        <section className="mb-10 mt-auto">
          <div className="flex items-center mb-6">
            <h3 className="text-xs font-bold text-[#86868B] uppercase tracking-widest flex items-center gap-2">
              <div className="w-1.5 h-1.5 rounded-full bg-blue-500" />
              추천 작업
            </h3>
          </div>

          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            {(() => {
              const lawyerPicks = ['lawyer-stats', 'mock-trial', 'case-precedent', 'storyboard']
              const userPicks = ['lawyer-finder', 'small-claims', 'case-precedent', 'storyboard']
              const targetPicks = role === 'lawyer' ? lawyerPicks : userPicks

              const picks = targetPicks
                .map(id => enabledModules.find(m => m.id === id))
                .filter(Boolean)

              return picks as Exclude<typeof picks[number], undefined>[]
            })().map((module) => (
              <Link
                key={module.id}
                href={module.href}
                className="group relative block p-4 bg-white border border-[#D2D2D7]/50 rounded-2xl hover:shadow-apple-hover hover:border-blue-200 transition-all duration-300 overflow-hidden"
              >
                {/* Action / Icon Indicator (Top Right) */}
                <div className="absolute top-2 right-2 pointer-events-none">
                  {/* Default Icon Background */}
                  <div className="opacity-[0.05] group-hover:opacity-0 transition-opacity">
                    {(() => {
                      const Icon = MODULE_ICONS[module.id] || Search
                      return <Icon size={48} strokeWidth={1} />
                    })()}
                  </div>
                </div>

                {/* Hover Action Text (Horizontal) */}
                <div className="absolute top-[18px] right-3 opacity-0 group-hover:opacity-100 transform translate-x-2 group-hover:translate-x-0 transition-all pointer-events-none">
                  <div className="text-[10px] font-bold text-blue-600 whitespace-nowrap bg-blue-50/80 backdrop-blur-sm px-2 py-1.5 rounded-lg flex items-center gap-1 shadow-sm border border-blue-100">
                    시작하기 <span className="text-xs">→</span>
                  </div>
                </div>

                <div className="relative z-10 flex flex-col h-full">
                  <div className="flex items-center gap-2.5 mb-2.5">
                    <div className="p-2 bg-blue-600/5 rounded-xl group-hover:bg-blue-600 group-hover:text-white transition-all duration-300 text-blue-600">
                      {(() => {
                        const Icon = MODULE_ICONS[module.id] || Search
                        return <Icon size={18} />
                      })()}
                    </div>
                    <h4 className="text-sm font-bold text-[#1D1D1F]/90 group-hover:text-blue-600 transition-colors line-clamp-1">
                      {module.name}
                    </h4>
                  </div>
                  <p className="text-[#86868B]/70 group-hover:text-[#86868B] text-[10px] leading-snug line-clamp-2 transition-colors">
                    {(() => {
                      const overrides: Record<string, string> = role === 'lawyer'
                        ? {
                          'lawyer-stats': '지역·전문분야별 법률시장 구조 분석 대시보드',
                          'case-precedent': '사건 맥락을 이해하는 AI 판례 분석 에이전트',
                          'mock-trial': 'AI 에이전트 기반 재판 시뮬레이션',
                          'storyboard': '사건 사실관계 구조화 및 타임라인 분석'
                        }
                        : {
                          'lawyer-finder': '사건 유형과 위치를 고려한 최적의 변호사 추천',
                          'case-precedent': '사건 내용을 기반으로 관련 판례를 찾아주는 AI 검색',
                          'small-claims': '단계별 안내에 따라 소액소송 서류 작성',
                          'storyboard': '복잡한 사건을 한눈에 정리'
                        }
                      return overrides[module.id] || module.description
                    })()}
                  </p>
                </div>
              </Link>
            ))}
          </div>
        </section>
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
