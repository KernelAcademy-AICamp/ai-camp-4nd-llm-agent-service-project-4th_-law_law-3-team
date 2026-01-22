'use client'

import { useChat } from '@/context/ChatContext'
import { useUI } from '@/context/UIContext'
import { LawyerView } from '@/features/case-precedent/components/LawyerView'
import { UserView } from '@/features/case-precedent/components/UserView'
import { BackButton } from '@/components/ui/BackButton'

export default function CasePrecedentPage() {
  const { userRole } = useChat()
  const { isChatOpen } = useUI()

  return (
    <div 
      className={`h-screen flex flex-col bg-gray-100 transition-all duration-500 ease-in-out ${
        isChatOpen ? 'w-1/2 border-r border-gray-200' : 'w-full'
      }`}
    >
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="flex items-center gap-3">
          <BackButton />
          <span className="text-2xl">📚</span>
          <div>
            <h1 className="text-xl font-bold text-gray-900">판례 검색</h1>
            <p className="text-sm text-gray-500">
              {userRole === 'lawyer' 
                ? '전문가용 판례 검색 및 분석 시스템' 
                : 'AI 기반 쉬운 판례/법령 열람'}
            </p>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 overflow-hidden">
        {userRole === 'lawyer' ? <LawyerView /> : <UserView />}
      </div>
    </div>
  )
}
