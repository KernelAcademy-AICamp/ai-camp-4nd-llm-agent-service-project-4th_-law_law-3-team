import { Scale } from 'lucide-react'
import { AGENT_DISPLAY_NAMES } from './constants'
import type { UserRole } from '@/context/ChatContext'

interface ChatHeaderProps {
  activeAgent: string | undefined
  userRole: UserRole
  onSetUserRole: (role: UserRole) => void
  supportsFloatingMode: boolean
  chatMode: 'split' | 'floating'
  onToggleViewMode: () => void
  onReset: () => void
  onClose: () => void
}

export function ChatHeader({
  activeAgent,
  userRole,
  onSetUserRole,
  supportsFloatingMode,
  chatMode,
  onToggleViewMode,
  onReset,
  onClose,
}: ChatHeaderProps) {
  const displayName = activeAgent
    ? AGENT_DISPLAY_NAMES[activeAgent] || activeAgent
    : null

  return (
    <div
      className={`p-4 md:p-6 flex justify-between items-center bg-white border-b border-black/[0.06] ${chatMode === 'floating' ? 'rounded-t-2xl' : ''}`}
    >
      <div className="flex items-start gap-2">
        <Scale className="w-5 h-5 text-[#1D1D1F] mt-0.5" />
        <div>
          <h3 className="font-bold text-lg text-[#1D1D1F]">AI 법률 어시스턴트</h3>
          {activeAgent ? (
            <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-green-50 text-green-700 text-xs font-medium">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500" />
              </span>
              {displayName} 모드
            </span>
          ) : (
            <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-red-50 text-red-600 text-xs font-medium">
              <span className="relative flex h-2 w-2">
                <span className="relative inline-flex rounded-full h-2 w-2 bg-red-400" />
              </span>
              에이전트 대기 중
            </span>
          )}
        </div>
      </div>
      <div className="flex items-center gap-2">
        {/* Reset Button */}
        <button
          onClick={onReset}
          className="px-3 py-1.5 rounded-lg text-xs font-medium transition-colors text-[#86868B] hover:bg-[#F5F5F7] hover:text-[#1D1D1F]"
          title="대화 초기화"
        >
          초기화
        </button>

        {/* Toggle View Mode Button */}
        {supportsFloatingMode && (
          <button
            onClick={onToggleViewMode}
            className="p-2 rounded-lg transition-colors text-[#86868B] hover:bg-[#F5F5F7] hover:text-[#1D1D1F]"
            title={chatMode === 'split' ? '작게 보기' : '크게 보기'}
          >
            {chatMode === 'split' ? (
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 14h6v6" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 10h-6V4" />
              </svg>
            ) : (
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 20h6v-6" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 4h-6v6" />
              </svg>
            )}
          </button>
        )}

        {/* Close Button */}
        <button
          onClick={onClose}
          className="p-2 rounded-lg transition-colors text-[#86868B] hover:bg-[#F5F5F7] hover:text-[#1D1D1F]"
        >
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
    </div>
  )
}
