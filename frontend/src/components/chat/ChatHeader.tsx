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
    ? AGENT_DISPLAY_NAMES[activeAgent] || 'Active Now'
    : 'Active Now'

  return (
    <div
      className={`p-4 md:p-6 flex justify-between items-center bg-white border-b border-black/[0.06] ${chatMode === 'floating' ? 'rounded-t-2xl' : ''}`}
    >
      <div className="flex items-center gap-4">
        <div className="w-10 h-10 rounded-full bg-blue-500 flex items-center justify-center shadow-lg">
          <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
            />
          </svg>
        </div>
        <div>
          <h3 className="font-bold text-lg text-[#1D1D1F]">AI 법률 상담</h3>
          <p className="text-xs font-bold uppercase tracking-widest text-[#007AFF]">
            {displayName}
          </p>
        </div>
      </div>
      <div className="flex items-center gap-2">
        {/* Role Selector */}
        <div className="flex rounded-lg border text-xs bg-[#F5F5F7] border-black/[0.06]">
          <button
            onClick={() => onSetUserRole('user')}
            className={`px-2 py-1 rounded-l-md transition-colors ${
              userRole === 'user'
                ? 'bg-[#007AFF] text-white'
                : 'text-[#86868B] hover:bg-black/[0.04]'
            }`}
          >
            일반인
          </button>
          <button
            onClick={() => onSetUserRole('lawyer')}
            className={`px-2 py-1 rounded-r-md transition-colors ${
              userRole === 'lawyer'
                ? 'bg-[#007AFF] text-white'
                : 'text-[#86868B] hover:bg-black/[0.04]'
            }`}
          >
            변호사
          </button>
        </div>

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
