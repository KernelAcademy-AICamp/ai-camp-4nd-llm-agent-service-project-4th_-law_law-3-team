'use client'

import { memo } from 'react'
import { Info } from 'lucide-react'
import type { StageInfo } from '../types'

interface StageGuideBannerProps {
  stageInfo: StageInfo
  onDismiss: () => void
}

export const StageGuideBanner = memo(function StageGuideBanner({
  stageInfo,
  onDismiss,
}: StageGuideBannerProps) {
  return (
    <div className="bg-blue-50 border-b border-blue-200 px-4 py-2 flex items-start gap-2">
      <Info className="w-4 h-4 text-blue-500 mt-0.5 shrink-0" />
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold text-blue-700">
            {stageInfo.name}
          </span>
          <span className="text-[10px] text-blue-500 bg-blue-100 px-1.5 py-0.5 rounded">
            {stageInfo.legal_basis}
          </span>
          <span className="text-[10px] text-gray-400">
            {stageInfo.duration_hint}
          </span>
        </div>
        <p className="text-xs text-blue-600 mt-0.5">
          {stageInfo.description} &middot;{' '}
          <span className="font-medium">{stageInfo.user_action}</span>
        </p>
      </div>
      <button
        onClick={onDismiss}
        className="text-blue-400 hover:text-blue-600 text-xs shrink-0"
        aria-label="가이드 닫기"
      >
        닫기
      </button>
    </div>
  )
})
