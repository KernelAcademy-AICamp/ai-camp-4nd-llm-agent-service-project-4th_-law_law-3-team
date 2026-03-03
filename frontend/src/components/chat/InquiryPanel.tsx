'use client'

import React, { Suspense } from 'react'
import { X } from 'lucide-react'
import { useUI } from '@/context/UIContext'
import { LawyerFinderMain } from '@/features/lawyer-finder/components/LawyerFinderMain'

interface InquiryPanelProps {
  panelId: string
}

export default function InquiryPanel({ panelId }: InquiryPanelProps) {
  const { setActivePanel } = useUI()

  const renderContent = () => {
    switch (panelId) {
      case 'lawyer-finder':
        return <LawyerFinderMain isInline />
      // case 'case-detail': ...
      default:
        return (
          <div className="flex items-center justify-center h-full text-gray-400">
            Unknown Panel: {panelId}
          </div>
        )
    }
  }

  return (
    <div className="h-screen flex flex-col bg-white">
      {/* Panel Header */}
      <div className="h-14 border-b border-gray-200 flex items-center justify-between px-6 bg-[#F5F5F7]/50 backdrop-blur-md sticky top-0 z-10">
        <div className="flex items-center gap-3">
          <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
          <span className="text-sm font-bold text-[#1D1D1F] uppercase tracking-wider">
            {panelId.replace('-', ' ')}
          </span>
        </div>
        <button
          onClick={() => setActivePanel(null)}
          className="p-2 hover:bg-black/5 rounded-full transition-colors text-gray-500"
        >
          <X size={18} />
        </button>
      </div>

      {/* Panel Content */}
      <div className="flex-1 overflow-hidden">
        <Suspense fallback={
          <div className="h-full flex items-center justify-center">
            <div className="w-6 h-6 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
          </div>
        }>
          {renderContent()}
        </Suspense>
      </div>
    </div>
  )
}
