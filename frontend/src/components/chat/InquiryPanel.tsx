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
        return <LawyerFinderMain isInline={false} />
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
    <div className="h-screen flex flex-col bg-white overflow-hidden">
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
