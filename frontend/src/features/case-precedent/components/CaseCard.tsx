'use client'

import { memo } from 'react'
import Image from 'next/image'
import type { PrecedentItem } from '../types'
import { getCourtLogo, getDocTypeLogo, DEFAULT_GOV_LOGO } from '../utils/lawTypeLogo'

interface CaseCardProps {
  case_: PrecedentItem
  selected: boolean
  onClick: () => void
}

function CaseCardComponent({ case_, selected, onClick }: CaseCardProps) {
  const getDocTypeLabel = (docType: string) => {
    const labels: Record<string, string> = {
      precedent: '판례',
      constitutional: '헌재결정',
      administrative: '행정심판',
      interpretation: '법령해석',
    }
    return labels[docType] || docType
  }

  const getDocTypeColor = (docType: string) => {
    const colors: Record<string, string> = {
      precedent: 'bg-blue-100 text-blue-800',
      constitutional: 'bg-purple-100 text-purple-800',
      administrative: 'bg-green-100 text-green-800',
      interpretation: 'bg-yellow-100 text-yellow-800',
    }
    return colors[docType] || 'bg-gray-100 text-gray-800'
  }

  const isLaw = case_.doc_type === 'law'
  const logoPath = isLaw
    ? null
    : getCourtLogo(case_.court) || getDocTypeLogo(case_.doc_type)

  return (
    <button
      onClick={onClick}
      className={`w-full text-left p-4 rounded-lg border transition-all ${
        selected
          ? 'border-blue-500 bg-blue-50 shadow-md'
          : 'border-gray-200 bg-white hover:border-gray-300 hover:shadow-sm'
      }`}
    >
      <div className="flex items-start gap-3">
        {/* 로고 */}
        <div className="shrink-0 w-10 h-10 flex items-center justify-center bg-gray-50 rounded-lg">
          <Image
            src={logoPath || DEFAULT_GOV_LOGO}
            alt={case_.court || '대한민국 정부'}
            width={32}
            height={32}
            className="object-contain"
            unoptimized
          />
        </div>

        <div className="flex-1 min-w-0">
          <h3 className="font-medium text-gray-900 line-clamp-2">
            {[case_.court, case_.case_number].filter(Boolean).join(' ') || '제목 없음'}
          </h3>
          {case_.date && (
            <p className="text-xs text-gray-400 mt-1">{case_.date}</p>
          )}
        </div>
      </div>
    </button>
  )
}

export const CaseCard = memo(CaseCardComponent)
