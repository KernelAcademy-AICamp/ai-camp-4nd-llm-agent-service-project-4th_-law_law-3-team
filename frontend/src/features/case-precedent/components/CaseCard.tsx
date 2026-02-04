'use client'

import { memo } from 'react'
import type { PrecedentItem } from '../types'

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

  return (
    <button
      onClick={onClick}
      className={`w-full text-left p-4 rounded-lg border transition-all ${
        selected
          ? 'border-blue-500 bg-blue-50 shadow-md'
          : 'border-gray-200 bg-white hover:border-gray-300 hover:shadow-sm'
      }`}
    >
      <div className="flex items-start justify-between gap-2 mb-2">
        <h3 className="font-medium text-gray-900 line-clamp-2 flex-1">
          {case_.case_name || '제목 없음'}
        </h3>
        <span
          className={`shrink-0 px-2 py-0.5 text-xs font-medium rounded ${getDocTypeColor(
            case_.doc_type
          )}`}
        >
          {getDocTypeLabel(case_.doc_type)}
        </span>
      </div>

      <div className="text-sm text-gray-500 mb-2 space-y-1">
        {case_.case_number && (
          <p className="font-mono text-xs">{case_.case_number}</p>
        )}
        <div className="flex items-center gap-2 text-xs">
          {case_.court && <span>{case_.court}</span>}
          {case_.date && (
            <>
              {case_.court && <span>|</span>}
              <span>{case_.date}</span>
            </>
          )}
        </div>
      </div>

      <p className="text-sm text-gray-600 line-clamp-2">{case_.summary}</p>
    </button>
  )
}

export const CaseCard = memo(CaseCardComponent)
