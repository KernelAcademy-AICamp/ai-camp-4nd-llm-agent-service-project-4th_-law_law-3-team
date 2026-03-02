'use client'

import type { NewsSource } from '../types'

interface NewsSourceBadgeProps {
  source: NewsSource
}

const SOURCE_STYLES: Record<NewsSource, { label: string; className: string }> = {
  lawtimes: {
    label: '법률신문',
    className: 'bg-blue-100 text-blue-700',
  },
  naver: {
    label: '네이버',
    className: 'bg-green-100 text-green-700',
  },
}

export function NewsSourceBadge({ source }: NewsSourceBadgeProps) {
  const style = SOURCE_STYLES[source] ?? {
    label: source,
    className: 'bg-gray-100 text-gray-700',
  }

  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${style.className}`}>
      {style.label}
    </span>
  )
}
