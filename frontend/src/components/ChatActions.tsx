'use client'

import Link from 'next/link'

export type ActionType = 'button' | 'link' | 'request_location'

export interface ChatAction {
  type: ActionType
  label: string
  action?: string
  url?: string
}

interface ChatActionsProps {
  actions: ChatAction[]
  onAction: (action: string) => void
  onRequestLocation: () => void
  isLightTheme?: boolean
}

export default function ChatActions({
  actions,
  onAction,
  onRequestLocation,
  isLightTheme = false,
}: ChatActionsProps) {
  if (!actions || actions.length === 0) return null

  const buttonClasses = isLightTheme
    ? 'bg-blue-100 text-blue-700 hover:bg-blue-200 border border-blue-200'
    : 'bg-blue-600/20 text-blue-400 hover:bg-blue-600/30 border border-blue-500/30'

  const linkClasses = isLightTheme
    ? 'bg-green-100 text-green-700 hover:bg-green-200 border border-green-200'
    : 'bg-green-600/20 text-green-400 hover:bg-green-600/30 border border-green-500/30'

  const locationClasses = isLightTheme
    ? 'bg-orange-100 text-orange-700 hover:bg-orange-200 border border-orange-200'
    : 'bg-orange-600/20 text-orange-400 hover:bg-orange-600/30 border border-orange-500/30'

  return (
    <div className="flex flex-wrap gap-2 mt-3">
      {actions.map((action, index) => {
        const key = `action-${index}-${action.label}`

        if (action.type === 'link' && action.url) {
          // 외부 링크 또는 내부 링크
          const isExternal = action.url.startsWith('http')

          if (isExternal) {
            return (
              <a
                key={key}
                href={action.url}
                target="_blank"
                rel="noopener noreferrer"
                className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${linkClasses}`}
              >
                {action.label}
                <svg
                  className="inline-block w-3 h-3 ml-1"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"
                  />
                </svg>
              </a>
            )
          }

          return (
            <Link
              key={key}
              href={action.url}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${linkClasses}`}
            >
              {action.label}
            </Link>
          )
        }

        if (action.type === 'request_location') {
          return (
            <button
              key={key}
              onClick={onRequestLocation}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors flex items-center gap-1 ${locationClasses}`}
            >
              <svg
                className="w-4 h-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"
                />
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"
                />
              </svg>
              {action.label}
            </button>
          )
        }

        // 기본 버튼
        return (
          <button
            key={key}
            onClick={() => action.action && onAction(action.action)}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${buttonClasses}`}
          >
            {action.label}
          </button>
        )
      })}
    </div>
  )
}
