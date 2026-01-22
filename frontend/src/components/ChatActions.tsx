'use client'

import Link from 'next/link'
import { useRouter } from 'next/navigation'

export type ActionType = 'button' | 'link' | 'request_location' | 'navigate'

export interface ChatAction {
  type: ActionType
  label: string
  action?: string
  url?: string
  params?: Record<string, string | number | boolean>  // navigate params
}

interface ChatActionsProps {
  actions: ChatAction[]
  onAction: (action: string) => void
  onRequestLocation: () => void
  isLightTheme?: boolean
}

/**
 * Render interactive action controls for a chat message.
 *
 * Renders buttons, links, a navigate button (which navigates to a URL with optional query params), and a location request button based on the provided actions array; returns null when `actions` is empty or not provided.
 *
 * @param actions - Array of chat action descriptors determining which controls are rendered and their behavior
 * @param onAction - Callback invoked with the action identifier when a standard button action is activated
 * @param onRequestLocation - Callback invoked when a `request_location` action is activated
 * @param isLightTheme - If true, use light-theme styling for action controls
 * @returns A React element containing the rendered action controls, or `null` if there are no actions
 */
export default function ChatActions({
  actions,
  onAction,
  onRequestLocation,
  isLightTheme = false,
}: ChatActionsProps) {
  const router = useRouter()

  if (!actions || actions.length === 0) return null

  // NAVIGATE 액션 핸들러 - URL + params로 페이지 이동
  const handleNavigate = (url: string, params?: Record<string, string | number | boolean>) => {
    if (!params || Object.keys(params).length === 0) {
      router.push(url)
      return
    }

    const searchParams = new URLSearchParams()
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null) {
        searchParams.set(key, String(value))
      }
    })

    const fullUrl = `${url}?${searchParams.toString()}`
    router.push(fullUrl)
  }

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

        // NAVIGATE 타입 - 쿼리 파라미터와 함께 페이지 이동
        if (action.type === 'navigate' && action.url) {
          return (
            <button
              key={key}
              onClick={() => handleNavigate(action.url!, action.params)}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors flex items-center gap-1 ${linkClasses}`}
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
                  d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7"
                />
              </svg>
              {action.label}
            </button>
          )
        }

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