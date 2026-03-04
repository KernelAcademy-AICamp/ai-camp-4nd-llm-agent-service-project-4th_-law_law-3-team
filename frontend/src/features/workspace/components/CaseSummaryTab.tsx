import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { FileText, RefreshCw } from 'lucide-react'
import type { WorkspaceCaseDetail, TimelineItem } from '@/features/workspace/types'
import { TAG_TYPE_COLORS } from '@/features/workspace/types'
import { getCaseSummary, triggerCaseSummarize } from '@/features/workspace/services'

const SUMMARY_FIELD_LABELS: Record<string, { label: string; icon: string }> = {
  facts: { label: '사실관계', icon: '📋' },
  issues: { label: '법적 쟁점', icon: '⚖️' },
  evidence: { label: '증거 자료', icon: '📎' },
  open_questions: { label: '미확인 사항', icon: '❓' },
  next_steps: { label: '다음 단계', icon: '➡️' },
}

interface CaseSummaryTabProps {
  caseData: WorkspaceCaseDetail
  timeline: TimelineItem[]
}

export function CaseSummaryTab({ caseData, timeline }: CaseSummaryTabProps) {
  const queryClient = useQueryClient()
  const [summarizeMessage, setSummarizeMessage] = useState<string | null>(null)

  const tagsByType: Record<string, number> = {}
  for (const tag of caseData.tagged_items) {
    tagsByType[tag.type] = (tagsByType[tag.type] || 0) + 1
  }

  const { data: summaryData } = useQuery({
    queryKey: ['case-summary', caseData.id],
    queryFn: () => getCaseSummary(caseData.id),
  })

  const summarizeMutation = useMutation({
    mutationFn: () => triggerCaseSummarize(caseData.id),
    onSuccess: (result) => {
      queryClient.invalidateQueries({ queryKey: ['case-summary', caseData.id] })
      setSummarizeMessage(result.message || '요약이 생성되었습니다.')
      setTimeout(() => setSummarizeMessage(null), 3000)
    },
    onError: () => {
      setSummarizeMessage('요약 생성에 실패했습니다.')
      setTimeout(() => setSummarizeMessage(null), 3000)
    },
  })

  const caseSummary = summaryData?.case_summary

  const hasSummaryContent = caseSummary && Object.values(caseSummary).some(
    (items) => Array.isArray(items) && items.length > 0,
  )

  return (
    <div className="space-y-6">
      {/* 통계 카드 */}
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-white rounded-lg border p-4">
          <p className="text-sm text-gray-500">대화</p>
          <p className="text-2xl font-bold text-gray-900 mt-1">
            {caseData.conversations.length}
          </p>
        </div>
        <div className="bg-white rounded-lg border p-4">
          <p className="text-sm text-gray-500">태그</p>
          <p className="text-2xl font-bold text-gray-900 mt-1">
            {caseData.tagged_items.length}
          </p>
        </div>
        <div className="bg-white rounded-lg border p-4">
          <p className="text-sm text-gray-500">타임라인</p>
          <p className="text-2xl font-bold text-gray-900 mt-1">
            {timeline.length}
          </p>
        </div>
      </div>

      {/* 구조화 요약 */}
      <div className="bg-white rounded-lg border p-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-medium text-gray-700 flex items-center gap-2">
            <FileText size={16} />
            구조화 요약
          </h3>
          <button
            onClick={() => summarizeMutation.mutate()}
            disabled={summarizeMutation.isPending}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-blue-600 bg-blue-50 rounded-lg hover:bg-blue-100 disabled:opacity-50 transition-colors"
          >
            <RefreshCw
              size={14}
              className={summarizeMutation.isPending ? 'animate-spin' : ''}
            />
            {summarizeMutation.isPending ? '생성 중...' : '요약 생성'}
          </button>
        </div>

        {summarizeMessage && (
          <p className="text-xs text-gray-500 mb-3">{summarizeMessage}</p>
        )}

        {hasSummaryContent ? (
          <div className="space-y-4">
            {Object.entries(SUMMARY_FIELD_LABELS).map(([field, { label, icon }]) => {
              const items = caseSummary[field]
              if (!Array.isArray(items) || items.length === 0) return null
              return (
                <div key={field}>
                  <h4 className="text-xs font-medium text-gray-500 mb-1.5">
                    {icon} {label}
                  </h4>
                  <ul className="space-y-1">
                    {items.map((item, i) => (
                      <li
                        key={`summary-${field}-${i}`}
                        className="text-sm text-gray-800 pl-4 relative before:content-['•'] before:absolute before:left-0 before:text-gray-400"
                      >
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>
              )
            })}
          </div>
        ) : (
          <p className="text-sm text-gray-400 text-center py-4">
            대화가 연결되면 AI가 사건 요약을 생성할 수 있습니다
          </p>
        )}
      </div>

      {/* 태그 유형 분포 */}
      {Object.keys(tagsByType).length > 0 && (
        <div className="bg-white rounded-lg border p-4">
          <h3 className="text-sm font-medium text-gray-700 mb-3">태그 유형 분포</h3>
          <div className="flex flex-wrap gap-2">
            {Object.entries(tagsByType).map(([type, count]) => (
              <span
                key={type}
                className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium ${
                  TAG_TYPE_COLORS[type] ?? 'bg-gray-100 text-gray-700'
                }`}
              >
                {type}: {count}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* 최근 대화 */}
      {caseData.conversations.length > 0 && (
        <div className="bg-white rounded-lg border p-4">
          <h3 className="text-sm font-medium text-gray-700 mb-3">연결된 대화</h3>
          <div className="space-y-2">
            {caseData.conversations.map((conv) => (
              <div
                key={conv.id}
                className="flex items-center justify-between py-1.5 text-sm"
              >
                <span className="text-gray-900 truncate">
                  {conv.title || '제목 없는 대화'}
                </span>
                <span className="text-xs text-gray-500 shrink-0 ml-2">
                  {conv.message_count}개 메시지
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
