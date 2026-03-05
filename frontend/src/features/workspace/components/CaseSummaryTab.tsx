import type { WorkspaceCaseDetail, TimelineItem } from '@/features/workspace/types'
import { TAG_TYPE_COLORS } from '@/features/workspace/types'

interface CaseSummaryTabProps {
  caseData: WorkspaceCaseDetail
  timeline: TimelineItem[]
}

export function CaseSummaryTab({ caseData, timeline }: CaseSummaryTabProps) {
  const tagsByType: Record<string, number> = {}
  for (const tag of caseData.tagged_items) {
    tagsByType[tag.type] = (tagsByType[tag.type] || 0) + 1
  }

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
