import type { TimelineItem } from '@/features/workspace/types'

interface CaseTimelineTabProps {
  timeline: TimelineItem[]
  isRebuilding: boolean
  onRebuild: () => void
  onOpenStoryboard: () => void
}

export function CaseTimelineTab({
  timeline,
  isRebuilding,
  onRebuild,
  onOpenStoryboard,
}: CaseTimelineTabProps) {
  const sorted = [...timeline].sort((a, b) => {
    if (a.date_normalized && b.date_normalized) {
      return a.date_normalized.localeCompare(b.date_normalized)
    }
    if (a.date_normalized) return -1
    if (b.date_normalized) return 1
    return 0
  })

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium text-gray-700">
          타임라인 ({timeline.length}개 항목)
        </h3>
        <div className="flex items-center gap-2">
          <button
            onClick={onOpenStoryboard}
            className="px-3 py-1.5 text-xs font-medium text-indigo-600 border border-indigo-200 rounded-lg hover:bg-indigo-50 transition-colors"
          >
            스토리보드 AI로 생성
          </button>
          <button
            onClick={onRebuild}
            disabled={isRebuilding}
            className="px-3 py-1.5 text-xs font-medium text-blue-600 border border-blue-200 rounded-lg hover:bg-blue-50 disabled:opacity-50 transition-colors"
          >
            {isRebuilding ? '재생성 중...' : '타임라인 재생성'}
          </button>
        </div>
      </div>

      {sorted.length === 0 ? (
        <div className="text-center py-12 text-gray-500">
          <p className="text-lg">타임라인 항목이 없습니다</p>
          <p className="text-sm mt-1">
            &ldquo;스토리보드 AI로 생성&rdquo;으로 대화 내용에서 타임라인을 추출하거나,
            &ldquo;타임라인 재생성&rdquo;으로 태그에서 타임라인을 만들어 보세요
          </p>
        </div>
      ) : (
        <div className="relative pl-6 space-y-4">
          {/* 세로 선 */}
          <div className="absolute left-2 top-0 bottom-0 w-0.5 bg-gray-200" />

          {sorted.map((item) => (
            <div key={item.id} className="relative">
              {/* 점 */}
              <div
                className={`absolute -left-4 top-1 w-3 h-3 rounded-full border-2 border-white ${
                  item.source_type === 'manual'
                    ? 'bg-blue-500'
                    : item.confidence >= 0.8
                      ? 'bg-green-500'
                      : item.confidence >= 0.5
                        ? 'bg-yellow-500'
                        : 'bg-red-500'
                }`}
              />
              <div className="bg-white rounded-lg border p-3">
                <div className="flex items-start justify-between">
                  <div>
                    <p className="font-medium text-gray-900 text-sm">
                      {item.title}
                    </p>
                    {item.description && (
                      <p className="text-xs text-gray-500 mt-1">
                        {item.description}
                      </p>
                    )}
                  </div>
                  <div className="text-right shrink-0 ml-4">
                    {item.date_text && (
                      <p className="text-xs font-medium text-gray-700">
                        {item.date_text}
                      </p>
                    )}
                    {item.category && (
                      <span className="inline-flex items-center px-1.5 py-0.5 rounded text-xs bg-gray-100 text-gray-600 mt-1">
                        {item.category}
                      </span>
                    )}
                  </div>
                </div>
                <div className="flex items-center gap-2 mt-2 text-xs text-gray-400">
                  <span>
                    신뢰도: {Math.round(item.confidence * 100)}%
                  </span>
                  <span>|</span>
                  <span>{item.source_type}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
