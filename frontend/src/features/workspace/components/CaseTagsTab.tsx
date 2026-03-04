import type { WorkspaceCaseDetail } from '@/features/workspace/types'
import { TAG_TYPE_COLORS } from '@/features/workspace/types'

interface CaseTagsTabProps {
  caseData: WorkspaceCaseDetail
}

export function CaseTagsTab({ caseData }: CaseTagsTabProps) {
  if (caseData.tagged_items.length === 0) {
    return (
      <div className="text-center py-12 text-gray-500">
        <p className="text-lg">추출된 태그가 없습니다</p>
        <p className="text-sm mt-1">대화를 시작하면 자동으로 태그가 수집됩니다</p>
      </div>
    )
  }

  const grouped: Record<string, typeof caseData.tagged_items> = {}
  for (const tag of caseData.tagged_items) {
    const type = tag.type || 'other'
    if (!grouped[type]) grouped[type] = []
    grouped[type].push(tag)
  }

  return (
    <div className="space-y-4">
      {Object.entries(grouped).map(([type, tags]) => (
        <div key={type} className="bg-white rounded-lg border p-4">
          <h3 className="text-sm font-medium text-gray-700 mb-3 flex items-center gap-2">
            <span
              className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs ${
                TAG_TYPE_COLORS[type] ?? 'bg-gray-100 text-gray-700'
              }`}
            >
              {type}
            </span>
            <span className="text-gray-400 text-xs">{tags.length}개</span>
          </h3>
          <div className="flex flex-wrap gap-2">
            {tags.map((tag, i) => (
              <div
                key={`case-tag-${i}`}
                className="bg-gray-50 rounded-lg px-3 py-2 text-sm"
              >
                <span className="font-medium text-gray-900">
                  {tag.label || tag.value}
                </span>
                {tag.confidence !== undefined && tag.confidence < 1 && (
                  <span className="ml-1 text-xs text-gray-400">
                    ({Math.round(tag.confidence * 100)}%)
                  </span>
                )}
                {tag.source && (
                  <span className="ml-1 text-xs text-gray-400">
                    - {tag.source}
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}
