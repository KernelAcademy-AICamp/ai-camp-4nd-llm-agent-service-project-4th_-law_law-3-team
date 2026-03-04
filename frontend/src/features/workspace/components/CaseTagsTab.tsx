'use client'

import { useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { X } from 'lucide-react'
import type { WorkspaceCaseDetail } from '@/features/workspace/types'
import { TAG_TYPE_COLORS } from '@/features/workspace/types'
import { deleteCaseTag } from '@/features/workspace/services'

interface CaseTagsTabProps {
  caseData: WorkspaceCaseDetail
}

export function CaseTagsTab({ caseData }: CaseTagsTabProps) {
  const queryClient = useQueryClient()
  const [minConfidence, setMinConfidence] = useState(0)

  const deleteTagMutation = useMutation({
    mutationFn: ({ tagIndex }: { tagIndex: number }) =>
      deleteCaseTag(caseData.id, tagIndex),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['case', caseData.id] })
    },
  })

  if (caseData.tagged_items.length === 0) {
    return (
      <div className="text-center py-12 text-gray-500">
        <p className="text-lg">추출된 태그가 없습니다</p>
        <p className="text-sm mt-1">대화를 시작하면 자동으로 태그가 수집됩니다</p>
      </div>
    )
  }

  // 신뢰도 필터 적용 (원본 인덱스 유지)
  const filteredTags = caseData.tagged_items
    .map((tag, originalIndex) => ({ tag, originalIndex }))
    .filter(({ tag }) => (tag.confidence ?? 1) >= minConfidence)

  // 유형별 그룹핑
  const grouped: Record<string, { tag: typeof caseData.tagged_items[0]; originalIndex: number }[]> = {}
  for (const entry of filteredTags) {
    const type = entry.tag.type || 'other'
    if (!grouped[type]) grouped[type] = []
    grouped[type].push(entry)
  }

  const hasConfidenceData = caseData.tagged_items.some(
    (tag) => tag.confidence !== undefined && tag.confidence < 1,
  )

  return (
    <div className="space-y-4">
      {/* 신뢰도 필터 */}
      {hasConfidenceData && (
        <div className="bg-white rounded-lg border p-4">
          <div className="flex items-center justify-between mb-2">
            <label className="text-sm font-medium text-gray-700">
              신뢰도 필터
            </label>
            <span className="text-xs text-gray-500">
              {Math.round(minConfidence * 100)}% 이상
              <span className="ml-2 text-gray-400">
                ({filteredTags.length}/{caseData.tagged_items.length})
              </span>
            </span>
          </div>
          <input
            type="range"
            min={0}
            max={100}
            step={5}
            value={minConfidence * 100}
            onChange={(e) => setMinConfidence(Number(e.target.value) / 100)}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
          />
          <div className="flex justify-between text-xs text-gray-400 mt-1">
            <span>0%</span>
            <span>50%</span>
            <span>100%</span>
          </div>
        </div>
      )}

      {/* 태그 그룹 */}
      {Object.entries(grouped).map(([type, entries]) => (
        <div key={type} className="bg-white rounded-lg border p-4">
          <h3 className="text-sm font-medium text-gray-700 mb-3 flex items-center gap-2">
            <span
              className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs ${
                TAG_TYPE_COLORS[type] ?? 'bg-gray-100 text-gray-700'
              }`}
            >
              {type}
            </span>
            <span className="text-gray-400 text-xs">{entries.length}개</span>
          </h3>
          <div className="flex flex-wrap gap-2">
            {entries.map(({ tag, originalIndex }) => (
              <div
                key={`case-tag-${originalIndex}`}
                className="group bg-gray-50 rounded-lg px-3 py-2 text-sm flex items-center gap-1.5"
              >
                <span className="font-medium text-gray-900">
                  {tag.label || tag.value}
                </span>
                {tag.confidence !== undefined && tag.confidence < 1 && (
                  <span className="text-xs text-gray-400">
                    ({Math.round(tag.confidence * 100)}%)
                  </span>
                )}
                {tag.source && (
                  <span className="text-xs text-gray-400">
                    - {tag.source}
                  </span>
                )}
                <button
                  onClick={() => deleteTagMutation.mutate({ tagIndex: originalIndex })}
                  disabled={deleteTagMutation.isPending}
                  className="ml-1 p-0.5 text-gray-300 hover:text-red-500 opacity-0 group-hover:opacity-100 transition-all"
                  aria-label="태그 삭제"
                >
                  <X size={14} />
                </button>
              </div>
            ))}
          </div>
        </div>
      ))}

      {filteredTags.length === 0 && (
        <div className="text-center py-8 text-gray-400">
          <p className="text-sm">현재 필터 조건에 맞는 태그가 없습니다</p>
        </div>
      )}
    </div>
  )
}
