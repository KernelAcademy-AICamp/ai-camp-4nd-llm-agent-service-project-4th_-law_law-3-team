'use client'

import type { ScriptMetadata } from '../types'

interface MetadataPanelProps {
  metadata: ScriptMetadata
  onRefresh: () => void
}

export function MetadataPanel({ metadata, onRefresh }: MetadataPanelProps) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-5 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-bold text-gray-800">메타데이터</h3>
        <button
          onClick={onRefresh}
          className="text-xs text-blue-600 hover:text-blue-700"
        >
          재생성
        </button>
      </div>

      {/* 영상 설명문 */}
      <div>
        <p className="text-xs font-medium text-gray-500 mb-1">영상 설명문</p>
        <p className="text-sm text-gray-700 bg-gray-50 rounded-lg p-3">
          {metadata.description}
        </p>
      </div>

      {/* SEO 태그 */}
      <div>
        <p className="text-xs font-medium text-gray-500 mb-1">SEO 태그</p>
        <div className="flex flex-wrap gap-1.5">
          {metadata.tags.map((tag, index) => (
            <span
              key={`tag-${index}`}
              className="px-2 py-0.5 text-xs bg-blue-50 text-blue-600 rounded-full"
            >
              {tag}
            </span>
          ))}
        </div>
      </div>

      {/* 해시태그 */}
      <div>
        <p className="text-xs font-medium text-gray-500 mb-1">해시태그</p>
        <p className="text-sm text-gray-600">
          {metadata.hashtags.join(' ')}
        </p>
      </div>

      {/* CTA */}
      <div>
        <p className="text-xs font-medium text-gray-500 mb-1">CTA</p>
        <p className="text-sm text-gray-700 bg-amber-50 rounded-lg p-3">
          {metadata.cta_text}
        </p>
      </div>
    </div>
  )
}
