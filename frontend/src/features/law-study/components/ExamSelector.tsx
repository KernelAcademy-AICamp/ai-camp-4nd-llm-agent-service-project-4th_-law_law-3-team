'use client'

import { CATEGORY_MAP, CATEGORY_ORDER } from '../types'
import type { ExamFile } from '../types'

interface ExamSelectorProps {
  selectedCategory: string
  onCategoryChange: (category: string) => void
  exams: ExamFile[]
  selectedExam: ExamFile | null
  onExamSelect: (exam: ExamFile) => void
  isLoading: boolean
}

const CATEGORY_COLORS: Record<string, string> = {
  CIVIL: 'border-blue-500 bg-blue-50 text-blue-700',
  CRIMINAL: 'border-red-500 bg-red-50 text-red-700',
  PUBLIC: 'border-green-500 bg-green-50 text-green-700',
}

const CATEGORY_INACTIVE = 'border-gray-200 bg-white text-gray-600 hover:bg-gray-50'

export function ExamSelector({
  selectedCategory,
  onCategoryChange,
  exams,
  selectedExam,
  onExamSelect,
  isLoading,
}: ExamSelectorProps) {
  return (
    <div className="flex flex-col h-full">
      {/* 카테고리 탭 */}
      <div className="flex gap-1 p-2 border-b border-gray-200">
        {CATEGORY_ORDER.map((cat) => {
          const info = CATEGORY_MAP[cat]
          const isActive = selectedCategory === cat
          return (
            <button
              key={cat}
              onClick={() => onCategoryChange(cat)}
              className={`flex-1 px-2 py-1.5 text-xs font-medium rounded border transition-colors ${
                isActive ? CATEGORY_COLORS[cat] : CATEGORY_INACTIVE
              }`}
            >
              {info.label}
            </button>
          )
        })}
      </div>

      {/* 회차 목록 */}
      <div className="flex-1 overflow-y-auto p-2 space-y-1">
        {isLoading ? (
          <div className="flex items-center justify-center py-8 text-sm text-gray-400">
            불러오는 중...
          </div>
        ) : exams.length === 0 ? (
          <div className="flex items-center justify-center py-8 text-sm text-gray-400">
            문제가 없습니다
          </div>
        ) : (
          exams
            .sort((a, b) => a.session - b.session)
            .map((exam) => {
              const isSelected =
                selectedExam?.category === exam.category &&
                selectedExam?.session === exam.session
              return (
                <button
                  key={`${exam.category}-${exam.session}`}
                  onClick={() => onExamSelect(exam)}
                  className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors ${
                    isSelected
                      ? 'border border-blue-500 bg-blue-50 text-blue-900'
                      : 'border border-transparent hover:bg-gray-50 text-gray-700'
                  }`}
                >
                  <div className="font-medium">제{exam.session}회</div>
                  <div className="text-xs text-gray-500 mt-0.5">{exam.year}년</div>
                </button>
              )
            })
        )}
      </div>
    </div>
  )
}
