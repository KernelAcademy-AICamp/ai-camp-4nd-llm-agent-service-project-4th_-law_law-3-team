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

const CATEGORY_TAB_COLORS: Record<
  string,
  { active: string; inactive: string }
> = {
  CIVIL: {
    active: 'bg-blue-600 text-white shadow-sm',
    inactive: 'text-blue-700 hover:bg-blue-50',
  },
  CRIMINAL: {
    active: 'bg-red-600 text-white shadow-sm',
    inactive: 'text-red-700 hover:bg-red-50',
  },
  PUBLIC: {
    active: 'bg-green-600 text-white shadow-sm',
    inactive: 'text-green-700 hover:bg-green-50',
  },
}

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
      {/* 과목 선택 헤더 */}
      <div className="px-3 pt-3 pb-1.5">
        <h3 className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider">
          과목 선택
        </h3>
      </div>

      {/* 카테고리 탭 */}
      <div className="flex gap-1 px-2 pb-2 border-b border-gray-200">
        {CATEGORY_ORDER.map((cat) => {
          const info = CATEGORY_MAP[cat]
          const isActive = selectedCategory === cat
          const colors = CATEGORY_TAB_COLORS[cat]
          return (
            <button
              key={cat}
              onClick={() => onCategoryChange(cat)}
              className={`flex-1 px-2 py-1.5 text-xs font-semibold rounded-md transition-all ${
                isActive ? colors.active : colors.inactive
              }`}
            >
              {info.label}
            </button>
          )
        })}
      </div>

      {/* 회차 목록 헤더 */}
      <div className="px-3 pt-3 pb-1.5">
        <h3 className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider">
          회차 ({exams.length})
        </h3>
      </div>

      {/* 회차 목록 */}
      <div className="flex-1 overflow-y-auto px-2 pb-2 space-y-1">
        {isLoading ? (
          <div className="flex items-center justify-center py-8 text-sm text-gray-400">
            불러오는 중...
          </div>
        ) : exams.length === 0 ? (
          <div className="flex items-center justify-center py-8 text-sm text-gray-400">
            문제가 없습니다
          </div>
        ) : (
          [...exams]
            .sort((a, b) => a.session - b.session)
            .map((exam) => {
              const isSelected =
                selectedExam?.category === exam.category &&
                selectedExam?.session === exam.session
              return (
                <button
                  key={`${exam.category}-${exam.session}`}
                  onClick={() => onExamSelect(exam)}
                  className={`w-full text-left px-3 py-2.5 rounded-lg text-sm transition-all ${
                    isSelected
                      ? 'bg-gray-900 text-white shadow-sm'
                      : 'hover:bg-gray-50 text-gray-700'
                  }`}
                >
                  <div className="flex items-center gap-2.5">
                    <span
                      className={`shrink-0 w-8 h-8 rounded-lg flex items-center justify-center text-xs font-bold ${
                        isSelected
                          ? 'bg-white/20 text-white'
                          : 'bg-gray-100 text-gray-600'
                      }`}
                    >
                      {exam.session}
                    </span>
                    <div>
                      <div className="font-medium">제{exam.session}회</div>
                      <div
                        className={`text-xs mt-0.5 ${
                          isSelected ? 'text-gray-300' : 'text-gray-500'
                        }`}
                      >
                        {exam.year}년
                      </div>
                    </div>
                  </div>
                </button>
              )
            })
        )}
      </div>
    </div>
  )
}
