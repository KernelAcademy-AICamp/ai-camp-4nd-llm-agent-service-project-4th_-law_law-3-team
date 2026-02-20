'use client'

import type { JudgmentResult } from '../types'

interface JudgmentDisplayProps {
  result: JudgmentResult
  onClose: () => void
  onRestart: () => void
}

export function JudgmentDisplay({
  result,
  onClose,
  onRestart,
}: JudgmentDisplayProps) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-white rounded-xl shadow-xl max-w-2xl w-full mx-4 max-h-[80vh] flex flex-col">
        {/* 헤더 */}
        <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between">
          <h2 className="text-lg font-bold text-gray-900">판결문</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 text-xl"
            aria-label="닫기"
          >
            \u00d7
          </button>
        </div>

        {/* 본문 */}
        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-6">
          {/* 판결 */}
          <section>
            <h3 className="text-sm font-semibold text-gray-700 mb-2">판결 요지</h3>
            <div className="p-4 bg-gray-50 rounded-lg text-sm text-gray-800 whitespace-pre-wrap leading-relaxed">
              {result.judgment}
            </div>
          </section>

          {/* 피드백 */}
          <section>
            <h3 className="text-sm font-semibold text-gray-700 mb-2">
              수행 평가 (강점/약점)
            </h3>
            <div className="p-4 bg-blue-50 rounded-lg text-sm text-gray-700 whitespace-pre-wrap leading-relaxed">
              {result.feedback}
            </div>
          </section>

          {/* 인용 판례 */}
          {result.cited_cases.length > 0 && (
            <section>
              <h3 className="text-sm font-semibold text-gray-700 mb-2">
                인용 판례 ({result.cited_cases.length}건)
              </h3>
              <ul className="space-y-1">
                {result.cited_cases.map((item) => (
                  <li key={item.id} className="text-xs text-gray-600">
                    - {item.title}
                  </li>
                ))}
              </ul>
            </section>
          )}

          {/* 인용 법령 */}
          {result.cited_articles.length > 0 && (
            <section>
              <h3 className="text-sm font-semibold text-gray-700 mb-2">
                인용 법령 ({result.cited_articles.length}건)
              </h3>
              <ul className="space-y-1">
                {result.cited_articles.map((item) => (
                  <li key={item.id} className="text-xs text-gray-600">
                    - {item.title}
                  </li>
                ))}
              </ul>
            </section>
          )}
        </div>

        {/* 푸터 */}
        <div className="px-6 py-4 border-t border-gray-200 flex gap-3 justify-end">
          <button
            onClick={onRestart}
            className="px-4 py-2 text-sm text-blue-600 border border-blue-300 rounded-lg hover:bg-blue-50 transition-colors"
          >
            다시 시작
          </button>
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            닫기
          </button>
        </div>
      </div>
    </div>
  )
}
