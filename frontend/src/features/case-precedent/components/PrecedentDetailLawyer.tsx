import type { ChatSource } from '../types'
import { PrecedentFullTextViewer } from './PrecedentFullTextViewer'

interface PrecedentDetailLawyerProps {
  source: ChatSource
}

export function PrecedentDetailLawyer({ source }: PrecedentDetailLawyerProps) {
  return (
    <div className="space-y-4">
      {/* 판결문 전체 표시 */}
      <PrecedentFullTextViewer data={source} mode="direct" highlightContent={source.content} />

      {/* 그래프 보강 정보 */}
      {(source.cited_statutes?.length || source.similar_cases?.length) ? (
        <div className="mt-6 pt-4 border-t border-gray-200">
          <h3 className="font-bold text-gray-700 mb-3">📊 관련 정보</h3>
          {source.cited_statutes && source.cited_statutes.length > 0 && (
            <p className="text-sm text-gray-600 mb-2">
              <span className="font-medium">인용 법령:</span>{' '}
              {source.cited_statutes.join(', ')}
            </p>
          )}
          {source.similar_cases && source.similar_cases.length > 0 && (
            <p className="text-sm text-gray-600">
              <span className="font-medium">유사 판례:</span>{' '}
              {source.similar_cases.join(', ')}
            </p>
          )}
        </div>
      ) : null}
    </div>
  )
}
