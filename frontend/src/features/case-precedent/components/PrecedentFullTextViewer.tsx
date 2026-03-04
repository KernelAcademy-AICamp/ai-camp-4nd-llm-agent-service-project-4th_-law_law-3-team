'use client'

import type { ChatSource, PrecedentDetail } from '../types'
import { PrecedentDocumentViewer } from './PrecedentDocumentViewer'

/**
 * 판례 데이터를 PrecedentDocumentViewer props로 변환
 *
 * ChatSource (채팅 참조)와 PrecedentDetail (API 응답) 모두 지원
 */
function mapToPrecedentViewerProps(data: ChatSource | PrecedentDetail) {
  // PrecedentDetail에만 있는 필드 (court, date)
  const detail = data as PrecedentDetail

  return {
    courtName: data.court_name || detail.court,
    caseNumber: data.case_number,
    caseName: data.case_name,
    decisionDate: data.decision_date || detail.date,
    summary: data.summary,
    reasoning: data.reasoning,
    referenceProvisions: data.reference_provisions,
    referenceCases: data.reference_cases,
    ruling: data.ruling,
    claim: data.claim,
    fullReason: data.full_reason,
    fullText: data.full_text,
  }
}

interface PrecedentFullTextViewerProps {
  /** 판례 데이터 (ChatSource 또는 PrecedentDetail) */
  data: ChatSource | PrecedentDetail
  /** 표시 모드: accordion (접힘/펼침) | direct (직접 표시) */
  mode?: 'accordion' | 'direct'
  /** 아코디언 모드일 때 제목 */
  title?: string
  /** 아코디언 모드일 때 기본 펼침 상태 */
  defaultOpen?: boolean
  /** RAG 검색 청크 텍스트 (하이라이팅 대상) */
  highlightContent?: string
}

/**
 * 판례 전문 뷰어 컨테이너 컴포넌트
 *
 * 일반인/변호사 서비스 모두에서 사용 가능한 재사용 컴포넌트
 *
 * @example
 * // 일반인 서비스: 아코디언 모드
 * <PrecedentFullTextViewer
 *   data={chatSource}
 *   mode="accordion"
 *   title="📄 판결문 전체 보기"
 * />
 *
 * @example
 * // 변호사 서비스: 직접 표시
 * <PrecedentFullTextViewer
 *   data={precedentDetail}
 *   mode="direct"
 * />
 */
export function PrecedentFullTextViewer({
  data,
  mode = 'direct',
  title = '📄 판결문 전체 보기',
  defaultOpen = false,
  highlightContent,
}: PrecedentFullTextViewerProps) {
  const viewerProps = mapToPrecedentViewerProps(data)

  if (mode === 'accordion') {
    return (
      <div className="pt-4">
        <details
          className="border border-gray-200 rounded-lg overflow-hidden"
          open={defaultOpen}
        >
          <summary className="w-full px-4 py-3 bg-gray-50 flex justify-between items-center hover:bg-gray-100 transition-colors cursor-pointer font-medium text-gray-700">
            {title}
          </summary>
          <div className="p-4 bg-white">
            <PrecedentDocumentViewer {...viewerProps} highlightContent={highlightContent} />
          </div>
        </details>
      </div>
    )
  }

  return <PrecedentDocumentViewer {...viewerProps} highlightContent={highlightContent} />
}

/**
 * Props 매핑 유틸리티 함수 (외부에서 직접 사용 가능)
 */
export { mapToPrecedentViewerProps }
