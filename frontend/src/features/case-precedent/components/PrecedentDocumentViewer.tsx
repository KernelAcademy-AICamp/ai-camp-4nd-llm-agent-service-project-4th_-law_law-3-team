'use client'

import { useRef, useEffect, type ReactNode } from 'react'
import { findHighlightRanges, splitByHighlights } from '../utils/highlightUtils'

interface PrecedentDocumentViewerProps {
  courtName?: string
  caseNumber?: string
  caseName?: string
  decisionDate?: string
  // 판결문 필드들
  summary?: string           // 판시사항
  reasoning?: string         // 판결요지
  referenceProvisions?: string  // 참조조문
  referenceCases?: string    // 참조판례
  ruling?: string            // 주문
  claim?: string             // 청구취지
  fullReason?: string        // 이유
  fullText?: string          // 판례내용 전문
  /** RAG 검색 청크 텍스트 (하이라이팅 대상) */
  highlightContent?: string
}

/**
 * 판결문 원문을 공식 문서 형태로 렌더링하는 컴포넌트
 * 대법원 종합법률정보 스타일 적용
 */
export function PrecedentDocumentViewer({
  courtName,
  caseNumber,
  caseName,
  decisionDate,
  summary,
  reasoning,
  referenceProvisions,
  referenceCases,
  ruling,
  claim,
  fullReason,
  fullText,
  highlightContent,
}: PrecedentDocumentViewerProps) {
  const firstHighlightRef = useRef<HTMLElement | null>(null)
  // 하이라이팅 위치로 자동 스크롤을 위한 ref 사용 여부 추적
  const hasScrolledRef = useRef(false)

  useEffect(() => {
    hasScrolledRef.current = false
  }, [highlightContent])

  useEffect(() => {
    if (firstHighlightRef.current && !hasScrolledRef.current) {
      hasScrolledRef.current = true
      // 렌더링 완료 후 스크롤
      const timer = setTimeout(() => {
        firstHighlightRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' })
      }, 300)
      return () => clearTimeout(timer)
    }
  })

  // 선고일 포맷팅 (19900612 → 1990. 6. 12.)
  const formatDate = (dateStr?: string) => {
    if (!dateStr) return ''
    const cleaned = String(dateStr).replace(/-/g, '')
    if (cleaned.length === 8) {
      const year = cleaned.slice(0, 4)
      const month = parseInt(cleaned.slice(4, 6), 10)
      const day = parseInt(cleaned.slice(6, 8), 10)
      return `${year}. ${month}. ${day}.`
    }
    return dateStr
  }

  // 글자 사이에 공백 추가 (판 시 사 항)
  const addLetterSpacing = (text: string) => {
    return text.split('').join(' ')
  }

  // 항목 패턴 앞에 줄바꿈 추가
  const formatListContent = (text: string) => {
    const singleLinePatterns = [
      /(?<=\s)(?<!\n)([가나다라마바사아자차카타파하]\.\s)/g,
      /(?<=\s)(?<!\n)(\d+\.\s)(?![\d선]|법률)/g,
      /(?<!\n)(\(\d+\)\s?)/g,
      /(?<!\n)(첫째,|둘째,|셋째,|넷째,|다섯째,|여섯째,|일곱째,|여덟째,|아홉째,|열째,)/g,
    ]
    const doubleLinePattern = /(?<!\n\n)(【[^】]+】)/g
    const judgePattern = /(?<!\n\n)((?:대법관|대법원장|판사)\s)/g

    let result = text
    for (const pattern of singleLinePatterns) {
      result = result.replace(pattern, '\n$1')
    }
    result = result.replace(doubleLinePattern, '\n\n$1')
    result = result.replace(judgePattern, '\n\n$1')
    return result.replace(/^\n+/, '')
  }

  // 섹션 타이틀 렌더링
  const SectionTitle = ({ title }: { title: string }) => (
    <h3 className="text-center text-xl font-bold text-gray-900 tracking-[0.3em] py-8">
      {addLetterSpacing(title)}
    </h3>
  )

  // 섹션 구분선
  const Divider = () => (
    <hr className="border-t border-gray-200 my-4" />
  )

  // 섹션 콘텐츠 (하이라이팅 지원)
  const SectionContent = ({ content }: { content?: string }) => {
    if (!content?.trim()) return null
    const formattedContent = formatListContent(content)

    let rendered: ReactNode = formattedContent
    if (highlightContent) {
      const ranges = findHighlightRanges(formattedContent, highlightContent)
      if (ranges.length > 0) {
        const refToUse = !hasScrolledRef.current ? firstHighlightRef : undefined
        rendered = splitByHighlights(formattedContent, ranges, refToUse)
      }
    }

    return (
      <div className="px-8 pb-8">
        <p className="text-gray-800 leading-loose whitespace-pre-wrap text-base">
          {rendered}
        </p>
      </div>
    )
  }

  const hasContent = summary || reasoning || referenceProvisions || ruling || fullReason || fullText

  if (!hasContent) {
    return (
      <div className="text-center text-gray-500 py-8">
        판례 원문 데이터가 없습니다.
      </div>
    )
  }

  // 헤더 타이틀 생성
  const headerTitle = `${courtName || '대법원'} ${formatDate(decisionDate)} 선고 ${caseNumber || ''} 판결`

  return (
    <div className="bg-white py-6">
      {/* 문서 헤더 */}
      <div className="text-center mb-10 mt-10 max-w-4xl mx-auto px-8">
        <h1 className="text-2xl font-bold text-gray-900 mb-3">
          {headerTitle}
        </h1>
        {caseName && (
          <p className="text-lg text-gray-600 mb-2 break-all">[{caseName}]</p>
        )}
        <p className="text-base text-gray-400">대법원 종합법률정보</p>
      </div>

      {/* 본문 카드 */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 max-w-4xl mx-auto">
        {/* 판시사항 */}
        {summary?.trim() && (
          <>
            <SectionTitle title="판시사항" />
            <SectionContent content={summary} />
            <Divider />
          </>
        )}

        {/* 판결요지 */}
        {reasoning?.trim() && (
          <>
            <SectionTitle title="판결요지" />
            <SectionContent content={reasoning} />
            <Divider />
          </>
        )}

        {/* 참조조문 */}
        {referenceProvisions?.trim() && (
          <>
            <SectionTitle title="참조조문" />
            <SectionContent content={referenceProvisions} />
            <Divider />
          </>
        )}

        {/* 참조판례 */}
        {referenceCases?.trim() && (
          <>
            <SectionTitle title="참조판례" />
            <SectionContent content={referenceCases} />
            <Divider />
          </>
        )}

        {/* 주문 */}
        {ruling?.trim() && (
          <>
            <SectionTitle title="주문" />
            <SectionContent content={ruling} />
            <Divider />
          </>
        )}

        {/* 청구취지 */}
        {claim?.trim() && (
          <>
            <SectionTitle title="청구취지" />
            <SectionContent content={claim} />
            <Divider />
          </>
        )}

        {/* 이유 */}
        {fullReason?.trim() && (
          <>
            <SectionTitle title="이유" />
            <SectionContent content={fullReason} />
          </>
        )}

        {/* 판례내용 전문 (fullReason이 없고 fullText가 있는 경우만) */}
        {!fullReason?.trim() && fullText?.trim() && fullText.length > 100 && (
          <>
            <SectionTitle title="전문" />
            <SectionContent content={fullText} />
          </>
        )}
      </div>

      {/* 푸터 */}
      <div className="text-center mt-6 text-xs text-gray-400">
        본 문서는 법률 정보 제공 목적으로만 사용됩니다
      </div>
    </div>
  )
}
