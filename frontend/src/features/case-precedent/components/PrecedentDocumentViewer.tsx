'use client'

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
}

/**
 * 판결문 원문을 문서 형태로 렌더링하는 컴포넌트
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
}: PrecedentDocumentViewerProps) {
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

  // 섹션 렌더링 컴포넌트
  const Section = ({
    title,
    content,
    bgColor = 'bg-white',
    borderColor = 'border-gray-200'
  }: {
    title: string
    content?: string
    bgColor?: string
    borderColor?: string
  }) => {
    if (!content?.trim()) return null

    return (
      <div className={`rounded-lg border ${borderColor} overflow-hidden mb-4`}>
        <div className={`px-4 py-2.5 ${bgColor} border-b ${borderColor}`}>
          <h3 className="font-bold text-slate-700 tracking-wide text-sm">
            【{title}】
          </h3>
        </div>
        <div className="px-4 py-3 bg-white">
          <p className="text-slate-700 leading-relaxed whitespace-pre-wrap text-[15px]">
            {content}
          </p>
        </div>
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

  return (
    <div className="bg-white rounded-lg shadow-lg overflow-hidden max-w-4xl mx-auto">
      {/* 문서 헤더 */}
      <div className="bg-gradient-to-b from-slate-800 to-slate-700 text-white py-6 px-6 text-center">
        <div className="text-sm tracking-[0.3em] text-slate-300 mb-1">
          {courtName || '대 법 원'}
        </div>
        <div className="text-2xl font-bold tracking-[0.2em] mb-3">
          판 결
        </div>
        <div className="w-16 h-0.5 bg-slate-500 mx-auto" />
      </div>

      {/* 사건 정보 */}
      <div className="bg-slate-100 px-6 py-4 border-b border-slate-300">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-sm">
          {caseNumber && (
            <div className="flex">
              <span className="text-slate-500 w-20 shrink-0">사건번호</span>
              <span className="font-semibold text-slate-800">{caseNumber}</span>
            </div>
          )}
          {caseName && (
            <div className="flex">
              <span className="text-slate-500 w-20 shrink-0">사 건 명</span>
              <span className="font-semibold text-slate-800">{caseName}</span>
            </div>
          )}
          {decisionDate && (
            <div className="flex">
              <span className="text-slate-500 w-20 shrink-0">선고일자</span>
              <span className="font-semibold text-slate-800">{formatDate(decisionDate)}</span>
            </div>
          )}
        </div>
      </div>

      {/* 본문 */}
      <div className="p-6">
        {/* 판시사항 */}
        <Section
          title="판시사항"
          content={summary}
          bgColor="bg-amber-50"
          borderColor="border-amber-200"
        />

        {/* 판결요지 */}
        <Section
          title="판결요지"
          content={reasoning}
          bgColor="bg-blue-50"
          borderColor="border-blue-200"
        />

        {/* 참조조문 */}
        <Section
          title="참조조문"
          content={referenceProvisions}
          bgColor="bg-gray-50"
          borderColor="border-gray-200"
        />

        {/* 참조판례 */}
        <Section
          title="참조판례"
          content={referenceCases}
          bgColor="bg-gray-50"
          borderColor="border-gray-200"
        />

        {/* 주문 */}
        <Section
          title="주    문"
          content={ruling}
          bgColor="bg-green-50"
          borderColor="border-green-200"
        />

        {/* 청구취지 */}
        <Section
          title="청구취지"
          content={claim}
          bgColor="bg-gray-50"
          borderColor="border-gray-200"
        />

        {/* 이유 */}
        <Section
          title="이    유"
          content={fullReason}
          bgColor="bg-slate-50"
          borderColor="border-slate-200"
        />

        {/* 판례내용 전문 (있는 경우) */}
        {fullText && fullText.length > 100 && (
          <Section
            title="전    문"
            content={fullText}
            bgColor="bg-gray-50"
            borderColor="border-gray-300"
          />
        )}
      </div>

      {/* 문서 푸터 */}
      <div className="bg-slate-100 px-6 py-3 text-center text-xs text-slate-500 border-t border-slate-300">
        본 문서는 법률 정보 제공 목적으로만 사용됩니다
      </div>
    </div>
  )
}
