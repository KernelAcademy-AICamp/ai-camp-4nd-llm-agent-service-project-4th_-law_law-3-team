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

  // 글자 사이에 공백 추가 (판 시 사 항)
  const addLetterSpacing = (text: string) => {
    return text.split('').join(' ')
  }

  // 항목 패턴 앞에 줄바꿈 추가
  const formatListContent = (text: string) => {
    // 패턴 정의 (줄바꿈 1번, 이미 줄바꿈이 있으면 적용 안함):
    // 1. 한글 항목: 가., 나., 다., ... (앞에 공백이 있고 줄바꿈이 아닌 경우)
    // 2. 숫자 항목: 1., 2., 3., ... (앞에 공백, 뒤에 숫자가 아닌 문자 - 날짜 제외)
    // 3. 괄호 숫자: (1), (2), (3), ... (앞에 줄바꿈이 없는 경우)
    // 4. 순서 표현: 첫째,, 둘째,, 셋째,, ... (앞에 줄바꿈이 없는 경우)
    const singleLinePatterns = [
      /(?<=\s)(?<!\n)([가나다라마바사아자차카타파하]\.\s)/g,  // 한글 항목
      /(?<=\s)(?<!\n)(\d+\.\s)(?![\d선]|법률)/g,  // 숫자 항목 (뒤에 숫자, "선고", "법률" 오면 날짜로 간주하여 제외)
      /(?<!\n)(\(\d+\)\s?)/g,  // 괄호 숫자
      /(?<!\n)(첫째,|둘째,|셋째,|넷째,|다섯째,|여섯째,|일곱째,|여덟째,|아홉째,|열째,)/g,  // 순서 표현
    ]

    // 대괄호 패턴 (줄바꿈 2번): 【...】
    // 이미 줄바꿈 2번이 있으면 적용 안함
    const doubleLinePattern = /(?<!\n\n)(【[^】]+】)/g

    let result = text

    // 줄바꿈 1번 추가
    for (const pattern of singleLinePatterns) {
      result = result.replace(pattern, '\n$1')
    }

    // 대괄호는 줄바꿈 2번 추가
    result = result.replace(doubleLinePattern, '\n\n$1')

    // 문서 시작 부분의 불필요한 줄바꿈 제거
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

  // 섹션 콘텐츠
  const SectionContent = ({ content }: { content?: string }) => {
    if (!content?.trim()) return null
    const formattedContent = formatListContent(content)
    return (
      <div className="px-8 pb-8">
        <p className="text-gray-800 leading-loose whitespace-pre-wrap text-[15px]">
          {formattedContent}
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

  // 헤더 타이틀 생성 (서울중앙지방법원 1995. 9. 28. 선고 95노1985 판결)
  const headerTitle = `${courtName || '대법원'} ${formatDate(decisionDate)} 선고 ${caseNumber || ''} 판결`

  return (
    <div className="bg-gray-50 py-6">
      {/* 문서 헤더 */}
      <div className="text-center mb-6">
        <h1 className="text-xl font-bold text-gray-900 mb-2">
          {headerTitle}
        </h1>
        {caseName && (
          <p className="text-gray-600 mb-1">[{caseName}]</p>
        )}
        <p className="text-sm text-gray-400">대법원 종합법률정보</p>
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
