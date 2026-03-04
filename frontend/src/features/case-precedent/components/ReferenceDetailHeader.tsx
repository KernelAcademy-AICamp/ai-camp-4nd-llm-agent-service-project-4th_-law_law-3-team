import Image from 'next/image'
import type { ChatSource } from '../types'
import { getLawTypeLogo, getLawTypeOrgName, getCourtLogo, getDocTypeLogo, DEFAULT_GOV_LOGO } from '../utils/lawTypeLogo'
import { getDocTypeLabel, getDocTypeBadgeColor } from '../utils/docTypeUtils'

interface ReferenceDetailHeaderProps {
  selectedRef: ChatSource
}

export function ReferenceDetailHeader({ selectedRef }: ReferenceDetailHeaderProps) {
  const isLaw = selectedRef.doc_type === 'law'
  const title = isLaw ? selectedRef.law_name : selectedRef.case_name
  const subtitle = isLaw
    ? (selectedRef.article_number
        ? `제${selectedRef.article_number}${selectedRef.article_title ? ` (${selectedRef.article_title})` : ''}`
        : selectedRef.law_type)
    : selectedRef.case_number

  const logoPath = isLaw
    ? getLawTypeLogo(selectedRef.law_type)
    : getCourtLogo(selectedRef.court_name) || getDocTypeLogo(selectedRef.doc_type)
  const altText = isLaw
    ? getLawTypeOrgName(selectedRef.law_type)
    : (selectedRef.court_name || '법원')
  const orgName = isLaw
    ? getLawTypeOrgName(selectedRef.law_type)
    : (selectedRef.court_name || (selectedRef.doc_type === 'constitutional' ? '헌법재판소' : '대법원'))

  return (
    <div className="mb-8 pb-6 border-b border-gray-100">
      {/* 발행기관/법원 로고 */}
      <div className="flex items-center gap-3 mb-4">
        <Image
          src={logoPath || DEFAULT_GOV_LOGO}
          alt={altText}
          width={48}
          height={48}
          className="object-contain"
          unoptimized
        />
        <span className="text-sm text-gray-500 font-medium">
          {orgName}
        </span>
      </div>

      {/* 배지 영역 */}
      <div className="flex items-center gap-2 flex-wrap mb-4">
        <span className={`px-3 py-1 rounded-full text-sm font-medium ${getDocTypeBadgeColor(selectedRef.doc_type)}`}>
          {getDocTypeLabel(selectedRef.doc_type)}
        </span>
        {/* 법령종류 배지 (법령일 때만) */}
        {isLaw && selectedRef.law_type && (
          <span className="px-3 py-1 rounded-full text-sm font-medium bg-emerald-50 text-emerald-700">
            {selectedRef.law_type}
          </span>
        )}
        {/* 사건유형 (민사/형사/행정) */}
        {selectedRef.case_type && (
          <span className="px-3 py-1 rounded-full text-sm font-medium bg-gray-100 text-gray-600">
            {selectedRef.case_type}
          </span>
        )}
        {/* 선고일 */}
        {selectedRef.decision_date && (
          <span className="px-3 py-1 rounded-full text-sm font-medium bg-gray-50 text-gray-500">
            {selectedRef.decision_date}
          </span>
        )}
      </div>

      {/* 제목 */}
      <h1 className="text-2xl font-bold text-gray-900 leading-tight mb-4">
        {title || '상세 정보'}
      </h1>

      {/* 부제목 */}
      {subtitle && (
        <div className="text-gray-500 font-mono text-sm bg-gray-50 px-3 py-1 rounded inline-block mb-4">
          {subtitle}
        </div>
      )}

      {/* 핵심 쟁점 (판시사항) */}
      {selectedRef.summary && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3 text-sm text-yellow-800">
          <strong>핵심 쟁점:</strong> {selectedRef.summary}
        </div>
      )}
    </div>
  )
}
