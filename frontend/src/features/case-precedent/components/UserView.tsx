'use client'

import { useState, useEffect } from 'react'
import Image from 'next/image'
import ReactMarkdown from 'react-markdown'
import { useChat } from '@/context/ChatContext'
import type { ChatSource } from '../types'
import { getLawTypeLogo, getLawTypeOrgName, getCourtLogo, getDocTypeLogo, DEFAULT_GOV_LOGO } from '../utils/lawTypeLogo'

// 아코디언 섹션 컴포넌트
function CollapsibleSection({
  title,
  content,
  defaultOpen = false,
}: {
  title: string
  content?: string
  defaultOpen?: boolean
}) {
  const [isOpen, setIsOpen] = useState(defaultOpen)

  if (!content) return null

  return (
    <div className="border border-gray-200 rounded-lg overflow-hidden mb-3">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-4 py-3 bg-gray-50 flex justify-between items-center hover:bg-gray-100 transition-colors"
      >
        <span className="font-medium text-gray-700">{title}</span>
        <span className="text-gray-400">{isOpen ? '▲' : '▼'}</span>
      </button>
      {isOpen && (
        <div className="p-4 bg-white text-gray-600 text-sm leading-relaxed whitespace-pre-wrap">
          {content}
        </div>
      )}
    </div>
  )
}

// doc_type을 한글로 변환
const docTypeLabels: Record<string, string> = {
  precedent: '판례',
  constitutional: '헌재결정',
  administration: '행정심판',
  legislation: '법령해석',
  committee: '위원회결정',
  law: '법령',
}

const getDocTypeLabel = (docType: string): string => {
  return docTypeLabels[docType] || docType
}

// doc_type별 배지 색상
const getDocTypeBadgeColor = (docType: string): string => {
  switch (docType) {
    case 'law':
      return 'bg-green-50 text-green-600'
    case 'precedent':
      return 'bg-blue-50 text-blue-600'
    case 'constitutional':
      return 'bg-purple-50 text-purple-600'
    case 'committee':
      return 'bg-orange-50 text-orange-600'
    case 'administration':
      return 'bg-yellow-50 text-yellow-700'
    case 'legislation':
      return 'bg-teal-50 text-teal-600'
    default:
      return 'bg-gray-50 text-gray-600'
  }
}

export function UserView() {
  const { sessionData, userRole } = useChat()
  const [references, setReferences] = useState<ChatSource[]>([])
  const [selectedRef, setSelectedRef] = useState<ChatSource | null>(null)
  const isLawyer = userRole === 'lawyer'

  useEffect(() => {
    // 세션 데이터에서 챗봇 참조 자료를 로드합니다.
    if (sessionData.aiReferences && Array.isArray(sessionData.aiReferences)) {
      setReferences(sessionData.aiReferences as ChatSource[])
    }
  }, [sessionData.aiReferences])

  if (references.length === 0) {
    return (
      <div className="h-full flex flex-col items-center justify-center bg-gray-50 text-center p-8 animate-in fade-in duration-500">
        <div className="bg-white p-8 rounded-3xl shadow-sm border border-gray-100 max-w-md">
          <span className="text-6xl mb-6 block">🤖</span>
          <h2 className="text-2xl font-bold text-gray-900 mb-3">
            챗봇에게 질문해보세요!
          </h2>
          <p className="text-gray-500 leading-relaxed">
            &quot;사기죄 성립 요건이 뭐야?&quot;<br />
            &quot;야간 주거침입 시 정당방위는?&quot;
            <br /><br />
            오른쪽 챗봇에게 법률 문제를 물어보면,<br />
            참고한 <strong>판례와 법령 상세 정보</strong>가 이곳에 표시됩니다.
          </p>
        </div>
      </div>
    )
  }

  // Detail View
  if (selectedRef) {
    const isLaw = selectedRef.doc_type === 'law'
    const isPrecedent = selectedRef.doc_type === 'precedent'
    const title = isLaw ? selectedRef.law_name : selectedRef.case_name
    const subtitle = isLaw ? selectedRef.law_type : selectedRef.case_number

    return (
      <div className="h-full flex flex-col bg-white animate-in slide-in-from-right duration-300">
        <div className="p-4 border-b border-gray-100 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setSelectedRef(null)}
              className="p-2 hover:bg-gray-100 rounded-full transition-colors text-gray-500"
            >
              ←
            </button>
            <span className="font-bold text-gray-900">상세 내용</span>
          </div>
          <span className={`px-2 py-1 rounded text-xs font-medium ${
            isLawyer ? 'bg-purple-100 text-purple-700' : 'bg-blue-100 text-blue-700'
          }`}>
            {isLawyer ? '변호사 모드' : '일반인 모드'}
          </span>
        </div>

        <div className="flex-1 overflow-y-auto p-6 md:p-8">
          <div className="max-w-3xl mx-auto">
            <div className="mb-8 pb-6 border-b border-gray-100">
              {/* 발행기관/법원 로고 표시 */}
              <div className="flex items-center gap-3 mb-4">
                {(() => {
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
                    <>
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
                    </>
                  )
                })()}
              </div>
              <span className={`inline-block px-3 py-1 rounded-full text-sm font-medium mb-4 ${getDocTypeBadgeColor(selectedRef.doc_type)}`}>
                {getDocTypeLabel(selectedRef.doc_type)}
              </span>
              <h1 className="text-2xl font-bold text-gray-900 leading-tight mb-4">
                {title || '상세 정보'}
              </h1>
              {subtitle && (
                <div className="text-gray-500 font-mono text-sm bg-gray-50 px-3 py-1 rounded inline-block">
                  {subtitle}
                </div>
              )}
            </div>

            {/* 판례인 경우 역할별 차등 표시 */}
            {isPrecedent ? (
              <div className="space-y-4">
                {/* 일반인 모드: 판결요지 + 주문 중심 */}
                {!isLawyer ? (
                  <>
                    {/* 판결요지 (핵심) */}
                    {selectedRef.reasoning && (
                      <div className="bg-blue-50 p-5 rounded-xl border border-blue-100">
                        <h3 className="font-bold text-blue-800 mb-3 flex items-center gap-2">
                          <span>📋</span> 판결 요지 (핵심 내용)
                        </h3>
                        <p className="text-gray-700 leading-relaxed whitespace-pre-wrap">
                          {selectedRef.reasoning}
                        </p>
                      </div>
                    )}

                    {/* 주문 (결과) */}
                    {selectedRef.ruling && (
                      <div className="bg-green-50 p-5 rounded-xl border border-green-100">
                        <h3 className="font-bold text-green-800 mb-3 flex items-center gap-2">
                          <span>⚖️</span> 주문 (판결 결과)
                        </h3>
                        <p className="text-gray-700 leading-relaxed whitespace-pre-wrap">
                          {selectedRef.ruling}
                        </p>
                      </div>
                    )}

                    {/* 더 보기 (접힘) */}
                    <div className="pt-4">
                      <p className="text-sm text-gray-500 mb-3">▼ 더 자세한 내용</p>
                      <CollapsibleSection title="청구취지" content={selectedRef.claim} />
                      <CollapsibleSection title="검색된 원문" content={selectedRef.content} />
                    </div>
                  </>
                ) : (
                  /* 변호사 모드: 전체 표시 */
                  <>
                    <CollapsibleSection title="판결요지" content={selectedRef.reasoning} defaultOpen />
                    <CollapsibleSection title="주문" content={selectedRef.ruling} defaultOpen />
                    <CollapsibleSection title="청구취지" content={selectedRef.claim} defaultOpen />
                    <CollapsibleSection title="이유 (상세)" content={selectedRef.full_reason} />
                    <CollapsibleSection title="검색된 원문" content={selectedRef.content} />
                  </>
                )}

                {/* 그래프 보강 정보 */}
                {(selectedRef.cited_statutes?.length || selectedRef.similar_cases?.length) && (
                  <div className="mt-6 pt-4 border-t border-gray-200">
                    <h3 className="font-bold text-gray-700 mb-3">📊 관련 정보</h3>
                    {selectedRef.cited_statutes && selectedRef.cited_statutes.length > 0 && (
                      <p className="text-sm text-gray-600 mb-2">
                        <span className="font-medium">인용 법령:</span>{' '}
                        {selectedRef.cited_statutes.join(', ')}
                      </p>
                    )}
                    {selectedRef.similar_cases && selectedRef.similar_cases.length > 0 && (
                      <p className="text-sm text-gray-600">
                        <span className="font-medium">유사 판례:</span>{' '}
                        {selectedRef.similar_cases.join(', ')}
                      </p>
                    )}
                  </div>
                )}
              </div>
            ) : (
              /* 법령인 경우 기존 방식 */
              <div className="prose prose-lg max-w-none text-gray-700">
                <div className="bg-gray-50 p-6 rounded-2xl border border-gray-100 leading-relaxed">
                  <ReactMarkdown>
                    {selectedRef.content || "상세 내용을 불러올 수 없습니다."}
                  </ReactMarkdown>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    )
  }

  // List View
  return (
    <div className="h-full flex flex-col bg-gray-50 animate-in slide-in-from-left duration-300">
      <div className="p-6 border-b border-gray-100 bg-white">
        <h2 className="text-xl font-bold text-gray-900 flex items-center gap-2">
          <span>📚</span> 챗봇 참조 자료
        </h2>
        <p className="text-sm text-gray-500 mt-2">
          챗봇이 답변을 생성할 때 참고한 근거 자료들입니다.<br/>
          자세히 보려면 항목을 클릭하세요.
        </p>
      </div>
      
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {references.map((ref, idx) => {
          const isLaw = ref.doc_type === 'law'
          const title = isLaw ? ref.law_name : ref.case_name
          const subtitle = isLaw ? ref.law_type : ref.case_number

          // 로고 경로 결정: 법령 → law_type 기준, 판례 → court_name 또는 doc_type 기준
          const logoPath = isLaw
            ? getLawTypeLogo(ref.law_type)
            : getCourtLogo(ref.court_name) || getDocTypeLogo(ref.doc_type)

          return (
            <button
              key={`${ref.case_number || ref.law_name}-${idx}`}
              onClick={() => setSelectedRef(ref)}
              className="w-full text-left p-5 rounded-xl bg-white border border-gray-100 hover:border-blue-300 hover:shadow-md transition-all duration-200 group"
            >
              <div className="flex items-start gap-3">
                {/* 로고 표시 (법령 및 판례 모두) */}
                <div className="shrink-0 w-10 h-10 flex items-center justify-center bg-gray-50 rounded-lg">
                  {logoPath ? (
                    <Image
                      src={logoPath}
                      alt={isLaw ? getLawTypeOrgName(ref.law_type) : (ref.court_name || '법원')}
                      width={32}
                      height={32}
                      className="object-contain"
                      unoptimized
                    />
                  ) : (
                    <Image
                      src={DEFAULT_GOV_LOGO}
                      alt="대한민국 정부"
                      width={32}
                      height={32}
                      className="object-contain"
                      unoptimized
                    />
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-2">
                    <span className={`px-2 py-0.5 rounded text-xs font-bold ${getDocTypeBadgeColor(ref.doc_type)}`}>
                      {getDocTypeLabel(ref.doc_type)}
                    </span>
                    <span className="text-xs text-gray-400 font-mono group-hover:text-blue-500 transition-colors">
                      {subtitle}
                    </span>
                  </div>
                  <h3 className="font-bold text-lg text-gray-900 mb-2 group-hover:text-blue-700 transition-colors">
                    {title || '제목 없음'}
                  </h3>
                  <p className="text-sm text-gray-500 line-clamp-2">
                    {ref.content ? ref.content.slice(0, 150) + '...' : '클릭하여 상세 내용을 확인하세요.'}
                  </p>
                </div>
              </div>
            </button>
          )
        })}
      </div>
    </div>
  )
}
