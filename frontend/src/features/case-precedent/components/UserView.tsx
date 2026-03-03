'use client'

import { useState, useEffect, useRef } from 'react'
import Image from 'next/image'
import { useChat } from '@/context/ChatContext'
import type { ChatSource } from '../types'
import { getLawTypeLogo, getLawTypeOrgName, getCourtLogo, getDocTypeLogo, DEFAULT_GOV_LOGO } from '../utils/lawTypeLogo'
import { getDocTypeLabel, getDocTypeBadgeColor } from '../utils/docTypeUtils'
import { ReferenceDetailHeader } from './ReferenceDetailHeader'
import { PrecedentDetailUser } from './PrecedentDetailUser'
import { PrecedentDetailLawyer } from './PrecedentDetailLawyer'
import { LawDetailUser } from './LawDetailUser'
import { LawDetailLawyer } from './LawDetailLawyer'

export function UserView() {
  const { sessionData, userRole, highlightedCaseNumber, setHighlightedCaseNumber } = useChat()
  const [references, setReferences] = useState<ChatSource[]>([])
  const [selectedRef, setSelectedRef] = useState<ChatSource | null>(null)
  const isLawyer = userRole === 'lawyer'
  const cardRefs = useRef<Map<string, HTMLButtonElement>>(new Map())
  const listContainerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (sessionData.aiReferences && Array.isArray(sessionData.aiReferences)) {
      const raw = sessionData.aiReferences as ChatSource[]
      const seen = new Set<string>()
      const unique = raw.filter((ref) => {
        const key = ref.doc_type === 'law' ? ref.law_name : ref.case_number
        if (!key || seen.has(key)) return false
        seen.add(key)
        return true
      })
      setReferences(unique)
    }
  }, [sessionData.aiReferences])

  useEffect(() => {
    if (!highlightedCaseNumber) return
    const matchingRef = references.find(
      (ref) => ref.case_number && ref.case_number.includes(highlightedCaseNumber)
    )
    if (matchingRef?.case_number) {
      const cardElement = cardRefs.current.get(matchingRef.case_number)
      if (cardElement) {
        cardElement.scrollIntoView({ behavior: 'smooth', block: 'center' })
        setTimeout(() => {
          setHighlightedCaseNumber(null)
        }, 3000)
      }
    }
  }, [highlightedCaseNumber, references, setHighlightedCaseNumber])

  // Empty state
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
            <ReferenceDetailHeader selectedRef={selectedRef} />
            {isLaw ? (
              isLawyer ? <LawDetailLawyer source={selectedRef} />
                       : <LawDetailUser source={selectedRef} />
            ) : (
              isLawyer ? <PrecedentDetailLawyer source={selectedRef} />
                       : <PrecedentDetailUser source={selectedRef} />
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

      <div ref={listContainerRef} className="flex-1 overflow-y-auto p-4 space-y-3">
        {references.map((ref, idx) => {
          const isLaw = ref.doc_type === 'law'
          const title = isLaw ? ref.law_name : ref.case_name
          const subtitle = isLaw ? ref.law_type : ref.case_number

          const logoPath = isLaw
            ? getLawTypeLogo(ref.law_type)
            : getCourtLogo(ref.court_name) || getDocTypeLogo(ref.doc_type)

          const isHighlighted = !isLaw && highlightedCaseNumber && ref.case_number?.includes(highlightedCaseNumber)

          return (
            <button
              key={`${ref.case_number || ref.law_name}-${idx}`}
              ref={(el) => {
                if (el && ref.case_number) {
                  cardRefs.current.set(ref.case_number, el)
                }
              }}
              onClick={() => setSelectedRef(ref)}
              className={`w-full text-left p-5 rounded-xl border transition-all duration-200 group ${
                isHighlighted
                  ? 'bg-yellow-50 border-yellow-400 shadow-lg ring-2 ring-yellow-300 animate-pulse'
                  : 'bg-white border-gray-100 hover:border-blue-300 hover:shadow-md'
              }`}
            >
              <div className="flex items-start gap-3">
                <div className="shrink-0 w-10 h-10 flex items-center justify-center bg-gray-50 rounded-lg">
                  <Image
                    src={logoPath || DEFAULT_GOV_LOGO}
                    alt={isLaw ? getLawTypeOrgName(ref.law_type) : (ref.court_name || '법원')}
                    width={32}
                    height={32}
                    className="object-contain"
                    unoptimized
                  />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-2 flex-wrap">
                    <span className={`px-2 py-0.5 rounded text-xs font-bold ${getDocTypeBadgeColor(ref.doc_type)}`}>
                      {getDocTypeLabel(ref.doc_type)}
                    </span>
                    {ref.case_type && (
                      <span className="px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-600">
                        {ref.case_type}
                      </span>
                    )}
                    {ref.court_name && (
                      <span className="text-xs text-gray-500">
                        {ref.court_name}
                      </span>
                    )}
                    {ref.decision_date && (
                      <span className="text-xs text-gray-400">
                        {ref.decision_date}
                      </span>
                    )}
                  </div>

                  <h3 className="font-bold text-lg text-gray-900 mb-2 group-hover:text-blue-700 transition-colors line-clamp-2">
                    {isLaw ? (title || '제목 없음') : (ref.summary || title || '제목 없음')}
                  </h3>

                  {!isLaw && (
                    <p className="text-sm text-gray-400 mb-2">
                      {title && <span>{title}</span>}
                      {title && subtitle && <span className="mx-1">|</span>}
                      {subtitle && <span className="font-mono">{subtitle}</span>}
                    </p>
                  )}

                  <p className="text-sm text-gray-500 line-clamp-2">
                    {ref.reasoning ? ref.reasoning.slice(0, 120) + '...' : (ref.content ? ref.content.slice(0, 120) + '...' : '클릭하여 상세 내용을 확인하세요.')}
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
