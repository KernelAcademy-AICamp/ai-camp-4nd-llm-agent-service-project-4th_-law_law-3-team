'use client'

import { useState, useEffect, useRef, useMemo, useCallback } from 'react'
import { useChat } from '@/context/ChatContext'
import type { ChatSource, PrecedentItem } from '../types'
import { CaseCard } from './CaseCard'
import { FilterPanel } from './FilterPanel'
import { ReferenceDetailHeader } from './ReferenceDetailHeader'
import { PrecedentDetailUser } from './PrecedentDetailUser'
import { PrecedentDetailLawyer } from './PrecedentDetailLawyer'
import { LawDetailUser } from './LawDetailUser'
import { LawDetailLawyer } from './LawDetailLawyer'
import { useClientFilter } from '../hooks/useClientFilter'

function getRefId(ref: ChatSource): string {
  return ref.doc_type === 'law' ? `law-${ref.law_name}` : `case-${ref.case_number}`
}

export function UserView() {
  const { sessionData, userRole, highlightedCaseNumber, setHighlightedCaseNumber } = useChat()
  const [references, setReferences] = useState<ChatSource[]>([])
  const [selectedRef, setSelectedRef] = useState<ChatSource | null>(null)
  const [showFilter, setShowFilter] = useState(true)
  const isLawyer = userRole === 'lawyer'
  const cardRefs = useRef<Map<string, HTMLDivElement>>(new Map())

  const {
    keyword, setKeyword,
    caseType, setCaseType,
    datePreset, setDatePreset,
    dateFrom, setDateFrom,
    dateTo, setDateTo,
    sortOrder, setSortOrder,
    caseTypes,
    filtered,
    total,
    search,
  } = useClientFilter(references)

  // 중복 제거
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

  // ChatSource[] → PrecedentItem[] 변환 (CaseCard 표시용)
  const precedentItems: PrecedentItem[] = useMemo(() =>
    filtered.map((ref) => ({
      id: getRefId(ref),
      case_name: ref.doc_type === 'law' ? (ref.law_name || '') : (ref.case_name || ''),
      case_number: ref.case_number || '',
      doc_type: ref.doc_type || 'precedent',
      court: ref.court_name || '',
      date: ref.decision_date || '',
      summary: ref.summary || '',
      similarity: ref.similarity || 0,
    }))
  , [filtered])

  // id → ChatSource 매핑 (선택 시 상세 조회용)
  const refById = useMemo(() => {
    const map = new Map<string, ChatSource>()
    filtered.forEach((ref) => {
      map.set(getRefId(ref), ref)
    })
    return map
  }, [filtered])

  // 카드 선택 → ChatSource 찾아서 상세 표시
  const handleSelect = useCallback((id: string) => {
    const ref = refById.get(id)
    if (ref) setSelectedRef(ref)
  }, [refById])

  // 선택된 항목 ID (CaseCard selected 표시용)
  const selectedId = useMemo(() => {
    if (!selectedRef) return null
    return getRefId(selectedRef)
  }, [selectedRef])

  // 채팅 답변에서 클릭 시 하이라이팅 + 자동 선택
  useEffect(() => {
    if (!highlightedCaseNumber) return
    const matchingRef = references.find(
      (ref) => ref.case_number && ref.case_number.includes(highlightedCaseNumber)
    )
    if (matchingRef) {
      setSelectedRef(matchingRef)
      const cardId = getRefId(matchingRef)
      const cardEl = cardRefs.current.get(cardId)
      if (cardEl) {
        cardEl.scrollIntoView({ behavior: 'smooth', block: 'center' })
      }
      setTimeout(() => {
        setHighlightedCaseNumber(null)
      }, 3000)
    }
  }, [highlightedCaseNumber, references, setHighlightedCaseNumber])

  // 빈 상태
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

  return (
    <div className="h-full flex overflow-hidden">
      {/* 왼쪽 패널: LawyerView와 동일한 구조 */}
      <div className="w-80 bg-white border-r border-gray-200 flex flex-col h-full">
        {/* 헤더: 결과 수 + 필터 토글 */}
        <div className="px-4 py-2 bg-gray-50 border-b flex items-center justify-between">
          <span className="text-sm text-gray-600">관련 문서 ({total}건)</span>
          <button
            onClick={() => setShowFilter(!showFilter)}
            className={`px-3 py-1 text-xs rounded-lg border transition-colors ${
              showFilter
                ? 'bg-blue-50 border-blue-300 text-blue-600'
                : 'bg-white border-gray-300 text-gray-600 hover:border-gray-400'
            }`}
          >
            {showFilter ? '필터 닫기' : '필터'}
          </button>
        </div>

        {/* 필터 패널 */}
        {showFilter && (
          <FilterPanel
            keyword={keyword}
            onKeywordChange={setKeyword}
            caseType={caseType}
            onCaseTypeChange={setCaseType}
            datePreset={datePreset}
            onDatePresetChange={setDatePreset}
            dateFrom={dateFrom}
            onDateFromChange={setDateFrom}
            dateTo={dateTo}
            onDateToChange={setDateTo}
            caseTypes={caseTypes}
            onSearch={search}
            isLoading={false}
          />
        )}

        {/* 정렬 */}
        <div className="flex items-center justify-between px-4 py-2 border-b border-gray-100">
          <span className="text-xs text-gray-500">총 {total}건</span>
          <div className="flex gap-1 text-xs">
            <button
              onClick={() => setSortOrder('relevance')}
              className={`px-2 py-0.5 rounded ${
                sortOrder === 'relevance'
                  ? 'text-blue-600 font-medium'
                  : 'text-gray-400 hover:text-gray-600'
              }`}
            >
              정확도순
            </button>
            <span className="text-gray-300">|</span>
            <button
              onClick={() => setSortOrder('latest')}
              className={`px-2 py-0.5 rounded ${
                sortOrder === 'latest'
                  ? 'text-blue-600 font-medium'
                  : 'text-gray-400 hover:text-gray-600'
              }`}
            >
              최신순
            </button>
          </div>
        </div>

        {/* 결과 목록 */}
        <div className="flex-1 overflow-y-auto p-3 space-y-2">
          {precedentItems.length > 0 ? (
            precedentItems.map((case_) => {
              const isHighlighted = !!(
                highlightedCaseNumber &&
                case_.case_number &&
                case_.case_number.includes(highlightedCaseNumber)
              )
              return (
                <div
                  key={case_.id}
                  ref={(el) => {
                    if (el) cardRefs.current.set(case_.id, el)
                  }}
                  className={isHighlighted ? 'ring-2 ring-yellow-400 rounded-lg animate-pulse' : ''}
                >
                  <CaseCard
                    case_={case_}
                    selected={selectedId === case_.id}
                    onSelect={handleSelect}
                  />
                </div>
              )
            })
          ) : (
            <div className="p-6 text-center text-gray-400">
              <svg
                className="w-12 h-12 mx-auto mb-3 text-gray-300"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                />
              </svg>
              <p className="text-sm">필터 조건에 맞는 결과가 없습니다</p>
              {datePreset !== 'all' && (
                <p className="text-xs text-gray-400 mt-2">
                  기간을 &apos;전체&apos;로 변경하면 더 많은 결과를 볼 수 있습니다
                </p>
              )}
            </div>
          )}
        </div>
      </div>

      {/* 오른쪽 패널: 사용자 친화적 상세 */}
      <div className="flex-1 overflow-hidden bg-white">
        {selectedRef ? (
          <div className="h-full flex flex-col">
            <div className="p-4 border-b border-gray-100 flex items-center justify-between">
              <span className="font-bold text-gray-900">상세 내용</span>
              <span className={`px-2 py-1 rounded text-xs font-medium ${
                isLawyer ? 'bg-purple-100 text-purple-700' : 'bg-blue-100 text-blue-700'
              }`}>
                {isLawyer ? '변호사 모드' : '일반인 모드'}
              </span>
            </div>

            <div className="flex-1 overflow-y-auto p-6 md:p-8">
              <div className="max-w-3xl mx-auto">
                <ReferenceDetailHeader selectedRef={selectedRef} />
                {selectedRef.doc_type === 'law' ? (
                  isLawyer ? <LawDetailLawyer source={selectedRef} />
                           : <LawDetailUser source={selectedRef} />
                ) : (
                  isLawyer ? <PrecedentDetailLawyer source={selectedRef} />
                           : <PrecedentDetailUser source={selectedRef} />
                )}
              </div>
            </div>
          </div>
        ) : (
          <div className="h-full flex flex-col items-center justify-center text-gray-400">
            <svg
              className="w-16 h-16 mb-4 text-gray-200"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1}
                d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
              />
            </svg>
            <p className="text-sm">판례 또는 법령을 선택하면 상세 내용이 표시됩니다</p>
          </div>
        )}
      </div>
    </div>
  )
}
