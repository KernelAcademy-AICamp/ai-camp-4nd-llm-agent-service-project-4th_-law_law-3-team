'use client'

import { useState, useMemo, useEffect } from 'react'
import type { ChatSource, DatePreset, SortOrder } from '../types'

function toYear(dateString: string | undefined | null): number | null {
  if (!dateString) return null
  const year = parseInt(dateString.slice(0, 4), 10)
  return isNaN(year) ? null : year
}

/**
 * aiReferences(ChatSource[])에 클라이언트 사이드 필터링을 적용하는 훅
 */
export function useClientFilter(references: ChatSource[]) {
  const [keyword, setKeyword] = useState('')
  const [caseType, setCaseType] = useState('')
  const [datePreset, setDatePreset] = useState<DatePreset>('all')
  const [dateFrom, setDateFrom] = useState('')
  const [dateTo, setDateTo] = useState('')
  const [sortOrder, setSortOrder] = useState<SortOrder>('relevance')
  const [hasSearched, setHasSearched] = useState(false)

  // 참조 목록에서 사건종류 옵션 추출
  const caseTypes = useMemo(() => {
    const types = new Set<string>()
    for (const ref of references) {
      if (ref.case_type) types.add(ref.case_type)
    }
    return Array.from(types).sort()
  }, [references])

  // 날짜 범위 계산
  const dateRange = useMemo(() => {
    if (datePreset === 'all') return { from: null, to: null }
    if (datePreset === 'custom') {
      return {
        from: dateFrom ? parseInt(dateFrom, 10) : null,
        to: dateTo ? parseInt(dateTo, 10) : null,
      }
    }
    const years = datePreset === '3y' ? 3 : datePreset === '5y' ? 5 : 10
    const currentYear = new Date().getFullYear()
    return { from: currentYear - years, to: currentYear }
  }, [datePreset, dateFrom, dateTo])

  // 필터링 + 정렬 적용
  const filtered = useMemo(() => {
    if (!hasSearched) return references

    let result = [...references]

    // 키워드 필터
    if (keyword) {
      const lower = keyword.toLowerCase()
      result = result.filter((ref) =>
        (ref.case_name?.toLowerCase().includes(lower)) ||
        (ref.case_number?.toLowerCase().includes(lower)) ||
        (ref.summary?.toLowerCase().includes(lower)) ||
        (ref.law_name?.toLowerCase().includes(lower))
      )
    }

    // 사건종류 필터
    if (caseType) {
      result = result.filter((ref) => ref.case_type === caseType)
    }

    // 날짜 범위 필터
    if (dateRange.from !== null || dateRange.to !== null) {
      result = result.filter((ref) => {
        const year = toYear(ref.decision_date)
        if (year === null) return false
        if (dateRange.from !== null && year < dateRange.from) return false
        if (dateRange.to !== null && year > dateRange.to) return false
        return true
      })
    }

    // 정렬
    if (sortOrder === 'relevance' && keyword) {
      const lower = keyword.toLowerCase()
      result.sort((a, b) => {
        const scoreA =
          (a.case_name?.toLowerCase().includes(lower) ? 3 : 0) +
          (a.case_number?.toLowerCase().includes(lower) ? 2 : 0) +
          (a.summary?.toLowerCase().includes(lower) ? 1 : 0)
        const scoreB =
          (b.case_name?.toLowerCase().includes(lower) ? 3 : 0) +
          (b.case_number?.toLowerCase().includes(lower) ? 2 : 0) +
          (b.summary?.toLowerCase().includes(lower) ? 1 : 0)
        return scoreB - scoreA
      })
    } else if (sortOrder === 'latest') {
      result.sort((a, b) => {
        const dateA = a.decision_date || ''
        const dateB = b.decision_date || ''
        return dateB.localeCompare(dateA)
      })
    }

    return result
  }, [references, keyword, caseType, dateRange, sortOrder, hasSearched])

  const search = () => {
    setHasSearched(true)
  }

  // 첫 렌더 시 필터 없이 전체 표시
  useEffect(() => {
    if (references.length > 0 && !hasSearched) {
      setHasSearched(true)
    }
  }, [references, hasSearched])

  return {
    keyword, setKeyword,
    caseType, setCaseType,
    datePreset, setDatePreset,
    dateFrom, setDateFrom,
    dateTo, setDateTo,
    sortOrder, setSortOrder,
    caseTypes,
    filtered,
    total: filtered.length,
    hasSearched,
    search,
  }
}
