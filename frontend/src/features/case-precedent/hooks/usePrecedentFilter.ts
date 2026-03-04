'use client'

import { useState, useEffect, useCallback } from 'react'
import { casePrecedentService } from '../services'
import type {
  DatePreset,
  FilteredPrecedentItem,
  PrecedentDetail,
  SortOrder,
} from '../types'

const LIMIT = 20

function calculateDateFrom(preset: DatePreset): string {
  if (preset === 'all' || preset === 'custom') return ''
  const now = new Date()
  const years = preset === '3y' ? 3 : preset === '5y' ? 5 : 10
  now.setFullYear(now.getFullYear() - years)
  return now.toISOString().slice(0, 10)
}

function formatToday(): string {
  return new Date().toISOString().slice(0, 10)
}

export function usePrecedentFilter() {
  // 필터 상태
  const [keyword, setKeyword] = useState('')
  const [caseType, setCaseType] = useState('')
  const [datePreset, setDatePreset] = useState<DatePreset>('all')
  const [dateFrom, setDateFrom] = useState('')
  const [dateTo, setDateTo] = useState('')
  const [sortOrder, setSortOrder] = useState<SortOrder>('relevance')

  // 사건종류 옵션
  const [caseTypes, setCaseTypes] = useState<string[]>([])

  // 결과 상태
  const [precedents, setPrecedents] = useState<FilteredPrecedentItem[]>([])
  const [total, setTotal] = useState(0)
  const [offset, setOffset] = useState(0)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [hasSearched, setHasSearched] = useState(false)

  // 상세 보기 상태
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [detail, setDetail] = useState<PrecedentDetail | null>(null)
  const [isDetailLoading, setIsDetailLoading] = useState(false)
  const [detailError, setDetailError] = useState<string | null>(null)

  // 마운트 시 사건종류 목록 로드
  useEffect(() => {
    casePrecedentService.getCaseTypes()
      .then(setCaseTypes)
      .catch(() => setCaseTypes([]))
  }, [])

  // 실제 API 호출 날짜 계산
  const getEffectiveDates = useCallback(() => {
    if (datePreset === 'custom') {
      return {
        date_from: dateFrom ? `${dateFrom}-01-01` : undefined,
        date_to: dateTo ? `${dateTo}-12-31` : undefined,
      }
    }
    if (datePreset === 'all') {
      return { date_from: undefined, date_to: undefined }
    }
    return {
      date_from: calculateDateFrom(datePreset),
      date_to: formatToday(),
    }
  }, [datePreset, dateFrom, dateTo])

  // 검색 실행
  const search = useCallback(async () => {
    setIsLoading(true)
    setError(null)
    setOffset(0)
    setHasSearched(true)
    try {
      const dates = getEffectiveDates()
      const result = await casePrecedentService.filterPrecedents({
        keyword: keyword || undefined,
        case_type: caseType || undefined,
        date_from: dates.date_from,
        date_to: dates.date_to,
        sort: sortOrder,
        offset: 0,
        limit: LIMIT,
      })
      setPrecedents(result.precedents)
      setTotal(result.total)
    } catch {
      setError('판례 검색 중 오류가 발생했습니다.')
      setPrecedents([])
      setTotal(0)
    } finally {
      setIsLoading(false)
    }
  }, [keyword, caseType, sortOrder, getEffectiveDates])

  // 더 보기
  const loadMore = useCallback(async () => {
    const nextOffset = offset + LIMIT
    setIsLoading(true)
    try {
      const dates = getEffectiveDates()
      const result = await casePrecedentService.filterPrecedents({
        keyword: keyword || undefined,
        case_type: caseType || undefined,
        date_from: dates.date_from,
        date_to: dates.date_to,
        sort: sortOrder,
        offset: nextOffset,
        limit: LIMIT,
      })
      setPrecedents((prev) => [...prev, ...result.precedents])
      setOffset(nextOffset)
    } catch {
      setError('추가 결과를 불러오는 중 오류가 발생했습니다.')
    } finally {
      setIsLoading(false)
    }
  }, [offset, keyword, caseType, sortOrder, getEffectiveDates])

  // 상세 조회
  const selectItem = useCallback(async (id: string) => {
    setSelectedId(id)
    setIsDetailLoading(true)
    setDetailError(null)
    try {
      const data = await casePrecedentService.getPrecedentDetail(id)
      setDetail(data)
    } catch {
      setDetailError('판례 상세 정보를 불러올 수 없습니다.')
      setDetail(null)
    } finally {
      setIsDetailLoading(false)
    }
  }, [])

  return {
    // 필터 상태
    keyword, setKeyword,
    caseType, setCaseType,
    datePreset, setDatePreset,
    dateFrom, setDateFrom,
    dateTo, setDateTo,
    sortOrder, setSortOrder,
    caseTypes,
    // 결과
    precedents, total, isLoading, error, hasSearched,
    hasMore: precedents.length < total,
    // 액션
    search, loadMore,
    // 상세
    selectedId, selectItem,
    detail, isDetailLoading, detailError,
  }
}
