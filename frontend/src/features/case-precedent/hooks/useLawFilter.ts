'use client'

import { useState, useEffect, useCallback } from 'react'
import { casePrecedentService } from '../services'
import type {
  DatePreset,
  FilteredLawItem,
  LawFullText,
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

/** YYYY-MM-DD → YYYYMMDD */
function toYYYYMMDD(dateStr: string): string {
  return dateStr.replace(/-/g, '')
}

export function useLawFilter() {
  // 필터 상태
  const [keyword, setKeyword] = useState('')
  const [lawType, setLawType] = useState('')
  const [ministry, setMinistry] = useState('')
  const [promulgationPreset, setPromulgationPreset] = useState<DatePreset>('all')
  const [promulgationFrom, setPromulgationFrom] = useState('')
  const [promulgationTo, setPromulgationTo] = useState('')
  const [enforcementPreset, setEnforcementPreset] = useState<DatePreset>('all')
  const [enforcementFrom, setEnforcementFrom] = useState('')
  const [enforcementTo, setEnforcementTo] = useState('')
  const [sortOrder, setSortOrder] = useState<SortOrder>('relevance')

  // 필터 옵션
  const [lawTypes, setLawTypes] = useState<string[]>([])
  const [ministries, setMinistries] = useState<string[]>([])

  // 결과 상태
  const [laws, setLaws] = useState<FilteredLawItem[]>([])
  const [total, setTotal] = useState(0)
  const [offset, setOffset] = useState(0)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [hasSearched, setHasSearched] = useState(false)

  // 상세 보기 상태
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [detail, setDetail] = useState<LawFullText | null>(null)
  const [isDetailLoading, setIsDetailLoading] = useState(false)
  const [detailError, setDetailError] = useState<string | null>(null)

  // 마운트 시 필터 옵션 로드
  useEffect(() => {
    casePrecedentService.getLawFilterOptions()
      .then((options) => {
        setLawTypes(options.law_types)
        setMinistries(options.ministries)
      })
      .catch(() => {
        setLawTypes([])
        setMinistries([])
      })
  }, [])

  // 공포일자 계산
  const getPromulgationDates = useCallback(() => {
    if (promulgationPreset === 'custom') {
      return {
        promulgation_from: promulgationFrom ? toYYYYMMDD(`${promulgationFrom}-01-01`) : undefined,
        promulgation_to: promulgationTo ? toYYYYMMDD(`${promulgationTo}-12-31`) : undefined,
      }
    }
    if (promulgationPreset === 'all') {
      return { promulgation_from: undefined, promulgation_to: undefined }
    }
    return {
      promulgation_from: toYYYYMMDD(calculateDateFrom(promulgationPreset)),
      promulgation_to: toYYYYMMDD(formatToday()),
    }
  }, [promulgationPreset, promulgationFrom, promulgationTo])

  // 시행일자 계산
  const getEnforcementDates = useCallback(() => {
    if (enforcementPreset === 'custom') {
      return {
        enforcement_from: enforcementFrom ? `${enforcementFrom}-01-01` : undefined,
        enforcement_to: enforcementTo ? `${enforcementTo}-12-31` : undefined,
      }
    }
    if (enforcementPreset === 'all') {
      return { enforcement_from: undefined, enforcement_to: undefined }
    }
    return {
      enforcement_from: calculateDateFrom(enforcementPreset),
      enforcement_to: formatToday(),
    }
  }, [enforcementPreset, enforcementFrom, enforcementTo])

  // 검색 실행
  const search = useCallback(async () => {
    setIsLoading(true)
    setError(null)
    setOffset(0)
    setHasSearched(true)
    try {
      const promDates = getPromulgationDates()
      const enfDates = getEnforcementDates()
      const result = await casePrecedentService.filterLaws({
        keyword: keyword || undefined,
        law_type: lawType || undefined,
        ministry: ministry || undefined,
        promulgation_from: promDates.promulgation_from,
        promulgation_to: promDates.promulgation_to,
        enforcement_from: enfDates.enforcement_from,
        enforcement_to: enfDates.enforcement_to,
        sort: sortOrder,
        offset: 0,
        limit: LIMIT,
      })
      setLaws(result.laws)
      setTotal(result.total)
    } catch {
      setError('법령 검색 중 오류가 발생했습니다.')
      setLaws([])
      setTotal(0)
    } finally {
      setIsLoading(false)
    }
  }, [keyword, lawType, ministry, sortOrder, getPromulgationDates, getEnforcementDates])

  // 더 보기
  const loadMore = useCallback(async () => {
    const nextOffset = offset + LIMIT
    setIsLoading(true)
    try {
      const promDates = getPromulgationDates()
      const enfDates = getEnforcementDates()
      const result = await casePrecedentService.filterLaws({
        keyword: keyword || undefined,
        law_type: lawType || undefined,
        ministry: ministry || undefined,
        promulgation_from: promDates.promulgation_from,
        promulgation_to: promDates.promulgation_to,
        enforcement_from: enfDates.enforcement_from,
        enforcement_to: enfDates.enforcement_to,
        sort: sortOrder,
        offset: nextOffset,
        limit: LIMIT,
      })
      setLaws((prev) => [...prev, ...result.laws])
      setOffset(nextOffset)
    } catch {
      setError('추가 결과를 불러오는 중 오류가 발생했습니다.')
    } finally {
      setIsLoading(false)
    }
  }, [offset, keyword, lawType, ministry, sortOrder, getPromulgationDates, getEnforcementDates])

  // 상세 조회 (법령 전문)
  const selectItem = useCallback(async (id: string) => {
    setSelectedId(id)
    setIsDetailLoading(true)
    setDetailError(null)
    try {
      const data = await casePrecedentService.getLawFullText(id)
      setDetail(data)
    } catch {
      setDetailError('법령 상세 정보를 불러올 수 없습니다.')
      setDetail(null)
    } finally {
      setIsDetailLoading(false)
    }
  }, [])

  return {
    // 필터 상태
    keyword, setKeyword,
    lawType, setLawType,
    ministry, setMinistry,
    promulgationPreset, setPromulgationPreset,
    promulgationFrom, setPromulgationFrom,
    promulgationTo, setPromulgationTo,
    enforcementPreset, setEnforcementPreset,
    enforcementFrom, setEnforcementFrom,
    enforcementTo, setEnforcementTo,
    sortOrder, setSortOrder,
    lawTypes, ministries,
    // 결과
    laws, total, isLoading, error, hasSearched,
    hasMore: laws.length < total,
    // 액션
    search, loadMore,
    // 상세
    selectedId, selectItem,
    detail, isDetailLoading, detailError,
  }
}
