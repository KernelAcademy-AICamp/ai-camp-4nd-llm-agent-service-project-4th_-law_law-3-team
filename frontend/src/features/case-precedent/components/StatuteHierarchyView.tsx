'use client'

import { useState, useCallback, useEffect, useMemo, useRef } from 'react'
import { Network, Search, X, Loader2, ChevronRight } from 'lucide-react'
import { ArrowLeft } from 'lucide-react'
import { useRouter, useSearchParams } from 'next/navigation'
import { StatuteForceGraph } from './StatuteForceGraph'
import { StatuteDetailPanel } from './StatuteDetailPanel'
import { casePrecedentService, type GraphNode } from '../services'
import type { StatuteNode, StatuteHierarchyResponse } from '../types'

export function StatuteHierarchyView() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const statuteId = searchParams.get('id')
  const statuteName = searchParams.get('name')
  const statuteType = searchParams.get('type')
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<StatuteNode[]>([])
  const [isSearching, setIsSearching] = useState(false)
  const [showDropdown, setShowDropdown] = useState(false)
  const [selectedStatute, setSelectedStatute] = useState<StatuteNode | null>(null)
  const [detailData, setDetailData] = useState<StatuteHierarchyResponse | null>(null)
  const [detailLoading, setDetailLoading] = useState(false)
  const [isPanelOpen, setIsPanelOpen] = useState(true)

  // 법령 유형별 필터
  const ALL_STATUTE_TYPES = ['헌법', '법률', '대통령령', '총리령·부령', '규칙'] as const
  const [visibleTypes, setVisibleTypes] = useState<Set<string>>(() => new Set(ALL_STATUTE_TYPES))

  const toggleType = useCallback((type: string) => {
    setVisibleTypes(prev => {
      const next = new Set(prev)
      if (next.has(type)) {
        if (next.size > 1) next.delete(type)
      } else {
        next.add(type)
      }
      return next
    })
  }, [])

  const TYPE_BADGE_COLORS: Record<string, string> = useMemo(() => ({
    '헌법': 'bg-orange-50 text-orange-700 border-orange-200',
    '법률': 'bg-amber-50 text-amber-700 border-amber-200',
    '대통령령': 'bg-blue-50 text-blue-700 border-blue-200',
    '총리령·부령': 'bg-emerald-50 text-emerald-700 border-emerald-200',
    '규칙': 'bg-violet-50 text-violet-700 border-violet-200',
  }), [])

  // URL 파라미터에서 선택된 법령 복원 + 상세 데이터 동시 로드 (API 워터폴 제거)
  useEffect(() => {
    let isCancelled = false
    const normalizeName = (value: string): string =>
      value.trim().replace(/\s+/g, '').toLowerCase()

    const resolveStatute = async () => {
      if (statuteId && statuteName) {
        try {
          const detail = await casePrecedentService.getStatuteHierarchy(statuteId)
          if (!detail.root) return
          const normalizedTargetName = normalizeName(statuteName)
          const normalizeRootName = normalizeName(detail.root.name)
          const normalizedAbbreviation = detail.root.abbreviation
            ? normalizeName(detail.root.abbreviation)
            : ''

          if (
            normalizeRootName === normalizedTargetName ||
            (normalizedAbbreviation && normalizedAbbreviation === normalizedTargetName)
          ) {
            if (!isCancelled && detail.root) {
              setSelectedStatute({
                id: detail.root.id || statuteId,
                name: detail.root.name,
                type: detail.root.type || statuteType || '',
                abbreviation: detail.root.abbreviation,
                citation_count: detail.root.citation_count,
              })
              setSearchQuery(detail.root.name)
              // 이미 받은 hierarchy 응답을 detailData에 직접 저장 (중복 API 호출 방지)
              setDetailData(detail)
            }
            return
          }

          console.info('법령 ID와 이름 불일치, 이름 기반으로 재검색:', {
            id: statuteId,
            name: statuteName,
          })
        } catch (error) {
          console.error('법령 ID 유효성 검증 실패:', error)
        }

        const response = await casePrecedentService.searchStatutes(statuteName, 1)
        const firstResult = response.results.find(
          (statute) =>
            statute.name === statuteName ||
            (statute.abbreviation && statute.abbreviation === statuteName)
        )
        const targetResult = firstResult || response.results[0]

        if (!targetResult || isCancelled) {
          if (!isCancelled) {
            setSelectedStatute({
              id: statuteId,
              name: statuteName,
              type: statuteType || '',
              citation_count: 0,
            })
            setSearchQuery(statuteName)
          }
          return
        }

        if (!isCancelled) {
          setSelectedStatute({
            id: targetResult.id,
            name: targetResult.name,
            type: targetResult.type,
            abbreviation: targetResult.abbreviation,
            citation_count: targetResult.citation_count,
          })
          setSearchQuery(targetResult.name)
        }
        return

      }

      if (!statuteName) {
        if (!isCancelled) {
          setSelectedStatute(null)
          setSearchQuery('')
        }
        return
      }

      setSearchQuery(statuteName)
      try {
        const response = await casePrecedentService.searchStatutes(statuteName, 1)
        const matchedResult = response.results.find(
          (statute) =>
            statute.name === statuteName ||
            (statute.abbreviation && statute.abbreviation === statuteName)
        )
        const targetResult = matchedResult || response.results[0]
        if (!targetResult) {
          if (!isCancelled) {
            setSelectedStatute(null)
          }
          return
        }

        if (!isCancelled) {
          setSelectedStatute({
            id: targetResult.id,
            name: targetResult.name,
            type: targetResult.type,
            abbreviation: targetResult.abbreviation,
            citation_count: targetResult.citation_count,
          })
        }
      } catch (error) {
        console.error('법령 검색으로 중심 법령 복원 실패:', error)
        if (isCancelled) return
        if (!isCancelled) {
          setSelectedStatute(null)
        }
      }
    }

    resolveStatute()

    return () => {
      isCancelled = true
    }
  }, [statuteId, statuteName, statuteType])

  // 검색 실행
  const handleSearch = useCallback(async (query: string) => {
    if (!query.trim()) {
      setSearchResults([])
      setShowDropdown(false)
      return
    }

    setIsSearching(true)
    try {
      const response = await casePrecedentService.searchStatutes(query, 10)
      setSearchResults(response.results)
      setShowDropdown(true)
    } catch (error) {
      console.error('검색 실패:', error)
      setSearchResults([])
    } finally {
      setIsSearching(false)
    }
  }, [])

  // 검색어 변경 (디바운스 — useRef로 타이머 관리)
  const debounceTimerRef = useRef<NodeJS.Timeout | null>(null)
  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value
    setSearchQuery(value)

    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current)
    }
    debounceTimerRef.current = setTimeout(() => {
      handleSearch(value)
    }, 300)
  }, [handleSearch])

  // 법령 선택 (URL에 추가하여 뒤로가기 지원)
  const handleSelect = useCallback((statute: StatuteNode) => {
    setShowDropdown(false)
    setIsPanelOpen(true)
    const params = new URLSearchParams()
    params.set('id', statute.id)
    params.set('name', statute.name)
    if (statute.type) params.set('type', statute.type)
    router.push(`/statute-hierarchy?${params.toString()}`)
  }, [router])

  // 선택 초기화 (URL에서 파라미터 제거)
  const handleClear = useCallback(() => {
    setSearchResults([])
    setShowDropdown(false)
    setDetailData(null)
    router.push('/statute-hierarchy')
  }, [router])

  // 상세 정보 로드
  const loadDetail = useCallback(async (statuteId: string) => {
    setDetailLoading(true)
    try {
      const response = await casePrecedentService.getStatuteHierarchy(statuteId)
      setDetailData(response)
    } catch (error) {
      console.error('상세 정보 로드 실패:', error)
      setDetailData(null)
    } finally {
      setDetailLoading(false)
    }
  }, [])

  // URL 파라미터에서 선택된 법령의 상세 정보 로드 (이미 로드된 경우 스킵)
  useEffect(() => {
    if (selectedStatute?.id) {
      // 첫 번째 useEffect에서 이미 detailData를 로드한 경우 중복 호출 방지
      if (detailData?.root?.id === selectedStatute.id) return
      loadDetail(selectedStatute.id)
    } else {
      setDetailData(null)
    }
  }, [selectedStatute?.id, loadDetail, detailData?.root?.id])

  // 그래프에서 노드 클릭 (URL에 추가하여 뒤로가기 지원)
  const handleNodeClick = useCallback((node: GraphNode) => {
    setIsPanelOpen(true)
    const params = new URLSearchParams()
    params.set('id', node.id)
    params.set('name', node.name)
    if (node.type) params.set('type', node.type)
    router.push(`/statute-hierarchy?${params.toString()}`)
  }, [router])

  // 패널 내 법령 클릭 → 그래프 이동
  const handlePanelNodeClick = useCallback((node: StatuteNode) => {
    handleNodeClick(node)
  }, [handleNodeClick])

  return (
    <div className="h-full w-full flex flex-col bg-slate-50">
      {/* 헤더 */}
      <div className="px-6 py-4 border-b border-amber-100/60 bg-gradient-to-r from-amber-50 via-white to-orange-50 shrink-0">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-3">
            <button
              onClick={() => router.back()}
              className="p-2 mr-2 text-gray-400 hover:text-gray-700 hover:bg-gray-100 rounded-full transition-colors flex items-center justify-center shrink-0"
              aria-label="뒤로 가기"
            >
              <ArrowLeft size={20} />
            </button>
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-amber-400 to-orange-500 flex items-center justify-center shadow-sm shadow-amber-200/50">
              <Network className="w-4 h-4 text-white" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-gray-900 tracking-tight">법령 체계도</h2>
              <p className="text-[11px] text-gray-400 -mt-0.5">법령 간 계층·인용 관계 시각화</p>
            </div>
          </div>
        </div>

        {/* 검색바 */}
        <div className="relative max-w-md">
          <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-amber-400" />
          <input
            type="text"
            value={searchQuery}
            onChange={handleInputChange}
            onFocus={() => searchResults.length > 0 && setShowDropdown(true)}
            placeholder="법령명 또는 약칭 검색 (민법, 민소법, 특가법...)"
            className="w-full pl-10 pr-10 py-2.5 bg-white/80 backdrop-blur-sm border border-amber-200/60 rounded-xl
                       text-gray-900 placeholder-gray-400 text-sm
                       focus:outline-none focus:ring-2 focus:ring-amber-400/40 focus:border-amber-300
                       shadow-sm shadow-amber-100/30 transition-all"
          />
          {isSearching && (
            <Loader2 className="absolute right-10 top-1/2 -translate-y-1/2 w-4 h-4 text-amber-400 animate-spin" />
          )}
          {(searchQuery || selectedStatute) && (
            <button
              onClick={handleClear}
              className="absolute right-3 top-1/2 -translate-y-1/2 p-1 hover:bg-amber-50 rounded-lg transition-colors"
            >
              <X className="w-4 h-4 text-gray-400" />
            </button>
          )}

          {/* 검색 결과 드롭다운 */}
          {showDropdown && searchResults.length > 0 && (
            <div className="absolute z-50 w-full mt-1.5 bg-white/95 backdrop-blur-md border border-gray-200/80 rounded-xl shadow-xl shadow-gray-200/50 max-h-64 overflow-y-auto">
              {searchResults.map((statute) => (
                <button
                  key={statute.id}
                  onClick={() => handleSelect(statute)}
                  className="w-full px-4 py-2.5 text-left hover:bg-amber-50/60 flex items-center justify-between transition-colors first:rounded-t-xl last:rounded-b-xl"
                >
                  <div className="flex flex-col">
                    <span className="text-sm font-medium text-gray-900">{statute.name}</span>
                    <span className="text-xs text-gray-500">
                      {statute.type}
                      {statute.abbreviation && ` (${statute.abbreviation})`}
                    </span>
                  </div>
                  {statute.citation_count > 0 && (
                    <span className="text-xs text-amber-600/70 bg-amber-50 px-1.5 py-0.5 rounded">
                      {statute.citation_count.toLocaleString()}
                    </span>
                  )}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* 유형 필터 */}
      <div className="px-5 py-2 border-b border-gray-200/60 bg-white/60 backdrop-blur-sm flex items-center gap-1.5 shrink-0">
        <span className="text-[11px] text-gray-400 mr-1.5 font-medium">필터</span>
        {ALL_STATUTE_TYPES.map((type) => (
          <button
            key={type}
            onClick={() => toggleType(type)}
            className={`px-2.5 py-1 text-xs rounded-lg border transition-all duration-200 font-medium ${
              visibleTypes.has(type)
                ? TYPE_BADGE_COLORS[type] + ' shadow-sm'
                : 'bg-gray-50 text-gray-400 border-gray-200/60 hover:bg-gray-100'
            }`}
          >
            {type}
          </button>
        ))}
      </div>

      {/* 상세 패널 (좌측) + 그래프 */}
      <div className="flex flex-1 overflow-hidden">
        {/* 상세 사이드 패널 (좌측) */}
        {isPanelOpen && (detailData || detailLoading) && (
          <div className="relative z-10 h-full">
            <StatuteDetailPanel
              data={detailData || { root: null, upper: [], lower: [], related: [] }}
              loading={detailLoading}
              onClose={() => setIsPanelOpen(false)}
              onNodeClick={handlePanelNodeClick}
            />
          </div>
        )}

        {/* 그래프 영역 */}
        <div className="relative flex-1 overflow-hidden">
          {/* 패널 열기 버튼 (패널이 닫혀 있고 데이터가 있을 때) */}
          {!isPanelOpen && detailData && (
            <button
              onClick={() => setIsPanelOpen(true)}
              className="absolute left-0 top-1/2 -translate-y-1/2 z-20 bg-white/90 backdrop-blur-sm border border-gray-200/60 border-l-0
                         rounded-r-lg px-1 py-3 shadow-sm hover:bg-amber-50 transition-colors"
              aria-label="패널 열기"
            >
              <ChevronRight className="w-4 h-4 text-amber-600" />
            </button>
          )}
          <div className="absolute inset-0">
            <StatuteForceGraph
              centerId={selectedStatute?.id}
              centerName={selectedStatute?.name}
              onNodeClick={handleNodeClick}
              visibleTypes={visibleTypes}
              highlightNodeId={selectedStatute?.id}
            />
          </div>
        </div>
      </div>
    </div>
  )
}
