'use client'

import { useState, useCallback, useEffect } from 'react'
import { Network, Search, X, Loader2, ArrowLeft } from 'lucide-react'
import { useRouter, useSearchParams } from 'next/navigation'
import { StatuteForceGraph } from './StatuteForceGraph'
import { casePrecedentService, type GraphNode } from '../services'
import { useChat } from '@/context/ChatContext'
import type { StatuteNode } from '../types'

export function StatuteHierarchyView() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const { userRole } = useChat()
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<StatuteNode[]>([])
  const [isSearching, setIsSearching] = useState(false)
  const [showDropdown, setShowDropdown] = useState(false)
  const [selectedStatute, setSelectedStatute] = useState<StatuteNode | null>(null)

  // URL 파라미터에서 선택된 법령 복원
  useEffect(() => {
    const statuteId = searchParams.get('id')
    const statuteName = searchParams.get('name')
    const statuteType = searchParams.get('type')

    if (statuteId && statuteName) {
      setSelectedStatute({
        id: statuteId,
        name: statuteName,
        type: statuteType || '',
        citation_count: 0,
      })
      setSearchQuery(statuteName)
    } else {
      setSelectedStatute(null)
      setSearchQuery('')
    }
  }, [searchParams])

  const handleBack = useCallback(() => {
    // 선택된 법령이 있으면 브라우저 히스토리로 뒤로가기
    if (selectedStatute) {
      router.back()
    } else {
      // 선택된 법령이 없으면 (초기 상태) 홈으로
      if (userRole) {
        router.push(`/?role=${userRole}`)
      } else {
        router.push('/')
      }
    }
  }, [router, userRole, selectedStatute])

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

  // 검색어 변경 (디바운스)
  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value
    setSearchQuery(value)

    // 간단한 디바운스
    const timer = setTimeout(() => {
      handleSearch(value)
    }, 300)
    return () => clearTimeout(timer)
  }, [handleSearch])

  // 법령 선택 (URL에 추가하여 뒤로가기 지원)
  const handleSelect = useCallback((statute: StatuteNode) => {
    setShowDropdown(false)
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
    router.push('/statute-hierarchy')
  }, [router])

  // 그래프에서 노드 클릭 (URL에 추가하여 뒤로가기 지원)
  const handleNodeClick = useCallback((node: GraphNode) => {
    const params = new URLSearchParams()
    params.set('id', node.id)
    params.set('name', node.name)
    if (node.type) params.set('type', node.type)
    router.push(`/statute-hierarchy?${params.toString()}`)
  }, [router])

  return (
    <div className="h-full w-full flex flex-col bg-slate-900">
      {/* 헤더 */}
      <div className="p-4 border-b border-slate-700 bg-slate-800 shrink-0">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <button
              onClick={handleBack}
              className="p-2 text-slate-400 hover:text-white hover:bg-slate-700 rounded-full transition-colors"
              aria-label="뒤로 가기"
            >
              <ArrowLeft size={20} />
            </button>
            <Network className="w-5 h-5 text-amber-400" />
            <h2 className="text-lg font-semibold text-white">법령 체계도</h2>
          </div>
          {selectedStatute && (
            <div className="flex items-center gap-2 text-sm">
              <span className="px-2 py-1 bg-amber-500/20 text-amber-400 rounded">
                {selectedStatute.type}
              </span>
              <span className="text-white font-medium">{selectedStatute.name}</span>
            </div>
          )}
        </div>

        {/* 검색바 */}
        <div className="relative max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
          <input
            type="text"
            value={searchQuery}
            onChange={handleInputChange}
            onFocus={() => searchResults.length > 0 && setShowDropdown(true)}
            placeholder="법령명 또는 약칭 검색 (민법, 민소법, 특가법...)"
            className="w-full pl-10 pr-10 py-2 bg-slate-700 border border-slate-600 rounded-lg
                       text-white placeholder-slate-400
                       focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-transparent"
          />
          {isSearching && (
            <Loader2 className="absolute right-10 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400 animate-spin" />
          )}
          {(searchQuery || selectedStatute) && (
            <button
              onClick={handleClear}
              className="absolute right-3 top-1/2 -translate-y-1/2 p-1 hover:bg-slate-600 rounded"
            >
              <X className="w-4 h-4 text-slate-400" />
            </button>
          )}

          {/* 검색 결과 드롭다운 */}
          {showDropdown && searchResults.length > 0 && (
            <div className="absolute z-50 w-full mt-1 bg-slate-800 border border-slate-600 rounded-lg shadow-lg max-h-64 overflow-y-auto">
              {searchResults.map((statute) => (
                <button
                  key={statute.id}
                  onClick={() => handleSelect(statute)}
                  className="w-full px-4 py-2 text-left hover:bg-slate-700 flex items-center justify-between"
                >
                  <div className="flex flex-col">
                    <span className="text-sm font-medium text-white">{statute.name}</span>
                    <span className="text-xs text-slate-400">
                      {statute.type}
                      {statute.abbreviation && ` (${statute.abbreviation})`}
                    </span>
                  </div>
                  {statute.citation_count > 0 && (
                    <span className="text-xs text-slate-500">
                      인용 {statute.citation_count.toLocaleString()}
                    </span>
                  )}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* 그래프 영역 */}
      <div className="relative flex-1 w-full">
        <div className="absolute inset-0">
          <StatuteForceGraph
            centerId={selectedStatute?.id}
            onNodeClick={handleNodeClick}
          />
        </div>
      </div>
    </div>
  )
}
