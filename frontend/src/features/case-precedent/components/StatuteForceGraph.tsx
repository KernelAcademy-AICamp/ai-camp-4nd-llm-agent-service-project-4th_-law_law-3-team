'use client'

import { useRef, useCallback, useEffect, useState } from 'react'
import dynamic from 'next/dynamic'
import { casePrecedentService, type GraphNode, type GraphLink } from '../services'
import { getLawTypeLogo, DEFAULT_GOV_LOGO } from '../utils/lawTypeLogo'

// Dynamic import to avoid SSR issues
const ForceGraph2D = dynamic(() => import('react-force-graph-2d'), {
  ssr: false,
})

// 이미지 캐시
const imageCache = new Map<string, HTMLImageElement>()

interface StatuteForceGraphProps {
  centerId?: string
  onNodeClick?: (node: GraphNode) => void
}

// 법령 유형별 색상
const TYPE_COLORS: Record<string, string> = {
  '법률': '#fbbf24',        // amber
  '대통령령': '#60a5fa',     // blue
  '총리령': '#34d399',       // emerald
  '대법원규칙': '#a78bfa',   // violet
  '헌법재판소규칙': '#f472b6', // pink
}

function getNodeColor(type: string): string {
  if (type in TYPE_COLORS) return TYPE_COLORS[type]
  if (type.endsWith('부령')) return '#fb923c' // orange for 부령
  return '#9ca3af' // gray default
}

// 법령 계급별 크기 (상위법일수록 큼)
function getHierarchySize(type: string): number {
  if (type === '헌법') return 14
  if (type === '법률') return 11
  if (type === '대통령령') return 8
  if (type === '총리령') return 7
  if (type.endsWith('부령')) return 6
  if (type.includes('규칙')) return 5
  return 5 // 기타
}

// 표시용 이름 (약어 또는 8자 제한)
function getDisplayName(node: GraphNode): string {
  if (node.abbreviation) return node.abbreviation
  if (node.name.length > 8) return node.name.slice(0, 8) + '…'
  return node.name
}

// 로고 이미지 로드 (캐싱)
function getLogoImage(type: string): HTMLImageElement | null {
  const logoPath = getLawTypeLogo(type) || DEFAULT_GOV_LOGO

  if (imageCache.has(logoPath)) {
    return imageCache.get(logoPath) || null
  }

  // 이미지 로드 시작
  const img = new Image()
  img.src = logoPath
  img.onload = () => {
    imageCache.set(logoPath, img)
  }
  img.onerror = () => {
    // 실패 시 기본 로고 사용
    const defaultImg = new Image()
    defaultImg.src = DEFAULT_GOV_LOGO
    defaultImg.onload = () => {
      imageCache.set(logoPath, defaultImg)
    }
  }

  // 로드 중에는 null 반환 (다음 렌더링에서 캐시 사용)
  imageCache.set(logoPath, img)
  return null
}

export function StatuteForceGraph({ centerId, onNodeClick }: StatuteForceGraphProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [graphData, setGraphData] = useState<{ nodes: GraphNode[]; links: GraphLink[] }>({ nodes: [], links: [] })
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 })
  const [isLoading, setIsLoading] = useState(true)
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null)

  // 그래프 데이터 로드
  useEffect(() => {
    const loadGraph = async () => {
      setIsLoading(true)
      try {
        const data = await casePrecedentService.getStatuteGraph(centerId, 150)
        setGraphData(data)
      } catch (error) {
        console.error('그래프 로드 실패:', error)
      } finally {
        setIsLoading(false)
      }
    }
    loadGraph()
  }, [centerId])

  // 컨테이너 크기 감지 (ResizeObserver 사용)
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const width = containerRef.current.clientWidth || window.innerWidth
        const height = containerRef.current.clientHeight || window.innerHeight - 120
        setDimensions({ width, height })
      } else {
        // fallback to window size
        setDimensions({
          width: window.innerWidth,
          height: window.innerHeight - 120,
        })
      }
    }

    // 초기 로드 시 약간의 지연 후 크기 계산 (레이아웃 완료 대기)
    const timer = setTimeout(updateDimensions, 100)

    // ResizeObserver로 컨테이너 크기 변경 감지
    let resizeObserver: ResizeObserver | null = null
    if (containerRef.current) {
      resizeObserver = new ResizeObserver(() => {
        updateDimensions()
      })
      resizeObserver.observe(containerRef.current)
    }

    window.addEventListener('resize', updateDimensions)

    return () => {
      clearTimeout(timer)
      resizeObserver?.disconnect()
      window.removeEventListener('resize', updateDimensions)
    }
  }, [])

  // 노드 크기 계산 (법령 계급 기반)
  const getNodeSize = useCallback((node: GraphNode) => {
    return getHierarchySize(node.type)
  }, [])

  // 노드 클릭 핸들러
  const handleNodeClick = useCallback((node: object) => {
    const n = node as GraphNode
    if (onNodeClick && n.id) {
      onNodeClick(n)
    }
  }, [onNodeClick])

  // 노드 호버 핸들러
  const handleNodeHover = useCallback((node: object | null) => {
    setHoveredNode(node as GraphNode | null)
  }, [])

  if (isLoading) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-slate-900">
        <div className="text-white">그래프 로딩 중...</div>
      </div>
    )
  }

  return (
    <div ref={containerRef} className="w-full h-full relative bg-slate-900">
      <ForceGraph2D
        width={dimensions.width}
        height={dimensions.height}
        graphData={graphData}
        nodeId="id"
        nodeLabel="name"
        nodeColor={(node) => getNodeColor((node as GraphNode).type)}
        nodeVal={(node) => getNodeSize(node as GraphNode)}
        nodeCanvasObject={(node, ctx, globalScale) => {
          const n = node as GraphNode & { x?: number; y?: number }
          if (!n.x || !n.y) return

          const size = getNodeSize(n)
          const fontSize = Math.max(12 / globalScale, 3)
          const isHovered = hoveredNode?.id === n.id
          const isCenter = centerId === n.id

          // 로고 이미지 가져오기
          const logoImg = getLogoImage(n.type)

          // 노드 배경 (흰색 원)
          ctx.beginPath()
          ctx.arc(n.x, n.y, size + 2, 0, 2 * Math.PI)
          ctx.fillStyle = '#ffffff'
          if (isHovered || isCenter) {
            ctx.shadowColor = getNodeColor(n.type)
            ctx.shadowBlur = 15
          }
          ctx.fill()
          ctx.shadowBlur = 0

          // 테두리 (타입별 색상)
          ctx.strokeStyle = getNodeColor(n.type)
          ctx.lineWidth = isHovered ? 3 / globalScale : 2 / globalScale
          ctx.stroke()

          // 로고 이미지 그리기
          if (logoImg && logoImg.complete && logoImg.naturalWidth > 0) {
            const imgSize = size * 1.5
            ctx.drawImage(
              logoImg,
              n.x - imgSize / 2,
              n.y - imgSize / 2,
              imgSize,
              imgSize
            )
          } else {
            // 이미지 로드 전/실패 시 색상 원으로 대체
            ctx.beginPath()
            ctx.arc(n.x, n.y, size * 0.7, 0, 2 * Math.PI)
            ctx.fillStyle = getNodeColor(n.type)
            ctx.fill()
          }

          // 라벨 표시: 호버 시 전체 이름, 아니면 약어/8자
          if (n.name) {
            const displayText = isHovered ? n.name : getDisplayName(n)

            ctx.font = `${fontSize}px Sans-Serif`
            ctx.textAlign = 'center'
            ctx.textBaseline = 'middle'

            // 텍스트 배경
            const textWidth = ctx.measureText(displayText).width
            ctx.fillStyle = 'rgba(0, 0, 0, 0.7)'
            ctx.fillRect(n.x - textWidth / 2 - 2, n.y + size + 4, textWidth + 4, fontSize + 4)

            ctx.fillStyle = '#ffffff'
            ctx.fillText(displayText, n.x, n.y + size + fontSize / 2 + 6)
          }
        }}
        linkColor={(link) => {
          const l = link as GraphLink
          // HIERARCHY_OF: 하위법령 → 상위법령 (시행령 → 법률)
          return l.relation === 'HIERARCHY_OF' ? 'rgba(251, 191, 36, 0.6)' : 'rgba(100, 116, 139, 0.4)'
        }}
        linkWidth={(link) => {
          const l = link as GraphLink
          return l.relation === 'HIERARCHY_OF' ? 1.5 : 0.5
        }}
        linkDirectionalArrowLength={(link) => {
          const l = link as GraphLink
          // 계급 관계에만 화살표 표시
          return l.relation === 'HIERARCHY_OF' ? 6 : 0
        }}
        linkDirectionalArrowRelPos={1}
        linkDirectionalParticles={0}
        onNodeClick={handleNodeClick}
        onNodeHover={handleNodeHover}
        cooldownTicks={100}
        d3AlphaDecay={0.02}
        d3VelocityDecay={0.3}
        enableNodeDrag={true}
        enableZoomInteraction={true}
        enablePanInteraction={true}
      />

      {/* 범례 */}
      <div className="absolute bottom-4 left-4 bg-black/70 rounded-lg p-3 text-xs text-white">
        <div className="font-bold mb-2">법령 유형</div>
        <div className="space-y-1.5">
          <div className="flex items-center gap-2">
            <img src="/data/logo/National_Assembly.png" alt="국회" className="w-5 h-5 object-contain bg-white rounded-full p-0.5" />
            <span>법률</span>
          </div>
          <div className="flex items-center gap-2">
            <img src="/data/logo/president.svg" alt="대통령" className="w-5 h-5 object-contain bg-white rounded-full p-0.5" />
            <span>대통령령</span>
          </div>
          <div className="flex items-center gap-2">
            <img src="/data/logo/government_of_Korea.svg" alt="정부" className="w-5 h-5 object-contain bg-white rounded-full p-0.5" />
            <span>부령/기타</span>
          </div>
        </div>
        <div className="mt-3 pt-2 border-t border-gray-600">
          <div className="font-bold mb-1">관계</div>
          <div className="flex items-center gap-2">
            <span className="text-amber-400">→</span>
            <span>계급 (하위법 → 상위법)</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-gray-400">—</span>
            <span>관련 법령</span>
          </div>
        </div>
        <div className="mt-2 text-gray-400">
          노드 크기 = 법령 계급
        </div>
      </div>

      {/* 호버 정보 */}
      {hoveredNode && (
        <div className="absolute top-4 right-4 bg-black/80 rounded-lg p-3 text-white max-w-xs">
          <div className="font-bold text-sm">{hoveredNode.name}</div>
          <div className="text-xs text-gray-300 mt-1">{hoveredNode.type}</div>
          <div className="text-xs text-gray-400 mt-1">
            인용 횟수: {hoveredNode.citation_count.toLocaleString()}
          </div>
        </div>
      )}
    </div>
  )
}
