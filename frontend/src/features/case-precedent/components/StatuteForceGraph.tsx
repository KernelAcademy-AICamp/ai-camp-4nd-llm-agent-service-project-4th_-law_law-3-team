'use client'

import { useRef, useCallback, useEffect, useState, useMemo } from 'react'
import dynamic from 'next/dynamic'
import NextImage from 'next/image'
import { Plus, Minus, Maximize2 } from 'lucide-react'
import { casePrecedentService, type GraphNode, type GraphLink } from '../services'
import type { SimulationNodeDatum } from 'd3-force'
import { forceCollide, forceManyBody, forceRadial } from 'd3-force'
import {
  getNodeColor,
  getHierarchySize,
  getRadialRadius,
  getFilterCategory,
  CONSTITUTION_NODE,
} from './StatuteForceGraph.constants'
import { renderNode, renderHoveredLabel, type NodeRenderContext } from './StatuteForceGraph.renderers'

// d3 시뮬레이션 노드 (런타임에 d3가 x, y 등을 주입)
type ForceNode = GraphNode & SimulationNodeDatum

// d3 ForceLink 인스턴스의 distance/strength 체이닝 타입
interface ForceLinkForce {
  distance: (fn: (link: GraphLink) => number) => ForceLinkForce
  strength: (fn: (link: GraphLink) => number) => ForceLinkForce
}

// react-force-graph-2d 인스턴스 타입 (패키지 타입 정의 불안정 대응)
interface ForceGraphInstance {
  d3Force(name: 'link'): ForceLinkForce | undefined
  d3Force(name: string, force: unknown): void
  d3ReheatSimulation: () => void
  zoom(k: number, ms?: number): void
  centerAt(x: number, y: number, ms?: number): void
  zoomToFit(ms?: number, padding?: number): void
}

// SSR 비활성화로 ForceGraph 로드 (Wrapper 컴포넌트 사용)
const ForceGraph2D = dynamic(
  () => import('./StatuteForceGraphWrapper'),
  {
    ssr: false,
    loading: () => (
      <div className="w-full h-full flex items-center justify-center bg-slate-900">
        <div className="text-white">그래프 엔진 로딩 중...</div>
      </div>
    )
  }
)

interface StatuteForceGraphProps {
  centerId?: string
  centerName?: string
  onNodeClick?: (node: GraphNode) => void
  visibleTypes?: Set<string>
  highlightNodeId?: string
}

export function StatuteForceGraph({ centerId, centerName, onNodeClick, visibleTypes, highlightNodeId }: StatuteForceGraphProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const fgRef = useRef<ForceGraphInstance | null>(null)
  const [graphData, setGraphData] = useState<{ nodes: GraphNode[]; links: GraphLink[] }>({ nodes: [], links: [] })
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 })
  const [isLoading, setIsLoading] = useState(true)
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null)
  // 원본 데이터 (필터링 전)
  const [rawGraphData, setRawGraphData] = useState<{ nodes: GraphNode[]; links: GraphLink[] }>({ nodes: [], links: [] })

  const resolveCenterId = useCallback((nodes: GraphNode[]): string | undefined => {
    const normalizedCenterName = centerName?.trim()
    const normalizedCenterId = centerId?.trim()

    if (normalizedCenterName) {
      const nameMatchedNode = nodes.find(
        (node) =>
          node.name === normalizedCenterName || node.abbreviation === normalizedCenterName
      )
      if (nameMatchedNode) {
        return nameMatchedNode.id
      }
    }

    if (normalizedCenterId) {
      const explicitCenterId = nodes.find((node) => node.id === normalizedCenterId)?.id
      if (explicitCenterId) {
        return explicitCenterId
      }
    }

    return undefined
  }, [centerId, centerName])

  // 그래프 데이터 로드
  useEffect(() => {
    const loadGraph = async () => {
      setIsLoading(true)
      try {
        const data = await casePrecedentService.getStatuteGraph(centerId, 150)

        let nodes = [...data.nodes]
        const normalizedCenterId = centerId?.trim()
        const normalizedCenterName = centerName?.trim()

        // centerId가 없을 때만 헌법을 추가하고, 하위 법령(총리령/부령/규칙)은 필터링
        if (!centerId) {
          const hasConstitution = nodes.some(n => n.type === '헌법')
          if (!hasConstitution) {
            nodes = [CONSTITUTION_NODE, ...nodes]
          }

          // 초기 화면에서는 총리령/부령/규칙 숨김 (대통령령까지만 표시)
          nodes = nodes.filter(n =>
            n.type === '헌법' || n.type === '법률' || n.type === '대통령령'
          )
        }

        // 중심 노드 결정: centerId가 실제 노드로 존재하면 해당 노드, 없으면 최소한 요청한 centerId를 우선 사용
        const resolvedCenterId = resolveCenterId(nodes)
        const centerNodeId = resolvedCenterId || normalizedCenterId || CONSTITUTION_NODE.id

        // API 반환 노드에 중심 노드가 누락되는 경우를 보정(네트워크/데이터 정합성 편차 대비)
        if (normalizedCenterId && !nodes.some((node) => node.id === normalizedCenterId)) {
          nodes.push({
            id: centerNodeId,
            name: normalizedCenterName || centerNodeId,
            type: centerNodeId === CONSTITUTION_NODE.id ? '헌법' : '법률',
            citation_count: 0,
          })
        }

        // 필터링된 노드 ID 집합 생성 (링크 필터링용)
        const visibleNodeIds = new Set(nodes.map((n) => n.id))

        // 유효한 노드끼리 연결된 링크만 남김 (dangling link 제거)
        const links = data.links.filter((link) => {
          const sourceId =
            typeof link.source === 'object' ? (link.source as { id: string }).id : link.source
          const targetId =
            typeof link.target === 'object' ? (link.target as { id: string }).id : link.target
          return visibleNodeIds.has(sourceId) && visibleNodeIds.has(targetId)
        })

        // HIERARCHY_OF 관계로 부모-자식 맵 생성 (법률 → 시행령)
        const parentToChildren: Record<string, string[]> = {}
        links.forEach(link => {
          if (link.relation === 'HIERARCHY_OF') {
            const sourceId = typeof link.source === 'object' ? (link.source as { id: string }).id : link.source
            const targetId = typeof link.target === 'object' ? (link.target as { id: string }).id : link.target

            const childId = sourceId
            const parentId = targetId

            if (!parentToChildren[parentId]) parentToChildren[parentId] = []
            parentToChildren[parentId].push(childId)
          }
        })

        // 법률 노드들 (상위법)에 각도 할당
        const lawNodes = nodes.filter(n => n.type === '법률' && n.id !== centerNodeId)
        const angleMap: Record<string, number> = {}
        lawNodes.forEach((node, index) => {
          const angle = (index / lawNodes.length) * 2 * Math.PI
          angleMap[node.id] = angle
          const children = parentToChildren[node.id] || []
          children.forEach(() => {
             angleMap[node.id] = angle + (Math.random() - 0.5) * 0.1
          })
        })

        // 부모가 없는 노드들(독립 노드)에 각도 할당
        const assignedIds = new Set(Object.keys(angleMap))
        const unassignedNodes = nodes.filter(n =>
          n.id !== centerNodeId && !assignedIds.has(n.id)
        )

        const unassignedByRadius: Record<number, typeof nodes> = {}
        unassignedNodes.forEach(node => {
          const radius = getRadialRadius(node.type)
          if (!unassignedByRadius[radius]) unassignedByRadius[radius] = []
          unassignedByRadius[radius].push(node)
        })
        Object.entries(unassignedByRadius).forEach(([, nodesInRadius]) => {
          nodesInRadius.forEach((node, index) => {
            const baseAngle = ((index + 0.5) / nodesInRadius.length) * 2 * Math.PI
            angleMap[node.id] = baseAngle
          })
        })

        const nodesWithPosition = nodes.map((node) => {
          if (node.id === centerNodeId) {
            return { ...node, x: 0, y: 0, fx: 0, fy: 0 }
          }

          const radius = getRadialRadius(node.type)
          const angle = angleMap[node.id] ?? Math.random() * 2 * Math.PI
          return {
            ...node,
            x: Math.cos(angle) * radius + (Math.random() - 0.5) * 10,
            y: Math.sin(angle) * radius + (Math.random() - 0.5) * 10,
          }
        })

        setRawGraphData({ nodes: nodesWithPosition, links: links })
        setGraphData({ nodes: nodesWithPosition, links: links })
      } catch (error) {
        console.error('그래프 로드 실패:', error)
      } finally {
        setIsLoading(false)
      }
    }
    loadGraph()
  }, [centerId, resolveCenterId, centerName])

  // visibleTypes 필터 적용
  useEffect(() => {
    if (!visibleTypes || rawGraphData.nodes.length === 0) return

    const filteredNodes = rawGraphData.nodes.filter(node => {
      const category = getFilterCategory(node.type)
      return visibleTypes.has(category)
    })
    const filteredNodeIds = new Set(filteredNodes.map(n => n.id))
    const filteredLinks = rawGraphData.links.filter(link => {
      const sourceId = typeof link.source === 'object' ? (link.source as { id: string }).id : link.source
      const targetId = typeof link.target === 'object' ? (link.target as { id: string }).id : link.target
      return filteredNodeIds.has(sourceId) && filteredNodeIds.has(targetId)
    })

    setGraphData({ nodes: filteredNodes, links: filteredLinks })
  }, [visibleTypes, rawGraphData])

  // 컨테이너 크기 감지 (ResizeObserver만 사용 — window resize 중복 제거)
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const width = containerRef.current.clientWidth || window.innerWidth
        const height = containerRef.current.clientHeight || window.innerHeight - 120
        setDimensions({ width, height })
      } else {
        setDimensions({
          width: window.innerWidth,
          height: window.innerHeight - 120,
        })
      }
    }

    requestAnimationFrame(updateDimensions)

    let resizeObserver: ResizeObserver | null = null
    if (containerRef.current) {
      resizeObserver = new ResizeObserver(() => {
        updateDimensions()
      })
      resizeObserver.observe(containerRef.current)
    }

    return () => {
      resizeObserver?.disconnect()
    }
  }, [])

  // D3 Force 설정 적용 함수
  const applyD3Forces = useCallback((fg: ForceGraphInstance) => {
    if (!fg) return

    try {
      fg.d3Force('radial', forceRadial<ForceNode>(
        (node) => {
          if (node.type === '헌법') return 0
          if (node.type === '법률') return 300
          return 300
        },
        0, 0
      ).strength((node) => {
         return (node.type === '헌법' || node.type === '법률') ? 0.7 : 0
      }))

      fg.d3Force('collide', forceCollide<ForceNode>((node) => {
        const size = getHierarchySize(node.type)
        return size * 2 + 10
      }).strength(0.8).iterations(3))

      fg.d3Force('charge', forceManyBody()
        .strength(-200)
        .distanceMin(10)
        .distanceMax(400)
      )

      fg.d3Force('link')
        ?.distance((link) => {
          if (link.relation === 'HIERARCHY_OF') return 50
          return 100
        })
        ?.strength((link) => {
           if (link.relation === 'HIERARCHY_OF') return 1.0
           return 0.3
        })

      fg.d3Force('center', null)
      fg.d3ReheatSimulation()
    } catch (err) {
      console.error('Failed to apply D3 forces:', err)
    }
  }, [])

  // ref callback (fgRef 설정 및 force 초기화)
  const handleGraphRef = useCallback((fg: ForceGraphInstance | null) => {
    fgRef.current = fg
    if (fg && graphData.nodes.length > 0) {
      setTimeout(() => applyD3Forces(fg), 10)
    }
  }, [graphData.nodes.length, applyD3Forces])

  // graphData가 변경될 때 force 설정 재적용
  useEffect(() => {
    const fg = fgRef.current
    if (fg && graphData.nodes.length > 0) {
      setTimeout(() => applyD3Forces(fg), 10)
    }
  }, [graphData, applyD3Forces])

  // 노드 클릭 핸들러 (센터링 포함)
  const handleNodeClick = useCallback((node: object) => {
    const n = node as GraphNode & { x?: number; y?: number }
    if (n.x !== undefined && n.y !== undefined && fgRef.current) {
      fgRef.current.centerAt(n.x, n.y, 500)
    }
    if (onNodeClick && n.id) {
      onNodeClick(n)
    }
  }, [onNodeClick])

  // 노드 호버 핸들러
  const handleNodeHover = useCallback((node: object | null) => {
    setHoveredNode(node as GraphNode | null)
  }, [])

  // 줌 컨트롤
  const handleZoomIn = useCallback(() => {
    fgRef.current?.zoom(2, 300)
  }, [])

  const handleZoomOut = useCallback(() => {
    fgRef.current?.zoom(0.5, 300)
  }, [])

  const handleZoomFit = useCallback(() => {
    fgRef.current?.zoomToFit(400, 40)
  }, [])

  // 그래프 통계
  const graphStats = useMemo(() => ({
    nodeCount: graphData.nodes.length,
    linkCount: graphData.links.length,
  }), [graphData.nodes.length, graphData.links.length])

  // 렌더링 컨텍스트 (nodeCanvasObject에 전달)
  const renderContext = useMemo<NodeRenderContext>(() => ({
    hoveredNodeId: hoveredNode?.id,
    highlightNodeId,
    centerNodeId: resolveCenterId(graphData.nodes) || CONSTITUTION_NODE.id,
  }), [hoveredNode?.id, highlightNodeId, resolveCenterId, graphData.nodes])

  // ForceGraph2D에 전달할 콜백들 (useCallback으로 안정적 참조)
  const nodeColor = useCallback(
    (node: object) => getNodeColor((node as GraphNode).type),
    [],
  )

  const nodeVal = useCallback(
    (node: object) => getHierarchySize((node as GraphNode).type),
    [],
  )

  const nodeCanvasObject = useCallback(
    (node: object, ctx: CanvasRenderingContext2D, globalScale: number) => {
      renderNode(node, ctx, globalScale, renderContext)
    },
    [renderContext],
  )

  const emptyLabel = useCallback(() => '', [])

  const linkVisibility = useCallback(
    (link: object) => {
      const l = link as unknown as Omit<GraphLink, 'source' | 'target'> & { source: GraphNode | string; target: GraphNode | string }
      if (l.relation === 'HIERARCHY_OF') return true
      if (!hoveredNode) return false
      const sourceId = typeof l.source === 'string' ? l.source : l.source.id
      const targetId = typeof l.target === 'string' ? l.target : l.target.id
      return sourceId === hoveredNode.id || targetId === hoveredNode.id
    },
    [hoveredNode],
  )

  const linkColor = useCallback(
    (link: object) => {
      const l = link as GraphLink
      return l.relation === 'HIERARCHY_OF' ? 'rgba(251, 191, 36, 0.8)' : 'rgba(148, 163, 184, 0.6)'
    },
    [],
  )

  const linkWidth = useCallback(
    (link: object) => {
      const l = link as GraphLink
      return l.relation === 'HIERARCHY_OF' ? 2 : 1
    },
    [],
  )

  const linkDirectionalArrowLength = useCallback(
    (link: object) => {
      const l = link as GraphLink
      return l.relation === 'HIERARCHY_OF' ? 6 : 0
    },
    [],
  )

  const onRenderFramePre = useCallback(
    (_ctx: CanvasRenderingContext2D) => {},
    [],
  )

  const onRenderFramePost = useCallback(
    (ctx: CanvasRenderingContext2D, globalScale: number) => {
      renderHoveredLabel(ctx, globalScale, hoveredNode)
    },
    [hoveredNode],
  )

  if (isLoading || dimensions.width === 0 || !ForceGraph2D) {
    return (
      <div ref={containerRef} className="w-full h-full flex items-center justify-center bg-slate-900">
        <div className="text-white">그래프 로딩 중...</div>
      </div>
    )
  }

  return (
    <div ref={containerRef} className="w-full h-full relative bg-slate-900">
      <ForceGraph2D
        graphRef={handleGraphRef as React.Ref<never>}
        width={dimensions.width}
        height={dimensions.height}
        graphData={graphData}
        nodeId="id"
        nodeLabel={emptyLabel}
        nodeColor={nodeColor}
        nodeVal={nodeVal}
        nodeCanvasObject={nodeCanvasObject}
        linkLabel={emptyLabel}
        linkVisibility={linkVisibility}
        linkColor={linkColor}
        linkWidth={linkWidth}
        linkDirectionalArrowLength={linkDirectionalArrowLength}
        linkDirectionalArrowRelPos={1}
        linkDirectionalParticles={0}
        onRenderFramePre={onRenderFramePre}
        onRenderFramePost={onRenderFramePost}
        onNodeClick={handleNodeClick}
        onNodeHover={handleNodeHover}
        cooldownTicks={150}
        d3AlphaDecay={0.02}
        d3VelocityDecay={0.3}
        warmupTicks={50}
        enableNodeDrag={true}
        enableZoomInteraction={true}
        enablePanInteraction={true}
      />

      {/* 통계 오버레이 */}
      <div className="absolute top-3 left-1/2 -translate-x-1/2 bg-white/80 border border-gray-200 rounded-full px-3 py-1 text-xs text-gray-500 pointer-events-none">
        노드 {graphStats.nodeCount}개 · 링크 {graphStats.linkCount}개
      </div>

      {/* 범례 */}
      <div className="absolute bottom-4 left-4 bg-white/90 border border-gray-200 rounded-lg p-3 text-xs text-gray-700 shadow-sm">
        <div className="font-bold mb-2">법령 계층</div>
        <div className="space-y-1.5">
          <div className="flex items-center gap-2">
            <div className="w-5 h-5 rounded-full bg-gradient-to-br from-orange-400 to-red-500 shadow-lg shadow-orange-500/50" />
            <span>헌법 (태양/중심)</span>
          </div>
          <div className="flex items-center gap-2">
            <NextImage src="/data/logo/National_Assembly.png" alt="국회" width={20} height={20} className="object-contain bg-white rounded-full p-0.5" />
            <span>법률</span>
          </div>
          <div className="flex items-center gap-2">
            <NextImage src="/data/logo/president.svg" alt="대통령" width={20} height={20} className="object-contain bg-white rounded-full p-0.5" />
            <span>대통령령</span>
          </div>
          <div className="flex items-center gap-2">
            <NextImage src="/data/logo/government_of_Korea.svg" alt="정부" width={20} height={20} className="object-contain bg-white rounded-full p-0.5" />
            <span>총리령/부령/규칙</span>
          </div>
        </div>
        <div className="mt-3 pt-2 border-t border-gray-200">
          <div className="font-bold mb-1">관계</div>
          <div className="flex items-center gap-2">
            <span className="text-amber-500">→</span>
            <span>계급 (하위법 → 상위법)</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-gray-400">—</span>
            <span className="text-gray-400">관련 법령 (호버 시)</span>
          </div>
        </div>
      </div>

      {/* 줌 컨트롤 */}
      <div className="absolute bottom-4 right-4 flex flex-col gap-1">
        <button
          onClick={handleZoomIn}
          className="w-8 h-8 bg-white/90 hover:bg-white border border-gray-200 text-gray-600 rounded flex items-center justify-center transition-colors shadow-sm"
          aria-label="확대"
        >
          <Plus className="w-4 h-4" />
        </button>
        <button
          onClick={handleZoomOut}
          className="w-8 h-8 bg-white/90 hover:bg-white border border-gray-200 text-gray-600 rounded flex items-center justify-center transition-colors shadow-sm"
          aria-label="축소"
        >
          <Minus className="w-4 h-4" />
        </button>
        <button
          onClick={handleZoomFit}
          className="w-8 h-8 bg-white/90 hover:bg-white border border-gray-200 text-gray-600 rounded flex items-center justify-center transition-colors shadow-sm"
          aria-label="전체 보기"
        >
          <Maximize2 className="w-4 h-4" />
        </button>
      </div>

      {/* 호버 정보 */}
      {hoveredNode && (
        <div className="absolute top-4 left-4 bg-black/80 rounded-lg p-3 text-white max-w-xs" style={{ top: '2.5rem' }}>
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
