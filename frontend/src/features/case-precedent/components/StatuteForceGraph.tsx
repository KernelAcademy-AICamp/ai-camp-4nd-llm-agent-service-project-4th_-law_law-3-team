'use client'

import { useRef, useCallback, useEffect, useState } from 'react'
import dynamic from 'next/dynamic'
import NextImage from 'next/image'
import { casePrecedentService, type GraphNode, type GraphLink } from '../services'
import { getLawTypeLogo, DEFAULT_GOV_LOGO } from '../utils/lawTypeLogo'
import { forceCollide, forceManyBody, forceRadial } from 'd3-force'
// import type { ForceGraphMethods } from 'react-force-graph-2d' // 타입 정의 문제 방지

// SSR 비활성화로 ForceGraph 로드
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

// 이미지 캐시
const imageCache = new Map<string, HTMLImageElement>()

interface StatuteForceGraphProps {
  centerId?: string
  onNodeClick?: (node: GraphNode) => void
}

// 법령 유형별 색상
const TYPE_COLORS: Record<string, string> = {
  '헌법': '#ff6b35',         // 태양 (주황-빨강)
  '법률': '#fbbf24',         // amber
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

// 법령 계급별 크기 (헌법은 태양처럼 크게)
function getHierarchySize(type: string): number {
  if (type === '헌법') return 25  // 태양
  if (type === '법률') return 12
  if (type === '대통령령') return 9
  if (type === '총리령') return 7
  if (type.endsWith('부령')) return 6
  if (type.includes('규칙')) return 5
  return 5 // 기타
}

// 법령 유형별 방사형 반경 (헌법은 중심, 나머지는 동심원)
function getRadialRadius(type: string): number {
  if (type === '헌법') return 0  // 태양 (중심)
  if (type === '법률') return 120
  if (type === '대통령령') return 220
  if (type === '총리령') return 300
  if (type.endsWith('부령')) return 300
  if (type.includes('규칙')) return 380
  return 380 // 기타
}

// 헌법 노드 (태양)
const CONSTITUTION_NODE: GraphNode = {
  id: '001444',
  name: '대한민국헌법',
  type: '헌법',
  abbreviation: '헌법',
  citation_count: 0,
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
  const fgRef = useRef<any>(null) // ForceGraphMethods 타입 제거
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

        let nodes = [...data.nodes]

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

        // 필터링된 노드 ID 집합 생성 (링크 필터링용)
        const visibleNodeIds = new Set(nodes.map(n => n.id))

        // 유효한 노드끼리 연결된 링크만 남김 (dangling link 제거)
        const links = data.links.filter(link => {
          // D3 초기화 전에는 source/target이 string ID임
          const sourceId = typeof link.source === 'object' ? (link.source as any).id : link.source
          const targetId = typeof link.target === 'object' ? (link.target as any).id : link.target
          return visibleNodeIds.has(sourceId) && visibleNodeIds.has(targetId)
        })

        // 중심 노드 결정: centerId가 있으면 해당 노드, 없으면 헌법
        const centerNodeId = centerId || CONSTITUTION_NODE.id

        // HIERARCHY_OF 관계로 부모-자식 맵 생성 (법률 → 시행령)
        const parentToChildren: Record<string, string[]> = {}
        const childToParent: Record<string, string> = {}
        links.forEach(link => {
          if (link.relation === 'HIERARCHY_OF') {
            // source가 하위법(시행령), target이 상위법(법률)
            const sourceId = typeof link.source === 'object' ? (link.source as any).id : link.source
            const targetId = typeof link.target === 'object' ? (link.target as any).id : link.target
            
            const childId = sourceId
            const parentId = targetId

            childToParent[childId] = parentId
            if (!parentToChildren[parentId]) parentToChildren[parentId] = []
            parentToChildren[parentId].push(childId)
          }
        })

        // ... (중략: angleMap 계산 로직은 유지하되, 헌법/법률에만 의미가 있음)
        
        // 법률 노드들 (상위법)에 각도 할당
        const lawNodes = nodes.filter(n => n.type === '법률' && n.id !== centerNodeId)
        const angleMap: Record<string, number> = {}
        lawNodes.forEach((node, index) => {
          const angle = (index / lawNodes.length) * 2 * Math.PI
          angleMap[node.id] = angle
          // 자식 노드들(시행령)도 초기 위치는 부모 근처로 설정
          const children = parentToChildren[node.id] || []
          children.forEach((childId, childIndex) => {
             // 부모 각도 주변으로 초기 배치
             angleMap[childId] = angle + (Math.random() - 0.5) * 0.1
          })
        })

        // 부모가 없는 노드들(독립 노드)에 각도 할당
        const assignedIds = new Set(Object.keys(angleMap))
        const unassignedNodes = nodes.filter(n =>
          n.id !== centerNodeId && !assignedIds.has(n.id)
        )
        // ... (나머지 초기 위치 로직 유지)

        // 계층별로 분류
        const unassignedByRadius: Record<number, typeof nodes> = {}
        unassignedNodes.forEach(node => {
          const radius = getRadialRadius(node.type)
          if (!unassignedByRadius[radius]) unassignedByRadius[radius] = []
          unassignedByRadius[radius].push(node)
        })
        // 각 계층 내에서 균등 배치 (기존 각도 피해서)
        Object.entries(unassignedByRadius).forEach(([, nodesInRadius]) => {
          nodesInRadius.forEach((node, index) => {
            // 기존 법률 노드들 사이에 배치
            const baseAngle = ((index + 0.5) / nodesInRadius.length) * 2 * Math.PI
            angleMap[node.id] = baseAngle
          })
        })

        const nodesWithPosition = nodes.map((node) => {
          // 중심 노드는 중앙에 고정
          if (node.id === centerNodeId) {
            return {
              ...node,
              x: 0,
              y: 0,
              fx: 0,
              fy: 0,
            }
          }

          const radius = getRadialRadius(node.type)
          const angle = angleMap[node.id] ?? Math.random() * 2 * Math.PI
          return {
            ...node,
            x: Math.cos(angle) * radius + (Math.random() - 0.5) * 10,
            y: Math.sin(angle) * radius + (Math.random() - 0.5) * 10,
          }
        })

        setGraphData({ nodes: nodesWithPosition, links: links })
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

  // D3 Force 설정 적용 함수
  const applyD3Forces = useCallback((fg: any) => {
    if (!fg) return
    
    console.log('applyD3Forces executing...', { nodes: graphData.nodes.length })

    try {
      // 1. 방사형 배치 (황도 십이궁 스타일) - 헌법과 법률만 궤도 강제
      // 대통령령 등 하위 법령은 궤도에 구속되지 않고 부모(법률) 주변에 위성처럼 위치함
      fg.d3Force('radial', forceRadial(
        (node: any) => {
          if (node.type === '헌법') return 0 // 태양
          if (node.type === '법률') return 300 // 법률 궤도
          return 300 // 위성 노드도 일단 300 반환
        },
        0, 0  // 중심점
      ).strength((node: any) => {
         // 헌법/법률은 고정하되, 링크에 의해 약간 움직일 수 있도록 강도 조절
         return (node.type === '헌법' || node.type === '법률') ? 0.7 : 0
      }))

      // 2. 충돌 방지: 노드가 겹치지 않도록
      fg.d3Force('collide', forceCollide((node: any) => {
        const size = getHierarchySize(node.type)
        return size * 2 + 10 // 간격 조정
      }).strength(0.8).iterations(3))

      // 3. 반발력 (Charge)
      fg.d3Force('charge', forceManyBody()
        .strength(-200) // 서로 밀어내는 힘 적절히 유지
        .distanceMin(10)
        .distanceMax(400)
      )

      // 4. 링크 힘 (위성 배치 & 관련 법령 응집)
      fg.d3Force('link')
        ?.distance((link: any) => {
          // 계급 관계(HIERARCHY_OF)는 짧게 -> 부모 옆에 착 붙게 (위성)
          if (link.relation === 'HIERARCHY_OF') return 50 
          // 관련 법령은 적당히 가깝게 (너무 멀지 않게)
          return 100 // 150 -> 100
        })
        ?.strength((link: any) => {
           // 계급 관계는 강하게 당김
           if (link.relation === 'HIERARCHY_OF') return 1.0
           // 관련 법령도 서로 끌어당기도록 힘 강화
           return 0.3 // 0.05 -> 0.3
        })

      // center force 제거
      fg.d3Force('center', null)

      // 시뮬레이션 재가열
      fg.d3ReheatSimulation()
      console.log('applyD3Forces applied successfully')
    } catch (err) {
      console.error('Failed to apply D3 forces:', err)
    }
  }, [graphData.nodes.length])

  // ref callback (fgRef 설정 및 force 초기화)
  const handleGraphRef = useCallback((fg: any) => {
    fgRef.current = fg
    if (fg && graphData.nodes.length > 0) {
      // 약간의 지연 후 적용 (초기화 안정성)
      setTimeout(() => applyD3Forces(fg), 10)
    }
  }, [graphData.nodes.length, applyD3Forces])

  // graphData가 변경될 때 force 설정 재적용
  useEffect(() => {
    if (fgRef.current && graphData.nodes.length > 0) {
      setTimeout(() => applyD3Forces(fgRef.current), 10)
    }
  }, [graphData, applyD3Forces])

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

  if (isLoading || !ForceGraph2D) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-slate-900">
        <div className="text-white">그래프 로딩 중...</div>
      </div>
    )
  }

  return (
    <div ref={containerRef} className="w-full h-full relative bg-slate-900">
      <ForceGraph2D
        graphRef={handleGraphRef} { ...({} as any) }
        width={dimensions.width}
        height={dimensions.height}
        graphData={graphData}
        nodeId="id"
        nodeLabel={() => ''}

        nodeColor={(node) => getNodeColor((node as GraphNode).type)}
        nodeVal={(node) => getNodeSize(node as GraphNode)}
        nodeCanvasObject={(node, ctx, globalScale) => {
          const n = node as GraphNode & { x?: number; y?: number }
          if (n.x === undefined || n.y === undefined) return

          const size = getNodeSize(n)
          const fontSize = Math.max(12 / globalScale, 3)
          const isHovered = hoveredNode?.id === n.id
          // 중심 노드: centerId가 있으면 해당 노드, 없으면 헌법
          const isCenterNode = centerId ? (n.id === centerId) : (n.type === '헌법')

          // 로고 이미지 가져오기
          const logoImg = getLogoImage(n.type)

          // 중심 노드(태양) 특별 효과
          if (isCenterNode) {
            // 태양 glow 효과
            const gradient = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, size * 2)
            gradient.addColorStop(0, 'rgba(255, 107, 53, 0.8)')
            gradient.addColorStop(0.5, 'rgba(255, 165, 0, 0.4)')
            gradient.addColorStop(1, 'rgba(255, 200, 0, 0)')
            ctx.beginPath()
            ctx.arc(n.x, n.y, size * 2, 0, 2 * Math.PI)
            ctx.fillStyle = gradient
            ctx.fill()

            // 태양 본체
            ctx.beginPath()
            ctx.arc(n.x, n.y, size, 0, 2 * Math.PI)
            const sunGradient = ctx.createRadialGradient(n.x - size * 0.3, n.y - size * 0.3, 0, n.x, n.y, size)
            sunGradient.addColorStop(0, '#ffdd00')
            sunGradient.addColorStop(0.5, '#ff8c00')
            sunGradient.addColorStop(1, '#ff4500')
            ctx.fillStyle = sunGradient
            ctx.shadowColor = '#ff6b35'
            ctx.shadowBlur = 20
            ctx.fill()
            ctx.shadowBlur = 0
          } else {
            // 일반 노드 배경 (흰색 원)
            ctx.beginPath()
            ctx.arc(n.x, n.y, size + 2, 0, 2 * Math.PI)
            ctx.fillStyle = '#ffffff'
            if (isHovered) {
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
          }

          // 라벨 표시: 호버되지 않은 경우에만 여기서 그림 (호버된 노드는 onRenderFramePost에서 최상단에 그림)
          if (n.name && !isHovered) {
            const displayText = getDisplayName(n)

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
        linkLabel={() => ''}
        linkVisibility={(link) => {
          const l = link as unknown as Omit<GraphLink, 'source' | 'target'> & { source: GraphNode | string; target: GraphNode | string }
          // 계급 관계(HIERARCHY_OF)는 항상 표시
          if (l.relation === 'HIERARCHY_OF') return true
          // 관련 법령은 호버된 노드와 연결된 경우에만 표시
          if (!hoveredNode) return false
          const sourceId = typeof l.source === 'string' ? l.source : l.source.id
          const targetId = typeof l.target === 'string' ? l.target : l.target.id
          return sourceId === hoveredNode.id || targetId === hoveredNode.id
        }}
        linkColor={(link) => {
          const l = link as GraphLink
          // HIERARCHY_OF: 계급 관계는 황금색, 그 외는 회색
          return l.relation === 'HIERARCHY_OF' ? 'rgba(251, 191, 36, 0.8)' : 'rgba(148, 163, 184, 0.6)'
        }}
        linkWidth={(link) => {
          const l = link as GraphLink
          return l.relation === 'HIERARCHY_OF' ? 2 : 1
        }}
        linkDirectionalArrowLength={(link) => {
          const l = link as GraphLink
          // 계급 관계에만 화살표 표시
          return l.relation === 'HIERARCHY_OF' ? 6 : 0
        }}
        linkDirectionalArrowRelPos={1}
        linkDirectionalParticles={0}
        onRenderFramePre={(ctx) => {
          // 궤도 원 그리기 (황도 십이궁 스타일)
          const orbits = [120, 220, 300, 380]  // 법률, 대통령령, 총리령/부령, 규칙
          const orbitColors = ['rgba(251, 191, 36, 0.15)', 'rgba(96, 165, 250, 0.15)', 'rgba(52, 211, 153, 0.12)', 'rgba(156, 163, 175, 0.1)']

          orbits.forEach((radius, i) => {
            ctx.beginPath()
            ctx.arc(0, 0, radius, 0, 2 * Math.PI)
            ctx.strokeStyle = orbitColors[i]
            ctx.lineWidth = 2
            ctx.setLineDash([5, 5])
            ctx.stroke()
            ctx.setLineDash([])
          })
        }}
        onRenderFramePost={(ctx, globalScale) => {
          if (hoveredNode) {
            const n = hoveredNode as GraphNode & { x?: number; y?: number }
            if (n.x === undefined || n.y === undefined) return

            const size = getNodeSize(n)
            const fontSize = Math.max(12 / globalScale, 3)
            const displayText = n.name

            ctx.font = `${fontSize}px Sans-Serif`
            ctx.textAlign = 'center'
            ctx.textBaseline = 'middle'

            // 텍스트 배경 (호버 시 더 명확하게)
            const textWidth = ctx.measureText(displayText).width
            ctx.fillStyle = 'rgba(0, 0, 0, 0.9)'
            ctx.fillRect(n.x - textWidth / 2 - 4, n.y + size + 4, textWidth + 8, fontSize + 6)

            ctx.fillStyle = '#ffffff'
            ctx.fillText(displayText, n.x, n.y + size + fontSize / 2 + 7)
          }
        }}
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

      {/* 범례 */}
      <div className="absolute bottom-4 left-4 bg-black/70 rounded-lg p-3 text-xs text-white">
        <div className="font-bold mb-2">법령 계층 (황도 십이궁)</div>
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
        <div className="mt-3 pt-2 border-t border-gray-600">
          <div className="font-bold mb-1">관계</div>
          <div className="flex items-center gap-2">
            <span className="text-amber-400">→</span>
            <span>계급 (하위법 → 상위법)</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-slate-400">—</span>
            <span className="text-slate-400">관련 법령 (호버 시)</span>
          </div>
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
