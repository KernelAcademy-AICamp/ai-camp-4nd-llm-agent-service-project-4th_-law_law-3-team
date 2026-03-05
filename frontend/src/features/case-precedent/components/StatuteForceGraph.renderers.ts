import type { GraphNode } from '../services'
import { getNodeColor, getHierarchySize, getDisplayName, getLogoImage } from './StatuteForceGraph.constants'

export interface NodeRenderContext {
  hoveredNodeId: string | undefined
  highlightNodeId: string | undefined
  centerNodeId: string
}

export function renderNode(
  node: object,
  ctx: CanvasRenderingContext2D,
  globalScale: number,
  context: NodeRenderContext,
): void {
  const n = node as GraphNode & { x?: number; y?: number }
  if (n.x === undefined || n.y === undefined) return

  const size = getHierarchySize(n.type)
  const fontSize = Math.max(12 / globalScale, 3)
  const isHovered = context.hoveredNodeId === n.id
  const isHighlighted = context.highlightNodeId === n.id
  const isCenterNode = n.id === context.centerNodeId
  const isSunNode = isCenterNode && n.type === '헌법'

  const logoImg = getLogoImage(n.type)

  // 하이라이트 링 (선택된 노드)
  if (isHighlighted && !isSunNode) {
    ctx.beginPath()
    ctx.arc(n.x, n.y, size + 6, 0, 2 * Math.PI)
    ctx.strokeStyle = '#fbbf24'
    ctx.lineWidth = 2 / globalScale
    ctx.setLineDash([4 / globalScale, 3 / globalScale])
    ctx.stroke()
    ctx.setLineDash([])
  }

  // 중심 노드(태양) 특별 효과
  if (isSunNode) {
    const gradient = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, size * 2)
    gradient.addColorStop(0, 'rgba(255, 107, 53, 0.8)')
    gradient.addColorStop(0.5, 'rgba(255, 165, 0, 0.4)')
    gradient.addColorStop(1, 'rgba(255, 200, 0, 0)')
    ctx.beginPath()
    ctx.arc(n.x, n.y, size * 2, 0, 2 * Math.PI)
    ctx.fillStyle = gradient
    ctx.fill()

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
      ctx.drawImage(logoImg, n.x - imgSize / 2, n.y - imgSize / 2, imgSize, imgSize)
    } else {
      ctx.beginPath()
      ctx.arc(n.x, n.y, size * 0.7, 0, 2 * Math.PI)
      ctx.fillStyle = getNodeColor(n.type)
      ctx.fill()
    }
  }

  // 라벨 (호버된 노드는 onRenderFramePost에서 최상단에 그림)
  if (n.name && !isHovered) {
    const displayText = getDisplayName(n)
    ctx.font = `${fontSize}px Sans-Serif`
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'

    const textWidth = ctx.measureText(displayText).width
    ctx.fillStyle = 'rgba(255, 255, 255, 0.9)'
    ctx.fillRect(n.x - textWidth / 2 - 2, n.y + size + 4, textWidth + 4, fontSize + 4)
    ctx.strokeStyle = 'rgba(209, 213, 219, 0.6)'
    ctx.lineWidth = 0.5
    ctx.strokeRect(n.x - textWidth / 2 - 2, n.y + size + 4, textWidth + 4, fontSize + 4)

    ctx.fillStyle = '#374151'
    ctx.fillText(displayText, n.x, n.y + size + fontSize / 2 + 6)
  }
}

const ORBIT_RADII = [120, 220, 300, 380]
const ORBIT_COLORS = [
  'rgba(251, 191, 36, 0.15)',
  'rgba(96, 165, 250, 0.15)',
  'rgba(52, 211, 153, 0.12)',
  'rgba(156, 163, 175, 0.1)',
]

// 궤도 원 그리기 (황도 십이궁 스타일)
export function renderOrbits(ctx: CanvasRenderingContext2D): void {
  ORBIT_RADII.forEach((radius, i) => {
    ctx.beginPath()
    ctx.arc(0, 0, radius, 0, 2 * Math.PI)
    ctx.strokeStyle = ORBIT_COLORS[i]
    ctx.lineWidth = 2
    ctx.setLineDash([5, 5])
    ctx.stroke()
    ctx.setLineDash([])
  })
}

// 호버된 노드의 라벨을 최상단에 렌더링
export function renderHoveredLabel(
  ctx: CanvasRenderingContext2D,
  globalScale: number,
  hoveredNode: GraphNode | null,
): void {
  if (!hoveredNode) return
  const n = hoveredNode as GraphNode & { x?: number; y?: number }
  if (n.x === undefined || n.y === undefined) return

  const size = getHierarchySize(n.type)
  const fontSize = Math.max(12 / globalScale, 3)
  const displayText = n.name

  ctx.font = `${fontSize}px Sans-Serif`
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'

  const textWidth = ctx.measureText(displayText).width
  ctx.fillStyle = 'rgba(255, 255, 255, 0.95)'
  ctx.fillRect(n.x - textWidth / 2 - 4, n.y + size + 4, textWidth + 8, fontSize + 6)
  ctx.strokeStyle = 'rgba(156, 163, 175, 0.8)'
  ctx.lineWidth = 1
  ctx.strokeRect(n.x - textWidth / 2 - 4, n.y + size + 4, textWidth + 8, fontSize + 6)

  ctx.fillStyle = '#1f2937'
  ctx.fillText(displayText, n.x, n.y + size + fontSize / 2 + 7)
}
