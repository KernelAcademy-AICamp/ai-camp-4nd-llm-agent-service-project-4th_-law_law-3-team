import type { GraphNode } from '../services'
import { getLawTypeLogo, DEFAULT_GOV_LOGO } from '../utils/lawTypeLogo'

// 법령 유형별 색상
export const TYPE_COLORS: Record<string, string> = {
  '헌법': '#ff6b35',
  '법률': '#fbbf24',
  '대통령령': '#60a5fa',
  '총리령': '#34d399',
  '대법원규칙': '#a78bfa',
  '헌법재판소규칙': '#f472b6',
}

export function getNodeColor(type: string): string {
  if (type in TYPE_COLORS) return TYPE_COLORS[type]
  if (type.endsWith('부령')) return '#fb923c'
  return '#9ca3af'
}

// 법령 계급별 크기 (헌법은 태양처럼 크게)
export function getHierarchySize(type: string): number {
  if (type === '헌법') return 25
  if (type === '법률') return 12
  if (type === '대통령령') return 9
  if (type === '총리령') return 7
  if (type.endsWith('부령')) return 6
  if (type.includes('규칙')) return 5
  return 5
}

// 법령 유형별 방사형 반경 (헌법은 중심, 나머지는 동심원)
export function getRadialRadius(type: string): number {
  if (type === '헌법') return 0
  if (type === '법률') return 120
  if (type === '대통령령') return 220
  if (type === '총리령') return 300
  if (type.endsWith('부령')) return 300
  if (type.includes('규칙')) return 380
  return 380
}

// 법령 유형 → 필터 카테고리 매핑
export function getFilterCategory(type: string): string {
  if (type === '헌법') return '헌법'
  if (type === '법률') return '법률'
  if (type === '대통령령') return '대통령령'
  if (type === '총리령' || type.endsWith('부령')) return '총리령·부령'
  if (type.includes('규칙')) return '규칙'
  return '규칙'
}

// 헌법 노드 (태양)
export const CONSTITUTION_NODE: GraphNode = {
  id: '001444',
  name: '대한민국헌법',
  type: '헌법',
  abbreviation: '헌법',
  citation_count: 0,
}

// 표시용 이름 (약어 또는 8자 제한)
export function getDisplayName(node: GraphNode): string {
  if (node.abbreviation) return node.abbreviation
  if (node.name.length > 8) return node.name.slice(0, 8) + '…'
  return node.name
}

// 이미지 캐시
const imageCache = new Map<string, HTMLImageElement>()

// 로고 이미지 로드 (캐싱)
export function getLogoImage(type: string): HTMLImageElement | null {
  const logoPath = getLawTypeLogo(type) || DEFAULT_GOV_LOGO

  if (imageCache.has(logoPath)) {
    return imageCache.get(logoPath) || null
  }

  const img = new Image()
  img.src = logoPath
  img.onload = () => {
    imageCache.set(logoPath, img)
  }
  img.onerror = () => {
    const defaultImg = new Image()
    defaultImg.src = DEFAULT_GOV_LOGO
    defaultImg.onload = () => {
      imageCache.set(logoPath, defaultImg)
    }
  }

  imageCache.set(logoPath, img)
  return null
}
