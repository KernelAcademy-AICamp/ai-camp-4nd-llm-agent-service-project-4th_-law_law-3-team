/**
 * 법령 계층도 관련 타입 정의
 */

export interface StatuteNode {
  id: string
  name: string
  type: string
  citation_count: number
  abbreviation?: string
}

export interface StatuteSearchResponse {
  results: StatuteNode[]
  total: number
}

export interface StatuteHierarchyResponse {
  statute: StatuteNode
  parents: StatuteNode[]
  children: StatuteNode[]
}

export interface StatuteChildrenResponse {
  statute: StatuteNode
  children: StatuteNode[]
  total: number
}
