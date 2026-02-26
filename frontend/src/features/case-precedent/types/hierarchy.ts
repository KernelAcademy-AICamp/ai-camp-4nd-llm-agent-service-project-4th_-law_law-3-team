/**
 * 법령 계층도 관련 타입 정의
 */

export interface StatuteNode {
  id: string
  name: string
  type: string
  citation_count: number
  abbreviation?: string
  content?: string | null
  supplementary?: string | null
}

export interface StatuteSearchResponse {
  query: string
  results: StatuteNode[]
}

export interface StatuteHierarchyResponse {
  root: StatuteNode | null
  upper: StatuteNode[]
  lower: StatuteNode[]
  related: StatuteNode[]
}

export interface StatuteChildrenResponse {
  statute_id: string
  children: StatuteNode[]
}
