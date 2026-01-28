/**
 * 변호사 통계 대시보드 타입 정의
 */

export interface StatusCount {
  status: string
  count: number
}

export interface OverviewResponse {
  total_lawyers: number
  status_counts: StatusCount[]
  coord_rate: number
  specialty_rate: number
}

export interface RegionStat {
  region: string
  count: number
}

export interface RegionStatResponse {
  data: RegionStat[]
}

export interface SpecialtyDetail {
  name: string
  count: number
}

export interface SpecialtyStat {
  category_id: string
  category_name: string
  count: number
  specialties: SpecialtyDetail[]
}

export interface SpecialtyStatResponse {
  data: SpecialtyStat[]
}

export interface CrossAnalysisCell {
  region: string
  category_id: string
  category_name: string
  count: number
}

export interface CrossAnalysisResponse {
  data: CrossAnalysisCell[]
  regions: string[]
  categories: string[]
}
