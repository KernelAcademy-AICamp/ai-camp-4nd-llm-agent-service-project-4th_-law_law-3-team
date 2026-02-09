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

export interface DensityStat {
  region: string
  count: number
  population: number
  density: number  // 인구 10만명당 변호사 수
  density_current?: number  // 예측 모드에서 현재 연도(2025) 기준 밀도
  change_percent?: number  // 예측 모드에서 현재 연도 대비 변화율
}

export interface DensityStatResponse {
  data: DensityStat[]
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

// 수요 통계 타입
export interface DemandStat {
  region: string
  case_count: number
  lawyer_count: number
  burden_index: number  // case_count / lawyer_count
  court_name: string    // 관할법원명
}

export interface DemandStatResponse {
  data: DemandStat[]
  year: number
  category: string
  available_years: number[]
  available_categories: string[]
}
