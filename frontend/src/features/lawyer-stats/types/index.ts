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

// 페이지 UI 상태 타입
export type IndicatorGroup = 'supply' | 'demand'
export type ViewMode = 'count' | 'density' | 'prediction' | 'case_count' | 'burden_index'
export type PredictionYear = 2030 | 2035 | 2040
export type DemandCategory = '민사' | '형사' | '가사' | '행정' | '소년보호' | '가정보호'

export const DEMAND_CATEGORIES: DemandCategory[] = ['민사', '형사', '가사', '행정', '소년보호', '가정보호']

export const PROVINCES = [
  '전체', '서울', '경기', '인천', '부산', '대구', '광주', '대전', '울산',
  '세종', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주',
] as const

/** 채팅 에이전트가 sessionData로 전달하는 대시보드 필터 */
export interface StatsFilter {
  viewMode?: 'count' | 'density' | 'prediction' | 'case_count' | 'burden_index'
  indicatorGroup?: 'supply' | 'demand'
  activeTab?: 'region' | 'cross'
  selectedProvince?: string | null
  predictionYear?: 2030 | 2035 | 2040
  demandCategory?: string
  demandYear?: number
}

/** 법원 단위로 그룹화된 수요 데이터 (지도 마커용) */
export interface CourtDemandMarker {
  court_name: string
  coordinates: [number, number]  // [lng, lat]
  case_count: number
  lawyer_count: number
  burden_index: number
  regions: string[]              // 관할 지역 목록
}
