/**
 * 변호사 통계 대시보드 API 서비스
 */

import { api, endpoints } from '@/lib/api'
import type {
  CrossAnalysisResponse,
  OverviewResponse,
  RegionStatResponse,
  SpecialtyStatResponse,
} from '../types'

export async function fetchOverview(): Promise<OverviewResponse> {
  const response = await api.get<OverviewResponse>(`${endpoints.lawyerStat}/overview`)
  return response.data
}

export async function fetchRegionStats(): Promise<RegionStatResponse> {
  const response = await api.get<RegionStatResponse>(`${endpoints.lawyerStat}/by-region`)
  return response.data
}

export async function fetchSpecialtyStats(): Promise<SpecialtyStatResponse> {
  const response = await api.get<SpecialtyStatResponse>(`${endpoints.lawyerStat}/by-specialty`)
  return response.data
}

export async function fetchCrossAnalysis(): Promise<CrossAnalysisResponse> {
  const response = await api.get<CrossAnalysisResponse>(`${endpoints.lawyerStat}/cross-analysis`)
  return response.data
}
