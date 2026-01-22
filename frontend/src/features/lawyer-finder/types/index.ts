/**
 * 변호사 찾기 모듈 타입 정의
 */

export interface Lawyer {
  id: number
  name: string
  status: string
  photo_url: string | null
  detail_id: string | null
  birth_year: number | null
  office_name: string | null
  address: string | null
  phone: string | null
  latitude: number | null
  longitude: number | null
  distance?: number // 검색 시 거리 (km)
  specialties?: string[] // 전문분야 목록
}

export interface NearbySearchResponse {
  lawyers: Lawyer[]
  total_count: number
  center: {
    lat: number
    lng: number
  }
  radius: number
}

export interface SearchResponse {
  lawyers: Lawyer[]
  total_count: number
}

export interface ClusterData {
  lat: number
  lng: number
  count: number
  lawyers: Lawyer[]
}

export interface ClusterResponse {
  clusters: ClusterData[]
  zoom_level: number
  total_count: number
}

export interface StatsResponse {
  total_lawyers: number
  with_coordinates: number
  without_coordinates: number
  with_specialties: number
  source: string | null
  crawled_at: string | null
  geocoded_at: string | null
  specialty_crawl_date: string | null
}

export interface SpecialtiesResponse {
  specialties: string[]
  count: number
}

export interface Category {
  id: string
  name: string
  icon: string
  description: string
}

export interface CategoriesResponse {
  categories: Category[]
  count: number
}

// 사무소 정보 (지도 팝업용)
export interface Office {
  name: string
  address: string | null
  lat: number
  lng: number
  lawyers: Lawyer[]
}
