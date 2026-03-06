/** 법률 뉴스 API 서비스 */

import { api, endpoints } from '@/lib/api'
import type {
  NewsArticleResponse,
  NewsCategoryStats,
  NewsListResponse,
  NewsSearchRequest,
  NewsSearchResponse,
  NewsSource,
  NewsStatsDaily,
} from '../types'

const BASE = endpoints.legalNews

/** 뉴스 목록 조회 (페이지네이션, 소스/날짜 필터) */
export async function fetchNewsList(
  params: {
    source?: NewsSource | null
    published_date?: string | null
    page?: number
    page_size?: number
  },
  signal?: AbortSignal,
): Promise<NewsListResponse> {
  const { data } = await api.get<NewsListResponse>(`${BASE}/list`, {
    params: {
      source: params.source || undefined,
      published_date: params.published_date || undefined,
      page: params.page ?? 1,
      page_size: params.page_size ?? 20,
    },
    signal,
  })
  return data
}

/** 뉴스 상세 조회 (doc_id로 조회) */
export async function fetchNewsDetail(
  articleId: string,
  signal?: AbortSignal,
): Promise<NewsArticleResponse> {
  const { data } = await api.get<NewsArticleResponse>(`${BASE}/${articleId}`, { signal })
  return data
}

/** 뉴스 하이브리드 검색 (Vector + FTS + 리랭커) */
export async function searchNews(
  request: NewsSearchRequest,
  signal?: AbortSignal,
): Promise<NewsSearchResponse> {
  const { data } = await api.post<NewsSearchResponse>(`${BASE}/search`, request, { signal })
  return data
}

/** 일별 수집 통계 조회 */
export async function fetchNewsStatsDaily(
  days: number = 7,
  signal?: AbortSignal,
): Promise<NewsStatsDaily> {
  const { data } = await api.get<NewsStatsDaily>(`${BASE}/stats/daily`, {
    params: { days },
    signal,
  })
  return data
}

/** 카테고리 분포 통계 조회 (기간 필터) */
export async function fetchNewsCategoryStats(
  days?: number,
  signal?: AbortSignal,
): Promise<NewsCategoryStats> {
  const { data } = await api.get<NewsCategoryStats>(`${BASE}/stats/category`, {
    params: days ? { days } : undefined,
    signal,
  })
  return data
}

