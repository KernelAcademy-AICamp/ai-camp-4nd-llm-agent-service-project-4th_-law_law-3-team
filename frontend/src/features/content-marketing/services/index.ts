/** 콘텐츠 마케팅 API 서비스
 * v2.0: Persona API 함수 5개 추가
 */

import { api, endpoints } from '@/lib/api'
import type {
  KeywordCollectRequest,
  KeywordCollectResponse,
  KeywordNewsRequest,
  KeywordNewsResponse,
  KeywordStreamEvent,
  LawyerPersona,
  MetadataRequest,
  PersonaAnalysisRequest,
  PersonaFeedbackRequest,
  PersonaOnboardingRequest,
  PersonaUpdateRequest,
  ScriptMetadata,
  ScriptRequest,
  ScriptStreamEvent,
  TrendDetailResponse,
  TrendRequest,
  TrendResponse,
} from '../types'

const BASE = endpoints.contentMarketing

// ── Persona API (v2.0 NEW) ──

/** Track 1: 대화 이력 기반 자동 페르소나 분석 */
export async function analyzePersona(
  request: PersonaAnalysisRequest,
): Promise<LawyerPersona> {
  const { data } = await api.post<LawyerPersona>(`${BASE}/persona/analyze`, request)
  return data
}

/** Track 2: 온보딩 결과로 페르소나 생성 */
export async function createPersonaFromOnboarding(
  request: PersonaOnboardingRequest,
): Promise<LawyerPersona> {
  const { data } = await api.post<LawyerPersona>(`${BASE}/persona/onboarding`, request)
  return data
}

/** 현재 페르소나 조회 */
export async function getCurrentPersona(): Promise<LawyerPersona | null> {
  const { data } = await api.get<LawyerPersona | null>(`${BASE}/persona/current`)
  return data
}

/** 페르소나 부분 수정 */
export async function updatePersona(
  request: PersonaUpdateRequest,
): Promise<LawyerPersona> {
  const { data } = await api.put<LawyerPersona>(`${BASE}/persona/update`, request)
  return data
}

/** 대본 생성 후 피드백 저장 */
export async function submitPersonaFeedback(
  request: PersonaFeedbackRequest,
): Promise<void> {
  await api.post(`${BASE}/persona/feedback`, request)
}

// ── Trend API ──

/** 트렌드 수집 + 스코어링 (다수 LLM + RAG 호출로 시간 소요) */
export async function fetchTrends(request: TrendRequest): Promise<TrendResponse> {
  const { data } = await api.post<TrendResponse>(`${BASE}/trends`, request, {
    timeout: 180000,
  })
  return data
}

/** 트렌드 상세 조회 */
export async function fetchTrendDetail(trendId: string): Promise<TrendDetailResponse> {
  const { data } = await api.get<TrendDetailResponse>(`${BASE}/trends/${trendId}`)
  return data
}

// ── Script API ──

/** 대본 SSE 스트리밍 생성 */
export function streamScript(
  request: ScriptRequest,
  onEvent: (event: ScriptStreamEvent) => void,
  onError: (error: string) => void,
  onDone: () => void,
): AbortController {
  const controller = new AbortController()

  fetch(`/api${BASE}/script/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
    signal: controller.signal,
  })
    .then(async (response) => {
      if (!response.ok) {
        onError(`서버 오류: ${response.status}`)
        return
      }

      const reader = response.body?.getReader()
      if (!reader) {
        onError('스트리밍을 시작할 수 없습니다.')
        return
      }

      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const event = JSON.parse(line.slice(6)) as ScriptStreamEvent
              if (event.event === 'done') {
                onDone()
              } else if (event.event === 'error') {
                onError(event.error || '알 수 없는 오류')
              } else {
                onEvent(event)
              }
            } catch {
              // JSON 파싱 실패 무시
            }
          }
        }
      }
    })
    .catch((error) => {
      if (error instanceof Error && error.name !== 'AbortError') {
        onError(error.message)
      }
    })

  return controller
}

/** 메타데이터 재생성 */
export async function regenerateMetadata(request: MetadataRequest): Promise<ScriptMetadata> {
  const { data } = await api.post<ScriptMetadata>(`${BASE}/script/metadata`, request)
  return data
}

// ── Keyword Flow API (v2.1 NEW) ──

/** 커뮤니티 트렌드 키워드 수집 + 4차원 스코어링 */
export async function collectKeywords(
  request: KeywordCollectRequest = {},
): Promise<KeywordCollectResponse> {
  const { data } = await api.post<KeywordCollectResponse>(
    `${BASE}/keywords/collect`,
    request,
    { timeout: 60000 },
  )
  return data
}

/** 키워드 수집 SSE 스트리밍 (§7.4) */
export function streamKeywordCollect(
  onEvent: (event: KeywordStreamEvent) => void,
  onError: (error: string) => void,
  maxKeywords: number = 10,
): AbortController {
  const controller = new AbortController()
  const url = `/api${BASE}/keywords/collect/stream?max_keywords=${maxKeywords}`

  fetch(url, {
    method: 'GET',
    headers: { Accept: 'text/event-stream' },
    signal: controller.signal,
  })
    .then(async (response) => {
      if (!response.ok) {
        onError(`서버 오류: ${response.status}`)
        return
      }

      const reader = response.body?.getReader()
      if (!reader) {
        onError('스트리밍을 시작할 수 없습니다.')
        return
      }

      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const event = JSON.parse(line.slice(6)) as KeywordStreamEvent
              onEvent(event)
            } catch {
              // JSON 파싱 실패 무시
            }
          }
        }
      }
    })
    .catch((error) => {
      if (error instanceof Error && error.name !== 'AbortError') {
        onError(error.message)
      }
    })

  return controller
}

/** 선택한 키워드로 뉴스 기사 검색 */
export async function searchKeywordNews(
  keywordId: string,
  request: KeywordNewsRequest = {},
): Promise<KeywordNewsResponse> {
  const { data } = await api.post<KeywordNewsResponse>(
    `${BASE}/keywords/${keywordId}/news`,
    request,
    { timeout: 30000 },
  )
  return data
}
