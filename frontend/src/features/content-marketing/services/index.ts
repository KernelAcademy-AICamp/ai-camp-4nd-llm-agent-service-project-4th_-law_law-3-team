/** 콘텐츠 마케팅 API 서비스
 * v2.0: Persona API 함수 5개 추가
 */

import { api, endpoints } from '@/lib/api'
import type {
  ChatHistoryCountResponse,
  KeywordCollectRequest,
  KeywordCollectResponse,
  KeywordNewsRequest,
  KeywordNewsResponse,
  KeywordStreamEvent,
  LawyerPersona,
  MetadataRequest,
  PersonaAnalysisRequest,
  PersonaAnalysisResponse,
  PersonaFeedbackRequest,
  PersonaOnboardingRequest,
  PersonaUpdateRequest,
  ScriptMetadata,
  ScriptRequest,
  ScriptStreamEvent,
  TrendDetailResponse,
  TrendRequest,
  TrendResponse,
  WebtoonGenerateRequest,
  WebtoonJobResponse,
  WebtoonJobStatus,
  WebtoonStreamEvent,
} from '../types'

const BASE = endpoints.contentMarketing

// ── Persona API (v2.0 NEW) ──

/** Track 1: 대화 이력 기반 자동 페르소나 분석 (v3.0: 응답 확장 + timeout) */
export async function analyzePersona(
  request: PersonaAnalysisRequest,
): Promise<PersonaAnalysisResponse> {
  const { data } = await api.post<PersonaAnalysisResponse>(
    `${BASE}/persona/analyze`,
    request,
    { timeout: 30000 },
  )
  return data
}

/** [v3.0 신규] 대화 이력 건수 조회 */
export async function fetchChatHistoryCount(): Promise<ChatHistoryCountResponse> {
  const { data } = await api.get<ChatHistoryCountResponse>(
    `${BASE}/persona/chat-history-count`,
  )
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

/** 대본 SSE 스트리밍 생성 (v2.2: 유휴 타이머로 전환 — 데이터 수신 시 리셋) */
export function streamScript(
  request: ScriptRequest,
  onEvent: (event: ScriptStreamEvent) => void,
  onError: (error: string) => void,
  onDone: () => void,
): AbortController {
  // 유휴 타임아웃: 마지막 데이터 수신 후 90초 동안 무응답이면 중단
  // (하드 타이머 대신 — 데이터가 흐르는 한 연결 유지)
  const IDLE_TIMEOUT_MS = 90_000
  const controller = new AbortController()
  let idleTimer: ReturnType<typeof setTimeout> | null = null

  function resetIdleTimer(): void {
    if (idleTimer !== null) clearTimeout(idleTimer)
    idleTimer = setTimeout(() => controller.abort(), IDLE_TIMEOUT_MS)
  }

  // 초기 연결 대기 타이머 시작
  resetIdleTimer()

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

      // 응답 시작 — 유휴 타이머 리셋
      resetIdleTimer()

      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        // 데이터 수신마다 유휴 타이머 리셋
        resetIdleTimer()

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
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          onError('대본 생성 시간이 초과되었습니다. 다시 시도해주세요.')
        } else {
          onError(error.message)
        }
      }
    })
    .finally(() => {
      if (idleTimer !== null) clearTimeout(idleTimer)
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
  timeRange: string = '48h',
  forceRefresh: boolean = false,
  category: string = 'all',
  personaId: string | null = null,
): AbortController {
  const controller = new AbortController()
  let url = `/api${BASE}/keywords/collect/stream?max_keywords=${maxKeywords}&time_range=${timeRange}&category=${category}`
  if (forceRefresh) {
    url += '&force_refresh=true'
  }
  if (personaId) {
    url += `&persona_id=${encodeURIComponent(personaId)}`
  }

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

/** 키워드/뉴스/트렌드 캐시 전체 클리어 */
export async function clearKeywordCache(): Promise<{
  status: string
  keyword_cache: number
  trend_detail_cache: number
  collector_cache: number
}> {
  const { data } = await api.delete(`${BASE}/keywords/cache`, { timeout: 10000 })
  return data
}


// ── Webtoon Storyboard API (v3.0 NEW) ──

/** 웹툰 스토리보드 Job 생성 */
export async function createWebtoonJob(
  request: WebtoonGenerateRequest,
): Promise<WebtoonJobResponse> {
  const { data } = await api.post<WebtoonJobResponse>(
    `${BASE}/script/webtoon`,
    request,
  )
  return data
}

/** 웹툰 Job 상태 폴링 */
export async function getWebtoonJobStatus(
  jobId: string,
): Promise<WebtoonJobStatus> {
  const { data } = await api.get<WebtoonJobStatus>(
    `${BASE}/script/webtoon/${jobId}`,
  )
  return data
}

/** 웹툰 SSE 스트리밍 (재연결 최대 3회) */
export function streamWebtoonProgress(
  jobId: string,
  onEvent: (event: WebtoonStreamEvent) => void,
  onError: (error: string) => void,
  onDone: () => void,
): AbortController {
  const controller = new AbortController()
  const url = `/api${BASE}/script/webtoon/${jobId}/stream`
  const MAX_RETRIES = 3
  let retryCount = 0

  function connectSSE(): void {
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
          onError('스트리밍 불가')
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
                const event = JSON.parse(line.slice(6)) as WebtoonStreamEvent
                if (event.event === 'all_done') {
                  onDone()
                } else if (event.event === 'error') {
                  onError(event.error || '알 수 없는 오류')
                } else {
                  onEvent(event)
                }
              } catch {
                /* JSON 파싱 실패 무시 */
              }
            }
          }
        }
      })
      .catch((error: unknown) => {
        if (error instanceof Error && error.name !== 'AbortError') {
          if (retryCount < MAX_RETRIES) {
            retryCount++
            setTimeout(() => connectSSE(), 2000 * retryCount)
          } else {
            onError(error.message)
          }
        }
      })
  }

  connectSSE()
  return controller
}

/** 개별 패널 재생성 */
export async function regenerateWebtoonPanel(
  jobId: string,
  panelNumber: number,
): Promise<WebtoonStreamEvent> {
  const { data } = await api.post<WebtoonStreamEvent>(
    `${BASE}/script/webtoon/${jobId}/regenerate`,
    { panel_number: panelNumber },
  )
  return data
}
