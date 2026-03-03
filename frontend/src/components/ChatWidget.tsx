'use client'

import { useState, useRef, useEffect, useCallback, useMemo, memo } from 'react'
import { usePathname, useRouter, useSearchParams } from 'next/navigation'
import { useUI } from '@/context/UIContext'
import { useChat } from '@/context/ChatContext'
import { useStreamingChat } from '@/hooks/useStreamingChat'
import { useSmallClaimsSync } from '@/hooks/useSmallClaimsSync'
import { api } from '@/lib/api'
import axios from 'axios'
import ReactMarkdown from 'react-markdown'
import ChatActions, { ChatAction } from './ChatActions'
import type { ChatSource } from '@/features/case-precedent/types'

// 판례번호 패턴: 2023다12345, 88도820, 99가합1234 등
const CASE_NUMBER_PATTERN = /(\d{2,4}[가-힣]{1,3}\d{1,6})/g

// 텍스트에서 판례번호를 클릭 가능한 버튼으로 변환하는 컴포넌트
function CaseNumberLink({
  text,
  onCaseClick,
  isLightTheme
}: {
  text: string
  onCaseClick: (caseNumber: string) => void
  isLightTheme: boolean
}) {
  const parts = text.split(CASE_NUMBER_PATTERN)

  return (
    <>
      {parts.map((part, index) => {
        if (CASE_NUMBER_PATTERN.test(part)) {
          // Reset the regex lastIndex
          CASE_NUMBER_PATTERN.lastIndex = 0
          return (
            <button
              key={index}
              onClick={(e) => {
                e.preventDefault()
                e.stopPropagation()
                onCaseClick(part)
              }}
              className={`inline px-1 py-0.5 mx-0.5 rounded text-sm font-mono font-bold transition-all hover:scale-105 ${
                isLightTheme
                  ? 'bg-blue-100 text-blue-700 hover:bg-blue-200'
                  : 'bg-blue-500/30 text-blue-300 hover:bg-blue-500/50'
              }`}
              title={`${part} 판례 보기`}
            >
              {part}
            </button>
          )
        }
        return <span key={index}>{part}</span>
      })}
    </>
  )
}

// Memoized markdown components factory to prevent recreation on every render
function useMarkdownComponents(
  onCaseClick: (caseNumber: string) => void,
  isLightTheme: boolean
) {
  return useMemo(() => ({
    p: ({ children }: { children?: React.ReactNode }) => (
      <p>
        {typeof children === 'string' ? (
          <CaseNumberLink
            text={children}
            onCaseClick={onCaseClick}
            isLightTheme={isLightTheme}
          />
        ) : Array.isArray(children) ? (
          children.map((child, i) =>
            typeof child === 'string' ? (
              <CaseNumberLink
                key={i}
                text={child}
                onCaseClick={onCaseClick}
                isLightTheme={isLightTheme}
              />
            ) : (
              <span key={i}>{child}</span>
            )
          )
        ) : (
          children
        )}
      </p>
    ),
    li: ({ children }: { children?: React.ReactNode }) => (
      <li>
        {typeof children === 'string' ? (
          <CaseNumberLink
            text={children}
            onCaseClick={onCaseClick}
            isLightTheme={isLightTheme}
          />
        ) : Array.isArray(children) ? (
          children.map((child, i) =>
            typeof child === 'string' ? (
              <CaseNumberLink
                key={i}
                text={child}
                onCaseClick={onCaseClick}
                isLightTheme={isLightTheme}
              />
            ) : (
              <span key={i}>{child}</span>
            )
          )
        ) : (
          children
        )}
      </li>
    ),
  }), [onCaseClick, isLightTheme])
}

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  actions?: ChatAction[]
  agentUsed?: string
  sources?: ChatSource[]  // 참조 자료 (카드 연결용)
}

interface MultiAgentChatResponse {
  response: string
  agent_used: string
  sources: ChatSource[]
  actions: ChatAction[]
  session_data: Record<string, unknown>
}

// pathname → agent 매핑 (페이지 진입 시 자동 선택)
const PATHNAME_AGENT_MAP: Record<string, string> = {
  '/lawyer-finder': 'lawyer_finder',
  '/storyboard': 'storyboard',
  '/lawyer-stats': 'lawyer_stats',
  '/law-study': 'law_study',
  '/small-claims': 'small_claims',
  '/statute-hierarchy': 'law_search',
  // /case-precedent는 기존 ?agent= URL 파라미터 사용
}

// agent → 이동할 페이지 매핑 (에이전트 선택 시 자동 이동)
const AGENT_PAGE_MAP: Record<string, string> = {
  'lawyer_finder': '/lawyer-finder',
  'case_search': '/case-precedent?agent=case_search',
  'law_search': '/case-precedent?agent=law_search',
  'legal_search': '/case-precedent',
  'legal_answer': '/case-precedent',
  'storyboard': '/storyboard',
  'lawyer_stats': '/lawyer-stats',
  'law_study': '/law-study',
  'small_claims': '/small-claims',
}

// agent 한글명 (헤더 표시용)
const AGENT_DISPLAY_NAMES: Record<string, string> = {
  'lawyer_finder': '변호사 찾기',
  'case_search': '판례 검색',
  'law_search': '법령 검색',
  'legal_search': '법률 검색',
  'storyboard': '스토리보드',
  'lawyer_stats': '변호사 통계',
  'law_study': '로스쿨 학습',
  'small_claims': '소액소송',
  'general': '일반 채팅',
}

// 에이전트별 초기 인사 메시지
const AGENT_GREETINGS: Record<string, string> = {
  'case_search': '안녕하세요! 판례 검색 AI입니다.\n\n**판례에 대해 질문해주세요.**\n- 관련 판례 검색\n- 법률 상담',
  'law_search': '안녕하세요! 법령 검색 AI입니다.\n\n**법령에 대해 질문해주세요.**\n- 관련 법령 조항 검색\n- 법령 해석 및 적용 사례',
  'lawyer_finder': '안녕하세요! 변호사 찾기 AI입니다.\n\n**주변 변호사를 찾아드릴게요.**\n- 위치 기반 변호사 검색\n- 전문분야별 추천',
  'storyboard': '안녕하세요! 스토리보드 AI입니다.\n\n**사건 타임라인을 정리해드릴게요.**\n- 사건 경위 정리\n- 시간순 타임라인 생성',
  'lawyer_stats': '안녕하세요! 변호사 통계 AI입니다.\n\n**변호사 통계 정보를 안내해드릴게요.**\n- 지역별 변호사 현황\n- 전문분야별 분포',
  'law_study': '안녕하세요! 법학 학습 AI입니다.\n\n**법학 공부를 도와드릴게요.**\n- 법령 학습 자료\n- 학습 가이드',
  'small_claims': '안녕하세요! 소액소송 가이드 AI입니다.\n\n**소액소송 절차를 안내해드릴게요.**\n- 내용증명 작성\n- 지급명령 신청\n- 소액심판 절차',
}

// floating 모드 기본 적용 페이지
const FLOATING_MODE_PATHS = new Set([
  '/lawyer-finder',
  '/small-claims',
  '/lawyer-stats',
  '/storyboard',
  '/statute-hierarchy',
  '/workspace',
])

// --- Memoized MessageBubble ---

type MarkdownComponentsType = ReturnType<typeof useMarkdownComponents>

interface MessageBubbleProps {
  msg: Message
  isStreamingMessage: boolean
  messageUserClass: string
  messageBotClass: string
  isLightTheme: boolean
  markdownComponents: MarkdownComponentsType
  loadingStatus: { title: string; detail: string }
  onAction: (action: string) => void
  onRequestLocation: () => void
}

const MessageBubble = memo(function MessageBubble({
  msg,
  isStreamingMessage,
  messageUserClass,
  messageBotClass,
  isLightTheme,
  markdownComponents,
  loadingStatus,
  onAction,
  onRequestLocation,
}: MessageBubbleProps) {
  return (
    <div className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[85%] p-4 rounded-2xl text-base leading-relaxed ${
          msg.role === 'user'
            ? `${messageUserClass} rounded-tr-none`
            : `${messageBotClass} rounded-tl-none`
        }`}
      >
        {msg.role === 'assistant' ? (
          isStreamingMessage && !msg.content.trim() ? (
            <div className="space-y-3 min-w-[260px]">
              <div className="flex items-start gap-3">
                <div className="mt-1.5 flex h-3 w-3">
                  <span className="relative inline-flex h-3 w-3">
                    <span className="absolute inline-flex h-full w-full rounded-full bg-blue-500 opacity-75 animate-ping" />
                    <span className="relative inline-flex h-3 w-3 rounded-full bg-blue-500" />
                  </span>
                </div>
                <div className="min-w-0">
                  <p className="text-sm font-semibold">{loadingStatus.title}</p>
                  <p className="text-xs opacity-70 mt-1">{loadingStatus.detail}</p>
                </div>
              </div>
              <div className="h-1.5 w-full rounded-full bg-blue-500/20 overflow-hidden">
                <div className="h-full w-1/3 rounded-full bg-blue-500 animate-pulse" />
              </div>
            </div>
          ) : (
            <div className={`prose prose-sm max-w-none prose-p:my-2 prose-ul:my-2 prose-ol:my-2 prose-li:my-0 prose-headings:my-2 prose-strong:text-inherit ${!isLightTheme ? 'prose-invert' : ''}`}>
              <ReactMarkdown components={markdownComponents}>
                {msg.content}
              </ReactMarkdown>
              {isStreamingMessage && (
                <span className="inline-block w-2 h-4 bg-current animate-pulse ml-0.5" />
              )}
              {msg.actions && msg.actions.length > 0 && (
                <ChatActions
                  actions={msg.actions}
                  onAction={onAction}
                  onRequestLocation={onRequestLocation}
                  isLightTheme={isLightTheme}
                />
              )}
            </div>
          )
        ) : (
          <span className="whitespace-pre-wrap">{msg.content}</span>
        )}
      </div>
    </div>
  )
}, (prevProps, nextProps) => {
  // 완료된 메시지는 스트리밍 중 재렌더 방지 (ReactMarkdown 파싱 비용 절감)
  if (prevProps.msg !== nextProps.msg) return false
  if (prevProps.isStreamingMessage !== nextProps.isStreamingMessage) return false
  if (prevProps.isLightTheme !== nextProps.isLightTheme) return false
  if (prevProps.isStreamingMessage && prevProps.loadingStatus !== nextProps.loadingStatus) return false
  return true
})

export default function ChatWidget() {
  const router = useRouter()
  const pathname = usePathname()
  const searchParams = useSearchParams()
  const agentFromUrl = searchParams.get('agent')
  const effectiveAgent = agentFromUrl || PATHNAME_AGENT_MAP[pathname] || null
  const { isChatOpen, toggleChat, setChatOpen, chatMode, setChatMode } = useUI()
  const {
    userRole,
    setUserRole,
    sessionData,
    setSessionData,
    userLocation,
    requestUserLocation,
    resetSession,
    setHighlightedCaseNumber,
    conversationId,
    setConversationId,
    caseId,
  } = useChat()

  // 소액소송 UI 동기화 훅
  const { readState, setChatDisputeType, setChatClaimAmount, setChatStep } = useSmallClaimsSync()

  // Determine if we are on pages that support floating mode
  const isMapPage = pathname === '/lawyer-finder'
  const supportsFloatingMode = FLOATING_MODE_PATHS.has(pathname) || pathname.startsWith('/workspace')

  // Global state for view mode is now handled by UIContext

  // agent 타입에 따른 초기 메시지 생성
  const getInitialMessage = useCallback((agent: string | null): Message => {
    if (agent && AGENT_GREETINGS[agent]) {
      return { id: '1', role: 'assistant', content: AGENT_GREETINGS[agent] }
    }
    return {
      id: '1',
      role: 'assistant',
      content: '안녕하세요! 저는 당신의 법률 AI 어시스턴트입니다.\n\n**무엇을 도와드릴까요?**\n- 변호사 찾기\n- 판례 검색\n- 소액소송 가이드',
    }
  }, [])

  const [messages, setMessages] = useState<Message[]>([getInitialMessage(effectiveAgent)])

  // 스트리밍 관련 상태
  const { sendStreamingMessage, isStreaming, abortStream } = useStreamingChat()
  const [streamingMessageId, setStreamingMessageId] = useState<string | null>(null)
  const [requestStartedAt, setRequestStartedAt] = useState<number | null>(null)
  const [loadingElapsedSeconds, setLoadingElapsedSeconds] = useState(0)
  const [hasReceivedFirstToken, setHasReceivedFirstToken] = useState(false)
  const hasReceivedFirstTokenRef = useRef(false)

  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)

  // 스트리밍 성능 최적화: rAF 기반 배치 업데이트
  const rafIdRef = useRef<number | null>(null)

  // 컴포넌트 언마운트 시 rAF 정리
  useEffect(() => {
    return () => {
      if (rafIdRef.current !== null) {
        cancelAnimationFrame(rafIdRef.current)
      }
    }
  }, [])

  // 페이지 변경 시 모드 설정
  const prevPathnameRef = useRef<string | null>(null)

  // 챗봇을 숨겨야 하는 페이지 (자체 채팅 UI가 있는 경우)
  const isChatHiddenPage = pathname === '/mock-trial'

  useEffect(() => {
    // 같은 페이지에서는 모드 변경 안 함 (사용자가 토글한 상태 유지)
    if (prevPathnameRef.current === pathname) return
    prevPathnameRef.current = pathname

    if (isChatHiddenPage) {
      // 자체 채팅 UI가 있는 페이지에서는 챗봇 최소화
      setChatOpen(false)
    } else if (supportsFloatingMode) {
      // floating 모드 지원 페이지 첫 진입 시 floating 모드로 시작
      setChatMode('floating')
      setChatOpen(true)
    } else {
      // 다른 페이지 진입 시 Split 모드 사용
      setChatMode('split')
    }
  }, [pathname, setChatMode, setChatOpen, supportsFloatingMode, isChatHiddenPage])

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages, isChatOpen, chatMode, isLoading])

  useEffect(() => {
    if (!(isLoading || isStreaming) || requestStartedAt === null) {
      setLoadingElapsedSeconds(0)
      return
    }

    const tick = () => {
      setLoadingElapsedSeconds(Math.floor((Date.now() - requestStartedAt) / 1000))
    }

    tick()
    const intervalId = window.setInterval(tick, 1000)
    return () => window.clearInterval(intervalId)
  }, [isLoading, isStreaming, requestStartedAt])

  const loadingStatus = useMemo(() => {
    const elapsedText = `${loadingElapsedSeconds}초 경과`

    if (hasReceivedFirstToken) {
      return {
        title: '답변을 작성하고 있습니다...',
        detail: elapsedText,
      }
    }

    if (loadingElapsedSeconds < 6) {
      return {
        title: '질문을 분석하고 있습니다...',
        detail: elapsedText,
      }
    }

    if (loadingElapsedSeconds < 14) {
      return {
        title: 'AI 모델을 호출하고 있습니다...',
        detail: `${elapsedText} · 응답 준비에 시간이 걸릴 수 있습니다`,
      }
    }

    if (loadingElapsedSeconds < 25) {
      return {
        title: '서버 응답이 지연되고 있습니다...',
        detail: `${elapsedText} · 연결 재시도를 진행 중일 수 있습니다`,
      }
    }

    return {
      title: '응답을 계속 기다리는 중입니다...',
      detail: `${elapsedText} · 잠시만 더 기다려 주세요`,
    }
  }, [hasReceivedFirstToken, loadingElapsedSeconds])

  const handleResetChat = useCallback(() => {
    abortStream()
    if (rafIdRef.current !== null) {
      cancelAnimationFrame(rafIdRef.current)
      rafIdRef.current = null
    }
    setIsLoading(false)
    setStreamingMessageId(null)
    setRequestStartedAt(null)
    setLoadingElapsedSeconds(0)
    setHasReceivedFirstToken(false)
    hasReceivedFirstTokenRef.current = false
    setInput('')
    resetSession()
    setMessages([getInitialMessage(effectiveAgent)])
  }, [abortStream, effectiveAgent, getInitialMessage, resetSession])

  const handleSend = async (overrideMessage?: string, overrideLocation?: { latitude: number; longitude: number } | null) => {
    const messageToSend = overrideMessage || input
    if (!messageToSend.trim() || isLoading || isStreaming) return

    const userMsg: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: messageToSend,
    }
    setMessages((prev) => [...prev, userMsg])

    if (!overrideMessage) {
      setInput('')
    }
    setIsLoading(true)
    setRequestStartedAt(Date.now())
    setLoadingElapsedSeconds(0)
    setHasReceivedFirstToken(false)
    hasReceivedFirstTokenRef.current = false

    // 대화 기록 준비 (최근 10개)
    const history = messages.slice(-10).map((msg) => ({
      role: msg.role,
      content: msg.content,
    }))

    // 위치 정보: override > context
    const locationToSend = overrideLocation !== undefined ? overrideLocation : userLocation

    // 스트리밍 메시지 ID 생성
    const streamingMsgId = (Date.now() + 1).toString()
    setStreamingMessageId(streamingMsgId)

    // 빈 어시스턴트 메시지 추가 (스트리밍 응답용)
    const initialAssistantMsg: Message = {
      id: streamingMsgId,
      role: 'assistant',
      content: '',
    }
    setMessages((prev) => [...prev, initialAssistantMsg])

    // 스트리밍 결과 저장용
    let accumulatedContent = ''
    let receivedSources: ChatSource[] = []
    let receivedActions: ChatAction[] = []
    let receivedSessionData: Record<string, unknown> = {}
    let agentUsed = ''

    // 소액소송인 경우 UI 상태를 session_data에 병합
    const isSmallClaims = effectiveAgent === 'small_claims' || sessionData.active_agent === 'small_claims'
    let finalSessionData = { ...sessionData }

    if (isSmallClaims) {
      const wizardState = readState()
      if (wizardState) {
        finalSessionData = {
          ...finalSessionData,
          wizard_state: {
            dispute_type: wizardState.disputeType,
            current_step: wizardState.currentStep,
            case_info: wizardState.caseInfo,
            checked_evidence: wizardState.checkedEvidence
          }
        }
      }
    }

    // Helper to clean AI response for case detail view
    const cleanAIResponse = (text: string) => {
      let cleaned = text
        .replace(/^(안녕하세요|반갑습니다).*?(\n|$)/g, '')
        .replace(/^.*?AI.*?입니다.*?(\n|$)/g, '')
        .replace(/^무엇을 도와드릴까요.*?(\n|$)/g, '')
        .trim()
      return cleaned
    }

    try {
      await sendStreamingMessage(
        {
          message: messageToSend,
          user_role: userRole,
          history: history,
          session_data: finalSessionData,
          user_location: locationToSend,
          agent: effectiveAgent || undefined,
          conversation_id: conversationId || undefined,
          case_id: caseId || undefined,
        },
        {
          onToken: (content) => {
            if (content && !hasReceivedFirstTokenRef.current) {
              hasReceivedFirstTokenRef.current = true
              setHasReceivedFirstToken(true)
            }
            accumulatedContent += content
            // rAF 기반 배치 업데이트: 토큰마다 setState 대신 프레임당 1회만
            if (rafIdRef.current === null) {
              rafIdRef.current = requestAnimationFrame(() => {
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === streamingMsgId
                      ? { ...msg, content: accumulatedContent }
                      : msg
                  )
                )
                rafIdRef.current = null
              })
            }
          },
          onSources: (sources) => {
            receivedSources = sources
            // sources가 오면 메시지에 추가
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === streamingMsgId
                  ? { ...msg, sources: sources }
                  : msg
              )
            )
          },
          onMetadata: (metadata) => {
            agentUsed = metadata.agent_used
            receivedActions = metadata.actions
            receivedSessionData = metadata.session_data
            
            // 소액소송 에이전트 응답 시 UI 동기화
            if (metadata.agent_used === 'small_claims' && metadata.session_data) {
              const sessionDataTyped = metadata.session_data as Record<string, unknown>
              // 분쟁 유형 동기화
              if (sessionDataTyped.dispute_type) {
                setChatDisputeType(sessionDataTyped.dispute_type as string)
              }
              // 청구 금액 동기화
              if (sessionDataTyped.claim_amount) {
                setChatClaimAmount(sessionDataTyped.claim_amount as number)
              }
              // 현재 단계 동기화
              if (sessionDataTyped.step) {
                setChatStep(sessionDataTyped.step as string)
              }
            }
            
            // 메타데이터 업데이트
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === streamingMsgId
                  ? { ...msg, agentUsed: metadata.agent_used, actions: metadata.actions }
                  : msg
              )
            )
          },
          onDone: (doneData) => {
            // Flush pending rAF update
            if (rafIdRef.current !== null) {
              cancelAnimationFrame(rafIdRef.current)
              rafIdRef.current = null
            }
            // 마지막 토큰까지 반영
            if (accumulatedContent) {
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === streamingMsgId
                    ? { ...msg, content: accumulatedContent }
                    : msg
                )
              )
            }
            // 스트리밍 완료
            setStreamingMessageId(null)
            setIsLoading(false)
            setRequestStartedAt(null)
            setLoadingElapsedSeconds(0)
            setHasReceivedFirstToken(false)
            hasReceivedFirstTokenRef.current = false

            // conversation_id 추적 (이어가기 지원)
            if (doneData?.conversation_id) {
              setConversationId(doneData.conversation_id as string)
            }

            // 세션 데이터 업데이트 (metadata + done 이벤트 병합)
            const newSessionData = { ...receivedSessionData }
            if (doneData?.thread_id) newSessionData.thread_id = doneData.thread_id
            if (doneData?.session_secret) newSessionData.session_secret = doneData.session_secret
            if (doneData?.active_agent) newSessionData.active_agent = doneData.active_agent

            // 판례/법령 검색 결과 → aiCase 데이터 구성
            const searchAgents = ['legal_search', 'case_search', 'legal_answer', 'law_search']
            if (receivedSources?.length > 0 && searchAgents.includes(agentUsed)) {
              const mainSource = receivedSources[0]

              const aiCase = {
                id: 'ai-generated-' + Date.now(),
                case_name: mainSource?.case_name || '',
                case_number: mainSource?.case_number || '',
                doc_type: mainSource?.doc_type || 'precedent',
                content: mainSource?.content || cleanAIResponse(accumulatedContent),
                summary: mainSource?.summary || '',
                court: mainSource?.court_name || '',
                court_name: mainSource?.court_name || '',
                date: mainSource?.decision_date || '',
                decision_date: mainSource?.decision_date || '',
                reasoning: mainSource?.reasoning || '',
                ruling: mainSource?.ruling || '',
                claim: mainSource?.claim || '',
                full_reason: mainSource?.full_reason || '',
                full_text: mainSource?.full_text || '',
                reference_provisions: mainSource?.reference_provisions || '',
                reference_cases: mainSource?.reference_cases || '',
              }

              newSessionData.aiGeneratedCase = aiCase
              // 중복 제거: case_number(판례) 또는 law_name(법령) 기준
              const seen = new Set<string>()
              const uniqueSources = (receivedSources || []).filter((ref) => {
                const key = ref.doc_type === 'law' ? ref.law_name : ref.case_number
                if (!key || seen.has(key)) return false
                seen.add(key)
                return true
              })
              newSessionData.aiReferences = uniqueSources
            }

            // 세션 데이터 업데이트
            if (Object.keys(newSessionData).length > 0) {
              setSessionData({ ...sessionData, ...newSessionData })
            } else if (receivedSessionData) {
              setSessionData(receivedSessionData)
            }

            // NAVIGATE 액션 처리 (좌표/파라미터 포함 → 우선 적용)
            let hasNavigated = false
            const navigateAction = receivedActions?.find(
              (action) => action.type === 'navigate' && action.url
            )

            if (navigateAction && navigateAction.url) {
              const params = navigateAction.params as Record<string, string | number | boolean> | undefined
              let fullUrl = navigateAction.url
              if (params && Object.keys(params).length > 0) {
                const urlSearchParams = new URLSearchParams()
                Object.entries(params).forEach(([key, value]) => {
                  if (value !== undefined && value !== null) {
                    urlSearchParams.set(key, String(value))
                  }
                })
                fullUrl = `${navigateAction.url}?${urlSearchParams.toString()}`
              }
              router.push(fullUrl)
              hasNavigated = true
            }

            // NAVIGATE 액션이 없으면 에이전트 → 페이지 매핑으로 이동
            // 단, 현재 페이지가 이미 에이전트 매핑된 페이지라면 리다이렉트하지 않음
            // (예: /statute-hierarchy에서 법령 검색 시 /case-precedent로 이동 방지)
            const currentPageAgent = PATHNAME_AGENT_MAP[pathname]
            if (!hasNavigated && agentUsed && AGENT_PAGE_MAP[agentUsed] && !currentPageAgent) {
              const targetPage = AGENT_PAGE_MAP[agentUsed]
              const targetPathname = targetPage.split('?')[0]
              if (pathname !== targetPathname) {
                router.push(targetPage)
              }
            }
          },
          onError: (errorMessage) => {
            console.error('Streaming error:', errorMessage)
            if (rafIdRef.current !== null) {
              cancelAnimationFrame(rafIdRef.current)
              rafIdRef.current = null
            }
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === streamingMsgId
                  ? { ...msg, content: '죄송합니다. 응답을 생성하는 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.' }
                  : msg
              )
            )
            setStreamingMessageId(null)
            setIsLoading(false)
            setRequestStartedAt(null)
            setLoadingElapsedSeconds(0)
            setHasReceivedFirstToken(false)
            hasReceivedFirstTokenRef.current = false
          },
        }
      )
    } catch (error) {
      console.error('Chat API error:', error)
      if (rafIdRef.current !== null) {
        cancelAnimationFrame(rafIdRef.current)
        rafIdRef.current = null
      }

      let errorContent: string

      if (axios.isAxiosError(error)) {
        const status = error.response?.status
        const isNetworkError =
          error.code === 'ECONNREFUSED' ||
          error.code === 'ERR_NETWORK' ||
          status === 502 ||
          (status === 500 && !error.response?.data)

        if (isNetworkError) {
          errorContent = '서버에 연결할 수 없습니다. 서버가 시작 중일 수 있으니 잠시 후 다시 시도해주세요.'
        } else if (error.code === 'ECONNABORTED') {
          errorContent = '응답 시간이 초과되었습니다. 다시 시도해주세요.'
        } else if (status === 503) {
          errorContent = '서비스가 준비 중입니다. 잠시 후 다시 시도해주세요.'
        } else {
          errorContent = '죄송합니다. 응답을 생성하는 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.'
        }
      } else {
        errorContent = '죄송합니다. 알 수 없는 오류가 발생했습니다.'
      }

      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === streamingMsgId
            ? { ...msg, content: errorContent }
            : msg
        )
      )
      setStreamingMessageId(null)
      setIsLoading(false)
      setRequestStartedAt(null)
      setLoadingElapsedSeconds(0)
      setHasReceivedFirstToken(false)
      hasReceivedFirstTokenRef.current = false
    }
  }

  const handleAction = async (action: string) => {
    // 액션에 따른 처리
    switch (action) {
      case 'reset_search':
      case 'reset_session':
        handleResetChat()
        break

      case 'expand_search':
        // 범위 넓혀 검색 - 메시지로 전달
        handleSend('더 넓은 범위에서 변호사를 검색해주세요')
        break

      case 'dispute_type_goods':
        handleSend('물품 대금 미지급 관련 소액소송을 진행하고 싶습니다')
        break

      case 'dispute_type_fraud':
        handleSend('중고거래 사기 관련 소액소송을 진행하고 싶습니다')
        break

      case 'dispute_type_deposit':
        handleSend('임대차 보증금 관련 소액소송을 진행하고 싶습니다')
        break

      case 'draft_demand_letter':
        handleSend('내용증명 작성을 도와주세요')
        break

      case 'skip_to_court':
        handleSend('바로 소송 절차를 진행하고 싶습니다')
        break

      case 'draft_complaint':
        handleSend('소장 작성을 도와주세요')
        break

      default:
        // 기타 액션은 메시지로 전달
        handleSend(action)
    }
  }

  const handleRequestLocation = async () => {
    // 로딩 메시지 표시
    setMessages((prev) => [
      ...prev,
      {
        id: Date.now().toString(),
        role: 'assistant',
        content: '📍 현재 위치를 확인하고 있습니다...',
      },
    ])

    const location = await requestUserLocation()

    if (location) {
      // 위치 획득 성공 - 위치를 직접 전달하여 변호사 검색
      handleSend('현재 위치 주변 변호사를 찾아주세요', location)
    } else {
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now().toString(),
          role: 'assistant',
          content:
            '위치 정보를 가져올 수 없습니다. 브라우저 설정에서 위치 권한을 허용하거나, 특정 지역명을 입력해주세요.',
        },
      ])
    }
  }

  // Toggle view mode manually
  const toggleViewMode = () => {
    setChatMode(chatMode === 'split' ? 'floating' : 'split')
  }

  // Theme: always light (Apple HIG white theme)
  const isLightTheme = true

  // Memoized markdown components for ReactMarkdown
  const markdownComponents = useMarkdownComponents(setHighlightedCaseNumber, isLightTheme)

  // Apple HIG white theme
  const themeClasses = {
    container:
      'bg-white/95 backdrop-blur-xl border-l border-black/[0.06] shadow-2xl text-[#1D1D1F]',
    header: 'bg-white border-b border-black/[0.06]',
    headerTitle: 'text-[#1D1D1F]',
    headerSubtitle: 'text-[#007AFF]',
    messageUser: 'bg-[#007AFF] text-white shadow-sm',
    messageBot: 'bg-[#F5F5F7] text-[#1D1D1F] border border-black/[0.04]',
    inputArea: 'bg-white border-t border-black/[0.06]',
    input:
      'bg-[#F5F5F7] border-black/[0.06] text-[#1D1D1F] placeholder-[#86868B] focus:border-[#007AFF] focus:bg-white',
    closeBtn: 'text-[#86868B] hover:bg-[#F5F5F7] hover:text-[#1D1D1F]',
    roleSelector: 'bg-[#F5F5F7] border-black/[0.06]',
    roleActive: 'bg-[#007AFF] text-white',
    roleInactive: 'text-[#86868B] hover:bg-black/[0.04]',
  }

  // Layout classes based on viewMode
  const layoutClasses =
    chatMode === 'split'
      ? 'fixed top-0 right-0 w-1/2 h-screen z-50 flex flex-col animate-in slide-in-from-right duration-500'
      : 'fixed bottom-6 right-6 w-[380px] h-[600px] z-50 rounded-2xl flex flex-col animate-in slide-in-from-bottom zoom-in duration-300'

  // Floating Button (Collapsed) - 자체 채팅 UI가 있는 페이지에서는 버튼도 숨김
  if (!isChatOpen) {
    if (isChatHiddenPage) return null
    return (
      <button
        onClick={toggleChat}
        className="fixed bottom-6 right-6 w-14 h-14 bg-[#007AFF] hover:bg-[#0056CC] text-white rounded-full shadow-apple-hover flex items-center justify-center z-50 transition-all duration-200 hover:scale-105 active:scale-95 cursor-pointer"
      >
        <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
          />
        </svg>
      </button>
    )
  }

  // Expanded Chat Interface
  return (
    <div className={`${layoutClasses} ${themeClasses.container}`}>
      {/* Header */}
      <div
        className={`p-4 md:p-6 flex justify-between items-center ${themeClasses.header} ${chatMode === 'floating' ? 'rounded-t-2xl' : ''}`}
      >
        <div className="flex items-center gap-4">
          <div className="w-10 h-10 rounded-full bg-blue-500 flex items-center justify-center shadow-lg">
            <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
              />
            </svg>
          </div>
          <div>
            <h3 className={`font-bold text-lg ${themeClasses.headerTitle}`}>AI 법률 상담</h3>
            <p className={`text-xs font-bold uppercase tracking-widest ${themeClasses.headerSubtitle}`}>
              {sessionData.active_agent
                ? AGENT_DISPLAY_NAMES[sessionData.active_agent as string] || 'Active Now'
                : 'Active Now'}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {/* Role Selector */}
          <div
            className={`flex rounded-lg border text-xs ${themeClasses.roleSelector}`}
          >
            <button
              onClick={() => setUserRole('user')}
              className={`px-2 py-1 rounded-l-md transition-colors ${
                userRole === 'user'
                  ? themeClasses.roleActive
                  : themeClasses.roleInactive
              }`}
            >
              일반인
            </button>
            <button
              onClick={() => setUserRole('lawyer')}
              className={`px-2 py-1 rounded-r-md transition-colors ${
                userRole === 'lawyer'
                  ? themeClasses.roleActive
                  : themeClasses.roleInactive
              }`}
            >
              변호사
            </button>
          </div>

          {/* Reset Button */}
          <button
            onClick={handleResetChat}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${themeClasses.closeBtn}`}
            title="대화 초기화"
          >
            초기화
          </button>

          {/* Toggle View Mode Button (floating mode 지원 페이지에서만) */}
          {supportsFloatingMode && (
            <button
              onClick={toggleViewMode}
              className={`p-2 rounded-lg transition-colors ${themeClasses.closeBtn}`}
              title={chatMode === 'split' ? '작게 보기' : '크게 보기'}
            >
              {chatMode === 'split' ? (
                <svg
                  className="w-5 h-5"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 14h6v6" />
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M20 10h-6V4"
                  />
                </svg>
              ) : (
                <svg
                  className="w-5 h-5"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M4 20h6v-6"
                  />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 4h-6v6" />
                </svg>
              )}
            </button>
          )}

          {/* Close Button */}
          <button
            onClick={() => setChatOpen(false)}
            className={`p-2 rounded-lg transition-colors ${themeClasses.closeBtn}`}
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>
      </div>

      {/* Messages */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto p-6 space-y-6 scrollbar-thin scrollbar-thumb-gray-300/20 scrollbar-track-transparent"
      >
        {messages.map((msg) => (
          <MessageBubble
            key={msg.id}
            msg={msg}
            isStreamingMessage={msg.id === streamingMessageId}
            messageUserClass={themeClasses.messageUser}
            messageBotClass={themeClasses.messageBot}
            isLightTheme={isLightTheme}
            markdownComponents={markdownComponents}
            loadingStatus={loadingStatus}
            onAction={handleAction}
            onRequestLocation={handleRequestLocation}
          />
        ))}
        {/* Loading indicator (스트리밍 중이 아닐 때만 표시) */}
        {isLoading && !streamingMessageId && (
          <div className="flex justify-start">
            <div className={`max-w-[85%] p-4 rounded-2xl rounded-tl-none ${themeClasses.messageBot}`}>
              <div className="flex items-center gap-2">
                <div
                  className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"
                  style={{ animationDelay: '0ms' }}
                />
                <div
                  className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"
                  style={{ animationDelay: '150ms' }}
                />
                <div
                  className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"
                  style={{ animationDelay: '300ms' }}
                />
                <span className="ml-2 text-sm opacity-70">{loadingStatus.title}</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Input */}
      <div
        className={`p-4 md:p-6 ${themeClasses.inputArea} ${chatMode === 'floating' ? 'rounded-b-2xl' : ''}`}
      >
        <div className="relative flex items-center gap-3">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && !isLoading && !isStreaming && handleSend()}
            placeholder={isLoading || isStreaming ? loadingStatus.title : '법률 질문을 입력하세요...'}
            disabled={isLoading || isStreaming}
            className={`flex-1 rounded-xl px-4 py-3 md:px-6 md:py-4 text-sm md:text-base focus:outline-none transition-all shadow-sm ${themeClasses.input} ${isLoading || isStreaming ? 'opacity-50 cursor-not-allowed' : ''}`}
          />
          <button
            onClick={() => handleSend()}
            disabled={isLoading || isStreaming || !input.trim()}
            className={`p-3 md:p-4 bg-[#007AFF] hover:bg-[#0056CC] text-white rounded-xl transition-colors shadow-sm active:scale-95 cursor-pointer ${isLoading || isStreaming || !input.trim() ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            {isLoading || isStreaming ? (
              <svg className="w-5 h-5 md:w-6 md:h-6 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                />
              </svg>
            ) : (
              <svg
                className="w-5 h-5 md:w-6 md:h-6"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                />
              </svg>
            )}
          </button>
        </div>
        {(isLoading || isStreaming) && (
          <p className="mt-2 text-xs text-[#86868B]">
            {loadingStatus.detail}
          </p>
        )}
      </div>
    </div>
  )
}
