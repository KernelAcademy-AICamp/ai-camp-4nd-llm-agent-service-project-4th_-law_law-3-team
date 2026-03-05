import { useState, useRef, useEffect, useCallback } from 'react'
import { usePathname, useRouter, useSearchParams } from 'next/navigation'
import { useUI } from '@/context/UIContext'
import { useChat } from '@/context/ChatContext'
import { useStreamingChat } from '@/hooks/useStreamingChat'
import { useSmallClaimsSync } from '@/hooks/useSmallClaimsSync'
import axios from 'axios'
import type { ChatAction } from '../ChatActions'
import type { ChatSource } from '@/features/case-precedent/types'

import {
  PATHNAME_AGENT_MAP,
  AGENT_PAGE_MAP,
  AGENT_GREETINGS,
  FLOATING_MODE_PATHS,
} from './constants'
import type { Message } from './constants'
import { useLoadingStatus } from './useLoadingStatus'

export function useChatMessages() {
  const router = useRouter()
  const pathname = usePathname()
  const searchParams = useSearchParams()
  const agentFromUrl = searchParams.get('agent')
  const effectiveAgent = agentFromUrl || PATHNAME_AGENT_MAP[pathname] || null
  const {
    isChatOpen,
    toggleChat,
    setChatOpen,
    chatMode,
    setChatMode,
    activePanel,
    setActivePanel,
    pendingMessage,
    setPendingMessage,
  } = useUI()
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

  const { readState, setChatDisputeType, setChatClaimAmount, setChatStep } = useSmallClaimsSync()

  const supportsFloatingMode = FLOATING_MODE_PATHS.has(pathname) || pathname.startsWith('/workspace')
  const isChatHiddenPage = pathname === '/mock-trial' || (pathname === '/' && !searchParams.get('role'))

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
  const { sendStreamingMessage, isStreaming, abortStream } = useStreamingChat()
  const [streamingMessageId, setStreamingMessageId] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  const {
    loadingStatus,
    resetLoadingState,
    startLoading,
    markFirstToken,
  } = useLoadingStatus(isLoading, isStreaming)

  const [input, setInput] = useState('')
  const [pendingNavTarget, setPendingNavTarget] = useState<string | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  const rafIdRef = useRef<number | null>(null)

  // rAF 정리
  useEffect(() => {
    return () => {
      if (rafIdRef.current !== null) {
        cancelAnimationFrame(rafIdRef.current)
      }
    }
  }, [])

  // SSE 콜백에서 설정된 네비게이션 대상을 React 라이프사이클에서 실행
  useEffect(() => {
    if (pendingNavTarget) {
      router.push(pendingNavTarget)
      setPendingNavTarget(null)
    }
  }, [pendingNavTarget, router])

  // pendingMessage 감지 시 자동 전송
  useEffect(() => {
    if (pendingMessage && !isLoading && !isStreaming) {
      handleSend(pendingMessage)
      setPendingMessage(null)
    }
  }, [pendingMessage, isLoading, isStreaming]) // eslint-disable-line react-hooks/exhaustive-deps

  // 페이지 변경 시 모드 설정
  const prevPathnameRef = useRef<string | null>(null)

  useEffect(() => {
    if (prevPathnameRef.current === pathname) return
    prevPathnameRef.current = pathname

    setChatOpen(false)
    if (isChatHiddenPage) return
    setChatMode('floating')
  }, [pathname, setChatMode, setChatOpen, supportsFloatingMode, isChatHiddenPage])

  // 자동 스크롤
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages, isChatOpen, chatMode, isLoading])

  const handleResetChat = useCallback(() => {
    abortStream()
    if (rafIdRef.current !== null) {
      cancelAnimationFrame(rafIdRef.current)
      rafIdRef.current = null
    }
    setIsLoading(false)
    setStreamingMessageId(null)
    resetLoadingState()
    setInput('')
    resetSession()
    setMessages([getInitialMessage(effectiveAgent)])
  }, [abortStream, effectiveAgent, getInitialMessage, resetSession, resetLoadingState])

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
    startLoading()

    const history = messages.slice(-10).map((msg) => ({
      role: msg.role,
      content: msg.content,
    }))

    const locationToSend = overrideLocation !== undefined ? overrideLocation : userLocation

    const streamingMsgId = (Date.now() + 1).toString()
    setStreamingMessageId(streamingMsgId)

    const initialAssistantMsg: Message = {
      id: streamingMsgId,
      role: 'assistant',
      content: '',
    }
    setMessages((prev) => [...prev, initialAssistantMsg])

    let accumulatedContent = ''
    let receivedSources: ChatSource[] = []
    let receivedActions: ChatAction[] = []
    let receivedSessionData: Record<string, unknown> = {}
    let agentUsed = ''
    let hasNavigatedOnRouting = false

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
            checked_evidence: wizardState.checkedEvidence,
          },
        }
      }
    }

    const cleanAIResponse = (text: string) => {
      return text
        .replace(/^(안녕하세요|반갑습니다).*?(\n|$)/g, '')
        .replace(/^.*?AI.*?입니다.*?(\n|$)/g, '')
        .replace(/^무엇을 도와드릴까요.*?(\n|$)/g, '')
        .trim()
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
            if (content) {
              markFirstToken()
            }
            accumulatedContent += content
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
          onRouting: (data) => {
            if (data.selected_agent) {
              agentUsed = data.selected_agent
              setSessionData({ ...sessionData, active_agent: data.selected_agent })

              // 라우팅 즉시 페이지 전환 (React 라이프사이클에서 실행)
              const targetPage = AGENT_PAGE_MAP[data.selected_agent]
              if (targetPage) {
                const targetPathname = targetPage.split('?')[0]
                if (pathname !== targetPathname) {
                  setPendingNavTarget(targetPage)
                  hasNavigatedOnRouting = true
                }
              }
            }
          },
          onSources: (sources) => {
            receivedSources = sources
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === streamingMsgId
                  ? { ...msg, sources: sources }
                  : msg
              )
            )

            // 판례/법령 검색 에이전트: sources 도착 즉시 왼쪽 패널 업데이트
            // (LLM 답변 스트리밍 전에 검색 결과를 먼저 표시)
            const currentAgent = agentUsed || effectiveAgent || ''
            const searchAgents = ['legal_search', 'case_search', 'legal_answer', 'law_search']
            if (sources?.length > 0 && searchAgents.includes(currentAgent)) {
              const mainSource = sources[0]
              const aiCase = {
                id: 'ai-generated-' + Date.now(),
                case_name: mainSource?.case_name || '',
                case_number: mainSource?.case_number || '',
                doc_type: mainSource?.doc_type || 'precedent',
                content: mainSource?.content || '',
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

              const seen = new Set<string>()
              const uniqueSources = sources.filter((ref) => {
                const key = ref.doc_type === 'law' ? ref.law_name : ref.case_number
                if (!key || seen.has(key)) return false
                seen.add(key)
                return true
              })

              setSessionData({
                ...sessionData,
                aiGeneratedCase: aiCase,
                aiReferences: uniqueSources,
              })
            }
          },
          onMetadata: (metadata) => {
            agentUsed = metadata.agent_used
            receivedActions = metadata.actions
            receivedSessionData = metadata.session_data

            // 에이전트 활성화 상태 즉시 반영 (답변 출력 전)
            if (metadata.agent_used) {
              setSessionData({ ...sessionData, active_agent: metadata.agent_used })
            }

            // Proactive UI Trigger: lawyer_finder 자동으로 패널 열기
            if (metadata.agent_used === 'lawyer_finder' && activePanel !== 'lawyer-finder') {
              setActivePanel('lawyer-finder')
            }

            if (metadata.agent_used === 'small_claims' && metadata.session_data) {
              const sessionDataTyped = metadata.session_data as Record<string, unknown>
              if (sessionDataTyped.dispute_type) {
                setChatDisputeType(sessionDataTyped.dispute_type as string)
              }
              if (sessionDataTyped.claim_amount) {
                setChatClaimAmount(sessionDataTyped.claim_amount as number)
              }
              if (sessionDataTyped.step) {
                setChatStep(sessionDataTyped.step as string)
              }
            }

            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === streamingMsgId
                  ? { ...msg, agentUsed: metadata.agent_used, actions: metadata.actions }
                  : msg
              )
            )
          },
          onDone: (doneData) => {
            if (rafIdRef.current !== null) {
              cancelAnimationFrame(rafIdRef.current)
              rafIdRef.current = null
            }
            if (accumulatedContent) {
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === streamingMsgId
                    ? { ...msg, content: accumulatedContent }
                    : msg
                )
              )
            }
            setStreamingMessageId(null)
            setIsLoading(false)
            resetLoadingState()

            if (doneData?.conversation_id) {
              setConversationId(doneData.conversation_id as string)
            }

            const newSessionData = { ...receivedSessionData }
            if (doneData?.thread_id) newSessionData.thread_id = doneData.thread_id
            if (doneData?.session_secret) newSessionData.session_secret = doneData.session_secret
            if (doneData?.active_agent) newSessionData.active_agent = doneData.active_agent

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
              const seen = new Set<string>()
              const uniqueSources = (receivedSources || []).filter((ref) => {
                const key = ref.doc_type === 'law' ? ref.law_name : ref.case_number
                if (!key || seen.has(key)) return false
                seen.add(key)
                return true
              })
              newSessionData.aiReferences = uniqueSources
            }

            if (Object.keys(newSessionData).length > 0) {
              setSessionData({ ...sessionData, ...newSessionData })
            } else if (receivedSessionData) {
              setSessionData(receivedSessionData)
            }

            // NAVIGATE 액션 처리
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
              setPendingNavTarget(fullUrl)
              hasNavigated = true
            }

            if (!hasNavigated && !hasNavigatedOnRouting && agentUsed && AGENT_PAGE_MAP[agentUsed]) {
              const targetPage = AGENT_PAGE_MAP[agentUsed]
              const targetPathname = targetPage.split('?')[0]
              if (pathname !== targetPathname) {
                setPendingNavTarget(targetPage)
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
            resetLoadingState()
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
      resetLoadingState()
    }
  }

  const handleAction = async (action: string) => {
    switch (action) {
      case 'reset_search':
      case 'reset_session':
        handleResetChat()
        break
      case 'expand_search':
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
        handleSend(action)
    }
  }

  const handleRequestLocation = async () => {
    setMessages((prev) => [
      ...prev,
      {
        id: Date.now().toString(),
        role: 'assistant' as const,
        content: '현재 위치를 확인하고 있습니다...',
      },
    ])

    const location = await requestUserLocation()

    if (location) {
      handleSend('현재 위치 주변 변호사를 찾아주세요', location)
    } else {
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now().toString(),
          role: 'assistant' as const,
          content:
            '위치 정보를 가져올 수 없습니다. 브라우저 설정에서 위치 권한을 허용하거나, 특정 지역명을 입력해주세요.',
        },
      ])
    }
  }

  const toggleViewMode = () => {
    setChatMode(chatMode === 'split' ? 'floating' : 'split')
  }

  return {
    // 상태
    messages,
    input,
    setInput,
    isLoading,
    isStreaming,
    streamingMessageId,
    loadingStatus,
    scrollRef,

    // UI 상태
    isChatOpen,
    toggleChat,
    setChatOpen,
    chatMode,
    setChatMode,
    supportsFloatingMode,
    isChatHiddenPage,

    // 사용자/세션
    userRole,
    setUserRole,
    sessionData,
    setHighlightedCaseNumber,

    // 핸들러
    handleSend,
    handleAction,
    handleResetChat,
    handleRequestLocation,
    toggleViewMode,
  }
}
