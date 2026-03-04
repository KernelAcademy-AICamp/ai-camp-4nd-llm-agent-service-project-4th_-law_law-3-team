'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import { usePathname, useRouter, useSearchParams } from 'next/navigation'
import { useUI } from '@/context/UIContext'
import { useChat } from '@/context/ChatContext'
import { useStreamingChat } from '@/hooks/useStreamingChat'
import { useSmallClaimsSync } from '@/hooks/useSmallClaimsSync'
import axios from 'axios'
import type { ChatAction } from './ChatActions'
import type { ChatSource } from '@/features/case-precedent/types'

import {
  PATHNAME_AGENT_MAP,
  AGENT_PAGE_MAP,
  AGENT_DISPLAY_NAMES,
  AGENT_GREETINGS,
  FLOATING_MODE_PATHS,
} from './chat/constants'
import type { Message } from './chat/constants'
import MessageBubble, { useMarkdownComponents } from './chat/MessageBubble'
import { useLoadingStatus } from './chat/useLoadingStatus'

export default function ChatWidget() {
  const router = useRouter()
  const pathname = usePathname()
  const searchParams = useSearchParams()
  const agentFromUrl = searchParams.get('agent')
  const effectiveAgent = agentFromUrl || PATHNAME_AGENT_MAP[pathname] || null
  const { isChatOpen, toggleChat, setChatOpen, chatMode, setChatMode, activePanel, setActivePanel, pendingMessage, setPendingMessage } = useUI()
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
  const supportsFloatingMode = FLOATING_MODE_PATHS.has(pathname) || pathname.startsWith('/workspace')

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
  const [isLoading, setIsLoading] = useState(false)

  const {
    loadingStatus,
    hasReceivedFirstTokenRef,
    resetLoadingState,
    startLoading,
    markFirstToken,
  } = useLoadingStatus(isLoading, isStreaming)

  const [input, setInput] = useState('')
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

  // Interactive Hub 연동: pendingMessage 감지 시 자동 전송
  useEffect(() => {
    if (pendingMessage && !isLoading && !isStreaming) {
      handleSend(pendingMessage)
      setPendingMessage(null)
    }
  }, [pendingMessage, isLoading, isStreaming])

  // 페이지 변경 시 모드 설정
  const prevPathnameRef = useRef<string | null>(null)

  // 챗봇을 숨겨야 하는 페이지 (자체 채팅 UI가 있는 경우)
  const isChatHiddenPage = pathname === '/mock-trial'

  useEffect(() => {
    if (prevPathnameRef.current === pathname) return
    prevPathnameRef.current = pathname

    if (isChatHiddenPage) {
      setChatOpen(false)
    } else if (supportsFloatingMode) {
      setChatMode('floating')
      setChatOpen(true)
    } else {
      setChatMode('split')
    }
  }, [pathname, setChatMode, setChatOpen, supportsFloatingMode, isChatHiddenPage])

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
          onSources: (sources) => {
            receivedSources = sources
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

            // Proactive UI Trigger: lawyer_finder 자동으로 패널 열기
            if (metadata.agent_used === 'lawyer_finder' && activePanel !== 'lawyer-finder') {
              setActivePanel('lawyer-finder')
            }

            // 소액소송 에이전트 응답 시 UI 동기화
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
            // Flush pending rAF update
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
        role: 'assistant',
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
          role: 'assistant',
          content:
            '위치 정보를 가져올 수 없습니다. 브라우저 설정에서 위치 권한을 허용하거나, 특정 지역명을 입력해주세요.',
        },
      ])
    }
  }

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
      'bg-white/95 backdrop-blur-xl border-l border-blue-500/20 shadow-[-20px_0_80px_-20px_rgba(0,0,0,0.15),-4px_0_20px_rgba(59,130,246,0.03)] text-[#1D1D1F]',
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

  const layoutClasses =
    chatMode === 'split'
      ? 'fixed top-0 right-0 w-1/2 h-screen z-50 flex flex-col animate-in slide-in-from-right duration-500'
      : 'fixed bottom-6 right-6 w-[380px] h-[600px] z-50 rounded-2xl flex flex-col animate-in slide-in-from-bottom zoom-in duration-300'

  // Floating Button (Collapsed)
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
              className={`px-2 py-1 rounded-l-md transition-colors ${userRole === 'user'
                ? themeClasses.roleActive
                : themeClasses.roleInactive
                }`}
            >
              일반인
            </button>
            <button
              onClick={() => setUserRole('lawyer')}
              className={`px-2 py-1 rounded-r-md transition-colors ${userRole === 'lawyer'
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

          {/* Toggle View Mode Button */}
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
