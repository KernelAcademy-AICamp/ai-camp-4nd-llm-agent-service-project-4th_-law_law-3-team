'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import { usePathname, useRouter, useSearchParams } from 'next/navigation'
import { useUI } from '@/context/UIContext'
import { useChat } from '@/context/ChatContext'
import { useStreamingChat } from '@/hooks/useStreamingChat'
import { api } from '@/lib/api'
import axios from 'axios'
import ReactMarkdown from 'react-markdown'
import ChatActions, { ChatAction } from './ChatActions'

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

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  actions?: ChatAction[]
  agentUsed?: string
  sources?: ChatSource[]  // 참조 자료 (카드 연결용)
}

interface ChatSource {
  case_name?: string
  case_number?: string
  doc_type: string
  similarity: number
  summary?: string
  content?: string
  // 법령용 필드
  law_name?: string
  law_type?: string
  // 그래프 보강 정보
  cited_statutes?: string[]
  similar_cases?: string[]
}

interface MultiAgentChatResponse {
  response: string
  agent_used: string
  sources: ChatSource[]
  actions: ChatAction[]
  session_data: Record<string, unknown>
}

export default function ChatWidget() {
  const router = useRouter()
  const pathname = usePathname()
  const searchParams = useSearchParams()
  const agentFromUrl = searchParams.get('agent')
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
  } = useChat()

  // Determine if we are on pages that support floating mode
  const isMapPage = pathname === '/lawyer-finder'
  const isStatuteHierarchyPage = pathname === '/statute-hierarchy'
  const supportsFloatingMode = isMapPage || isStatuteHierarchyPage

  // Global state for view mode is now handled by UIContext

  // agent 타입에 따른 초기 메시지 생성
  const getInitialMessage = useCallback((agent: string | null): Message => {
    if (agent === 'law_search') {
      return {
        id: '1',
        role: 'assistant',
        content: '안녕하세요! 법령 검색 AI입니다.\n\n**법령에 대해 질문해주세요.**\n- 관련 법령 조항 검색\n- 법령 해석 및 적용 사례',
      }
    } else if (agent === 'case_search') {
      return {
        id: '1',
        role: 'assistant',
        content: '안녕하세요! 판례 검색 AI입니다.\n\n**판례에 대해 질문해주세요.**\n- 관련 판례 검색\n- 법률 상담',
      }
    }
    return {
      id: '1',
      role: 'assistant',
      content: '안녕하세요! 저는 당신의 법률 AI 어시스턴트입니다.\n\n**무엇을 도와드릴까요?**\n- 변호사 찾기\n- 판례 검색\n- 소액소송 가이드',
    }
  }, [])

  const [messages, setMessages] = useState<Message[]>([getInitialMessage(null)])

  // 스트리밍 관련 상태
  const { sendStreamingMessage, isStreaming, abortStream } = useStreamingChat()
  const [streamingMessageId, setStreamingMessageId] = useState<string | null>(null)

  // agent 파라미터 변경 시 채팅 초기화
  const prevAgentRef = useRef<string | null>(null)
  useEffect(() => {
    // 첫 렌더링이 아니고 agent가 변경된 경우에만 초기화
    if (prevAgentRef.current !== null && prevAgentRef.current !== agentFromUrl) {
      setMessages([getInitialMessage(agentFromUrl)])
      resetSession()
    }
    prevAgentRef.current = agentFromUrl
  }, [agentFromUrl, getInitialMessage, resetSession])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)

  // 페이지 변경 시 초기화 및 모드 설정
  const prevPathnameRef = useRef<string | null>(null)

  // 페이지 변경 시 채팅 초기화
  useEffect(() => {
    if (prevPathnameRef.current !== null && prevPathnameRef.current !== pathname) {
      // 다른 페이지로 이동 시 채팅 초기화
      setMessages([getInitialMessage(agentFromUrl)])
    }
  }, [pathname, agentFromUrl, getInitialMessage])

  useEffect(() => {
    // 같은 페이지에서는 모드 변경 안 함 (사용자가 토글한 상태 유지)
    if (prevPathnameRef.current === pathname) return
    prevPathnameRef.current = pathname

    if (supportsFloatingMode) {
      // floating 모드 지원 페이지 첫 진입 시 floating 모드로 시작
      setChatMode('floating')
      if (isMapPage) {
        setChatOpen(true)
      }
    } else {
      // 다른 페이지 진입 시 Split 모드 사용
      setChatMode('split')
    }
  }, [pathname, setChatMode, setChatOpen, isMapPage, supportsFloatingMode])

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages, isChatOpen, chatMode, isLoading])

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
          session_data: sessionData,
          user_location: locationToSend,
          agent: agentFromUrl || undefined,
        },
        {
          onToken: (content) => {
            accumulatedContent += content
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === streamingMsgId
                  ? { ...msg, content: accumulatedContent }
                  : msg
              )
            )
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
            // 메타데이터 업데이트
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === streamingMsgId
                  ? { ...msg, agentUsed: metadata.agent_used, actions: metadata.actions }
                  : msg
              )
            )
          },
          onDone: () => {
            // 스트리밍 완료
            setStreamingMessageId(null)
            setIsLoading(false)

            // 세션 데이터 업데이트
            const newSessionData = { ...receivedSessionData }

            // 판례 검색이나 법률 질문인 경우 페이지 이동
            if (
              (receivedSources && receivedSources.length > 0) ||
              agentUsed === 'legal_search' ||
              agentUsed === 'case_search' ||
              agentUsed === 'legal_answer'
            ) {
              const mainSource = receivedSources?.[0]

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

              setSessionData({ ...sessionData, ...newSessionData })

              if (pathname !== '/case-precedent') {
                router.push('/case-precedent')
              }
            } else if (receivedSessionData) {
              setSessionData(receivedSessionData)
            }

            // NAVIGATE 액션 처리
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
            }
          },
          onError: (errorMessage) => {
            console.error('Streaming error:', errorMessage)
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === streamingMsgId
                  ? { ...msg, content: '죄송합니다. 응답을 생성하는 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.' }
                  : msg
              )
            )
            setStreamingMessageId(null)
            setIsLoading(false)
          },
        }
      )
    } catch (error) {
      console.error('Chat API error:', error)

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
    }
  }

  const handleAction = async (action: string) => {
    // 액션에 따른 처리
    switch (action) {
      case 'reset_search':
      case 'reset_session':
        resetSession()
        setMessages((prev) => [
          ...prev,
          {
            id: Date.now().toString(),
            role: 'assistant',
            content: '세션을 초기화했습니다. 새로운 질문을 해주세요.',
          },
        ])
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

  // Theme configuration (map page uses light theme, others use dark)
  const isLightTheme = isMapPage

  // Styles based on theme
  const themeClasses = isLightTheme
    ? {
        container:
          'bg-white/90 backdrop-blur-xl border-l border-gray-200 shadow-2xl text-gray-900',
        header: 'bg-gray-50 border-b border-gray-200',
        headerTitle: 'text-gray-900',
        headerSubtitle: 'text-blue-600',
        messageUser: 'bg-blue-600 text-white shadow-md',
        messageBot: 'bg-gray-100 text-gray-800 border border-gray-200',
        inputArea: 'bg-gray-50 border-t border-gray-200',
        input:
          'bg-white border-gray-200 text-gray-900 placeholder-gray-400 focus:border-blue-500 focus:bg-white',
        closeBtn: 'text-gray-400 hover:bg-gray-100 hover:text-gray-600',
        roleSelector: 'bg-gray-100 border-gray-200',
        roleActive: 'bg-blue-600 text-white',
        roleInactive: 'text-gray-600 hover:bg-gray-200',
      }
    : {
        container:
          'bg-slate-900/95 backdrop-blur-xl border-l border-white/10 shadow-2xl text-white',
        header: 'bg-blue-600/10 border-b border-white/10',
        headerTitle: 'text-white',
        headerSubtitle: 'text-blue-400',
        messageUser: 'bg-blue-600 text-white shadow-lg',
        messageBot: 'bg-white/5 text-gray-200 border border-white/5',
        inputArea: 'bg-white/5 border-t border-white/10',
        input:
          'bg-black/20 border-white/10 text-white placeholder-white/30 focus:border-blue-500/50 focus:bg-white/5',
        closeBtn: 'text-white/50 hover:bg-white/10 hover:text-white',
        roleSelector: 'bg-white/10 border-white/10',
        roleActive: 'bg-blue-600 text-white',
        roleInactive: 'text-white/70 hover:bg-white/10',
      }

  // Layout classes based on viewMode
  const layoutClasses =
    chatMode === 'split'
      ? 'fixed top-0 right-0 w-1/2 h-screen z-50 flex flex-col animate-in slide-in-from-right duration-500'
      : 'fixed bottom-6 right-6 w-[380px] h-[600px] z-50 rounded-2xl flex flex-col animate-in slide-in-from-bottom zoom-in duration-300'

  // Floating Button (Collapsed)
  if (!isChatOpen) {
    return (
      <button
        onClick={toggleChat}
        className="fixed bottom-6 right-6 w-16 h-16 bg-blue-600 hover:bg-blue-500 text-white rounded-full shadow-lg flex items-center justify-center text-3xl z-50 transition-all hover:scale-110 active:scale-95 animate-in fade-in zoom-in duration-300"
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
                ? `${sessionData.active_agent === 'lawyer_finder' ? '변호사 찾기' : sessionData.active_agent === 'small_claims' ? '소액소송' : '판례 검색'}`
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
          <div
            key={msg.id}
            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[85%] p-4 rounded-2xl text-base leading-relaxed ${
                msg.role === 'user'
                  ? `${themeClasses.messageUser} rounded-tr-none`
                  : `${themeClasses.messageBot} rounded-tl-none`
              }`}
            >
              {msg.role === 'assistant' ? (
                <div className={`prose prose-sm max-w-none prose-p:my-2 prose-ul:my-2 prose-ol:my-2 prose-li:my-0 prose-headings:my-2 prose-strong:text-inherit ${!isLightTheme ? 'prose-invert' : ''}`}>
                  <ReactMarkdown
                    components={{
                      // 텍스트 노드에서 판례번호를 클릭 가능하게 변환
                      p: ({ children }) => (
                        <p>
                          {typeof children === 'string' ? (
                            <CaseNumberLink
                              text={children}
                              onCaseClick={setHighlightedCaseNumber}
                              isLightTheme={isLightTheme}
                            />
                          ) : Array.isArray(children) ? (
                            children.map((child, i) =>
                              typeof child === 'string' ? (
                                <CaseNumberLink
                                  key={i}
                                  text={child}
                                  onCaseClick={setHighlightedCaseNumber}
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
                      li: ({ children }) => (
                        <li>
                          {typeof children === 'string' ? (
                            <CaseNumberLink
                              text={children}
                              onCaseClick={setHighlightedCaseNumber}
                              isLightTheme={isLightTheme}
                            />
                          ) : Array.isArray(children) ? (
                            children.map((child, i) =>
                              typeof child === 'string' ? (
                                <CaseNumberLink
                                  key={i}
                                  text={child}
                                  onCaseClick={setHighlightedCaseNumber}
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
                    }}
                  >
                    {msg.content}
                  </ReactMarkdown>
                  {/* 스트리밍 중일 때 깜빡이는 커서 표시 */}
                  {msg.id === streamingMessageId && (
                    <span className="inline-block w-2 h-4 bg-current animate-pulse ml-0.5" />
                  )}
                  {msg.actions && msg.actions.length > 0 && (
                    <ChatActions
                      actions={msg.actions}
                      onAction={handleAction}
                      onRequestLocation={handleRequestLocation}
                      isLightTheme={isLightTheme}
                    />
                  )}
                </div>
              ) : (
                <span className="whitespace-pre-wrap">{msg.content}</span>
              )}
            </div>
          </div>
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
                <span className="ml-2 text-sm opacity-70">응답을 생성하고 있습니다...</span>
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
            placeholder={isLoading || isStreaming ? '응답을 기다리는 중...' : '법률 질문을 입력하세요...'}
            disabled={isLoading || isStreaming}
            className={`flex-1 rounded-xl px-4 py-3 md:px-6 md:py-4 text-sm md:text-base focus:outline-none transition-all shadow-sm ${themeClasses.input} ${isLoading || isStreaming ? 'opacity-50 cursor-not-allowed' : ''}`}
          />
          <button
            onClick={() => handleSend()}
            disabled={isLoading || isStreaming || !input.trim()}
            className={`p-3 md:p-4 bg-blue-600 hover:bg-blue-500 text-white rounded-xl transition-colors shadow-lg active:scale-95 ${isLoading || isStreaming || !input.trim() ? 'opacity-50 cursor-not-allowed' : ''}`}
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
      </div>
    </div>
  )
}
