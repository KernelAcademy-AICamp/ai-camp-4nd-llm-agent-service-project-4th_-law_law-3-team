'use client'

import { useState, useRef, useEffect } from 'react'
import { usePathname } from 'next/navigation'
import { useUI } from '@/context/UIContext'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
}

export default function ChatWidget() {
  const pathname = usePathname()
  const { isChatOpen, toggleChat, setChatOpen } = useUI()
  
  // Determine if we are on the map page
  const isMapPage = pathname === '/lawyer-finder'

  // Local state for view mode: 'split' or 'floating'
  // Default to 'floating' on map page, 'split' elsewhere
  const [viewMode, setViewMode] = useState<'split' | 'floating'>('split')

  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      role: 'assistant',
      content: '안녕하세요! 저는 당신의 법률 AI 어시스턴트입니다. 무엇을 도와드릴까요?',
    },
  ])
  const [input, setInput] = useState('')
  const scrollRef = useRef<HTMLDivElement>(null)

  // Sync view mode with page change
  useEffect(() => {
    if (isMapPage) {
      setViewMode('floating')
    } else {
      setViewMode('split')
    }
  }, [isMapPage])

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages, isChatOpen, viewMode])

  const handleSend = () => {
    if (!input.trim()) return

    const userMsg: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
    }
    setMessages((prev) => [...prev, userMsg])
    setInput('')

    // Fake AI response
    setTimeout(() => {
      const assistantMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `"${input}"에 대해 법률 정보를 분석 중입니다. (이것은 데모 응용 프로그램입니다)`,
      }
      setMessages((prev) => [...prev, assistantMsg])
    }, 1000)
  }

  // Toggle view mode manually
  const toggleViewMode = () => {
    setViewMode((prev) => (prev === 'split' ? 'floating' : 'split'))
  }

  // Theme configuration
  // Map page uses 'light' theme (unless in split mode? or always light? User said design match)
  // Let's stick to Light theme for Map Page regardless of mode, or mainly for floating.
  // Actually, split screen might look odd if it's light theme next to map? No, map is light.
  const isLightTheme = isMapPage

  // Styles based on theme
  const themeClasses = isLightTheme
    ? {
        container: 'bg-white/90 backdrop-blur-xl border-l border-gray-200 shadow-2xl text-gray-900',
        header: 'bg-gray-50 border-b border-gray-200',
        headerTitle: 'text-gray-900',
        headerSubtitle: 'text-blue-600',
        messageUser: 'bg-blue-600 text-white shadow-md',
        messageBot: 'bg-gray-100 text-gray-800 border border-gray-200',
        inputArea: 'bg-gray-50 border-t border-gray-200',
        input: 'bg-white border-gray-200 text-gray-900 placeholder-gray-400 focus:border-blue-500 focus:bg-white',
        closeBtn: 'text-gray-400 hover:bg-gray-100 hover:text-gray-600',
      }
    : {
        container: 'bg-slate-900/95 backdrop-blur-xl border-l border-white/10 shadow-2xl text-white',
        header: 'bg-blue-600/10 border-b border-white/10',
        headerTitle: 'text-white',
        headerSubtitle: 'text-blue-400',
        messageUser: 'bg-blue-600 text-white shadow-lg',
        messageBot: 'bg-white/5 text-gray-200 border border-white/5',
        inputArea: 'bg-white/5 border-t border-white/10',
        input: 'bg-black/20 border-white/10 text-white placeholder-white/30 focus:border-blue-500/50 focus:bg-white/5',
        closeBtn: 'text-white/50 hover:bg-white/10 hover:text-white',
      }

  // Layout classes based on viewMode
  // If undefined/hidden, we don't render this part (handled below)
  const layoutClasses = viewMode === 'split'
    ? 'fixed top-0 right-0 w-1/2 h-screen z-50 flex flex-col animate-in slide-in-from-right duration-500'
    : 'fixed bottom-6 right-6 w-[380px] h-[600px] z-50 rounded-2xl flex flex-col animate-in slide-in-from-bottom zoom-in duration-300' // Floating

  // Floating Button (Collapsed)
  if (!isChatOpen) {
    return (
      <button
        onClick={toggleChat}
        className="fixed bottom-6 right-6 w-16 h-16 bg-blue-600 hover:bg-blue-500 text-white rounded-full shadow-lg flex items-center justify-center text-3xl z-50 transition-all hover:scale-110 active:scale-95 animate-in fade-in zoom-in duration-300"
      >
        🤖
      </button>
    )
  }

  // Expanded Chat Interface
  return (
    <div className={`${layoutClasses} ${themeClasses.container}`}>
      {/* Header */}
      <div className={`p-4 md:p-6 flex justify-between items-center ${themeClasses.header} ${viewMode === 'floating' ? 'rounded-t-2xl' : ''}`}>
        <div className="flex items-center gap-4">
          <div className="w-10 h-10 rounded-full bg-blue-500 flex items-center justify-center text-xl shadow-lg">🤖</div>
          <div>
            <h3 className={`font-bold text-lg ${themeClasses.headerTitle}`}>AI 법률 상담</h3>
            <p className={`text-xs font-bold uppercase tracking-widest ${themeClasses.headerSubtitle}`}>Active Now</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {/* Toggle View Mode Button (Only on Map Page) */}
          {isMapPage && (
            <button
              onClick={toggleViewMode}
              className={`p-2 rounded-lg transition-colors ${themeClasses.closeBtn}`}
              title={viewMode === 'split' ? "작게 보기" : "크게 보기"}
            >
              {viewMode === 'split' ? (
                 // Minimize Icon
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 14h6v6" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 10h-6V4" />
                </svg>
              ) : (
                // Maximize Icon
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 20h6v-6" />
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
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
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
              {msg.content}
            </div>
          </div>
        ))}
      </div>

      {/* Input */}
      <div className={`p-4 md:p-6 ${themeClasses.inputArea} ${viewMode === 'floating' ? 'rounded-b-2xl' : ''}`}>
        <div className="relative flex items-center gap-3">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            placeholder="법률 질문을 입력하세요..."
            className={`flex-1 rounded-xl px-4 py-3 md:px-6 md:py-4 text-sm md:text-base focus:outline-none transition-all shadow-sm ${themeClasses.input}`}
          />
          <button
            onClick={handleSend}
            className="p-3 md:p-4 bg-blue-600 hover:bg-blue-500 text-white rounded-xl transition-colors shadow-lg active:scale-95"
          >
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
          </button>
        </div>
      </div>
    </div>
  )
}
