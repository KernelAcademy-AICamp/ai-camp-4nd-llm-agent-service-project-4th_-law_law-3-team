'use client'

import { createContext, useContext, useState, useCallback, ReactNode } from 'react'

interface UIContextType {
  isChatOpen: boolean
  chatMode: 'split' | 'floating'
  activePanel: string | null
  pendingMessage: string | null
  isRouting: boolean
  toggleChat: () => void
  setChatOpen: (isOpen: boolean) => void
  setChatMode: (mode: 'split' | 'floating') => void
  setActivePanel: (panel: string | null) => void
  setPendingMessage: (message: string | null) => void
  setIsRouting: (isRouting: boolean) => void
}

const UIContext = createContext<UIContextType | undefined>(undefined)

export function UIProvider({ children }: { children: ReactNode }) {
  const [isChatOpen, setIsChatOpen] = useState(false)
  const [chatMode, setChatModeState] = useState<'split' | 'floating'>('split')
  const [activePanel, setActivePanelState] = useState<string | null>(null)
  const [pendingMessage, setPendingMessageState] = useState<string | null>(null)
  const [isRouting, setIsRoutingState] = useState(false)

  const toggleChat = useCallback(() => setIsChatOpen((prev) => !prev), [])
  const setChatOpen = useCallback((isOpen: boolean) => setIsChatOpen(isOpen), [])
  const setChatMode = useCallback((mode: 'split' | 'floating') => setChatModeState(mode), [])
  const setActivePanel = useCallback((panel: string | null) => setActivePanelState(panel), [])
  const setPendingMessage = useCallback((message: string | null) => setPendingMessageState(message), [])
  const setIsRouting = useCallback((routing: boolean) => setIsRoutingState(routing), [])

  return (
    <UIContext.Provider value={{
      isChatOpen,
      chatMode,
      activePanel,
      pendingMessage,
      isRouting,
      toggleChat,
      setChatOpen,
      setChatMode,
      setActivePanel,
      setPendingMessage,
      setIsRouting,
    }}>
      {children}
    </UIContext.Provider>
  )
}

export function useUI() {
  const context = useContext(UIContext)
  if (context === undefined) {
    throw new Error('useUI must be used within a UIProvider')
  }
  return context
}
