'use client'

import { createContext, useContext, useState, useCallback, ReactNode } from 'react'

interface UIContextType {
  isChatOpen: boolean
  chatMode: 'split' | 'floating'
  toggleChat: () => void
  setChatOpen: (isOpen: boolean) => void
  setChatMode: (mode: 'split' | 'floating') => void
}

const UIContext = createContext<UIContextType | undefined>(undefined)

export function UIProvider({ children }: { children: ReactNode }) {
  const [isChatOpen, setIsChatOpen] = useState(false)
  const [chatMode, setChatModeState] = useState<'split' | 'floating'>('split')

  const toggleChat = useCallback(() => setIsChatOpen((prev) => !prev), [])
  const setChatOpen = useCallback((isOpen: boolean) => setIsChatOpen(isOpen), [])
  const setChatMode = useCallback((mode: 'split' | 'floating') => setChatModeState(mode), [])

  return (
    <UIContext.Provider value={{ isChatOpen, chatMode, toggleChat, setChatOpen, setChatMode }}>
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
