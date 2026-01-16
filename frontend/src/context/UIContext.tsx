'use client'

import { createContext, useContext, useState, ReactNode } from 'react'

interface UIContextType {
  isChatOpen: boolean
  toggleChat: () => void
  setChatOpen: (isOpen: boolean) => void
}

const UIContext = createContext<UIContextType | undefined>(undefined)

export function UIProvider({ children }: { children: ReactNode }) {
  const [isChatOpen, setIsChatOpen] = useState(false)

  const toggleChat = () => setIsChatOpen((prev) => !prev)
  const setChatOpen = (isOpen: boolean) => setIsChatOpen(isOpen)

  return (
    <UIContext.Provider value={{ isChatOpen, toggleChat, setChatOpen }}>
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
