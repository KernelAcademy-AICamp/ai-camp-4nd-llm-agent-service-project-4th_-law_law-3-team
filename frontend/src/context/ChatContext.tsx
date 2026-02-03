'use client'

import { createContext, useContext, useState, useCallback, ReactNode } from 'react'

export type UserRole = 'user' | 'lawyer'

interface UserLocation {
  latitude: number
  longitude: number
}

interface SessionData {
  active_agent?: string
  step?: string
  [key: string]: unknown
}

interface ChatContextType {
  userRole: UserRole
  setUserRole: (role: UserRole) => void
  sessionData: SessionData
  setSessionData: (data: SessionData) => void
  userLocation: UserLocation | null
  setUserLocation: (location: UserLocation | null) => void
  resetSession: () => void
  requestUserLocation: () => Promise<UserLocation | null>
  // 챗봇-카드 연결용 (판례번호로 하이라이트)
  highlightedCaseNumber: string | null
  setHighlightedCaseNumber: (caseNumber: string | null) => void
}

const ChatContext = createContext<ChatContextType | undefined>(undefined)

export function ChatProvider({ children }: { children: ReactNode }) {
  const [userRole, setUserRole] = useState<UserRole>('user')
  const [sessionData, setSessionData] = useState<SessionData>({})
  const [userLocation, setUserLocation] = useState<UserLocation | null>(null)
  const [highlightedCaseNumber, setHighlightedCaseNumber] = useState<string | null>(null)

  const resetSession = useCallback(() => {
    setSessionData({})
    setHighlightedCaseNumber(null)
  }, [])

  const requestUserLocation = useCallback(async (): Promise<UserLocation | null> => {
    if (!navigator.geolocation) {
      console.warn('Geolocation is not supported by this browser')
      return null
    }

    return new Promise((resolve) => {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const location = {
            latitude: position.coords.latitude,
            longitude: position.coords.longitude,
          }
          setUserLocation(location)
          resolve(location)
        },
        (error) => {
          console.warn('Error getting location:', error.message)
          resolve(null)
        },
        {
          enableHighAccuracy: true,
          timeout: 10000,
          maximumAge: 300000, // 5분 캐시
        }
      )
    })
  }, [])

  return (
    <ChatContext.Provider
      value={{
        userRole,
        setUserRole,
        sessionData,
        setSessionData,
        userLocation,
        setUserLocation,
        resetSession,
        requestUserLocation,
        highlightedCaseNumber,
        setHighlightedCaseNumber,
      }}
    >
      {children}
    </ChatContext.Provider>
  )
}

export function useChat() {
  const context = useContext(ChatContext)
  if (context === undefined) {
    throw new Error('useChat must be used within a ChatProvider')
  }
  return context
}
