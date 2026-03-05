'use client'

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import { api } from '@/lib/api'

export interface AuthUser {
  user_id: string
  email: string
  display_name: string | null
  avatar_url: string | null
  role: string
  email_verified: boolean
}

interface AuthContextType {
  user: AuthUser | null
  isLoading: boolean
  isAuthenticated: boolean
  login: (email: string, password: string) => Promise<AuthUser>
  register: (email: string, password: string, displayName?: string, role?: string) => Promise<AuthUser>
  logout: () => Promise<void>
  refreshUser: () => Promise<void>
}

const AuthContext = createContext<AuthContextType | null>(null)

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  const refreshUser = useCallback(async () => {
    try {
      const res = await api.get('/auth/me')
      setUser(res.data)
    } catch {
      setUser(null)
    }
  }, [])

  useEffect(() => {
    refreshUser().finally(() => setIsLoading(false))
  }, [refreshUser])

  const login = useCallback(async (email: string, password: string): Promise<AuthUser> => {
    const res = await api.post('/auth/login', { email, password })
    setUser(res.data.user)
    return res.data.user
  }, [])

  const register = useCallback(async (
    email: string, password: string, displayName?: string, role?: string
  ): Promise<AuthUser> => {
    const res = await api.post('/auth/register', {
      email,
      password,
      display_name: displayName,
      role: role || 'user',
    })
    setUser(res.data.user)
    return res.data.user
  }, [])

  const logout = useCallback(async () => {
    await api.post('/auth/logout')
    setUser(null)
  }, [])

  const value = useMemo<AuthContextType>(
    () => ({
      user,
      isLoading,
      isAuthenticated: user !== null,
      login,
      register,
      logout,
      refreshUser,
    }),
    [user, isLoading, login, register, logout, refreshUser],
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth(): AuthContextType {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider')
  }
  return context
}
