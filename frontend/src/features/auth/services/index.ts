import { api } from '@/lib/api'
import type { AuthUser } from '@/context/AuthContext'

interface AuthResponse {
  user: AuthUser
  message: string
}

export async function loginApi(email: string, password: string): Promise<AuthResponse> {
  const res = await api.post('/auth/login', { email, password })
  return res.data
}

export async function registerApi(
  email: string,
  password: string,
  displayName?: string,
): Promise<AuthResponse> {
  const res = await api.post('/auth/register', { email, password, display_name: displayName })
  return res.data
}

export async function logoutApi(): Promise<void> {
  await api.post('/auth/logout')
}

export async function getMe(): Promise<AuthUser> {
  const res = await api.get('/auth/me')
  return res.data
}

export async function updateProfile(data: {
  display_name?: string
  avatar_url?: string
}): Promise<AuthUser> {
  const res = await api.patch('/auth/me', data)
  return res.data
}

export async function updateRole(role: string): Promise<AuthUser> {
  const res = await api.patch('/auth/me/role', { role })
  return res.data
}

export async function getGoogleOAuthUrl(): Promise<{ url: string; state: string }> {
  const res = await api.get('/auth/oauth/google')
  return res.data
}

export async function getKakaoOAuthUrl(): Promise<{ url: string; state: string }> {
  const res = await api.get('/auth/oauth/kakao')
  return res.data
}
