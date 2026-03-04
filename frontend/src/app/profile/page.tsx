'use client'

import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { LogOut, Save, User } from 'lucide-react'
import { useRouter } from 'next/navigation'
import { useUI } from '@/context/UIContext'
import { useAuth } from '@/context/AuthContext'
import { useChat, type UserRole } from '@/context/ChatContext'
import { BackButton } from '@/components/ui/BackButton'
import { api } from '@/lib/api'

const ROLE_OPTIONS: { value: UserRole; label: string; description: string }[] = [
  { value: 'user', label: '일반 사용자', description: '법률 상담, 변호사 찾기, 소액소송 등' },
  { value: 'lawyer', label: '변호사', description: '판례 분석, 법령 체계도, 사건 워크스페이스 등' },
]

export default function ProfilePage() {
  const router = useRouter()
  const { isChatOpen, chatMode } = useUI()
  const { user, isLoading, isAuthenticated, logout, refreshUser } = useAuth()
  const { userRole, setUserRole } = useChat()

  const [displayName, setDisplayName] = useState(user?.display_name || '')
  const [selectedRole, setSelectedRole] = useState<UserRole>(userRole)
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)

  const updateProfileMutation = useMutation({
    mutationFn: async (data: { display_name?: string; role?: string }) => {
      if (data.display_name !== undefined) {
        await api.patch('/auth/me', { display_name: data.display_name })
      }
      if (data.role) {
        await api.patch('/auth/me/role', { role: data.role })
      }
    },
    onSuccess: () => {
      refreshUser()
      setMessage({ type: 'success', text: '프로필이 업데이트되었습니다.' })
      setTimeout(() => setMessage(null), 3000)
    },
    onError: () => {
      setMessage({ type: 'error', text: '업데이트에 실패했습니다.' })
      setTimeout(() => setMessage(null), 3000)
    },
  })

  const handleSave = () => {
    if (isAuthenticated) {
      const updates: { display_name?: string; role?: string } = {}
      if (displayName !== (user?.display_name || '')) {
        updates.display_name = displayName
      }
      if (selectedRole !== user?.role) {
        updates.role = selectedRole
      }
      if (Object.keys(updates).length > 0) {
        updateProfileMutation.mutate(updates)
      }
    }
    // 역할 변경은 인증 여부와 무관하게 ChatContext에 반영
    setUserRole(selectedRole)
    if (!isAuthenticated) {
      setMessage({ type: 'success', text: '역할이 변경되었습니다.' })
      setTimeout(() => setMessage(null), 3000)
    }
  }

  const handleLogout = async () => {
    await logout()
    router.push('/')
  }

  if (isLoading) {
    return (
      <div className="h-screen flex items-center justify-center bg-gray-50">
        <div className="animate-spin w-8 h-8 border-2 border-gray-300 border-t-blue-500 rounded-full" />
      </div>
    )
  }

  return (
    <div
      className={`h-screen flex flex-col bg-gray-50 transition-all duration-500 ${
        isChatOpen && chatMode === 'split' ? 'w-1/2' : 'w-full'
      }`}
    >
      {/* 헤더 */}
      <header className="bg-white border-b px-6 py-4">
        <div className="max-w-2xl mx-auto">
          <div className="flex items-center gap-3">
            <BackButton />
            <div>
              <h1 className="text-xl font-bold text-gray-900">프로필 설정</h1>
              <p className="text-sm text-gray-500 mt-1">
                계정 정보와 역할을 관리합니다
              </p>
            </div>
          </div>
        </div>
      </header>

      <div className="flex-1 overflow-y-auto px-6 py-6">
        <div className="max-w-2xl mx-auto space-y-6">
          {/* 알림 메시지 */}
          {message && (
            <div
              className={`px-4 py-3 rounded-lg text-sm ${
                message.type === 'success'
                  ? 'bg-green-50 text-green-700 border border-green-200'
                  : 'bg-red-50 text-red-700 border border-red-200'
              }`}
            >
              {message.text}
            </div>
          )}

          {/* 프로필 카드 */}
          <div className="bg-white rounded-xl border border-gray-200 p-6">
            <div className="flex items-center gap-4 mb-6">
              <div className="w-16 h-16 rounded-full bg-blue-100 flex items-center justify-center">
                {user?.avatar_url ? (
                  <img
                    src={user.avatar_url}
                    alt="프로필"
                    className="w-16 h-16 rounded-full object-cover"
                  />
                ) : (
                  <User size={28} className="text-blue-600" />
                )}
              </div>
              <div>
                <p className="font-semibold text-gray-900">
                  {isAuthenticated ? (user?.display_name || user?.email) : '익명 사용자'}
                </p>
                {isAuthenticated && user?.email && (
                  <p className="text-sm text-gray-500">{user.email}</p>
                )}
                {!isAuthenticated && (
                  <p className="text-sm text-gray-500">
                    로그인하면 기기 간 데이터가 동기화됩니다
                  </p>
                )}
              </div>
            </div>

            {/* 표시 이름 (인증 사용자만) */}
            {isAuthenticated && (
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-1.5">
                  표시 이름
                </label>
                <input
                  type="text"
                  value={displayName}
                  onChange={(e) => setDisplayName(e.target.value)}
                  placeholder="이름을 입력하세요"
                  className="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
                />
              </div>
            )}
          </div>

          {/* 역할 선택 */}
          <div className="bg-white rounded-xl border border-gray-200 p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">역할 선택</h2>
            <p className="text-sm text-gray-500 mb-4">
              역할에 따라 표시되는 기능 모듈이 달라집니다
            </p>
            <div className="space-y-3">
              {ROLE_OPTIONS.map((option) => (
                <label
                  key={option.value}
                  className={`flex items-start gap-3 p-4 rounded-lg border cursor-pointer transition-colors ${
                    selectedRole === option.value
                      ? 'border-blue-500 bg-blue-50'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                >
                  <input
                    type="radio"
                    name="role"
                    value={option.value}
                    checked={selectedRole === option.value}
                    onChange={() => setSelectedRole(option.value)}
                    className="mt-0.5 text-blue-600 focus:ring-blue-500"
                  />
                  <div>
                    <p className="font-medium text-gray-900">{option.label}</p>
                    <p className="text-sm text-gray-500 mt-0.5">{option.description}</p>
                  </div>
                </label>
              ))}
            </div>
          </div>

          {/* 저장 버튼 */}
          <button
            onClick={handleSave}
            disabled={updateProfileMutation.isPending}
            className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors"
          >
            <Save size={18} />
            {updateProfileMutation.isPending ? '저장 중...' : '변경사항 저장'}
          </button>

          {/* 로그아웃 / 로그인 안내 */}
          {isAuthenticated ? (
            <button
              onClick={handleLogout}
              className="w-full flex items-center justify-center gap-2 px-4 py-3 border border-gray-300 text-gray-700 font-medium rounded-lg hover:bg-gray-50 transition-colors"
            >
              <LogOut size={18} />
              로그아웃
            </button>
          ) : (
            <div className="text-center">
              <button
                onClick={() => router.push('/login')}
                className="text-blue-600 hover:text-blue-700 text-sm font-medium"
              >
                로그인 / 회원가입
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
