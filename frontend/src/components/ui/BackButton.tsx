'use client'

import { useRouter } from 'next/navigation'
import { useChat } from '@/context/ChatContext'
import { ArrowLeft } from 'lucide-react'

interface BackButtonProps {
  className?: string
}

export function BackButton({ className }: BackButtonProps) {
  const router = useRouter()
  const { userRole, resetSession } = useChat()

  const handleBack = () => {
    // 세션 초기화 후 홈으로 이동
    resetSession()
    if (userRole) {
      router.push(`/?role=${userRole}`)
    } else {
      router.push('/')
    }
  }

  return (
    <button
      onClick={handleBack}
      className={className || "p-2 mr-2 text-gray-400 hover:text-gray-700 hover:bg-gray-100 rounded-full transition-colors flex items-center justify-center shrink-0"}
      aria-label="뒤로 가기"
    >
      <ArrowLeft size={20} />
    </button>
  )
}
