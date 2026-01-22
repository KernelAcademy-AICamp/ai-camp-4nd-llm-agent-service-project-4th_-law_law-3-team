'use client'

import { useRouter } from 'next/navigation'
import { useChat } from '@/context/ChatContext'
import { ArrowLeft } from 'lucide-react'

export function BackButton() {
  const router = useRouter()
  const { userRole } = useChat()
  
  const handleBack = () => {
    if (userRole) {
      router.push(`/?role=${userRole}`)
    } else {
      router.back()
    }
  }
  
  return (
    <button
      onClick={handleBack}
      className="p-2 mr-2 text-gray-400 hover:text-gray-700 hover:bg-gray-100 rounded-full transition-colors flex items-center justify-center shrink-0"
      aria-label="뒤로 가기"
    >
      <ArrowLeft size={20} />
    </button>
  )
}
