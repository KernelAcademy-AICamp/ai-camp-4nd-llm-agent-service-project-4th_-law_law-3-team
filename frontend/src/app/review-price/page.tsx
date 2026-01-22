'use client'

import { BackButton } from '@/components/ui/BackButton'

export default function ReviewPricePage() {
  return (
    <div className="min-h-screen p-8">
      <div className="flex items-center gap-4 mb-6">
        <BackButton />
        <h1 className="text-2xl font-bold">후기/가격 비교</h1>
      </div>
      <p className="text-gray-600">상담 후기 및 가격 정보 비교</p>
      {/* TODO: 후기 목록 및 가격 비교 차트 구현 */}
    </div>
  )
}
