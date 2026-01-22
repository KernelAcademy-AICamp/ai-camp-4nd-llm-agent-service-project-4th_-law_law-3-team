'use client'

import { BackButton } from '@/components/ui/BackButton'

export default function StoryboardPage() {
  return (
    <div className="min-h-screen p-8">
      <div className="flex items-center gap-4 mb-6">
        <BackButton />
        <h1 className="text-2xl font-bold">스토리보드</h1>
      </div>
      <p className="text-gray-600">타임라인 기반 사건 시각화</p>
      {/* TODO: 타임라인 입력 및 이미지 생성 구현 */}
    </div>
  )
}
