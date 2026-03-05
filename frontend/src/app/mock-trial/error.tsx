'use client'

import Link from 'next/link'
import { ArrowLeft } from 'lucide-react'

export default function Error({ error, reset }: { error: Error & { digest?: string }; reset: () => void }) {
  return (
    <div className="flex min-h-[60vh] flex-col items-center justify-center gap-4 p-8">
      <Link
        href="/"
        className="p-2 text-gray-400 hover:text-gray-700 hover:bg-gray-100 rounded-full transition-colors"
        aria-label="홈으로"
      >
        <ArrowLeft size={20} />
      </Link>
      <h2 className="text-xl font-semibold text-[#1D1D1F]">페이지를 불러오지 못했습니다</h2>
      <p className="text-sm text-gray-500">{error.message}</p>
      <button
        onClick={reset}
        className="rounded-lg bg-blue-600 px-4 py-2 text-white hover:opacity-90 transition-opacity"
      >
        다시 시도
      </button>
    </div>
  )
}
