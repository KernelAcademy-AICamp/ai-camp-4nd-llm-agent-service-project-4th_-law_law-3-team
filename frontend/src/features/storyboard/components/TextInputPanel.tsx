'use client'

import { useState, useRef } from 'react'

interface TextInputPanelProps {
  onExtract: (text: string) => Promise<void>
  onImport: (file: File) => Promise<void>
  isExtracting: boolean
  error: string | null
}

export function TextInputPanel({
  onExtract,
  onImport,
  isExtracting,
  error,
}: TextInputPanelProps) {
  const [text, setText] = useState('')
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleExtract = async () => {
    if (text.trim().length < 10) return
    await onExtract(text)
  }

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      await onImport(file)
      e.target.value = ''
    }
  }

  return (
    <div className="flex-shrink-0 w-96 border-r border-white/10 bg-slate-900/50 backdrop-blur-md p-6 flex flex-col h-full relative z-10">
      <div className="mb-6">
        <h2 className="text-xl font-bold text-white mb-2">사건 내용</h2>
        <p className="text-sm text-slate-400">
          사건의 전말을 입력하시면 AI가 타임라인과 시각 자료를 생성합니다.
        </p>
      </div>

      <div className="flex-1 flex flex-col min-h-0 bg-slate-800/50 rounded-2xl border border-white/5 p-1 mb-4">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="여기에 사건 내용을 자세히 서술해주세요.&#10;&#10;예시:&#10;2024년 1월 15일 A와 B는 강남구 소재의 건물을 50억원에 매매하기로 계약했다. 계약금 5억원이 당일 지급되었다..."
          className="flex-1 w-full p-4 bg-transparent resize-none text-slate-300 placeholder-slate-600 focus:outline-none text-base leading-relaxed custom-scrollbar"
          disabled={isExtracting}
        />
      </div>

      <div className="space-y-4">
        <button
          type="button"
          onClick={handleExtract}
          disabled={isExtracting || text.trim().length < 10}
          className="w-full py-4 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-xl font-bold text-lg hover:from-blue-500 hover:to-indigo-500 disabled:from-slate-700 disabled:to-slate-700 disabled:text-slate-500 disabled:cursor-not-allowed transition-all shadow-lg shadow-blue-900/20 flex items-center justify-center gap-3 group"
        >
          {isExtracting ? (
            <>
              <svg className="animate-spin h-5 w-5 text-white/70" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
              <span>분석 중...</span>
            </>
          ) : (
            <>
              <span className="group-hover:scale-105 transition-transform">타임라인 생성하기</span>
              <svg className="w-5 h-5 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </>
          )}
        </button>

        <div className="flex justify-center">
          <input
            ref={fileInputRef}
            type="file"
            accept=".json"
            onChange={handleFileChange}
            className="hidden"
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="text-xs text-slate-600 hover:text-slate-400 underline decoration-slate-700 underline-offset-4 transition-colors"
            title="기존 데이터 불러오기"
          >
            백업 파일(.json) 가져오기
          </button>
        </div>
      </div>

      {error && (
        <div className="absolute bottom-20 left-6 right-6 p-4 bg-red-500/10 border border-red-500/20 rounded-xl backdrop-blur-md">
          <div className="flex items-start gap-3">
            <svg className="w-5 h-5 text-red-400 shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <p className="text-sm text-red-200">{error}</p>
          </div>
        </div>
      )}
    </div>
  )
}
