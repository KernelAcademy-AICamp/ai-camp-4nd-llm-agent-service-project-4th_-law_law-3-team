'use client'

import { useState, useRef, useCallback } from 'react'
import type { InputMode, TimelineItem } from '../types'

interface MultiInputPanelProps {
  onExtractText: (text: string) => Promise<void>
  onExtractVoice: (file: File) => Promise<void>
  onExtractImage: (file: File, context: string) => Promise<void>
  onImport: (file: File) => Promise<void>
  isExtracting: boolean
  error: string | null
}

export function MultiInputPanel({
  onExtractText,
  onExtractVoice,
  onExtractImage,
  onImport,
  isExtracting,
  error,
}: MultiInputPanelProps) {
  const [inputMode, setInputMode] = useState<InputMode>('text')
  const [text, setText] = useState('')
  const [imageContext, setImageContext] = useState('')
  const [isRecording, setIsRecording] = useState(false)
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null)
  const [selectedImage, setSelectedImage] = useState<File | null>(null)
  const [imagePreview, setImagePreview] = useState<string | null>(null)

  const fileInputRef = useRef<HTMLInputElement>(null)
  const audioInputRef = useRef<HTMLInputElement>(null)
  const imageInputRef = useRef<HTMLInputElement>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])

  // 텍스트 추출
  const handleExtractText = async () => {
    if (text.trim().length < 10) return
    await onExtractText(text)
  }

  // 음성 녹음 시작
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mediaRecorder = new MediaRecorder(stream)
      mediaRecorderRef.current = mediaRecorder
      audioChunksRef.current = []

      mediaRecorder.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data)
      }

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' })
        setAudioBlob(audioBlob)
        stream.getTracks().forEach((track) => track.stop())
      }

      mediaRecorder.start()
      setIsRecording(true)
    } catch (error) {
      console.error('마이크 접근 오류:', error)
    }
  }

  // 음성 녹음 중지
  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
    }
  }

  // 음성 파일 선택
  const handleAudioFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setAudioBlob(file)
    }
  }

  // 음성 추출
  const handleExtractVoice = async () => {
    if (!audioBlob) return
    const file = new File([audioBlob], 'recording.webm', { type: 'audio/webm' })
    await onExtractVoice(file)
  }

  // 이미지 파일 선택
  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setSelectedImage(file)
      const reader = new FileReader()
      reader.onload = (e) => {
        setImagePreview(e.target?.result as string)
      }
      reader.readAsDataURL(file)
    }
  }

  // 이미지 추출
  const handleExtractImage = async () => {
    if (!selectedImage) return
    await onExtractImage(selectedImage, imageContext)
  }

  // JSON 파일 가져오기
  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      await onImport(file)
      e.target.value = ''
    }
  }

  const tabs: { mode: InputMode; label: string; icon: JSX.Element }[] = [
    {
      mode: 'text',
      label: '텍스트',
      icon: (
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
      ),
    },
    {
      mode: 'voice',
      label: '음성',
      icon: (
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
        </svg>
      ),
    },
    {
      mode: 'image',
      label: '이미지',
      icon: (
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
        </svg>
      ),
    },
  ]

  return (
    <div className="w-96 h-full border-r border-white/10 bg-slate-900/50 backdrop-blur-md p-6 flex flex-col relative z-10">
      <div className="mb-6">
        <h2 className="text-xl font-bold text-white mb-2">사건 내용</h2>
        <p className="text-sm text-slate-400">
          텍스트, 음성, 이미지로 사건 내용을 입력하면 AI가 타임라인을 생성합니다.
        </p>
      </div>

      {/* 탭 */}
      <div className="flex gap-1 mb-4 p-1 bg-slate-800/50 rounded-xl">
        {tabs.map((tab) => (
          <button
            key={tab.mode}
            type="button"
            onClick={() => setInputMode(tab.mode)}
            disabled={isExtracting}
            className={`
              flex-1 flex items-center justify-center gap-2 py-2 px-3 rounded-lg text-sm font-medium transition-all
              ${inputMode === tab.mode
                ? 'bg-blue-600 text-white shadow-lg'
                : 'text-slate-400 hover:text-white hover:bg-slate-700/50'
              }
              disabled:opacity-50 disabled:cursor-not-allowed
            `}
          >
            {tab.icon}
            {tab.label}
          </button>
        ))}
      </div>

      {/* 텍스트 입력 */}
      {inputMode === 'text' && (
        <div className="flex-1 flex flex-col min-h-0">
          <div className="flex-1 bg-slate-800/50 rounded-2xl border border-white/5 p-1 mb-4">
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="여기에 사건 내용을 자세히 서술해주세요.&#10;&#10;예시:&#10;2024년 1월 15일 A와 B는 강남구 소재의 건물을 50억원에 매매하기로 계약했다..."
              className="flex-1 w-full h-full p-4 bg-transparent resize-none text-slate-300 placeholder-slate-600 focus:outline-none text-base leading-relaxed custom-scrollbar"
              disabled={isExtracting}
            />
          </div>

          <button
            type="button"
            onClick={handleExtractText}
            disabled={isExtracting || text.trim().length < 10}
            className="w-full py-4 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-xl font-bold text-lg hover:from-blue-500 hover:to-indigo-500 disabled:from-slate-700 disabled:to-slate-700 disabled:text-slate-500 disabled:cursor-not-allowed transition-all shadow-lg shadow-blue-900/20 flex items-center justify-center gap-3"
          >
            {isExtracting ? (
              <>
                <svg className="animate-spin h-5 w-5" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                <span>분석 중...</span>
              </>
            ) : (
              <>
                <span>타임라인 생성하기</span>
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </>
            )}
          </button>
        </div>
      )}

      {/* 음성 입력 */}
      {inputMode === 'voice' && (
        <div className="flex-1 flex flex-col min-h-0">
          <div className="flex-1 bg-slate-800/50 rounded-2xl border border-white/5 p-6 mb-4 flex flex-col items-center justify-center">
            {/* 녹음 버튼 */}
            <button
              type="button"
              onClick={isRecording ? stopRecording : startRecording}
              disabled={isExtracting}
              className={`
                w-24 h-24 rounded-full flex items-center justify-center transition-all mb-6
                ${isRecording
                  ? 'bg-red-500 animate-pulse shadow-lg shadow-red-500/50'
                  : 'bg-blue-600 hover:bg-blue-500 shadow-lg shadow-blue-500/30'
                }
                disabled:opacity-50 disabled:cursor-not-allowed
              `}
            >
              {isRecording ? (
                <svg className="w-10 h-10 text-white" fill="currentColor" viewBox="0 0 24 24">
                  <rect x="6" y="6" width="12" height="12" rx="2" />
                </svg>
              ) : (
                <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                </svg>
              )}
            </button>

            <p className="text-slate-400 text-sm mb-4">
              {isRecording ? '녹음 중... 클릭하여 중지' : '클릭하여 녹음 시작'}
            </p>

            {audioBlob && !isRecording && (
              <div className="flex items-center gap-2 p-3 bg-slate-700/50 rounded-xl">
                <svg className="w-5 h-5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                <span className="text-slate-300 text-sm">녹음 완료</span>
              </div>
            )}

            <div className="w-full border-t border-slate-700 my-6" />

            <p className="text-slate-500 text-xs mb-3">또는 파일 업로드</p>
            <input
              ref={audioInputRef}
              type="file"
              accept="audio/*"
              onChange={handleAudioFileChange}
              className="hidden"
            />
            <button
              type="button"
              onClick={() => audioInputRef.current?.click()}
              disabled={isExtracting}
              className="px-4 py-2 bg-slate-700 text-slate-300 rounded-lg text-sm hover:bg-slate-600 transition-colors disabled:opacity-50"
            >
              음성 파일 선택
            </button>
          </div>

          <button
            type="button"
            onClick={handleExtractVoice}
            disabled={isExtracting || !audioBlob}
            className="w-full py-4 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-xl font-bold text-lg hover:from-blue-500 hover:to-indigo-500 disabled:from-slate-700 disabled:to-slate-700 disabled:text-slate-500 disabled:cursor-not-allowed transition-all shadow-lg shadow-blue-900/20 flex items-center justify-center gap-3"
          >
            {isExtracting ? (
              <>
                <svg className="animate-spin h-5 w-5" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                <span>음성 분석 중...</span>
              </>
            ) : (
              <>
                <span>음성으로 타임라인 생성</span>
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </>
            )}
          </button>
        </div>
      )}

      {/* 이미지 입력 */}
      {inputMode === 'image' && (
        <div className="flex-1 flex flex-col min-h-0">
          <div className="flex-1 bg-slate-800/50 rounded-2xl border border-white/5 p-4 mb-4 flex flex-col">
            <input
              ref={imageInputRef}
              type="file"
              accept="image/*"
              onChange={handleImageChange}
              className="hidden"
            />

            {imagePreview ? (
              <div className="relative flex-1 min-h-0 mb-4">
                <img
                  src={imagePreview}
                  alt="선택된 이미지"
                  className="w-full h-full object-contain rounded-lg"
                />
                <button
                  type="button"
                  onClick={() => {
                    setSelectedImage(null)
                    setImagePreview(null)
                  }}
                  className="absolute top-2 right-2 p-2 bg-slate-900/80 rounded-full hover:bg-slate-800 transition-colors"
                >
                  <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            ) : (
              <button
                type="button"
                onClick={() => imageInputRef.current?.click()}
                disabled={isExtracting}
                className="flex-1 border-2 border-dashed border-slate-600 rounded-xl flex flex-col items-center justify-center hover:border-blue-500 hover:bg-blue-500/5 transition-all mb-4 disabled:opacity-50"
              >
                <svg className="w-12 h-12 text-slate-500 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
                <span className="text-slate-400 text-sm">이미지를 선택하세요</span>
                <span className="text-slate-500 text-xs mt-1">문서, 스크린샷, 사진 등</span>
              </button>
            )}

            <textarea
              value={imageContext}
              onChange={(e) => setImageContext(e.target.value)}
              placeholder="추가 설명 (선택사항)&#10;예: 이 문서는 계약서입니다..."
              className="w-full p-3 bg-slate-700/50 rounded-xl resize-none text-slate-300 placeholder-slate-500 focus:outline-none text-sm h-20"
              disabled={isExtracting}
            />
          </div>

          <button
            type="button"
            onClick={handleExtractImage}
            disabled={isExtracting || !selectedImage}
            className="w-full py-4 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-xl font-bold text-lg hover:from-blue-500 hover:to-indigo-500 disabled:from-slate-700 disabled:to-slate-700 disabled:text-slate-500 disabled:cursor-not-allowed transition-all shadow-lg shadow-blue-900/20 flex items-center justify-center gap-3"
          >
            {isExtracting ? (
              <>
                <svg className="animate-spin h-5 w-5" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                <span>이미지 분석 중...</span>
              </>
            ) : (
              <>
                <span>이미지로 타임라인 생성</span>
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </>
            )}
          </button>
        </div>
      )}

      {/* JSON 가져오기 버튼 */}
      <div className="flex justify-center mt-4">
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

      {/* 에러 표시 */}
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
