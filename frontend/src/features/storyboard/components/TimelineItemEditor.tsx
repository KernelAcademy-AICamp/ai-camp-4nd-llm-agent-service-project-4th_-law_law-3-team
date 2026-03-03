'use client'

import { useState, useEffect } from 'react'
import type { TimelineItem } from '../types'

interface TimelineItemEditorProps {
  item?: TimelineItem | null
  onSave: (data: Omit<TimelineItem, 'id' | 'order'>) => void
  onCancel: () => void
  isNew?: boolean
}

export function TimelineItemEditor({
  item,
  onSave,
  onCancel,
  isNew = false,
}: TimelineItemEditorProps) {
  const [date, setDate] = useState('')
  const [title, setTitle] = useState('')
  const [description, setDescription] = useState('')
  const [participantsText, setParticipantsText] = useState('')
  const [imageUrl, setImageUrl] = useState('')
  const [imageUrlError, setImageUrlError] = useState<string | null>(null)

  useEffect(() => {
    if (item) {
      setDate(item.date)
      setTitle(item.title)
      setDescription(item.description)
      setParticipantsText(item.participants.join(', '))
      setImageUrl(item.imageUrl || '')
    } else {
      setDate('')
      setTitle('')
      setDescription('')
      setParticipantsText('')
      setImageUrl('')
    }
    setImageUrlError(null)
  }, [item])

  // ESC 키로 에디터 닫기
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onCancel()
    }
    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [onCancel])

  const isValidImageUrl = (url: string): boolean => {
    if (!url) return true
    return url.startsWith('/media/') || url.startsWith('https://')
  }

  const handleImageUrlChange = (value: string) => {
    setImageUrl(value)
    if (value && !isValidImageUrl(value)) {
      setImageUrlError('/media/로 시작하는 내부 URL 또는 https://로 시작하는 외부 URL만 허용됩니다')
    } else {
      setImageUrlError(null)
    }
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!date.trim() || !title.trim()) return
    if (imageUrl && !isValidImageUrl(imageUrl)) return

    const participants = participantsText
      .split(',')
      .map((p) => p.trim())
      .filter(Boolean)

    onSave({
      date: date.trim(),
      title: title.trim(),
      description: description.trim(),
      participants,
      imageUrl: imageUrl.trim() || undefined,
    })
  }

  return (
    <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50 backdrop-blur-sm">
      <div className="bg-white rounded-2xl shadow-apple-hover w-full max-w-md mx-4 border border-black/[0.06]">
        <div className="p-6 border-b border-black/[0.06]">
          <h3 className="text-xl font-bold text-[#1D1D1F]">
            {isNew ? '새 항목 추가' : '항목 편집'}
          </h3>
        </div>

        <form onSubmit={handleSubmit} className="p-6 space-y-5">
          <div className="space-y-4">
            <div>
              <label htmlFor="date" className="block text-sm font-medium text-[#86868B] mb-1.5">
                날짜
              </label>
              <input
                id="date"
                type="text"
                value={date}
                onChange={(e) => setDate(e.target.value)}
                placeholder="예: 2024-01-15 또는 2024년 1월 초"
                className="w-full px-4 py-2.5 bg-[#F5F5F7] border border-black/[0.06] rounded-xl text-[#1D1D1F] placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-[#007AFF] focus:border-transparent transition-all"
                required
              />
            </div>

            <div>
              <label htmlFor="title" className="block text-sm font-medium text-[#86868B] mb-1.5">
                제목
              </label>
              <input
                id="title"
                type="text"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="이벤트 제목"
                className="w-full px-4 py-2.5 bg-[#F5F5F7] border border-black/[0.06] rounded-xl text-[#1D1D1F] placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-[#007AFF] focus:border-transparent transition-all"
                required
              />
            </div>

            <div>
              <label htmlFor="description" className="block text-sm font-medium text-[#86868B] mb-1.5">
                설명
              </label>
              <textarea
                id="description"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="이벤트 상세 설명"
                rows={3}
                className="w-full px-4 py-2.5 bg-[#F5F5F7] border border-black/[0.06] rounded-xl text-[#1D1D1F] placeholder-gray-400 resize-none focus:outline-none focus:ring-2 focus:ring-[#007AFF] focus:border-transparent transition-all"
              />
            </div>

            <div>
              <label htmlFor="imageUrl" className="block text-sm font-medium text-[#86868B] mb-1.5">
                이미지 URL (선택)
              </label>
              <input
                id="imageUrl"
                type="text"
                value={imageUrl}
                onChange={(e) => handleImageUrlChange(e.target.value)}
                placeholder="https://example.com/image.jpg"
                className={`w-full px-4 py-2.5 bg-[#F5F5F7] border rounded-xl text-[#1D1D1F] placeholder-gray-400 focus:outline-none focus:ring-2 focus:border-transparent transition-all ${imageUrlError ? 'border-red-400 focus:ring-red-400' : 'border-black/[0.06] focus:ring-[#007AFF]'}`}
              />
              {imageUrlError && (
                <p className="text-xs text-red-500 mt-1">{imageUrlError}</p>
              )}
            </div>

            <div>
              <label htmlFor="participants" className="block text-sm font-medium text-[#86868B] mb-1.5">
                관련자 (쉼표로 구분)
              </label>
              <input
                id="participants"
                type="text"
                value={participantsText}
                onChange={(e) => setParticipantsText(e.target.value)}
                placeholder="예: 홍길동, 김철수, A회사"
                className="w-full px-4 py-2.5 bg-[#F5F5F7] border border-black/[0.06] rounded-xl text-[#1D1D1F] placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-[#007AFF] focus:border-transparent transition-all"
              />
            </div>
          </div>

          <div className="flex gap-3 pt-4">
            <button
              type="button"
              onClick={onCancel}
              className="flex-1 py-3 border border-black/[0.06] rounded-xl text-[#3C3C43] font-medium hover:bg-[#F5F5F7] transition-colors"
            >
              취소
            </button>
            <button
              type="submit"
              disabled={!date.trim() || !title.trim() || (!!imageUrl && !isValidImageUrl(imageUrl))}
              className="flex-1 py-3 bg-[#007AFF] text-white rounded-xl font-bold hover:bg-[#0056CC] disabled:bg-gray-200 disabled:text-gray-400 disabled:cursor-not-allowed transition-all shadow-apple"
            >
              {isNew ? '추가' : '저장'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
