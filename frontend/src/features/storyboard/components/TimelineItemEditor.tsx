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
  }, [item])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!date.trim() || !title.trim()) return

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
    <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 backdrop-blur-sm">
      <div className="bg-slate-900 rounded-2xl shadow-2xl w-full max-w-md mx-4 border border-slate-700">
        <div className="p-6 border-b border-slate-700">
          <h3 className="text-xl font-bold text-white">
            {isNew ? '새 항목 추가' : '항목 편집'}
          </h3>
        </div>

        <form onSubmit={handleSubmit} className="p-6 space-y-5">
          <div className="space-y-4">
            <div>
              <label htmlFor="date" className="block text-sm font-medium text-slate-400 mb-1.5">
                날짜
              </label>
              <input
                id="date"
                type="text"
                value={date}
                onChange={(e) => setDate(e.target.value)}
                placeholder="예: 2024-01-15 또는 2024년 1월 초"
                className="w-full px-4 py-2.5 bg-slate-800 border border-slate-600 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                required
              />
            </div>

            <div>
              <label htmlFor="title" className="block text-sm font-medium text-slate-400 mb-1.5">
                제목
              </label>
              <input
                id="title"
                type="text"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="이벤트 제목"
                className="w-full px-4 py-2.5 bg-slate-800 border border-slate-600 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                required
              />
            </div>

            <div>
              <label htmlFor="description" className="block text-sm font-medium text-slate-400 mb-1.5">
                설명
              </label>
              <textarea
                id="description"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="이벤트 상세 설명"
                rows={3}
                className="w-full px-4 py-2.5 bg-slate-800 border border-slate-600 rounded-xl text-white placeholder-slate-500 resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
              />
            </div>

            <div>
              <label htmlFor="imageUrl" className="block text-sm font-medium text-slate-400 mb-1.5">
                이미지 URL (선택)
              </label>
              <input
                id="imageUrl"
                type="text"
                value={imageUrl}
                onChange={(e) => setImageUrl(e.target.value)}
                placeholder="https://example.com/image.jpg"
                className="w-full px-4 py-2.5 bg-slate-800 border border-slate-600 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
              />
            </div>

            <div>
              <label htmlFor="participants" className="block text-sm font-medium text-slate-400 mb-1.5">
                관련자 (쉼표로 구분)
              </label>
              <input
                id="participants"
                type="text"
                value={participantsText}
                onChange={(e) => setParticipantsText(e.target.value)}
                placeholder="예: 홍길동, 김철수, A회사"
                className="w-full px-4 py-2.5 bg-slate-800 border border-slate-600 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
              />
            </div>
          </div>

          <div className="flex gap-3 pt-4">
            <button
              type="button"
              onClick={onCancel}
              className="flex-1 py-3 border border-slate-600 rounded-xl text-slate-300 font-medium hover:bg-slate-800 transition-colors"
            >
              취소
            </button>
            <button
              type="submit"
              disabled={!date.trim() || !title.trim()}
              className="flex-1 py-3 bg-blue-600 text-white rounded-xl font-bold hover:bg-blue-500 disabled:bg-slate-700 disabled:text-slate-500 disabled:cursor-not-allowed transition-all shadow-lg shadow-blue-900/30"
            >
              {isNew ? '추가' : '저장'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
