'use client'

import { useState } from 'react'
import type { TimelineItem, TransitionType, VideoSettings } from '../types'
import { TRANSITION_OPTIONS } from '../types'

interface VideoGenerationModalProps {
  isOpen: boolean
  onClose: () => void
  items: TimelineItem[]
  onGenerate: (imageUrls: string[], settings: VideoSettings) => Promise<void>
  isGenerating: boolean
  videoUrl?: string | null
}

export function VideoGenerationModal({
  isOpen,
  onClose,
  items,
  onGenerate,
  isGenerating,
  videoUrl,
}: VideoGenerationModalProps) {
  const [selectedItems, setSelectedItems] = useState<Set<string>>(
    new Set(items.filter((item) => item.imageUrl).map((item) => item.id))
  )
  const [settings, setSettings] = useState<VideoSettings>({
    durationPerImage: 6,
    transition: 'fade',
    transitionDuration: 1,
    resolution: [1280, 720],
  })

  const itemsWithImages = items.filter((item) => item.imageUrl)

  const handleToggleItem = (id: string) => {
    const newSelected = new Set(selectedItems)
    if (newSelected.has(id)) {
      newSelected.delete(id)
    } else {
      newSelected.add(id)
    }
    setSelectedItems(newSelected)
  }

  const handleSelectAll = () => {
    setSelectedItems(new Set(itemsWithImages.map((item) => item.id)))
  }

  const handleDeselectAll = () => {
    setSelectedItems(new Set())
  }

  const handleGenerate = async () => {
    const selectedImages = items
      .filter((item) => selectedItems.has(item.id) && item.imageUrl)
      .map((item) => item.imageUrl!)

    if (selectedImages.length < 2) return
    await onGenerate(selectedImages, settings)
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* 배경 오버레이 */}
      <div
        className="absolute inset-0 bg-black/70 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* 모달 */}
      <div className="relative bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl w-full max-w-2xl max-h-[80vh] overflow-hidden flex flex-col">
        {/* 헤더 */}
        <div className="flex items-center justify-between p-6 border-b border-slate-700">
          <div>
            <h2 className="text-xl font-bold text-white">영상 생성</h2>
            <p className="text-sm text-slate-400 mt-1">
              이미지를 선택하고 설정을 조정하여 영상을 생성합니다
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-2 text-slate-400 hover:text-white hover:bg-slate-700 rounded-lg transition-colors"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* 컨텐츠 */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* 생성된 영상 미리보기 */}
          {videoUrl && (
            <div className="mb-6">
              <h3 className="text-sm font-medium text-slate-300 mb-3">생성된 영상</h3>
              <div className="bg-slate-800 rounded-xl overflow-hidden">
                <video
                  src={videoUrl}
                  controls
                  className="w-full"
                  autoPlay={false}
                />
              </div>
              <div className="flex gap-2 mt-3">
                <a
                  href={videoUrl}
                  download
                  className="flex-1 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-500 transition-colors text-center"
                >
                  다운로드
                </a>
              </div>
            </div>
          )}

          {/* 이미지 선택 */}
          <div>
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-medium text-slate-300">
                이미지 선택 ({selectedItems.size}/{itemsWithImages.length})
              </h3>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={handleSelectAll}
                  className="text-xs text-blue-400 hover:text-blue-300"
                >
                  전체 선택
                </button>
                <button
                  type="button"
                  onClick={handleDeselectAll}
                  className="text-xs text-slate-400 hover:text-slate-300"
                >
                  전체 해제
                </button>
              </div>
            </div>

            {itemsWithImages.length === 0 ? (
              <div className="text-center py-8 bg-slate-800/50 rounded-xl">
                <svg className="w-12 h-12 text-slate-500 mx-auto mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
                <p className="text-slate-400 text-sm">생성된 이미지가 없습니다</p>
                <p className="text-slate-500 text-xs mt-1">먼저 타임라인 항목의 이미지를 생성해주세요</p>
              </div>
            ) : (
              <div className="grid grid-cols-4 gap-2">
                {itemsWithImages.map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => handleToggleItem(item.id)}
                    className={`
                      relative aspect-video rounded-lg overflow-hidden border-2 transition-all
                      ${selectedItems.has(item.id)
                        ? 'border-blue-500 ring-2 ring-blue-500/30'
                        : 'border-transparent hover:border-slate-600'}
                    `}
                  >
                    <img
                      src={item.imageUrl}
                      alt={item.title}
                      className="w-full h-full object-cover"
                    />
                    {selectedItems.has(item.id) && (
                      <div className="absolute top-1 right-1 w-5 h-5 bg-blue-500 rounded-full flex items-center justify-center">
                        <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                        </svg>
                      </div>
                    )}
                    <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-1">
                      <p className="text-white text-[10px] truncate">{item.title}</p>
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* 설정 */}
          <div className="space-y-4">
            <h3 className="text-sm font-medium text-slate-300">영상 설정</h3>

            {/* 이미지당 시간 */}
            <div>
              <label className="block text-xs text-slate-400 mb-2">
                이미지당 표시 시간: {settings.durationPerImage}초
              </label>
              <input
                type="range"
                min={3}
                max={15}
                step={1}
                value={settings.durationPerImage}
                onChange={(e) =>
                  setSettings({ ...settings, durationPerImage: Number(e.target.value) })
                }
                className="w-full accent-blue-500"
              />
              <div className="flex justify-between text-xs text-slate-500 mt-1">
                <span>3초</span>
                <span>15초</span>
              </div>
            </div>

            {/* 전환 효과 */}
            <div>
              <label className="block text-xs text-slate-400 mb-2">전환 효과</label>
              <div className="flex gap-2">
                {TRANSITION_OPTIONS.map((option) => (
                  <button
                    key={option.value}
                    type="button"
                    onClick={() =>
                      setSettings({ ...settings, transition: option.value })
                    }
                    className={`
                      flex-1 py-2 px-3 rounded-lg text-sm font-medium transition-colors
                      ${settings.transition === option.value
                        ? 'bg-blue-600 text-white'
                        : 'bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-white'}
                    `}
                  >
                    {option.label}
                  </button>
                ))}
              </div>
            </div>

            {/* 전환 시간 */}
            {settings.transition !== 'none' && (
              <div>
                <label className="block text-xs text-slate-400 mb-2">
                  전환 시간: {settings.transitionDuration}초
                </label>
                <input
                  type="range"
                  min={0.5}
                  max={3}
                  step={0.5}
                  value={settings.transitionDuration}
                  onChange={(e) =>
                    setSettings({
                      ...settings,
                      transitionDuration: Number(e.target.value),
                    })
                  }
                  className="w-full accent-blue-500"
                />
                <div className="flex justify-between text-xs text-slate-500 mt-1">
                  <span>0.5초</span>
                  <span>3초</span>
                </div>
              </div>
            )}

            {/* 예상 영상 길이 */}
            <div className="p-3 bg-slate-800/50 rounded-lg">
              <p className="text-xs text-slate-400">
                예상 영상 길이:{' '}
                <span className="text-white font-medium">
                  {selectedItems.size * settings.durationPerImage}초
                </span>
              </p>
            </div>
          </div>
        </div>

        {/* 푸터 */}
        <div className="flex items-center justify-end gap-3 p-6 border-t border-slate-700">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 text-slate-400 hover:text-white transition-colors"
          >
            닫기
          </button>
          <button
            type="button"
            onClick={handleGenerate}
            disabled={isGenerating || selectedItems.size < 2}
            className="px-6 py-2 bg-indigo-600 text-white rounded-lg font-medium hover:bg-indigo-500 disabled:bg-slate-700 disabled:text-slate-500 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
          >
            {isGenerating ? (
              <>
                <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                <span>영상 생성 중...</span>
              </>
            ) : (
              <>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
                <span>영상 생성</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  )
}
