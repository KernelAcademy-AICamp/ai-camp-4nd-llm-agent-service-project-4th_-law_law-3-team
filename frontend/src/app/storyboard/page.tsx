'use client'

import { useState, useCallback } from 'react'
import dynamic from 'next/dynamic'
import { BackButton } from '@/components/ui/BackButton'
import { useUI } from '@/context/UIContext'
import { TimelineToolbar } from '@/features/storyboard/components/TimelineToolbar'
import { useTimelineState } from '@/features/storyboard/hooks'
import type { TimelineItem, VideoSettings } from '@/features/storyboard/types'

// Dynamic imports for heavy components (reduces initial bundle size)
const MultiInputPanel = dynamic(
  () => import('@/features/storyboard/components/MultiInputPanel').then(m => m.MultiInputPanel),
  { loading: () => <div className="flex h-full items-center justify-center"><div className="h-8 w-8 animate-spin rounded-full border-4 border-blue-500 border-t-transparent" /></div> }
)

const TimelineView = dynamic(
  () => import('@/features/storyboard/components/TimelineView').then(m => m.TimelineView),
  { loading: () => <div className="flex h-full items-center justify-center"><div className="h-8 w-8 animate-spin rounded-full border-4 border-blue-500 border-t-transparent" /></div> }
)

const TimelineItemEditor = dynamic(
  () => import('@/features/storyboard/components/TimelineItemEditor').then(m => m.TimelineItemEditor),
  { ssr: false }
)

const VideoGenerationModal = dynamic(
  () => import('@/features/storyboard/components/VideoGenerationModal').then(m => m.VideoGenerationModal),
  { ssr: false }
)

export default function StoryboardPage() {
  const { isChatOpen, chatMode } = useUI()
  // 입력 패널 접기/펼치기 상태
  const [isInputPanelOpen, setIsInputPanelOpen] = useState(true)
  const {
    items,
    title,
    summary,
    editMode,
    selectedItemId,
    isExtracting,
    extractError,
    generatingImageIds,
    isGeneratingBatch,
    batchProgress,
    itemsWithImagesCount,
    isGeneratingVideo,
    generatedVideoUrl,
    showVideoModal,
    setShowVideoModal,
    extractTimeline,
    extractFromVoice,
    extractFromImage,
    generateItemImage,
    generateAllImages,
    generateVideo,
    setTitle,
    addItem,
    updateItem,
    deleteItem,
    toggleEditMode,
    selectItem,
    exportToJson,
    importFromJson,
    resetTimeline,
  } = useTimelineState()

  // 편집 모달 상태
  const [isEditorOpen, setIsEditorOpen] = useState(false)
  const [editingItem, setEditingItem] = useState<TimelineItem | null>(null)
  const [isNewItem, setIsNewItem] = useState(false)

  // 새 항목 추가
  const handleAddItem = useCallback(() => {
    setEditingItem(null)
    setIsNewItem(true)
    setIsEditorOpen(true)
  }, [])

  // 항목 편집
  const handleEditItem = useCallback(
    (id: string) => {
      const item = items.find((i) => i.id === id)
      if (item) {
        setEditingItem(item)
        setIsNewItem(false)
        setIsEditorOpen(true)
      }
    },
    [items]
  )

  // 편집 저장
  const handleSaveEdit = useCallback(
    (data: Omit<TimelineItem, 'id' | 'order'>) => {
      if (isNewItem) {
        addItem(data)
      } else if (editingItem) {
        updateItem(editingItem.id, data)
      }
      setIsEditorOpen(false)
      setEditingItem(null)
    },
    [isNewItem, editingItem, addItem, updateItem]
  )

  // 편집 취소
  const handleCancelEdit = useCallback(() => {
    setIsEditorOpen(false)
    setEditingItem(null)
  }, [])

  // 스토리보드 이미지 생성
  const handleGenerateImage = useCallback(
    (id: string) => {
      generateItemImage(id)
    },
    [generateItemImage]
  )

  // 전체 스토리보드 이미지 생성
  const handleGenerateAllImages = useCallback(() => {
    generateAllImages()
  }, [generateAllImages])

  // 영상 생성 모달 열기
  const handleOpenVideoModal = useCallback(() => {
    setShowVideoModal(true)
  }, [setShowVideoModal])

  // 영상 생성
  const handleGenerateVideo = useCallback(
    async (imageUrls: string[], settings: VideoSettings) => {
      await generateVideo(imageUrls, settings)
    },
    [generateVideo]
  )

  return (
    <div
      className={`h-screen flex flex-col bg-slate-950 overflow-hidden relative transition-all duration-500 ease-in-out ${
        isChatOpen && chatMode === 'split' ? 'w-1/2 border-r border-slate-800' : 'w-full'
      }`}
    >
      {/* Background Gradients (Global Effect) */}
      <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-blue-600/10 rounded-full blur-[120px] pointer-events-none" />
      <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-indigo-600/10 rounded-full blur-[120px] pointer-events-none" />

      {/* 헤더 */}
      <header className="flex-shrink-0 bg-slate-900/50 backdrop-blur-xl border-b border-white/5 px-8 py-4 relative z-20">
        <div className="flex items-center justify-between max-w-[1920px] mx-auto">
          <div className="flex items-center gap-4">
            <BackButton />
            <div>
              <h1 className="text-2xl font-bold text-white tracking-tight flex items-center gap-2">
                <span className="text-3xl">🎞️</span>
                스토리보드
                <span className="text-xs px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400 font-medium border border-blue-500/20">BETA</span>
              </h1>
              <p className="text-sm text-slate-400 mt-1 font-light">
                사건의 흐름을 시각화하고 AI로 이미지와 영상을 생성합니다
              </p>
            </div>
          </div>
          {summary && (
            <div className="text-right hidden md:block">
              <span className="text-xs font-bold text-blue-500 uppercase tracking-wider">Case Summary</span>
              <p className="text-sm font-medium text-slate-300 max-w-xl truncate">{summary}</p>
            </div>
          )}
        </div>
      </header>

      {/* 메인 컨텐츠 */}
      <div className="flex-1 flex overflow-hidden relative z-10">
        {/* 왼쪽 패널: 멀티모달 입력 (접기/펼치기) */}
        <div
          className={`
            flex-shrink-0 transition-all duration-300 ease-in-out relative
            ${isInputPanelOpen ? 'w-96' : 'w-0'}
          `}
        >
          <div className={`
            absolute inset-0 overflow-hidden
            ${isInputPanelOpen ? 'opacity-100' : 'opacity-0 pointer-events-none'}
            transition-opacity duration-300
          `}>
            <MultiInputPanel
              onExtractText={extractTimeline}
              onExtractVoice={extractFromVoice}
              onExtractImage={extractFromImage}
              onImport={importFromJson}
              isExtracting={isExtracting}
              error={extractError}
            />
          </div>
        </div>

        {/* 패널 토글 버튼 */}
        <button
          type="button"
          onClick={() => setIsInputPanelOpen(!isInputPanelOpen)}
          className={`
            flex-shrink-0 w-6 flex items-center justify-center
            bg-slate-800/50 hover:bg-slate-700/50 border-r border-white/5
            transition-colors group
          `}
          title={isInputPanelOpen ? '입력 패널 접기' : '입력 패널 펼치기'}
        >
          <svg
            className={`w-4 h-4 text-slate-500 group-hover:text-slate-300 transition-all duration-300 ${isInputPanelOpen ? '' : 'rotate-180'}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
        </button>

        {/* 오른쪽 패널: 타임라인 */}
        <div className="flex-1 flex flex-col min-w-0 bg-transparent">
          <div className="max-w-5xl mx-auto w-full h-full flex flex-col">
            {/* 상단 툴바 영역 */}
            <div className="flex-shrink-0 px-8 py-6 flex items-center justify-between gap-4">
              {/* 타이틀 편집 */}
              <div className="flex-1">
                {items.length > 0 ? (
                  <input
                    type="text"
                    value={title}
                    onChange={(e) => setTitle(e.target.value)}
                    className="w-full text-3xl font-bold bg-transparent border-none text-white focus:ring-0 placeholder-slate-700"
                    placeholder="무제 타임라인"
                  />
                ) : (
                  <h2 className="text-3xl font-bold text-slate-700">새 타임라인</h2>
                )}
              </div>

              {/* 툴바 (버튼들) */}
              <TimelineToolbar
                editMode={editMode}
                onToggleEditMode={toggleEditMode}
                onAddItem={handleAddItem}
                onExport={exportToJson}
                onReset={resetTimeline}
                onGenerateAllImages={handleGenerateAllImages}
                onGenerateVideo={handleOpenVideoModal}
                hasItems={items.length > 0}
                hasImages={itemsWithImagesCount >= 2}
                isGeneratingBatch={isGeneratingBatch}
                batchProgress={batchProgress}
              />
            </div>

            {/* 타임라인 뷰 */}
            <TimelineView
              items={items}
              editMode={editMode}
              selectedItemId={selectedItemId}
              onItemSelect={selectItem}
              onItemEdit={handleEditItem}
              onItemDelete={deleteItem}
              onItemGenerateImage={handleGenerateImage}
              generatingImageIds={generatingImageIds}
            />
          </div>
        </div>
      </div>

      {/* 편집 모달 */}
      {isEditorOpen && (
        <TimelineItemEditor
          item={editingItem}
          onSave={handleSaveEdit}
          onCancel={handleCancelEdit}
          isNew={isNewItem}
        />
      )}

      {/* 영상 생성 모달 */}
      <VideoGenerationModal
        isOpen={showVideoModal}
        onClose={() => setShowVideoModal(false)}
        items={items}
        onGenerate={handleGenerateVideo}
        isGenerating={isGeneratingVideo}
        videoUrl={generatedVideoUrl}
      />
    </div>
  )
}
