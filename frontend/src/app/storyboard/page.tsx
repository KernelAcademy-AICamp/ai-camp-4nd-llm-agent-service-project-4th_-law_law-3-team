'use client'

import { useState, useCallback } from 'react'
import dynamic from 'next/dynamic'
import { BackButton } from '@/components/ui/BackButton'
import { useUI } from '@/context/UIContext'
import { useChat } from '@/context/ChatContext'
import { TimelineToolbar } from '@/features/storyboard/components/TimelineToolbar'
import { useTimelineState } from '@/features/storyboard/hooks'
import type { TimelineItem, ViewMode } from '@/features/storyboard/types'

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

const GanttChartView = dynamic(
  () => import('@/features/storyboard/components/GanttChartView').then(m => m.GanttChartView),
  { ssr: false, loading: () => <div className="flex h-full items-center justify-center"><div className="h-8 w-8 animate-spin rounded-full border-4 border-blue-500 border-t-transparent" /></div> }
)

const GanttDetailPanel = dynamic(
  () => import('@/features/storyboard/components/GanttDetailPanel').then(m => m.GanttDetailPanel),
  { ssr: false }
)

const VideoGenerationModal = dynamic(
  () => import('@/features/storyboard/components/VideoGenerationModal').then(m => m.VideoGenerationModal),
  { ssr: false }
)

export default function StoryboardPage() {
  const { isChatOpen, chatMode } = useUI()
  const { resetSession } = useChat()
  // 뷰 모드 (카드 / 간트)
  const [viewMode, setViewMode] = useState<ViewMode>('card')
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

  // 새 사건 타임라인 만들기 (타임라인 초기화 + 채팅 새 대화)
  const handleNewTimeline = useCallback(() => {
    resetTimeline()
    resetSession()
  }, [resetTimeline, resetSession])

  // 영상 생성 모달 열기
  const handleOpenVideoModal = useCallback(() => {
    setShowVideoModal(true)
  }, [setShowVideoModal])

  return (
    <div
      className={`h-screen flex flex-col bg-[#F5F5F7] overflow-hidden relative transition-all duration-500 ease-in-out ${
        isChatOpen && chatMode === 'split' ? 'w-1/2 border-r border-black/[0.06]' : 'w-full'
      }`}
    >
      {/* 헤더 */}
      <header className="flex-shrink-0 bg-white/80 backdrop-blur-xl border-b border-black/[0.06] px-8 py-4 relative z-20">
        <div className="flex items-center justify-between max-w-[1920px] mx-auto">
          <div className="flex items-center gap-4">
            <BackButton />
            <div>
              <h1 className="text-2xl font-bold text-[#1D1D1F] tracking-tight flex items-center gap-2">
                스토리보드
                <span className="text-xs px-2 py-0.5 rounded-full bg-[#007AFF]/10 text-[#007AFF] font-medium border border-[#007AFF]/20">BETA</span>
              </h1>
            </div>
          </div>
          <div className="flex items-center gap-4">
            {summary && (
              <div className="text-right hidden md:block">
                <span className="text-xs font-bold text-[#007AFF] uppercase tracking-wider">Case Summary</span>
                <p className="text-sm font-medium text-[#3C3C43] max-w-xl truncate">{summary}</p>
              </div>
            )}
            {items.length > 0 && (
              <button
                type="button"
                onClick={handleNewTimeline}
                className="flex-shrink-0 px-4 py-2 bg-white border border-black/[0.08] rounded-xl text-sm font-medium text-[#1D1D1F] hover:bg-[#F5F5F7] transition-colors shadow-sm"
              >
                새 사건 타임라인 만들기
              </button>
            )}
          </div>
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
            bg-[#F5F5F7] hover:bg-gray-200 border-r border-black/[0.06]
            transition-colors group
          `}
          title={isInputPanelOpen ? '입력 패널 접기' : '입력 패널 펼치기'}
        >
          <svg
            className={`w-4 h-4 text-[#86868B] group-hover:text-[#3C3C43] transition-all duration-300 ${isInputPanelOpen ? '' : 'rotate-180'}`}
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
                    className="w-full text-3xl font-bold bg-transparent border-none text-[#1D1D1F] focus:ring-0 placeholder-gray-300"
                    placeholder="무제 타임라인"
                  />
                ) : (
                  <h2 className="text-3xl font-bold text-gray-300">새 타임라인</h2>
                )}
              </div>

              {/* 툴바 (버튼들) */}
              <TimelineToolbar
                editMode={editMode}
                onToggleEditMode={toggleEditMode}
                onAddItem={handleAddItem}
                onExport={exportToJson}
                onReset={resetTimeline}
                onGenerateAllImages={generateAllImages}
                onGenerateVideo={handleOpenVideoModal}
                hasItems={items.length > 0}
                hasImages={itemsWithImagesCount >= 2}
                isGeneratingBatch={isGeneratingBatch}
                batchProgress={batchProgress}
                viewMode={viewMode}
                onViewModeChange={setViewMode}
              />
            </div>

            {/* 타임라인 뷰 */}
            {viewMode === 'gantt' ? (
              <div className="flex-1 flex overflow-hidden">
                <div className="flex-1 min-w-0">
                  <GanttChartView
                    items={items}
                    evidenceFiles={[]}
                    onItemSelect={(item) => selectItem(item.id)}
                    onEvidenceClick={() => {}}
                  />
                </div>
                {selectedItemId && (
                  <GanttDetailPanel
                    selectedItem={items.find(i => i.id === selectedItemId) ?? null}
                    evidenceFiles={[]}
                    onClose={() => selectItem('')}
                    onEvidenceClick={() => {}}
                  />
                )}
              </div>
            ) : (
              <TimelineView
                items={items}
                editMode={editMode}
                selectedItemId={selectedItemId}
                onItemSelect={selectItem}
                onItemEdit={handleEditItem}
                onItemDelete={deleteItem}
                onItemGenerateImage={generateItemImage}
                generatingImageIds={generatingImageIds}
              />
            )}
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
        onGenerate={generateVideo}
        isGenerating={isGeneratingVideo}
        videoUrl={generatedVideoUrl}
      />
    </div>
  )
}
