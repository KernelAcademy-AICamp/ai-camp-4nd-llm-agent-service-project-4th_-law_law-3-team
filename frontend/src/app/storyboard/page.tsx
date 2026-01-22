'use client'

import { useState, useCallback } from 'react'
import { BackButton } from '@/components/ui/BackButton'
import {
  TextInputPanel,
  TimelineView,
  TimelineToolbar,
  TimelineItemEditor,
} from '@/features/storyboard/components'
import { useTimelineState } from '@/features/storyboard/hooks'
import type { TimelineItem } from '@/features/storyboard/types'

export default function StoryboardPage() {
  const {
    items,
    title,
    summary,
    editMode,
    selectedItemId,
    isExtracting,
    extractError,
    extractTimeline,
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

  return (
    <div className="h-screen flex flex-col bg-slate-950 overflow-hidden relative">
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
                사건의 흐름을 시각화하고 AI로 이미지를 생성합니다
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
        {/* 왼쪽 패널: 텍스트 입력 */}
        <TextInputPanel
          onExtract={extractTimeline}
          onImport={importFromJson}
          isExtracting={isExtracting}
          error={extractError}
        />

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
                hasItems={items.length > 0}
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
    </div>
  )
}
