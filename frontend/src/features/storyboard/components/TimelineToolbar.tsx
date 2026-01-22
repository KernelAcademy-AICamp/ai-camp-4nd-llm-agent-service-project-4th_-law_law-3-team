'use client'

import type { EditMode } from '../types'

interface TimelineToolbarProps {
  editMode: EditMode
  onToggleEditMode: () => void
  onAddItem: () => void
  onExport: () => void
  onReset: () => void
  onGenerateAllImages: () => void
  onGenerateVideo: () => void
  hasItems: boolean
  hasImages: boolean
  isGeneratingBatch: boolean
  batchProgress?: { current: number; total: number }
}

export function TimelineToolbar({
  editMode,
  onToggleEditMode,
  onAddItem,
  onExport,
  onReset,
  onGenerateAllImages,
  onGenerateVideo,
  hasItems,
  hasImages,
  isGeneratingBatch,
  batchProgress,
}: TimelineToolbarProps) {

  return (
    <div className="flex items-center gap-3 flex-wrap">
      {/* 편집 모드 토글 */}
      <button
        type="button"
        onClick={onToggleEditMode}
        className={`
          px-4 py-2 rounded-xl text-sm font-bold transition-all flex items-center gap-2 border
          ${editMode === 'edit'
            ? 'bg-blue-600 border-blue-500 text-white shadow-lg shadow-blue-500/20'
            : 'bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-700 hover:text-white hover:border-slate-600'}
        `}
      >
        <svg
          className="w-4 h-4"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
          aria-hidden="true"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z"
          />
        </svg>
        {editMode === 'edit' ? '편집 종료' : '편집 모드'}
      </button>

      {/* 항목 추가 (편집 모드에서만) */}
      {editMode === 'edit' && (
        <button
          type="button"
          onClick={onAddItem}
          className="px-4 py-2 bg-emerald-600 border border-emerald-500 text-white rounded-xl text-sm font-bold hover:bg-emerald-500 transition-all shadow-lg shadow-emerald-500/20 flex items-center gap-2"
        >
          <svg
            className="w-4 h-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            aria-hidden="true"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 4v16m8-8H4"
            />
          </svg>
          항목 추가
        </button>
      )}

      {hasItems && (
        <>
          <div className="w-px h-6 bg-slate-700 mx-1" />

          {/* 전체 스토리보드 이미지 생성 */}
          <button
            type="button"
            onClick={onGenerateAllImages}
            disabled={isGeneratingBatch}
            className={`
              px-4 py-2 rounded-xl text-sm font-bold transition-all flex items-center gap-2 border
              ${isGeneratingBatch
                ? 'bg-purple-600 border-purple-500 text-white'
                : 'bg-slate-800 border-slate-700 text-slate-300 hover:bg-purple-600 hover:border-purple-500 hover:text-white'}
              disabled:opacity-70
            `}
          >
            {isGeneratingBatch ? (
              <>
                <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                {batchProgress ? (
                  <span>{batchProgress.current}/{batchProgress.total}</span>
                ) : (
                  <span>생성 중...</span>
                )}
              </>
            ) : (
              <>
                <svg
                  className="w-4 h-4"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                  aria-hidden="true"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                  />
                </svg>
                전체 이미지 생성
              </>
            )}
          </button>

          {/* 영상 생성 */}
          {hasImages && (
            <button
              type="button"
              onClick={onGenerateVideo}
              className="px-4 py-2 bg-slate-800 border border-slate-700 text-slate-300 rounded-xl text-sm font-medium hover:bg-indigo-600 hover:border-indigo-500 hover:text-white transition-all flex items-center gap-2"
            >
              <svg
                className="w-4 h-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                aria-hidden="true"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                />
              </svg>
              영상 생성
            </button>
          )}

          <div className="w-px h-6 bg-slate-700 mx-1" />

          {/* 내보내기 */}
          <button
            type="button"
            onClick={onExport}
            className="px-4 py-2 bg-slate-800 border border-slate-700 text-slate-300 rounded-xl text-sm font-medium hover:bg-slate-700 hover:text-white hover:border-slate-600 transition-all flex items-center gap-2"
          >
            <svg
              className="w-4 h-4"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              aria-hidden="true"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
              />
            </svg>
            내보내기
          </button>

          {/* 초기화 */}
          <button
            type="button"
            onClick={onReset}
            className="px-4 py-2 text-slate-400 hover:text-red-400 hover:bg-red-500/10 border border-transparent hover:border-red-500/20 rounded-xl text-sm font-medium transition-all flex items-center gap-2"
          >
            <svg
              className="w-4 h-4"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              aria-hidden="true"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
              />
            </svg>
            초기화
          </button>
        </>
      )}
    </div>
  )
}
