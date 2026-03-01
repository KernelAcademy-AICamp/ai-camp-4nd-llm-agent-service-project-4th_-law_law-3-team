'use client'

import { useCallback, useRef, useState } from 'react'
import type { BatchAnalysisResult, EvidenceFile, EvidenceType, MergeConflict, MergeTimelineResponse, TimelineItem } from '../types'
import { useEvidenceUpload } from '../hooks/useEvidenceUpload'
import { api, endpoints } from '@/lib/api'

const STORYBOARD_BASE = endpoints.storyboard

interface EvidenceUploadPanelProps {
  onUploadComplete: (result: BatchAnalysisResult) => void
  existingEvidence: EvidenceFile[]
  // 병합 모드 (기존 타임라인에 추가)
  mergeMode?: boolean
  existingItems?: TimelineItem[]
  onMergeComplete?: (response: MergeTimelineResponse) => void
}

const EVIDENCE_TYPE_ICONS: Record<EvidenceType, { icon: string; label: string }> = {
  kakao_txt: { icon: '📱', label: '카카오톡' },
  messenger_screenshot: { icon: '💬', label: '메신저 캡처' },
  voice_recording: { icon: '🎤', label: '음성 녹음' },
  document: { icon: '📄', label: '문서' },
  photo: { icon: '📷', label: '사진' },
  text_input: { icon: '✏️', label: '텍스트' },
  other: { icon: '📎', label: '기타' },
}

const MAX_FILES = 10

// 파일 확장자로 증거 유형 추정
function guessEvidenceType(file: File): EvidenceType {
  const name = file.name.toLowerCase()
  const mime = file.type.toLowerCase()

  if (name.endsWith('.txt')) return 'kakao_txt'
  if (mime.startsWith('audio/')) return 'voice_recording'
  if (mime.startsWith('image/')) return 'messenger_screenshot'
  if (mime === 'application/pdf' || name.endsWith('.docx') || name.endsWith('.doc')) return 'document'
  return 'other'
}

interface PendingFile {
  file: File
  evidenceType: EvidenceType
  id: string
}

export function EvidenceUploadPanel({
  onUploadComplete,
  existingEvidence,
  mergeMode = false,
  existingItems = [],
  onMergeComplete,
}: EvidenceUploadPanelProps) {
  const [pendingFiles, setPendingFiles] = useState<PendingFile[]>([])
  const [isDragOver, setIsDragOver] = useState(false)
  const [mergeConflicts, setMergeConflicts] = useState<MergeConflict[]>([])
  const [isMerging, setIsMerging] = useState(false)
  const [mergeError, setMergeError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const { uploadFiles, isUploading, progress, error, reset } = useEvidenceUpload()

  const addFiles = useCallback((newFiles: File[]) => {
    const remaining = MAX_FILES - pendingFiles.length
    const toAdd = newFiles.slice(0, remaining).map((file) => ({
      file,
      evidenceType: guessEvidenceType(file),
      id: `${file.name}-${Date.now()}-${Math.random()}`,
    }))
    setPendingFiles((prev) => [...prev, ...toAdd])
  }, [pendingFiles.length])

  const handleFileInputChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files ?? [])
    addFiles(files)
    // input 초기화 (같은 파일 재선택 허용)
    event.target.value = ''
  }, [addFiles])

  const handleDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault()
    setIsDragOver(true)
  }, [])

  const handleDragLeave = useCallback(() => {
    setIsDragOver(false)
  }, [])

  const handleDrop = useCallback((event: React.DragEvent) => {
    event.preventDefault()
    setIsDragOver(false)
    const files = Array.from(event.dataTransfer.files)
    addFiles(files)
  }, [addFiles])

  const removeFile = useCallback((id: string) => {
    setPendingFiles((prev) => prev.filter((f) => f.id !== id))
  }, [])

  const handleUpload = useCallback(async () => {
    if (pendingFiles.length === 0 || isUploading) return

    const result = await uploadFiles(pendingFiles.map((p) => p.file))
    if (result) {
      onUploadComplete(result)
      setPendingFiles([])
      reset()
    }
  }, [pendingFiles, isUploading, uploadFiles, onUploadComplete, reset])

  // 병합 모드: POST /merge
  const handleMerge = useCallback(async () => {
    if (pendingFiles.length === 0 || isMerging) return

    setIsMerging(true)
    setMergeError(null)
    setMergeConflicts([])

    try {
      const formData = new FormData()
      // MergeTimelineRequest: { existing_items, existing_evidence } 객체로 전송
      const mergeRequest = {
        existing_items: existingItems,
        existing_evidence: existingEvidence,
      }
      formData.append('existing_timeline', JSON.stringify(mergeRequest))
      for (const pf of pendingFiles) {
        formData.append('files', pf.file)
      }

      const response = await api.post<{
        success: boolean
        merged_items?: TimelineItem[]
        merged_evidence?: EvidenceFile[]
        merge_report?: {
          new_items_added: number
          duplicates_detected: number
          items_updated: number
          conflicts: Array<{
            existing_item_id: string
            new_item_id: string
            conflict_type: string
            description: string
          }>
        }
        error?: string
      }>(
        `${STORYBOARD_BASE}/merge`,
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' } },
      )

      const data = response.data
      if (!data.success) {
        setMergeError(data.error ?? '병합 실패')
        return
      }

      const conflicts: MergeConflict[] = (data.merge_report?.conflicts ?? []).map((c) => ({
        existingItemId: c.existing_item_id,
        newItemId: c.new_item_id,
        conflictType: c.conflict_type as MergeConflict['conflictType'],
        description: c.description,
      }))

      if (conflicts.length > 0) {
        setMergeConflicts(conflicts)
      }

      if (onMergeComplete) {
        onMergeComplete({
          success: data.success,
          mergedItems: data.merged_items ?? [],
          mergedEvidence: data.merged_evidence ?? [],
          mergeReport: data.merge_report
            ? {
                newItemsAdded: data.merge_report.new_items_added,
                duplicatesDetected: data.merge_report.duplicates_detected,
                itemsUpdated: data.merge_report.items_updated,
                conflicts,
              }
            : undefined,
        })
      }

      setPendingFiles([])
      reset()
    } catch (err) {
      setMergeError(err instanceof Error ? err.message : '병합 요청 실패')
    } finally {
      setIsMerging(false)
    }
  }, [pendingFiles, isMerging, existingItems, existingEvidence, onMergeComplete, reset])

  const handleReset = useCallback(() => {
    setPendingFiles([])
    setMergeConflicts([])
    setMergeError(null)
    reset()
  }, [reset])

  const isLoading = isUploading || isMerging
  const displayError = error ?? mergeError

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-5 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-bold text-gray-800">
          {mergeMode ? '추가 증거 병합' : '증거 파일 업로드'}
        </h3>
        {existingEvidence.length > 0 && (
          <span className="text-xs text-gray-400">
            기존 {existingEvidence.length}개 등록됨
          </span>
        )}
      </div>

      {/* 병합 모드 안내 */}
      {mergeMode && existingItems.length > 0 && (
        <div className="flex items-center gap-2 text-xs text-blue-600 bg-blue-50 border border-blue-200 rounded-lg px-3 py-2">
          <svg className="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          기존 {existingItems.length}개 항목에 새 증거를 병합합니다
        </div>
      )}

      {/* 드래그앤드롭 영역 */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        role="button"
        tabIndex={0}
        aria-label="파일 업로드 영역. 클릭하거나 파일을 드래그하세요"
        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') fileInputRef.current?.click() }}
        className={`
          border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-colors
          ${isDragOver
            ? 'border-blue-400 bg-blue-50'
            : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'}
          ${pendingFiles.length >= MAX_FILES ? 'opacity-50 pointer-events-none' : ''}
        `}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".txt,.pdf,.doc,.docx,.jpg,.jpeg,.png,.gif,.webp,.mp3,.mp4,.wav,.m4a"
          onChange={handleFileInputChange}
          className="hidden"
          aria-hidden="true"
        />
        <div className="flex flex-col items-center gap-2">
          <div className="flex gap-2 text-2xl">
            <span>📱</span>
            <span>🎤</span>
            <span>📄</span>
            <span>📷</span>
          </div>
          <p className="text-sm font-medium text-gray-600">
            파일을 드래그하거나 클릭하여 선택
          </p>
          <p className="text-xs text-gray-400">
            카카오톡 .txt, 이미지, 음성, 문서 (최대 {MAX_FILES}개)
          </p>
        </div>
      </div>

      {/* 선택된 파일 목록 */}
      {pendingFiles.length > 0 && (
        <ul className="space-y-2">
          {pendingFiles.map((pf) => {
            const typeInfo = EVIDENCE_TYPE_ICONS[pf.evidenceType]
            return (
              <li key={pf.id} className="flex items-center gap-3 px-3 py-2 bg-gray-50 rounded-lg">
                <span className="text-xl flex-shrink-0">{typeInfo.icon}</span>
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-medium text-gray-800 truncate">{pf.file.name}</p>
                  <p className="text-xs text-gray-400">
                    {typeInfo.label} · {(pf.file.size / 1024).toFixed(0)} KB
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => removeFile(pf.id)}
                  aria-label={`${pf.file.name} 제거`}
                  disabled={isLoading}
                  className="flex-shrink-0 p-1 rounded hover:bg-red-50 text-gray-400 hover:text-red-500 transition-colors disabled:opacity-40"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </li>
            )
          })}
        </ul>
      )}

      {/* SSE 진행률 */}
      {isUploading && progress && (
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs text-gray-500">
            <span>
              {progress.currentFile
                ? `분석 중: ${progress.currentFile}`
                : progress.message || '분석 중...'}
            </span>
            <span>{progress.currentFileIndex}/{progress.totalFiles}</span>
          </div>
          <div className="w-full bg-gray-100 rounded-full h-2 overflow-hidden">
            <div
              className="h-2 bg-blue-500 rounded-full transition-all duration-300"
              style={{ width: `${progress.progress}%` }}
              role="progressbar"
              aria-valuenow={progress.progress}
              aria-valuemin={0}
              aria-valuemax={100}
            />
          </div>
        </div>
      )}

      {/* 병합 로딩 */}
      {isMerging && (
        <div className="flex items-center gap-2 text-sm text-gray-500">
          <svg className="animate-spin h-4 w-4 text-blue-500" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
          </svg>
          병합 중...
        </div>
      )}

      {/* 로딩 (진행률 없는 초기 상태) */}
      {isUploading && !progress && (
        <div className="flex items-center gap-2 text-sm text-gray-500">
          <svg className="animate-spin h-4 w-4 text-blue-500" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
          </svg>
          업로드 중...
        </div>
      )}

      {/* 에러 */}
      {displayError && (
        <div className="flex items-center gap-2 text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg px-3 py-2">
          <svg className="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
          </svg>
          {displayError}
        </div>
      )}

      {/* 충돌 목록 */}
      {mergeConflicts.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-xs font-bold text-amber-700 flex items-center gap-1">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
            </svg>
            병합 충돌 {mergeConflicts.length}건 감지
          </h4>
          <ul className="space-y-1">
            {mergeConflicts.map((conflict) => (
              <li
                key={`${conflict.existingItemId}-${conflict.newItemId}`}
                className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2"
              >
                <span className="font-semibold">
                  {conflict.conflictType === 'date_overlap' ? '날짜 겹침' : '내용 모순'}:
                </span>{' '}
                {conflict.description}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* 액션 버튼 */}
      {pendingFiles.length > 0 && !isLoading && (
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={mergeMode ? handleMerge : handleUpload}
            className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-xl text-sm font-bold hover:bg-blue-700 transition-colors"
          >
            {mergeMode
              ? `${pendingFiles.length}개 파일 병합`
              : `${pendingFiles.length}개 파일 분석 시작`}
          </button>
          <button
            type="button"
            onClick={handleReset}
            className="px-3 py-2 text-gray-500 hover:text-red-500 hover:bg-red-50 rounded-xl text-sm transition-colors"
            aria-label="선택 초기화"
          >
            초기화
          </button>
        </div>
      )}
    </div>
  )
}
