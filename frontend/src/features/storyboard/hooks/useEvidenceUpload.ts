'use client'

import { useState, useCallback, useRef } from 'react'
import { api, endpoints } from '@/lib/api'
import type { BatchAnalysisResult, BatchJobProgress, EvidenceFile, EvidenceType, TimelineItem } from '../types'

const BASE_URL = endpoints.storyboard

const VALID_EVIDENCE_TYPES: EvidenceType[] = [
  'kakao_txt', 'messenger_screenshot', 'voice_recording',
  'document', 'photo', 'text_input', 'other',
]

function toEvidenceType(value: unknown): EvidenceType {
  if (typeof value === 'string' && (VALID_EVIDENCE_TYPES as string[]).includes(value)) {
    return value as EvidenceType
  }
  return 'other'
}

function transformEvidenceFile(raw: Record<string, unknown>): EvidenceFile {
  const extractedIds = Array.isArray(raw.extracted_timeline_ids)
    ? (raw.extracted_timeline_ids as unknown[]).filter((v): v is string => typeof v === 'string')
    : []
  const tags = Array.isArray(raw.tags)
    ? (raw.tags as unknown[]).filter((v): v is string => typeof v === 'string')
    : []

  return {
    evidenceId: typeof raw.evidence_id === 'string' ? raw.evidence_id : '',
    evidenceType: toEvidenceType(raw.evidence_type),
    filename: typeof raw.filename === 'string' ? raw.filename : '',
    uploadedAt: typeof raw.uploaded_at === 'string' ? raw.uploaded_at : '',
    fileSizeKb: typeof raw.file_size_kb === 'number' ? raw.file_size_kb : 0,
    fileHash: typeof raw.file_hash === 'string' ? raw.file_hash : undefined,
    sessionId: typeof raw.session_id === 'string' ? raw.session_id : '',
    extractedTimelineIds: extractedIds,
    tags,
    sourceDescription: typeof raw.source_description === 'string' ? raw.source_description : undefined,
  }
}

function transformTimelineItemFromBatch(raw: Record<string, unknown>): TimelineItem {
  const participants = Array.isArray(raw.participants)
    ? (raw.participants as unknown[]).filter((v): v is string => typeof v === 'string')
    : []
  const evidenceIds = Array.isArray(raw.evidence_ids)
    ? (raw.evidence_ids as unknown[]).filter((v): v is string => typeof v === 'string')
    : []

  return {
    id: typeof raw.id === 'string' ? raw.id : '',
    date: typeof raw.date === 'string' ? raw.date : '날짜 미상',
    title: typeof raw.title === 'string' ? raw.title : '제목 없음',
    description: typeof raw.description === 'string' ? raw.description : '',
    participants,
    order: typeof raw.order === 'number' ? raw.order : 0,
    topic: typeof raw.topic === 'string' ? raw.topic : undefined,
    dateStart: typeof raw.date_start === 'string' ? raw.date_start : undefined,
    dateEnd: typeof raw.date_end === 'string' ? raw.date_end : undefined,
    evidenceIds,
    confidence: typeof raw.confidence === 'number' ? raw.confidence : undefined,
    descriptionShort: typeof raw.description_short === 'string' ? raw.description_short : undefined,
    descriptionDetailed: typeof raw.description_detailed === 'string' ? raw.description_detailed : undefined,
    legalSignificance: typeof raw.legal_significance === 'string' ? raw.legal_significance : undefined,
  }
}

function transformBatchResult(raw: Record<string, unknown>): BatchAnalysisResult {
  const timelineItems = Array.isArray(raw.timeline_items)
    ? (raw.timeline_items as Record<string, unknown>[]).map(transformTimelineItemFromBatch)
    : []
  const evidenceFiles = Array.isArray(raw.evidence_files)
    ? (raw.evidence_files as Record<string, unknown>[]).map(transformEvidenceFile)
    : []
  const topics = Array.isArray(raw.topics)
    ? (raw.topics as unknown[]).filter((v): v is string => typeof v === 'string')
    : []
  const failedFiles = Array.isArray(raw.failed_files)
    ? (raw.failed_files as unknown[]).filter((v): v is string => typeof v === 'string')
    : []

  return {
    timelineItems,
    evidenceFiles,
    topics,
    summary: typeof raw.summary === 'string' ? raw.summary : undefined,
    totalFiles: typeof raw.total_files === 'number' ? raw.total_files : 0,
    successCount: typeof raw.success_count === 'number' ? raw.success_count : 0,
    failedFiles,
  }
}

interface UseEvidenceUploadReturn {
  uploadFiles: (files: File[], context?: string) => Promise<BatchAnalysisResult | null>
  isUploading: boolean
  progress: BatchJobProgress | null
  error: string | null
  reset: () => void
}

export function useEvidenceUpload(): UseEvidenceUploadReturn {
  const [isUploading, setIsUploading] = useState(false)
  const [progress, setProgress] = useState<BatchJobProgress | null>(null)
  const [error, setError] = useState<string | null>(null)
  const eventSourceRef = useRef<EventSource | null>(null)

  const reset = useCallback(() => {
    setIsUploading(false)
    setProgress(null)
    setError(null)
    if (eventSourceRef.current) {
      eventSourceRef.current.close()
      eventSourceRef.current = null
    }
  }, [])

  const uploadFiles = useCallback(async (
    files: File[],
    context: string = '',
  ): Promise<BatchAnalysisResult | null> => {
    if (files.length === 0) return null

    setIsUploading(true)
    setError(null)
    setProgress(null)

    try {
      // 1. FormData 구성 + POST /analyze-batch
      const formData = new FormData()
      for (const file of files) {
        formData.append('files', file)
      }
      formData.append('context', context)

      const response = await api.post<{ success: boolean; job_id?: string; error?: string }>(
        `${BASE_URL}/analyze-batch`,
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' } },
      )

      if (!response.data.success || !response.data.job_id) {
        throw new Error(response.data.error ?? '배치 분석 요청 실패')
      }

      const jobId = response.data.job_id

      // 2. SSE /jobs/{job_id}/status 구독 → Promise로 래핑
      return await new Promise<BatchAnalysisResult | null>((resolve, reject) => {
        const eventSource = new EventSource(`/api${BASE_URL}/jobs/${jobId}/status`)
        eventSourceRef.current = eventSource

        const handleProgressEvent = (event: MessageEvent) => {
          try {
            const data = JSON.parse(event.data as string) as Record<string, unknown>

            const jobProgress: BatchJobProgress = {
              jobId: typeof data.job_id === 'string' ? data.job_id : jobId,
              status: typeof data.status === 'string' ? data.status : 'processing',
              progress: typeof data.progress === 'number' ? data.progress : 0,
              currentFile: typeof data.current_file === 'string' ? data.current_file : undefined,
              currentFileIndex: typeof data.current_file_index === 'number' ? data.current_file_index : 0,
              totalFiles: typeof data.total_files === 'number' ? data.total_files : files.length,
              message: typeof data.message === 'string' ? data.message : '',
            }
            setProgress(jobProgress)

            if (data.status === 'completed' && data.result) {
              eventSource.close()
              eventSourceRef.current = null
              setIsUploading(false)
              const result = transformBatchResult(data.result as Record<string, unknown>)
              resolve(result)
            } else if (data.status === 'failed') {
              eventSource.close()
              eventSourceRef.current = null
              const errorMsg = typeof data.error === 'string' ? data.error : '분석 실패'
              setError(errorMsg)
              setIsUploading(false)
              reject(new Error(errorMsg))
            }
          } catch {
            // JSON 파싱 실패 시 무시
          }
        }

        // Backend가 named event("progress")로 전송하므로 addEventListener 사용
        // onmessage는 unnamed event(event: "message")만 수신
        eventSource.addEventListener('progress', handleProgressEvent)
        // fallback: unnamed event도 처리 (방어적 코딩)
        eventSource.onmessage = handleProgressEvent

        eventSource.onerror = () => {
          eventSource.close()
          eventSourceRef.current = null
          const errorMsg = 'SSE 연결 오류'
          setError(errorMsg)
          setIsUploading(false)
          reject(new Error(errorMsg))
        }
      })
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : '업로드 실패'
      setError(errorMsg)
      setIsUploading(false)
      return null
    }
  }, [])

  return { uploadFiles, isUploading, progress, error, reset }
}
