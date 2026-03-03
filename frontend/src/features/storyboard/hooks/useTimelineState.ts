'use client'

import { useState, useCallback, useRef, useEffect } from 'react'
import type { TimelineItem, TimelineData, EditMode } from '../types'
import { storyboardService } from '../services'
import { useImageGeneration } from './useImageGeneration'
import { useVideoGeneration } from './useVideoGeneration'
import { generateId } from '../utils/generateId'

export function useTimelineState() {
  // 타임라인 데이터
  const [items, setItems] = useState<TimelineItem[]>([])
  const itemsRef = useRef<TimelineItem[]>(items)
  useEffect(() => {
    itemsRef.current = items
  }, [items])
  const [title, setTitle] = useState('새 타임라인')
  const [originalText, setOriginalText] = useState<string | undefined>()
  const [summary, setSummary] = useState<string | undefined>()

  // UI 상태
  const [editMode, setEditMode] = useState<EditMode>('view')
  const [selectedItemId, setSelectedItemId] = useState<string | null>(null)
  const [isExtracting, setIsExtracting] = useState(false)
  const [extractError, setExtractError] = useState<string | null>(null)

  // 이미지/영상 생성 에러 상태
  const [actionError, setActionError] = useState<string | null>(null)

  // 이미지 생성 서브 훅
  const {
    generatingImageIds,
    isGeneratingBatch,
    batchProgress,
    generateItemImage,
    generateAllImages,
  } = useImageGeneration({ itemsRef, setItems, setActionError })

  // 영상 생성 서브 훅
  const {
    isGeneratingVideo,
    generatedVideoUrl,
    showVideoModal,
    setShowVideoModal,
    generateVideo,
  } = useVideoGeneration({ setActionError })

  // AI 타임라인 추출 (텍스트)
  const extractTimeline = useCallback(async (text: string) => {
    setIsExtracting(true)
    setExtractError(null)

    try {
      const response = await storyboardService.extractTimeline(text)
      if (response.success) {
        setItems(response.timeline)
        setOriginalText(text)
        setSummary(response.summary)
        setTitle(response.summary || '새 타임라인')
      } else {
        setExtractError('타임라인 추출에 실패했습니다')
      }
    } catch (error) {
      setExtractError('서버 오류가 발생했습니다')
      console.error('Extract timeline error:', error)
    } finally {
      setIsExtracting(false)
    }
  }, [])

  // 음성에서 타임라인 추출
  const extractFromVoice = useCallback(async (audioFile: File) => {
    setIsExtracting(true)
    setExtractError(null)

    try {
      // 1. 음성 → 텍스트
      const transcribeResponse = await storyboardService.transcribeAudio(audioFile)
      if (!transcribeResponse.success || !transcribeResponse.text) {
        setExtractError(transcribeResponse.error || '음성 변환에 실패했습니다')
        return
      }

      // 2. 텍스트 → 타임라인
      const extractResponse = await storyboardService.extractTimeline(transcribeResponse.text)
      if (extractResponse.success) {
        setItems(extractResponse.timeline)
        setOriginalText(transcribeResponse.text)
        setSummary(extractResponse.summary)
        setTitle(extractResponse.summary || '새 타임라인')
      } else {
        setExtractError('타임라인 추출에 실패했습니다')
      }
    } catch (error) {
      setExtractError('서버 오류가 발생했습니다')
      console.error('Extract from voice error:', error)
    } finally {
      setIsExtracting(false)
    }
  }, [])

  // 이미지에서 타임라인 추출
  const extractFromImage = useCallback(async (imageFile: File, context: string) => {
    setIsExtracting(true)
    setExtractError(null)

    try {
      const response = await storyboardService.analyzeImage(imageFile, context)
      if (response.success) {
        setItems(response.timeline)
        setSummary(response.summary)
        setTitle(response.summary || '새 타임라인')
      } else {
        setExtractError(response.error || '이미지 분석에 실패했습니다')
      }
    } catch (error) {
      setExtractError('서버 오류가 발생했습니다')
      console.error('Extract from image error:', error)
    } finally {
      setIsExtracting(false)
    }
  }, [])

  // 항목 추가
  const addItem = useCallback(
    (item: Omit<TimelineItem, 'id' | 'order'>) => {
      setItems((prev) => {
        const newItem: TimelineItem = {
          ...item,
          id: generateId(),
          order: prev.length,
        }
        return [...prev, newItem]
      })
    },
    []
  )

  // 항목 수정
  const updateItem = useCallback(
    (id: string, updates: Partial<TimelineItem>) => {
      setItems((prev) =>
        prev.map((item) => (item.id === id ? { ...item, ...updates } : item))
      )
    },
    []
  )

  // 항목 삭제
  const deleteItem = useCallback((id: string) => {
    setItems((prev) => {
      const filtered = prev.filter((item) => item.id !== id)
      return filtered.map((item, idx) => ({ ...item, order: idx }))
    })
    setSelectedItemId(null)
  }, [])

  // 항목 순서 변경
  const reorderItems = useCallback(
    (sourceIndex: number, destIndex: number) => {
      setItems((prev) => {
        const result = [...prev]
        const [removed] = result.splice(sourceIndex, 1)
        result.splice(destIndex, 0, removed)
        return result.map((item, idx) => ({ ...item, order: idx }))
      })
    },
    []
  )

  // 편집 모드 토글
  const toggleEditMode = useCallback(() => {
    setEditMode((prev) => (prev === 'view' ? 'edit' : 'view'))
  }, [])

  // 항목 선택
  const selectItem = useCallback((id: string | null) => {
    setSelectedItemId(id)
  }, [])

  // JSON 내보내기
  const exportToJson = useCallback(() => {
    const now = new Date().toISOString()
    const data: TimelineData = {
      title,
      created_at: now,
      updated_at: now,
      items,
      original_text: originalText,
    }

    const blob = new Blob([JSON.stringify(data, null, 2)], {
      type: 'application/json',
    })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `timeline_${Date.now()}.json`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }, [title, items, originalText])

  // JSON 가져오기
  const importFromJson = useCallback(async (file: File) => {
    try {
      const text = await file.text()
      const data = JSON.parse(text) as TimelineData

      const validation = await storyboardService.validateTimeline(data)
      if (!validation.valid) {
        setExtractError(validation.message)
        return
      }

      setTitle(data.title)
      setItems(data.items)
      setOriginalText(data.original_text)
      setExtractError(null)
    } catch (error) {
      setExtractError('JSON 파일을 읽는데 실패했습니다')
      console.error('Import JSON error:', error)
    }
  }, [])

  // 초기화
  const resetTimeline = useCallback(() => {
    setItems([])
    setTitle('새 타임라인')
    setOriginalText(undefined)
    setSummary(undefined)
    setSelectedItemId(null)
    setEditMode('view')
    setExtractError(null)
  }, [])

  // 이미지가 있는 항목 수
  const itemsWithImagesCount = items.filter((item) => item.imageUrl).length

  return {
    // 데이터
    items,
    title,
    originalText,
    summary,

    // UI 상태
    editMode,
    selectedItemId,
    isExtracting,
    extractError,
    actionError,

    // 이미지 생성 상태
    generatingImageIds,
    isGeneratingBatch,
    batchProgress,
    itemsWithImagesCount,

    // 영상 생성 상태
    isGeneratingVideo,
    generatedVideoUrl,
    showVideoModal,
    setShowVideoModal,

    // 추출 액션
    extractTimeline,
    extractFromVoice,
    extractFromImage,

    // 이미지 생성 액션
    generateItemImage,
    generateAllImages,

    // 영상 생성 액션
    generateVideo,

    // 기본 액션
    setTitle,
    addItem,
    updateItem,
    deleteItem,
    reorderItems,
    toggleEditMode,
    selectItem,
    exportToJson,
    importFromJson,
    resetTimeline,
  }
}
