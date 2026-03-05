'use client'

import { useState, useCallback, useRef } from 'react'
import type { MutableRefObject } from 'react'
import type { TimelineItem } from '../types'
import { storyboardService } from '../services'

interface UseImageGenerationOptions {
  itemsRef: MutableRefObject<TimelineItem[]>
  setItems: React.Dispatch<React.SetStateAction<TimelineItem[]>>
  setActionError: React.Dispatch<React.SetStateAction<string | null>>
}

export function useImageGeneration({
  itemsRef,
  setItems,
  setActionError,
}: UseImageGenerationOptions) {
  const [generatingImageIds, setGeneratingImageIds] = useState<Set<string>>(new Set())
  const [isGeneratingBatch, setIsGeneratingBatch] = useState(false)
  const [batchProgress, setBatchProgress] = useState<{ current: number; total: number } | undefined>()
  const cancelledRef = useRef(false)

  const generateItemImage = useCallback(async (itemId: string) => {
    setGeneratingImageIds((prev) => new Set(prev).add(itemId))
    setActionError(null)

    try {
      const targetItem = itemsRef.current.find((i) => i.id === itemId)

      if (!targetItem) {
        return
      }

      const response = await storyboardService.generateImage(targetItem)
      if (response.success && response.image_url) {
        setItems((prev) =>
          prev.map((i) =>
            i.id === itemId
              ? { ...i, imageUrl: response.image_url, imagePrompt: response.image_prompt }
              : i
          )
        )
      } else {
        console.error('Image generation failed:', response.error)
        setActionError(response.error || '이미지 생성에 실패했습니다')
      }
    } catch (error) {
      console.error('Generate image error:', error)
      setActionError('이미지 생성에 실패했습니다')
    } finally {
      setGeneratingImageIds((prev) => {
        const newSet = new Set(prev)
        newSet.delete(itemId)
        return newSet
      })
    }
  }, [itemsRef, setItems, setActionError])

  const generateAllImages = useCallback(async () => {
    const currentItems = itemsRef.current

    if (currentItems.length === 0) return

    cancelledRef.current = false
    setIsGeneratingBatch(true)
    setBatchProgress({ current: 0, total: currentItems.length })
    setActionError(null)

    try {
      for (let i = 0; i < currentItems.length; i++) {
        if (cancelledRef.current) break

        const item = currentItems[i]
        setBatchProgress({ current: i, total: currentItems.length })

        try {
          const response = await storyboardService.generateImage(item)
          if (response.success && response.image_url) {
            setItems((prev) =>
              prev.map((it) =>
                it.id === item.id
                  ? { ...it, imageUrl: response.image_url, imagePrompt: response.image_prompt }
                  : it
              )
            )
          }
        } catch (err) {
          console.error(`Image generation failed for ${item.id}:`, err)
        }
      }

      if (!cancelledRef.current) {
        setBatchProgress({ current: currentItems.length, total: currentItems.length })
      }
    } catch (error) {
      console.error('Generate all images error:', error)
      setActionError('일괄 이미지 생성 중 오류가 발생했습니다')
    } finally {
      setIsGeneratingBatch(false)
      setBatchProgress(undefined)
    }
  }, [itemsRef, setItems, setActionError])

  const cancelBatchGeneration = useCallback(() => {
    cancelledRef.current = true
  }, [])

  return {
    generatingImageIds,
    isGeneratingBatch,
    batchProgress,
    generateItemImage,
    generateAllImages,
    cancelBatchGeneration,
  }
}
