'use client'

import { useState, useCallback } from 'react'
import type { VideoSettings } from '../types'
import { storyboardService } from '../services'

const generateId = () =>
  `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 11)}`

interface UseVideoGenerationOptions {
  setActionError: React.Dispatch<React.SetStateAction<string | null>>
}

export function useVideoGeneration({ setActionError }: UseVideoGenerationOptions) {
  const [isGeneratingVideo, setIsGeneratingVideo] = useState(false)
  const [generatedVideoUrl, setGeneratedVideoUrl] = useState<string | null>(null)
  const [showVideoModal, setShowVideoModal] = useState(false)

  const generateVideo = useCallback(async (imageUrls: string[], settings: VideoSettings) => {
    if (imageUrls.length < 2) return

    setIsGeneratingVideo(true)
    setActionError(null)

    try {
      const response = await storyboardService.generateVideo({
        timeline_id: generateId(),
        image_urls: imageUrls,
        duration_per_image: settings.durationPerImage,
        transition: settings.transition,
        transition_duration: settings.transitionDuration,
        resolution: settings.resolution,
      })

      if (response.success && response.video_url) {
        setGeneratedVideoUrl(response.video_url)
      } else {
        console.error('Video generation failed:', response.error)
        setActionError(response.error || '영상 생성에 실패했습니다')
      }
    } catch (error) {
      console.error('Generate video error:', error)
      setActionError('영상 생성에 실패했습니다')
    } finally {
      setIsGeneratingVideo(false)
    }
  }, [setActionError])

  return {
    isGeneratingVideo,
    generatedVideoUrl,
    setGeneratedVideoUrl,
    showVideoModal,
    setShowVideoModal,
    generateVideo,
  }
}
