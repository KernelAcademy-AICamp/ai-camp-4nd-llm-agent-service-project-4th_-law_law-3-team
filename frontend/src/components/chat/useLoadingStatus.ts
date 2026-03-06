import { useState, useEffect, useMemo, useRef } from 'react'

export function useLoadingStatus(isLoading: boolean, isStreaming: boolean) {
  const [requestStartedAt, setRequestStartedAt] = useState<number | null>(null)
  const [loadingElapsedSeconds, setLoadingElapsedSeconds] = useState(0)
  const [hasReceivedFirstToken, setHasReceivedFirstToken] = useState(false)
  const hasReceivedFirstTokenRef = useRef(false)

  useEffect(() => {
    if (!(isLoading || isStreaming) || requestStartedAt === null) {
      setLoadingElapsedSeconds(0)
      return
    }

    const tick = () => {
      setRequestStartedAt((started) => {
        if (started !== null) {
          setLoadingElapsedSeconds(Math.floor((Date.now() - started) / 1000))
        }
        return started
      })
    }

    tick()
    const intervalId = window.setInterval(tick, 1000)
    return () => window.clearInterval(intervalId)
  }, [isLoading, isStreaming, requestStartedAt])

  const loadingStatus = useMemo(() => {
    if (hasReceivedFirstToken) {
      return { title: '답변을 완성하는 중입니다...' }
    }
    if (loadingElapsedSeconds < 3) {
      return { title: '질문 의도를 분석하고 있습니다...' }
    }
    if (loadingElapsedSeconds < 8) {
      return { title: '관련 데이터를 검색하고 있습니다...' }
    }
    if (loadingElapsedSeconds < 15) {
      return { title: '검색된 결과를 정제하고 있습니다...' }
    }
    if (loadingElapsedSeconds < 25) {
      return { title: '심층 분석을 진행하고 있습니다...' }
    }
    return { title: '응답 준비가 거의 완료되었습니다...' }
  }, [hasReceivedFirstToken, loadingElapsedSeconds])

  const resetLoadingState = () => {
    setRequestStartedAt(null)
    setLoadingElapsedSeconds(0)
    setHasReceivedFirstToken(false)
    hasReceivedFirstTokenRef.current = false
  }

  const startLoading = () => {
    setRequestStartedAt(Date.now())
    setLoadingElapsedSeconds(0)
    setHasReceivedFirstToken(false)
    hasReceivedFirstTokenRef.current = false
  }

  const markFirstToken = () => {
    if (!hasReceivedFirstTokenRef.current) {
      hasReceivedFirstTokenRef.current = true
      setHasReceivedFirstToken(true)
    }
  }

  return {
    loadingStatus,
    requestStartedAt,
    hasReceivedFirstTokenRef,
    resetLoadingState,
    startLoading,
    markFirstToken,
  }
}
