'use client'

import { useState, useCallback, useRef } from 'react'
import dynamic from 'next/dynamic'
import { useUI } from '@/context/UIContext'
import { BackButton } from '@/components/ui/BackButton'
import { DisclaimerBanner } from '@/features/mock-trial/components/DisclaimerBanner'
import { MockTrialSetup } from '@/features/mock-trial/components/MockTrialSetup'
import { StageProgress } from '@/features/mock-trial/components/StageProgress'
import { ChatPanel } from '@/features/mock-trial/components/ChatPanel'
import { eventBus } from '@/features/mock-trial/game/EventBus'
import type {
  CaseType,
  CaseCategory,
  UserRole,
  CourtEvent,
} from '@/features/mock-trial/types'
import { CRIMINAL_STAGES, CIVIL_STAGES } from '@/features/mock-trial/types'
import type { DemoScenario } from '@/features/mock-trial/demo/demo-scenarios'

const MockTrialGame = dynamic(
  () =>
    import('@/features/mock-trial/components/MockTrialGame').then(
      (m) => m.MockTrialGame
    ),
  { ssr: false }
)

type TrialPhase = 'setup' | 'trial' | 'verdict'

/** 데모 모드에서 mock AI 응답 간 딜레이 (ms) */
const DEMO_RESPONSE_DELAY = 1200

export default function MockTrialPage() {
  const { isChatOpen, chatMode } = useUI()

  const [phase, setPhase] = useState<TrialPhase>('setup')
  const [caseType, setCaseType] = useState<CaseType | null>(null)
  const [currentStageId, setCurrentStageId] = useState('identity')
  const [messages, setMessages] = useState<CourtEvent[]>([])
  const [isWaiting, setIsWaiting] = useState(false)

  // 데모 모드 상태
  const [isDemoMode, setIsDemoMode] = useState(false)
  const [demoScenario, setDemoScenario] = useState<DemoScenario | null>(null)
  /** 각 단계별 사용자 입력 인덱스 (몇 번째 입력을 사용할 차례인지) */
  const demoInputIndexRef = useRef<Record<string, number>>({})

  const stages = caseType === 'civil' ? CIVIL_STAGES : CRIMINAL_STAGES

  /** 현재 단계의 데모 데이터 */
  const currentDemoStage = demoScenario?.stages.find(
    (s) => s.stageId === currentStageId
  )

  /** 현재 단계에서 다음으로 입력할 데모 텍스트 */
  const nextDemoInput = (() => {
    if (!isDemoMode || !currentDemoStage) return null
    const index = demoInputIndexRef.current[currentStageId] ?? 0
    return currentDemoStage.userInputs[index] ?? null
  })()

  /** 현재 단계의 모든 사용자 입력을 소진했는지 */
  const isDemoStageInputsDone = (() => {
    if (!isDemoMode || !currentDemoStage) return false
    const index = demoInputIndexRef.current[currentStageId] ?? 0
    return index >= currentDemoStage.userInputs.length
  })()

  /** 다음 단계 ID를 반환 */
  const getNextStageId = useCallback((): string | null => {
    const currentIndex = stages.findIndex((s) => s.id === currentStageId)
    if (currentIndex < 0 || currentIndex >= stages.length - 1) return null
    return stages[currentIndex + 1].id
  }, [stages, currentStageId])

  /** 데모 mock AI 응답을 순차적으로 재생 */
  const playMockResponses = useCallback(
    (responses: { speaker: string; content: string }[], stageId: string) => {
      if (responses.length === 0) {
        setIsWaiting(false)
        return
      }

      let index = 0
      const playNext = (): void => {
        if (index >= responses.length) {
          setIsWaiting(false)
          return
        }
        const response = responses[index]
        const event: CourtEvent = {
          stage: stageId,
          speaker: response.speaker,
          content: response.content,
          timestamp: new Date().toISOString(),
        }
        setMessages((prev) => [...prev, event])
        eventBus.emit('agent:speak', {
          agent: response.speaker,
          text: response.content,
          streaming: false,
        })
        index++
        setTimeout(playNext, DEMO_RESPONSE_DELAY)
      }

      setIsWaiting(true)
      setTimeout(playNext, DEMO_RESPONSE_DELAY)
    },
    []
  )

  // ── 일반 모드 핸들러 ──

  const handleSetupComplete = useCallback(
    (setup: {
      caseType: CaseType
      caseCategory: CaseCategory
      userRole: UserRole
      caseSummary: string
    }) => {
      setCaseType(setup.caseType)
      setCurrentStageId(
        setup.caseType === 'criminal' ? 'identity' : 'pretrial'
      )
      setPhase('trial')

      eventBus.emit('setup:complete', {
        caseType: setup.caseType,
        userRole: setup.userRole,
        caseSummary: setup.caseSummary,
      })
    },
    []
  )

  const handleSendMessage = useCallback(
    (text: string) => {
      const newMessage: CourtEvent = {
        stage: currentStageId,
        speaker: 'user',
        content: text,
        timestamp: new Date().toISOString(),
      }
      setMessages((prev) => [...prev, newMessage])
      setIsWaiting(true)

      eventBus.emit('user:input', { text })

      if (isDemoMode && currentDemoStage) {
        // 데모 모드: mock AI 응답 재생
        playMockResponses(currentDemoStage.mockResponses, currentStageId)
        // 사용자 입력 인덱스 증가
        const currentIndex =
          demoInputIndexRef.current[currentStageId] ?? 0
        demoInputIndexRef.current[currentStageId] = currentIndex + 1
      } else {
        // 일반 모드: 백엔드 응답 대기 (플레이스홀더)
        setTimeout(() => setIsWaiting(false), 1000)
      }
    },
    [currentStageId, isDemoMode, currentDemoStage, playMockResponses]
  )

  // ── 데모 모드 핸들러 ──

  const handleDemoStart = useCallback(
    (scenario: DemoScenario) => {
      setIsDemoMode(true)
      setDemoScenario(scenario)
      demoInputIndexRef.current = {}

      // setup 데이터로 바로 시작
      const { setup } = scenario
      handleSetupComplete({
        caseType: setup.caseType,
        caseCategory: setup.caseCategory,
        userRole: setup.userRole,
        caseSummary: setup.caseSummary,
      })

      // 첫 단계가 자동 진행 단계(userInputs 없음)이면 mock 응답 자동 재생
      const firstStageId =
        setup.caseType === 'criminal' ? 'identity' : 'pretrial'
      const firstStage = scenario.stages.find(
        (s) => s.stageId === firstStageId
      )
      if (firstStage && firstStage.userInputs.length === 0) {
        setTimeout(() => {
          playMockResponses(firstStage.mockResponses, firstStageId)
        }, 500)
      }
    },
    [handleSetupComplete, playMockResponses]
  )

  /** 데모 자동 입력 버튼 클릭 */
  const handleDemoInput = useCallback(() => {
    if (!nextDemoInput) return
    handleSendMessage(nextDemoInput)
  }, [nextDemoInput, handleSendMessage])

  /** 다음 단계로 이동 */
  const handleNextStage = useCallback(() => {
    const nextId = getNextStageId()
    if (!nextId) {
      setPhase('verdict')
      return
    }

    setCurrentStageId(nextId)
    const nextIndex = stages.findIndex((s) => s.id === nextId)
    eventBus.emit('stage:change', {
      from: currentStageId,
      to: nextId,
      stageNumber: nextIndex + 1,
      totalStages: stages.length,
    })

    // 데모 모드: 자동 진행 단계면 mock 응답 자동 재생
    if (isDemoMode && demoScenario) {
      const nextStage = demoScenario.stages.find(
        (s) => s.stageId === nextId
      )
      if (nextStage && nextStage.userInputs.length === 0) {
        setTimeout(() => {
          playMockResponses(nextStage.mockResponses, nextId)
        }, 500)
      }
    }
  }, [getNextStageId, isDemoMode, demoScenario, playMockResponses])

  /** 현재 단계에서 "다음 단계" 버튼을 보여줄지 여부 */
  const showNextStageButton =
    isDemoMode && !isWaiting && phase === 'trial' && isDemoStageInputsDone

  return (
    <div
      className={`h-screen flex flex-col bg-gray-100 transition-all duration-500 ease-in-out ${
        isChatOpen && chatMode === 'split'
          ? 'w-1/2 border-r border-gray-200'
          : 'w-full'
      }`}
    >
      {/* 면책 고지 */}
      <DisclaimerBanner />

      {/* 헤더 */}
      <header className="bg-white border-b border-gray-200 px-6 py-3">
        <div className="flex items-center gap-3">
          <BackButton />
          <div className="flex-1">
            <h1 className="text-lg font-bold text-gray-900">모의 법정</h1>
            <p className="text-xs text-gray-500">
              AI 에이전트와 함께하는 모의재판 시뮬레이션
            </p>
          </div>
          {isDemoMode && (
            <span className="px-2 py-1 text-xs font-medium bg-amber-100 text-amber-700 rounded-full">
              데모 모드
            </span>
          )}
        </div>
      </header>

      {/* 단계 진행률 (재판 중일 때만) */}
      {phase === 'trial' && (
        <StageProgress stages={stages} currentStageId={currentStageId} />
      )}

      {/* 메인 콘텐츠 */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {phase === 'setup' ? (
          <div className="flex-1 flex overflow-hidden">
            {/* 왼쪽: Phaser 게임 (로비) */}
            <div className="flex-1 flex items-center justify-center p-4">
              <MockTrialGame />
            </div>
            {/* 오른쪽: 설정 UI */}
            <div className="w-96 border-l border-gray-200 bg-white overflow-y-auto p-6">
              <MockTrialSetup
                onComplete={handleSetupComplete}
                onDemoStart={handleDemoStart}
              />
            </div>
          </div>
        ) : (
          <div className="flex-1 flex flex-col overflow-hidden">
            {/* Phaser 게임 (법정) */}
            <div className="flex items-center justify-center bg-gray-50 py-2">
              <MockTrialGame />
            </div>

            {/* 다음 단계 버튼 (데모 모드) */}
            {showNextStageButton && (
              <div className="px-4 py-2 bg-amber-50 border-t border-amber-200 flex items-center justify-between">
                <span className="text-sm text-amber-700">
                  현재 단계가 완료되었습니다.
                </span>
                <button
                  onClick={handleNextStage}
                  className="px-4 py-1.5 text-sm font-medium bg-amber-500 text-white rounded-lg hover:bg-amber-600 transition-colors"
                >
                  {getNextStageId() ? '다음 단계로' : '판결 보기'}
                </button>
              </div>
            )}

            {/* 하단 채팅 패널 */}
            <div className="flex-1 min-h-0">
              <ChatPanel
                messages={messages}
                isWaiting={isWaiting}
                placeholder="발언을 입력하세요..."
                onSend={handleSendMessage}
                demoInput={nextDemoInput}
                onDemoInput={handleDemoInput}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
