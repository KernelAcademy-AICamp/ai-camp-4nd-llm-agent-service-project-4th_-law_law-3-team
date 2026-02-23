'use client'

import { useState, useCallback, useRef, useEffect } from 'react'
import dynamic from 'next/dynamic'
import { useUI } from '@/context/UIContext'
import { BackButton } from '@/components/ui/BackButton'
import { DisclaimerBanner } from '@/features/mock-trial/components/DisclaimerBanner'
import { MockTrialSetup } from '@/features/mock-trial/components/MockTrialSetup'
import { StageProgress } from '@/features/mock-trial/components/StageProgress'
import { ChatPanel } from '@/features/mock-trial/components/ChatPanel'
import { ChatBottomBar } from '@/features/mock-trial/components/ChatBottomBar'
import { ReferencePanel } from '@/features/mock-trial/components/ReferencePanel'
import { EvidencePanel } from '@/features/mock-trial/components/EvidencePanel'
import { eventBus } from '@/features/mock-trial/game/EventBus'
import { ChevronLeft, ChevronRight, Info } from 'lucide-react'
import type {
  CaseType,
  CaseCategory,
  UserRole,
  CourtEvent,
  ReferenceItem,
  EvidenceItem,
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

/** 판례번호 패턴 (예: 2023다12345) */
const CASE_NUMBER_PATTERN = /(\d{2,4}[가-힣]{1,3}\d{1,6})/g

/** 법령 참조 패턴 (예: 형사소송법 제284조) */
const LAW_REFERENCE_PATTERN =
  /((?:형사|민사|상법|민법|헌법|행정)(?:소송법|소송규칙)?)\s*(?:제?\s*(\d+)조(?:의?\d+)?)/g

export default function MockTrialPage() {
  const { isChatOpen, chatMode } = useUI()

  const [phase, setPhase] = useState<TrialPhase>('setup')
  const [caseType, setCaseType] = useState<CaseType | null>(null)
  const [currentStageId, setCurrentStageId] = useState('identity')
  const [messages, setMessages] = useState<CourtEvent[]>([])
  const [isWaiting, setIsWaiting] = useState(false)
  const [references, setReferences] = useState<ReferenceItem[]>([])
  const referenceIdsRef = useRef<Set<string>>(new Set())

  // 패널 토글 상태
  const [isReferencePanelOpen, setIsReferencePanelOpen] = useState(false)
  const [chatDisplayMode, setChatDisplayMode] = useState<'bar' | 'panel'>('bar')

  // H3: 증거 선택 상태
  const [evidenceCases, setEvidenceCases] = useState<EvidenceItem[]>([])
  const [evidenceArticles, setEvidenceArticles] = useState<EvidenceItem[]>([])
  const [selectedEvidenceIds, setSelectedEvidenceIds] = useState<Set<string>>(new Set())
  const [isEvidenceLoading, setIsEvidenceLoading] = useState(false)

  // H5: 단계 가이드 표시
  const [showStageGuide, setShowStageGuide] = useState(true)

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
    setShowStageGuide(true)
    // 증거 상태 초기화 (증거조사 단계를 벗어날 때)
    if (currentStageId === 'evidence') {
      setSelectedEvidenceIds(new Set())
    }
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

  // ── 참조 추출 ──

  /** 메시지에서 판례번호/법령 참조를 감지하여 참조 목록에 추가 */
  useEffect(() => {
    if (messages.length === 0) return
    const lastMessage = messages[messages.length - 1]
    if (lastMessage.speaker === 'user') return

    const content = lastMessage.content
    const newReferences: ReferenceItem[] = []

    // 판례번호 감지
    for (const match of Array.from(content.matchAll(CASE_NUMBER_PATTERN))) {
      const matchedText = match[1]
      const id = `case-${matchedText}`
      if (!referenceIdsRef.current.has(id)) {
        referenceIdsRef.current.add(id)
        newReferences.push({
          id,
          type: 'case',
          title: matchedText,
          summary: `${matchedText} 판결문 - 재판 중 언급됨`,
          relevance_score: 0.8,
          source: '재판 기록',
          matched_text: matchedText,
        })
      }
    }

    // 법령 참조 감지
    for (const match of Array.from(content.matchAll(LAW_REFERENCE_PATTERN))) {
      const lawName = match[1]
      const article = match[2]
      const matchedText = match[0]
      const id = `law-${lawName}-${article}`
      if (!referenceIdsRef.current.has(id)) {
        referenceIdsRef.current.add(id)
        newReferences.push({
          id,
          type: 'law',
          title: `${lawName} 제${article}조`,
          summary: `${lawName} 제${article}조 - 재판 중 언급됨`,
          relevance_score: 0.9,
          source: lawName,
          matched_text: matchedText,
        })
      }
    }

    if (newReferences.length > 0) {
      setReferences((prev) => [...prev, ...newReferences])
    }
  }, [messages])

  // H3: 증거 토글 핸들러
  const handleEvidenceToggle = useCallback((id: string) => {
    setSelectedEvidenceIds((prev) => {
      const next = new Set(prev)
      if (next.has(id)) {
        next.delete(id)
      } else {
        next.add(id)
      }
      return next
    })
  }, [])

  // H3: 증거 제출 핸들러
  const handleEvidenceSubmit = useCallback(() => {
    const allIds = [
      ...evidenceCases.map((c) => c.id),
      ...evidenceArticles.map((a) => a.id),
    ]
    const excludedIds = allIds.filter((id) => !selectedEvidenceIds.has(id))

    // 백엔드에 선택 결과 전달
    eventBus.emit('user:input', {
      text: JSON.stringify({
        selected_ids: Array.from(selectedEvidenceIds),
        excluded_ids: excludedIds,
        text: `증거 ${selectedEvidenceIds.size}건 제출`,
      }),
    })

    const submitMessage: CourtEvent = {
      stage: currentStageId,
      speaker: 'user',
      content: `증거 ${selectedEvidenceIds.size}건을 제출했습니다.`,
      timestamp: new Date().toISOString(),
    }
    setMessages((prev) => [...prev, submitMessage])
    setIsWaiting(true)

    if (!isDemoMode) {
      setTimeout(() => setIsWaiting(false), 1000)
    }
  }, [evidenceCases, evidenceArticles, selectedEvidenceIds, currentStageId, isDemoMode])

  /** 현재 단계가 증거조사인지 여부 */
  const isEvidenceStage = currentStageId === 'evidence' && phase === 'trial'

  /** 현재 단계의 StageInfo */
  const currentStageInfo = stages.find((s) => s.id === currentStageId)

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

      {/* H5: 단계 가이드 배너 */}
      {phase === 'trial' && showStageGuide && currentStageInfo && (
        <div className="bg-blue-50 border-b border-blue-200 px-4 py-2 flex items-start gap-2">
          <Info className="w-4 h-4 text-blue-500 mt-0.5 shrink-0" />
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold text-blue-700">
                {currentStageInfo.name}
              </span>
              <span className="text-[10px] text-blue-500 bg-blue-100 px-1.5 py-0.5 rounded">
                {currentStageInfo.legal_basis}
              </span>
              <span className="text-[10px] text-gray-400">
                {currentStageInfo.duration_hint}
              </span>
            </div>
            <p className="text-xs text-blue-600 mt-0.5">
              {currentStageInfo.description} &middot;{' '}
              <span className="font-medium">{currentStageInfo.user_action}</span>
            </p>
          </div>
          <button
            onClick={() => setShowStageGuide(false)}
            className="text-blue-400 hover:text-blue-600 text-xs shrink-0"
            aria-label="가이드 닫기"
          >
            닫기
          </button>
        </div>
      )}

      {/* 메인 콘텐츠 - MockTrialGame은 한 번만 렌더링하여 씬 전환 유지 */}
      <div className="flex-1 flex overflow-hidden">
        {/* 좌: 참조/증거 패널 토글 (trial phase에서만 표시) */}
        {phase === 'trial' && (
          <>
            {/* 참조/증거 패널 (접기/펴기) */}
            <div
              className={`${
                isReferencePanelOpen || isEvidenceStage ? 'w-72' : 'w-0'
              } transition-all duration-300 overflow-hidden border-r border-gray-200 bg-white`}
            >
              <div className="w-72 h-full overflow-y-auto">
                {isEvidenceStage ? (
                  <EvidencePanel
                    cases={evidenceCases}
                    articles={evidenceArticles}
                    selectedIds={selectedEvidenceIds}
                    onToggle={handleEvidenceToggle}
                    onSubmit={handleEvidenceSubmit}
                    isLoading={isEvidenceLoading}
                  />
                ) : (
                  <ReferencePanel references={references} />
                )}
              </div>
            </div>

            {/* 토글 버튼 (증거조사 단계에서는 항상 열림) */}
            {!isEvidenceStage && (
              <button
                onClick={() => setIsReferencePanelOpen((prev) => !prev)}
                className="w-6 shrink-0 flex flex-col items-center justify-center bg-gray-100 hover:bg-gray-200 border-r border-gray-200 transition-colors"
                aria-label={isReferencePanelOpen ? '참조 패널 접기' : '참조 패널 열기'}
              >
                {isReferencePanelOpen ? (
                  <ChevronLeft className="w-4 h-4 text-gray-500" />
                ) : (
                  <>
                    <ChevronRight className="w-4 h-4 text-gray-500" />
                    {references.length > 0 && (
                      <span className="mt-1 text-[10px] font-medium text-blue-600 bg-blue-100 rounded-full w-5 h-5 flex items-center justify-center">
                        {references.length}
                      </span>
                    )}
                  </>
                )}
              </button>
            )}
          </>
        )}

        {/* 중앙: Phaser 게임 + 하단 바 */}
        <div
          className={`flex flex-col overflow-hidden ${
            phase === 'setup' ? 'flex-1 bg-gray-100' : 'flex-1 bg-gray-50'
          }`}
        >
          <div className="flex-1 flex items-center justify-center p-2">
            <MockTrialGame />
          </div>

          {/* 다음 단계 버튼 (데모 모드, trial phase) */}
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

          {/* 하단 채팅 바 (trial phase, bar 모드일 때) */}
          {phase === 'trial' && chatDisplayMode === 'bar' && (
            <ChatBottomBar
              messages={messages}
              isWaiting={isWaiting}
              placeholder="발언을 입력하세요..."
              onSend={handleSendMessage}
              onExpand={() => setChatDisplayMode('panel')}
              demoInput={nextDemoInput}
              onDemoInput={handleDemoInput}
            />
          )}
        </div>

        {/* 우: setup phase → 설정 UI / trial phase + panel 모드 → 채팅 패널 */}
        {phase === 'setup' ? (
          <div className="w-96 border-l border-gray-200 bg-white overflow-y-auto p-6">
            <MockTrialSetup
              onComplete={handleSetupComplete}
              onDemoStart={handleDemoStart}
            />
          </div>
        ) : chatDisplayMode === 'panel' ? (
          <div className="flex-[1.4] border-l border-gray-200">
            <ChatPanel
              messages={messages}
              isWaiting={isWaiting}
              placeholder="발언을 입력하세요..."
              onSend={handleSendMessage}
              demoInput={nextDemoInput}
              onDemoInput={handleDemoInput}
              onCollapse={() => setChatDisplayMode('bar')}
            />
          </div>
        ) : null}
      </div>
    </div>
  )
}
