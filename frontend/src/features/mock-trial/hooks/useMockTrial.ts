import { useState, useCallback, useRef, useEffect, useMemo } from 'react'
import { eventBus } from '@/features/mock-trial/game/EventBus'
import { useStreamingChat, type ChatMetadata } from '@/hooks/useStreamingChat'
import { CASE_NUMBER_PATTERN, LAW_REFERENCE_PATTERN } from '@/features/mock-trial/constants'
import type {
  CaseType,
  CaseCategory,
  UserRole,
  CourtEvent,
  ReferenceItem,
  EvidenceItem,
  UserHint,
  EmotionType,
  DialogueSpeed,
  PhysicalEvidence,
  JudgmentResult,
} from '@/features/mock-trial/types'
import { CRIMINAL_STAGES, CIVIL_STAGES, DEFAULT_ROLE_EMOTION } from '@/features/mock-trial/types'
import type { DemoScenario } from '@/features/mock-trial/demo/demo-scenarios'

export type TrialPhase = 'setup' | 'briefing' | 'trial' | 'verdict'

export function useMockTrial() {
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

  // 증거 선택 상태
  const [evidenceCases, setEvidenceCases] = useState<EvidenceItem[]>([])
  const [evidenceArticles, setEvidenceArticles] = useState<EvidenceItem[]>([])
  const [selectedEvidenceIds, setSelectedEvidenceIds] = useState<Set<string>>(new Set())
  const [isEvidenceLoading, setIsEvidenceLoading] = useState(false)

  // 물적 증거 (시나리오 기반)
  const [physicalEvidence, setPhysicalEvidence] = useState<PhysicalEvidence[]>([])

  // RAG 사용자 힌트
  const [userHints, setUserHints] = useState<UserHint[]>([])

  // 단계 가이드 표시
  const [showStageGuide, setShowStageGuide] = useState(true)

  // 대화 속도
  const [dialogueSpeed, setDialogueSpeed] = useState<DialogueSpeed>('normal')

  // 판결 결과
  const [judgmentResult, setJudgmentResult] = useState<JudgmentResult | null>(null)

  // 데모 모드 상태
  const [isDemoMode, setIsDemoMode] = useState(false)
  const [demoScenario, setDemoScenario] = useState<DemoScenario | null>(null)
  const demoInputIndexRef = useRef<Record<string, number>>({})
  const [demoInputTick, setDemoInputTick] = useState(0)

  // 구체화 질문 상태
  const [clarificationQuestion, setClarificationQuestion] = useState<string | null>(null)
  const [isClarifying, setIsClarifying] = useState(false)

  // 일반 모드 SSE 상태
  const { sendStreamingMessage } = useStreamingChat()
  const sessionDataRef = useRef<Record<string, unknown>>({})
  const tokenBufferRef = useRef('')
  const setupInfoRef = useRef({
    caseType: '',
    caseCategory: '',
    userRole: '',
    caseSummary: '',
  })

  const stages = caseType === 'civil' ? CIVIL_STAGES : CRIMINAL_STAGES

  /** 현재 단계의 데모 데이터 */
  const currentDemoStage = demoScenario?.stages.find(
    (s) => s.stageId === currentStageId
  )

  /** 현재 단계에서 다음으로 입력할 데모 텍스트 */
  const nextDemoInput = useMemo(() => {
    if (!isDemoMode || !currentDemoStage) return null
    const index = demoInputIndexRef.current[currentStageId] ?? 0
    return currentDemoStage.userInputs[index] ?? null
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isDemoMode, currentDemoStage, currentStageId, demoInputTick])

  /** 현재 단계의 모든 사용자 입력을 소진했는지 */
  const isDemoStageInputsDone = useMemo(() => {
    if (!isDemoMode || !currentDemoStage) return false
    const index = demoInputIndexRef.current[currentStageId] ?? 0
    return index >= currentDemoStage.userInputs.length
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isDemoMode, currentDemoStage, currentStageId, demoInputTick])

  /** 다음 단계 ID를 반환 */
  const getNextStageId = useCallback((): string | null => {
    const currentIndex = stages.findIndex((s) => s.id === currentStageId)
    if (currentIndex < 0 || currentIndex >= stages.length - 1) return null
    return stages[currentIndex + 1].id
  }, [stages, currentStageId])

  // ── 데모 참조 추가 헬퍼 ──

  const addDemoReferences = useCallback((refs: ReferenceItem[] | undefined) => {
    if (!refs) return
    const newRefs = refs.filter((r) => !referenceIdsRef.current.has(r.id))
    for (const r of newRefs) referenceIdsRef.current.add(r.id)
    if (newRefs.length > 0) {
      setReferences((prev) => [...prev, ...newRefs])
      setIsReferencePanelOpen(true)
    }
  }, [])

  /** 데모 mock AI 응답을 대화 큐에 일괄 추가 */
  const playMockResponses = useCallback(
    (
      responses: { speaker: string; content: string; emotion?: EmotionType }[],
      stageId: string
    ) => {
      if (responses.length === 0) {
        setIsWaiting(false)
        return
      }

      setIsWaiting(true)

      const events: CourtEvent[] = responses.map((response) => ({
        stage: stageId,
        speaker: response.speaker,
        content: response.content,
        timestamp: new Date().toISOString(),
        emotion: response.emotion ?? DEFAULT_ROLE_EMOTION[response.speaker] ?? 'neutral',
      }))

      setMessages((prev) => [...prev, ...events])

      for (const event of events) {
        eventBus.emit('dialogue:enqueue', {
          agent: event.speaker,
          text: event.content,
          emotion: event.emotion,
        })
      }
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
      setupInfoRef.current = {
        caseType: setup.caseType,
        caseCategory: setup.caseCategory,
        userRole: setup.userRole,
        caseSummary: setup.caseSummary,
      }

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

  /** SSE 메타데이터에서 clarification step 감지 시 처리 */
  const processSetupSSEMetadata = useCallback(
    (metadata: ChatMetadata) => {
      const content = tokenBufferRef.current
      tokenBufferRef.current = ''

      if (metadata.session_data) {
        sessionDataRef.current = {
          thread_id: metadata.session_data.thread_id,
          session_secret: metadata.session_data.session_secret,
        }
      }

      if (metadata.step === 'clarification' && content) {
        setClarificationQuestion(content)
        setIsClarifying(false)
        return
      }

      // clarification이 아닌 경우 → 일반 trial 진행
      const info = setupInfoRef.current
      setCaseType(info.caseType as CaseType)
      setCurrentStageId(
        info.caseType === 'criminal' ? 'identity' : 'pretrial'
      )
      setPhase('trial')
      setIsClarifying(false)

      eventBus.emit('setup:complete', {
        caseType: info.caseType,
        userRole: info.userRole,
        caseSummary: info.caseSummary,
      })

      // 첫 번째 stage의 응답 처리
      if (content) {
        const speaker = metadata.speaking_agent || 'judge'
        const event: CourtEvent = {
          stage: info.caseType === 'criminal' ? 'identity' : 'pretrial',
          speaker,
          content,
          timestamp: new Date().toISOString(),
          emotion: (metadata.emotion || 'stern') as EmotionType,
        }
        setMessages((prev) => [...prev, event])
        eventBus.emit('dialogue:enqueue', {
          agent: speaker,
          text: content,
          emotion: event.emotion,
        })
      }
    },
    []
  )

  /** 구체화 답변 제출 */
  const handleClarificationSubmit = useCallback(
    (answer: string) => {
      setIsClarifying(true)
      setClarificationQuestion(null)

      const info = setupInfoRef.current
      sendStreamingMessage(
        {
          message: answer,
          agent: 'mock_trial',
          session_data: {
            ...sessionDataRef.current,
            stage: 'setup',
            case_type: info.caseType,
            case_category: info.caseCategory,
            user_role: info.userRole,
            case_summary: info.caseSummary,
          },
        },
        {
          onToken: (content) => {
            tokenBufferRef.current += content
          },
          onMetadata: processSetupSSEMetadata,
          onDone: () => {
            setIsClarifying(false)
          },
          onError: (error) => {
            console.error('[MockTrial] Clarification SSE error:', error)
            setIsClarifying(false)
            // 구체화 실패 시 원본 개요로 재판 시작
            const fallbackInfo = setupInfoRef.current
            setCaseType(fallbackInfo.caseType as CaseType)
            setCurrentStageId(
              fallbackInfo.caseType === 'criminal' ? 'identity' : 'pretrial'
            )
            setPhase('trial')
            eventBus.emit('setup:complete', {
              caseType: fallbackInfo.caseType,
              userRole: fallbackInfo.userRole,
              caseSummary: fallbackInfo.caseSummary,
            })
          },
        }
      )
    },
    [sendStreamingMessage, processSetupSSEMetadata]
  )

  /** SSE 메타데이터 응답 처리 (일반 모드 공용) */
  const processSSEMetadata = useCallback(
    (metadata: ChatMetadata) => {
      const content = tokenBufferRef.current
      tokenBufferRef.current = ''

      if (metadata.session_data) {
        sessionDataRef.current = {
          thread_id: metadata.session_data.thread_id,
          session_secret: metadata.session_data.session_secret,
        }
      }

      const speaker = metadata.speaking_agent || 'judge'
      const emotion =
        metadata.emotion || DEFAULT_ROLE_EMOTION[speaker] || 'neutral'

      if (content) {
        const event: CourtEvent = {
          stage: currentStageId,
          speaker,
          content,
          timestamp: new Date().toISOString(),
          emotion: emotion as EmotionType,
        }
        setMessages((prev) => [...prev, event])
        eventBus.emit('dialogue:enqueue', {
          agent: speaker,
          text: content,
          emotion: emotion as EmotionType,
        })
      }

      if (metadata.evidence) {
        setEvidenceCases(metadata.evidence.cases as EvidenceItem[])
        setEvidenceArticles(metadata.evidence.articles as EvidenceItem[])
      }

      if (metadata.user_hints) {
        setUserHints(metadata.user_hints as UserHint[])
      }

      // RAG 검색 결과를 ReferencePanel에 추가
      if (metadata.references && metadata.references.length > 0) {
        const newRefs: ReferenceItem[] = []
        for (const ref of metadata.references) {
          if (!referenceIdsRef.current.has(ref.id)) {
            referenceIdsRef.current.add(ref.id)
            newRefs.push({
              id: ref.id,
              type: ref.type as 'case' | 'law',
              title: ref.title,
              summary: ref.summary,
              relevance_score: ref.relevance_score,
              source: ref.source,
              matched_text: ref.title,
            })
          }
        }
        if (newRefs.length > 0) {
          setReferences((prev) => [...prev, ...newRefs])
          setIsReferencePanelOpen(true)
        }
      }

      if (metadata.stage && metadata.stage !== currentStageId) {
        setCurrentStageId(metadata.stage)
      }
    },
    [currentStageId]
  )

  const handleSendMessage = useCallback(
    (text: string) => {
      const userRole = setupInfoRef.current.userRole || 'user'
      const newMessage: CourtEvent = {
        stage: currentStageId,
        speaker: userRole,
        content: text,
        timestamp: new Date().toISOString(),
        emotion: DEFAULT_ROLE_EMOTION[userRole] ?? 'neutral',
        isUser: true,
      }
      setMessages((prev) => [...prev, newMessage])
      setIsWaiting(true)

      // 게임에서 사용자 역할 캐릭터가 발언하도록 전달
      eventBus.emit('dialogue:enqueue', {
        agent: userRole,
        text,
        emotion: DEFAULT_ROLE_EMOTION[userRole] ?? 'neutral',
      })

      if (isDemoMode && currentDemoStage) {
        playMockResponses(currentDemoStage.mockResponses, currentStageId)
        addDemoReferences(currentDemoStage.references)
        const currentIndex =
          demoInputIndexRef.current[currentStageId] ?? 0
        demoInputIndexRef.current[currentStageId] = currentIndex + 1
        setDemoInputTick((t) => t + 1)
      } else {
        const info = setupInfoRef.current
        sendStreamingMessage(
          {
            message: text,
            agent: 'mock_trial',
            session_data: {
              ...sessionDataRef.current,
              stage: currentStageId,
              case_type: info.caseType,
              user_role: info.userRole,
              case_summary: info.caseSummary,
            },
          },
          {
            onToken: (content) => {
              tokenBufferRef.current += content
            },
            onMetadata: processSSEMetadata,
            onDone: () => {
              setIsWaiting(false)
            },
            onError: (error) => {
              console.error('[MockTrial] SSE error:', error)
              setIsWaiting(false)
            },
          }
        )
      }
    },
    [currentStageId, isDemoMode, currentDemoStage, playMockResponses, addDemoReferences, sendStreamingMessage, processSSEMetadata]
  )

  // ── 데모 모드 핸들러 ──

  const handleDemoStart = useCallback(
    (scenario: DemoScenario) => {
      setIsDemoMode(true)
      setDemoScenario(scenario)
      setPhysicalEvidence(scenario.evidence ?? [])
      demoInputIndexRef.current = {}
      setPhase('briefing')
    },
    []
  )

  /** 브리핑 완료 → 재판 시작 */
  const handleBriefingComplete = useCallback(() => {
    if (!demoScenario) return
    const { setup } = demoScenario
    handleSetupComplete({
      caseType: setup.caseType,
      caseCategory: setup.caseCategory,
      userRole: setup.userRole,
      caseSummary: setup.caseSummary,
    })

    const firstStageId =
      setup.caseType === 'criminal' ? 'identity' : 'pretrial'
    const firstStage = demoScenario.stages.find(
      (s) => s.stageId === firstStageId
    )
    if (firstStage && firstStage.userInputs.length === 0) {
      const unsub = eventBus.on('court:entrance:complete', () => {
        unsub()
        playMockResponses(firstStage.mockResponses, firstStageId)
        addDemoReferences(firstStage.references)
      })
    }
  }, [demoScenario, handleSetupComplete, playMockResponses, addDemoReferences])

  /** 데모 자동 입력 버튼 클릭 */
  const handleDemoInput = useCallback(() => {
    if (!nextDemoInput) return
    handleSendMessage(nextDemoInput)
  }, [nextDemoInput, handleSendMessage])

  /** 다음 단계로 이동 */
  const handleNextStage = useCallback(() => {
    const nextId = getNextStageId()
    if (!nextId) {
      if (isDemoMode && demoScenario) {
        const verdictStage = demoScenario.stages.find(s => s.stageId === 'verdict')
        if (verdictStage) {
          playMockResponses(verdictStage.mockResponses, 'verdict')
          addDemoReferences(verdictStage.references)
          const judgeSpeech = verdictStage.mockResponses.find(r => r.speaker === 'judge')
          setJudgmentResult({
            judgment: judgeSpeech?.content ?? '판결 내용이 없습니다.',
            feedback: '데모 시나리오에서는 수행 평가가 제공되지 않습니다.',
            cited_cases: [],
            cited_articles: [],
          })
        }
      }
      setPhase('verdict')
      return
    }

    setCurrentStageId(nextId)
    setShowStageGuide(true)
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

    if (isDemoMode && demoScenario) {
      const nextStage = demoScenario.stages.find(
        (s) => s.stageId === nextId
      )
      if (nextStage && nextStage.userInputs.length === 0) {
        setTimeout(() => {
          playMockResponses(nextStage.mockResponses, nextId)
          addDemoReferences(nextStage.references)
        }, 500)
      }
    }
  }, [getNextStageId, isDemoMode, demoScenario, playMockResponses, addDemoReferences, currentStageId, stages])

  /** 처음부터 다시 시작 */
  const handleRestart = useCallback(() => {
    setPhase('setup')
    setMessages([])
    setJudgmentResult(null)
    setCurrentStageId('')
    setIsDemoMode(false)
    setDemoScenario(null)
  }, [])

  /** 판결 모달 닫기 → 재판 화면 복귀 */
  const handleCloseJudgment = useCallback(() => {
    setPhase('trial')
  }, [])

  // ── dialogue:queue:empty → isWaiting 해제 ──
  useEffect(() => {
    const unsub = eventBus.on('dialogue:queue:empty', () => {
      setIsWaiting(false)
    })
    return unsub
  }, [])

  // ── Space 키보드 리스너 (dialogue:advance) ──
  useEffect(() => {
    if (phase !== 'trial') return

    const handleKeyDown = (e: KeyboardEvent): void => {
      if (e.code !== 'Space') return
      const tag = (e.target as HTMLElement)?.tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA') return
      e.preventDefault()
      eventBus.emit('dialogue:advance', {} as Record<string, never>)
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [phase])

  // ── 참조 추출 ──
  useEffect(() => {
    if (messages.length === 0) return
    const lastMessage = messages[messages.length - 1]
    if (lastMessage.speaker === 'user') return

    const content = lastMessage.content
    const newReferences: ReferenceItem[] = []

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
      setIsReferencePanelOpen(true)
    }
  }, [messages])

  // 증거 토글 핸들러
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

  // 증거 제출 핸들러
  const handleEvidenceSubmit = useCallback(() => {
    const allIds = [
      ...evidenceCases.map((c) => c.id),
      ...evidenceArticles.map((a) => a.id),
    ]
    const excludedIds = allIds.filter((id) => !selectedEvidenceIds.has(id))

    eventBus.emit('user:input', {
      text: JSON.stringify({
        selected_ids: Array.from(selectedEvidenceIds),
        excluded_ids: excludedIds,
        text: `증거 ${selectedEvidenceIds.size}건 제출`,
      }),
    })

    const userRole = setupInfoRef.current.userRole || 'user'
    const submitMessage: CourtEvent = {
      stage: currentStageId,
      speaker: userRole,
      content: `증거 ${selectedEvidenceIds.size}건을 제출했습니다.`,
      timestamp: new Date().toISOString(),
      isUser: true,
    }
    setMessages((prev) => [...prev, submitMessage])
    setIsWaiting(true)

    if (!isDemoMode) {
      const info = setupInfoRef.current
      sendStreamingMessage(
        {
          message: JSON.stringify({
            selected_ids: Array.from(selectedEvidenceIds),
            excluded_ids: excludedIds,
            text: `증거 ${selectedEvidenceIds.size}건 제출`,
          }),
          agent: 'mock_trial',
          session_data: {
            ...sessionDataRef.current,
            stage: currentStageId,
            case_type: info.caseType,
            user_role: info.userRole,
          },
        },
        {
          onToken: (content) => {
            tokenBufferRef.current += content
          },
          onMetadata: processSSEMetadata,
          onDone: () => {
            setIsWaiting(false)
          },
          onError: (error) => {
            console.error('[MockTrial] SSE error:', error)
            setIsWaiting(false)
          },
        }
      )
    } else if (currentDemoStage) {
      playMockResponses(currentDemoStage.mockResponses, currentStageId)
      addDemoReferences(currentDemoStage.references)
      const currentIndex = demoInputIndexRef.current[currentStageId] ?? 0
      demoInputIndexRef.current[currentStageId] = currentIndex + 1
      setDemoInputTick((t) => t + 1)
    }
  }, [evidenceCases, evidenceArticles, selectedEvidenceIds, currentStageId, isDemoMode, currentDemoStage, playMockResponses, addDemoReferences, sendStreamingMessage, processSSEMetadata])

  // ── 파생 상태 ──

  const isEvidenceStage = currentStageId === 'evidence' && phase === 'trial'
  const currentStageInfo = useMemo(
    () => stages.find((s) => s.id === currentStageId),
    [stages, currentStageId]
  )
  const showNextStageButton =
    isDemoMode && !isWaiting && phase === 'trial' && isDemoStageInputsDone
  const nextStageId = getNextStageId()

  return {
    // 상태
    phase,
    caseType,
    currentStageId,
    messages,
    isWaiting,
    references,
    isReferencePanelOpen,
    setIsReferencePanelOpen,
    chatDisplayMode,
    setChatDisplayMode,
    evidenceCases,
    evidenceArticles,
    selectedEvidenceIds,
    isEvidenceLoading,
    physicalEvidence,
    userHints,
    showStageGuide,
    setShowStageGuide,
    dialogueSpeed,
    setDialogueSpeed,
    isDemoMode,
    demoScenario,
    stages,
    nextDemoInput,

    // 파생 상태
    isEvidenceStage,
    currentStageInfo,
    showNextStageButton,
    nextStageId,

    // 판결
    judgmentResult,

    // 구체화
    clarificationQuestion,
    isClarifying,

    // 핸들러
    handleSetupComplete,
    handleSendMessage,
    handleDemoStart,
    handleBriefingComplete,
    handleDemoInput,
    handleNextStage,
    handleEvidenceToggle,
    handleEvidenceSubmit,
    handleRestart,
    handleCloseJudgment,
    handleClarificationSubmit,
  }
}
