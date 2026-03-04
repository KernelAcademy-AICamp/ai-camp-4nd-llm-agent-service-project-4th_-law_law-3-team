'use client'

import dynamic from 'next/dynamic'
import { useCallback } from 'react'
import { useUI } from '@/context/UIContext'
import { BackButton } from '@/components/ui/BackButton'
import { DisclaimerBanner } from '@/features/mock-trial/components/DisclaimerBanner'
import { StageProgress } from '@/features/mock-trial/components/StageProgress'
import { DialogueControls } from '@/features/mock-trial/components/DialogueControls'
import { StageGuideBanner } from '@/features/mock-trial/components/StageGuideBanner'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import { useMockTrial } from '@/features/mock-trial/hooks/useMockTrial'

const MockTrialGame = dynamic(
  () =>
    import('@/features/mock-trial/components/MockTrialGame').then(
      (m) => m.MockTrialGame
    ),
  { ssr: false }
)

const MockTrialSetup = dynamic(
  () =>
    import('@/features/mock-trial/components/MockTrialSetup').then(
      (m) => m.MockTrialSetup
    )
)

const ScenarioBriefing = dynamic(
  () =>
    import('@/features/mock-trial/components/ScenarioBriefing').then(
      (m) => m.ScenarioBriefing
    )
)

const ChatPanel = dynamic(
  () =>
    import('@/features/mock-trial/components/ChatPanel').then(
      (m) => m.ChatPanel
    )
)

const ChatBottomBar = dynamic(
  () =>
    import('@/features/mock-trial/components/ChatBottomBar').then(
      (m) => m.ChatBottomBar
    )
)

const EvidencePanel = dynamic(
  () =>
    import('@/features/mock-trial/components/EvidencePanel').then(
      (m) => m.EvidencePanel
    )
)

const ReferencePanel = dynamic(
  () =>
    import('@/features/mock-trial/components/ReferencePanel').then(
      (m) => m.ReferencePanel
    )
)

export default function MockTrialPage() {
  const { isChatOpen, chatMode } = useUI()
  const {
    phase,
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
    isEvidenceStage,
    currentStageInfo,
    showNextStageButton,
    nextStageId,
    handleSetupComplete,
    handleSendMessage,
    handleDemoStart,
    handleBriefingComplete,
    handleDemoInput,
    handleNextStage,
    handleEvidenceToggle,
    handleEvidenceSubmit,
  } = useMockTrial()

  const handleDismissGuide = useCallback(() => setShowStageGuide(false), [setShowStageGuide])

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

      {/* 단계 가이드 배너 */}
      {phase === 'trial' && showStageGuide && currentStageInfo && (
        <StageGuideBanner
          stageInfo={currentStageInfo}
          onDismiss={handleDismissGuide}
        />
      )}

      {/* 메인 콘텐츠 */}
      <div className="flex-1 flex overflow-hidden">
        {/* 좌: 참조/증거 패널 토글 */}
        {phase === 'trial' && (
          <>
            <div
              className={`${
                isReferencePanelOpen || isEvidenceStage ? 'w-72' : 'w-0'
              } transition-[width] duration-300 overflow-hidden border-r border-gray-200 bg-white`}
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
                    userHints={userHints}
                    physicalEvidence={physicalEvidence}
                    references={references}
                  />
                ) : (
                  <ReferencePanel references={references} />
                )}
              </div>
            </div>

            {/* 토글 버튼 */}
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

          {/* 대화 컨트롤 */}
          {phase === 'trial' && (
            <DialogueControls
              currentSpeed={dialogueSpeed}
              onSpeedChange={setDialogueSpeed}
            />
          )}

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
                {nextStageId ? '다음 단계로' : '판결 보기'}
              </button>
            </div>
          )}

          {/* 하단 채팅 바 */}
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

        {/* 우: setup/briefing/chat 패널 */}
        {phase === 'setup' ? (
          <div className="w-96 border-l border-gray-200 bg-white overflow-y-auto p-6">
            <MockTrialSetup
              onComplete={handleSetupComplete}
              onDemoStart={handleDemoStart}
            />
          </div>
        ) : phase === 'briefing' && demoScenario ? (
          <div className="w-96 border-l border-gray-200 bg-white overflow-y-auto p-6">
            <ScenarioBriefing
              scenario={demoScenario}
              onStart={handleBriefingComplete}
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
