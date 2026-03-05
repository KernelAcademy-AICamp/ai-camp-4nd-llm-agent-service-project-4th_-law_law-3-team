'use client'

import { BookOpen, PanelLeftOpen, PanelLeftClose } from 'lucide-react'
import { useUI } from '@/context/UIContext'
import { BackButton } from '@/components/ui/BackButton'
import { useLawStudy } from '@/features/law-study/hooks/useLawStudy'
import { ExamSelector } from '@/features/law-study/components/ExamSelector'
import { ExamViewer } from '@/features/law-study/components/ExamViewer'
import { ReferencePanel } from '@/features/law-study/components/ReferencePanel'
import dynamic from 'next/dynamic'

const FeedbackPanel = dynamic(
  () =>
    import('@/features/law-study/components/FeedbackPanel').then((mod) => ({
      default: mod.FeedbackPanel,
    })),
)

export default function LawStudyPage() {
  const { isChatOpen, chatMode } = useUI()
  const {
    selectedCategory,
    setSelectedCategory,
    selectedExam,
    selectExam,
    answerText,
    setAnswerText,
    isReferencePanelOpen,
    setReferencePanelOpen,
    isFeedbackPanelOpen,
    setFeedbackPanelOpen,
    filteredExams,
    examListQuery,
    examContentQuery,
    feedbackMutation,
    referenceResults,
    isSearchingReference,
    searchReference,
  } = useLawStudy()

  const isSplit = isChatOpen && chatMode === 'split'

  return (
    <div
      className={`h-screen flex flex-col bg-gray-50 transition-all duration-300 ${
        isSplit ? 'w-1/2 border-r border-gray-200' : 'w-full'
      }`}
    >
      {/* 헤더 */}
      <header className="shrink-0 px-4 py-3 bg-white border-b border-gray-200">
        <div className="flex items-center gap-3">
          <BackButton />
          <BookOpen size={24} className="text-blue-600 shrink-0" />
          <div className="min-w-0">
            <h1 className="text-lg font-bold text-gray-900 truncate">
              변호사시험 기록형 연습
            </h1>
          </div>
        </div>
      </header>

      {/* 메인 콘텐츠 */}
      <div className="flex-1 flex overflow-hidden">
        {/* 좌측: 참조 패널 (토글) */}
        <div
          className={`shrink-0 border-r border-gray-200 bg-white transition-all duration-300 overflow-hidden ${
            isReferencePanelOpen ? 'w-80' : 'w-0'
          }`}
        >
          {isReferencePanelOpen && (
            <ReferencePanel
              results={referenceResults}
              isSearching={isSearchingReference}
              onSearch={searchReference}
            />
          )}
        </div>

        {/* 참조 패널 토글 버튼 */}
        <button
          onClick={() => setReferencePanelOpen(!isReferencePanelOpen)}
          className="shrink-0 w-6 flex items-center justify-center bg-gray-100 hover:bg-gray-200 border-r border-gray-200 transition-colors"
          aria-label={isReferencePanelOpen ? '참조 패널 닫기' : '참조 패널 열기'}
        >
          {isReferencePanelOpen ? (
            <PanelLeftClose size={14} className="text-gray-500" />
          ) : (
            <PanelLeftOpen size={14} className="text-gray-500" />
          )}
        </button>

        {/* 중앙-좌: 문제 선택 사이드바 */}
        <div className="shrink-0 w-56 border-r border-gray-200 bg-white">
          <ExamSelector
            selectedCategory={selectedCategory}
            onCategoryChange={setSelectedCategory}
            exams={filteredExams}
            selectedExam={selectedExam}
            onExamSelect={selectExam}
            isLoading={examListQuery.isLoading}
          />
        </div>

        {/* 중앙-우: 문제 뷰어 + 답안 작성 */}
        <div className="flex-1 min-w-0">
          <ExamViewer
            content={examContentQuery.data?.content}
            isLoading={examContentQuery.isLoading}
            title={selectedExam?.title ?? ''}
            answerText={answerText}
            onAnswerChange={setAnswerText}
            onRequestFeedback={() => feedbackMutation.mutate()}
            isFeedbackLoading={feedbackMutation.isPending}
            category={selectedExam?.category}
            session={selectedExam?.session}
            year={selectedExam?.year}
          />
        </div>

        {/* 우측: AI 피드백 패널 (조건부) */}
        {isFeedbackPanelOpen && (
          <div className="shrink-0 w-96 border-l border-gray-200 bg-white">
            <FeedbackPanel
              feedback={feedbackMutation.data?.feedback}
              isLoading={feedbackMutation.isPending}
              isError={feedbackMutation.isError}
              errorMessage={
                feedbackMutation.error instanceof Error
                  ? feedbackMutation.error.message
                  : undefined
              }
              onClose={() => setFeedbackPanelOpen(false)}
            />
          </div>
        )}
      </div>
    </div>
  )
}
