'use client'

import { Suspense, useState } from 'react'
import dynamic from 'next/dynamic'
import { ProgressBar } from '@/features/small-claims/components/ProgressBar'
import { DisputeTypeStep } from '@/features/small-claims/components/DisputeTypeStep'
import { useWizardState } from '@/features/small-claims/hooks/useWizardState'
import { DISPUTE_TYPE_OPTIONS } from '@/features/small-claims/types'
import { BackButton } from '@/components/ui/BackButton'
import { useUI } from '@/context/UIContext'
import { ChevronLeft, ChevronRight } from 'lucide-react'

const CaseInfoStep = dynamic(
  () => import('@/features/small-claims/components/CaseInfoStep').then((m) => m.CaseInfoStep),
  { ssr: false }
)

const CourtCostCalculator = dynamic(
  () =>
    import('@/features/small-claims/components/CourtCostCalculator').then(
      (m) => m.CourtCostCalculator
    ),
  { ssr: false }
)

const EvidenceStep = dynamic(
  () => import('@/features/small-claims/components/EvidenceStep').then((m) => m.EvidenceStep),
  { ssr: false }
)

const DocumentStep = dynamic(
  () => import('@/features/small-claims/components/DocumentStep').then((m) => m.DocumentStep),
  { ssr: false }
)

const RelatedCases = dynamic(
  () => import('@/features/small-claims/components/RelatedCases').then((m) => m.RelatedCases),
  { ssr: false }
)

function StepSkeleton() {
  return (
    <div className="max-w-2xl mx-auto">
      <div className="h-8 w-48 bg-navy-100 rounded animate-pulse mb-4" />
      <div className="h-32 bg-navy-100 rounded-lg animate-pulse" />
    </div>
  )
}

export default function SmallClaimsPage() {
  const { isChatOpen, chatMode } = useUI()
  const [isSidebarOpen, setIsSidebarOpen] = useState(true)
  const {
    currentStep,
    goToStep,
    goToNextStep,
    goToPreviousStep,
    disputeType,
    setDisputeType,
    caseInfo,
    updateCaseInfo,
    evidenceItems,
    checkedEvidence,
    toggleEvidence,
    isLoadingEvidence,
    uploadedFiles,
    handleFileUpload,
    removeUploadedFile,
    generatedDocument,
    isGenerating,
    generateError,
    generateDocument,
    relatedCases,
    isLoadingRelatedCases,
    resetWizard,
  } = useWizardState()

  const selectedDisputeOption = DISPUTE_TYPE_OPTIONS.find((opt) => opt.id === disputeType)

  const renderStep = () => {
    switch (currentStep) {
      case 'dispute_type':
        return (
          <DisputeTypeStep
            selectedType={disputeType}
            onSelect={setDisputeType}
            onNext={goToNextStep}
          />
        )
      case 'case_info':
        return (
          <>
            <CaseInfoStep
              caseInfo={caseInfo}
              onUpdate={updateCaseInfo}
              onNext={goToNextStep}
              onPrevious={goToPreviousStep}
            />
            <div className="max-w-2xl mx-auto mt-6">
              <CourtCostCalculator claimAmount={caseInfo.amount} />
            </div>
          </>
        )
      case 'evidence':
        return (
          <EvidenceStep
            items={evidenceItems}
            checkedItems={checkedEvidence}
            isLoading={isLoadingEvidence}
            uploadedFiles={uploadedFiles}
            onToggle={toggleEvidence}
            onFileUpload={handleFileUpload}
            onFileRemove={removeUploadedFile}
            onNext={goToNextStep}
            onPrevious={goToPreviousStep}
          />
        )
      case 'document':
        return (
          <DocumentStep
            generatedDocument={generatedDocument}
            isGenerating={isGenerating}
            error={generateError}
            onGenerate={generateDocument}
            onPrevious={goToPreviousStep}
            onReset={resetWizard}
          />
        )
      default:
        return null
    }
  }

  return (
    <div
      className={`h-screen flex flex-col bg-gray-100 transition-all duration-500 ease-in-out ${
        isChatOpen && chatMode === 'split' ? 'w-1/2 border-r border-gray-200' : 'w-full'
      }`}
    >
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <BackButton />
            <span className="text-2xl">⚖️</span>
            <div>
              <h1 className="text-xl font-bold text-gray-900">소액소송 도우미</h1>
              <p className="text-sm text-gray-500">
                {selectedDisputeOption
                  ? `${selectedDisputeOption.icon} ${selectedDisputeOption.name}`
                  : '나홀로 소송 지원 - 내용증명, 지급명령, 소액심판'}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {disputeType && (
              <button
                onClick={resetWizard}
                className="text-sm text-gray-500 hover:text-gray-700 flex items-center gap-1"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                  />
                </svg>
                처음부터 다시
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Progress Bar */}
      <ProgressBar currentStep={currentStep} onStepClick={goToStep} />

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Related Cases Sidebar - Show after selecting dispute type */}
        {disputeType && currentStep !== 'dispute_type' && (
          <div className="hidden md:flex shrink-0">
            {isSidebarOpen ? (
              <div className="w-72 h-full relative">
                <Suspense fallback={<div className="w-72 h-full bg-white border-r border-gray-100 animate-pulse" />}>
                  <RelatedCases
                    cases={relatedCases}
                    isLoading={isLoadingRelatedCases}
                    disputeType={selectedDisputeOption?.name || null}
                    onClose={() => setIsSidebarOpen(false)}
                  />
                </Suspense>
              </div>
            ) : (
              <button
                onClick={() => setIsSidebarOpen(true)}
                className="flex flex-col items-center justify-center w-8 bg-white border-r border-gray-200 hover:bg-gray-50 transition-colors gap-1"
                aria-label="유사 판례 패널 열기"
              >
                <ChevronRight className="w-4 h-4 text-gray-400" />
                <span className="text-[10px] text-gray-500 [writing-mode:vertical-rl]">유사 판례</span>
                {relatedCases.length > 0 && (
                  <span className="inline-flex items-center justify-center w-5 h-5 text-[10px] font-medium bg-blue-100 text-blue-600 rounded-full">
                    {relatedCases.length}
                  </span>
                )}
              </button>
            )}
          </div>
        )}

        {/* Wizard Content */}
        <div className="flex-1 overflow-y-auto p-8">
          <Suspense fallback={<StepSkeleton />}>
            {renderStep()}
          </Suspense>
        </div>
      </div>
    </div>
  )
}
