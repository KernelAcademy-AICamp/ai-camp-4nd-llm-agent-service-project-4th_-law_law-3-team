'use client'

import {
  ProgressBar,
  DisputeTypeStep,
  CaseInfoStep,
  EvidenceStep,
  DocumentStep,
  RelatedCases,
} from '@/features/small-claims/components'
import { useWizardState } from '@/features/small-claims/hooks/useWizardState'
import { DISPUTE_TYPE_OPTIONS } from '@/features/small-claims/types'

export default function SmallClaimsPage() {
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
          <CaseInfoStep
            caseInfo={caseInfo}
            onUpdate={updateCaseInfo}
            onNext={goToNextStep}
            onPrevious={goToPreviousStep}
          />
        )
      case 'evidence':
        return (
          <EvidenceStep
            items={evidenceItems}
            checkedItems={checkedEvidence}
            isLoading={isLoadingEvidence}
            onToggle={toggleEvidence}
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
    <div className="h-screen flex flex-col bg-gray-100">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
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
      </header>

      {/* Progress Bar */}
      <ProgressBar currentStep={currentStep} onStepClick={goToStep} />

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Wizard Content */}
        <div className="flex-1 overflow-y-auto p-8">{renderStep()}</div>

        {/* Related Cases Sidebar - Show after selecting dispute type */}
        {disputeType && currentStep !== 'dispute_type' && (
          <RelatedCases
            cases={relatedCases}
            isLoading={isLoadingRelatedCases}
            disputeType={selectedDisputeOption?.name || null}
          />
        )}
      </div>
    </div>
  )
}
