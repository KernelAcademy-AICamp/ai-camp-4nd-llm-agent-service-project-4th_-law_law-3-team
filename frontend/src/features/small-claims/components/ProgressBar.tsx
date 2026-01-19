'use client'

import type { WizardStep } from '../types'
import { WIZARD_STEPS } from '../types'

interface ProgressBarProps {
  currentStep: WizardStep
  onStepClick?: (step: WizardStep) => void
}

export function ProgressBar({ currentStep, onStepClick }: ProgressBarProps) {
  const currentIndex = WIZARD_STEPS.findIndex((s) => s.step === currentStep)

  return (
    <div className="w-full py-4 px-6 bg-white border-b border-gray-200">
      <div className="flex items-center justify-between max-w-3xl mx-auto">
        {WIZARD_STEPS.map((step, index) => {
          const isCompleted = index < currentIndex
          const isCurrent = index === currentIndex
          const isClickable = index <= currentIndex && onStepClick

          return (
            <div key={step.step} className="flex items-center flex-1">
              {/* Step Circle */}
              <button
                onClick={() => isClickable && onStepClick(step.step)}
                disabled={!isClickable}
                className={`
                  relative flex items-center justify-center w-10 h-10 rounded-full font-medium text-sm transition-all
                  ${
                    isCompleted
                      ? 'bg-blue-600 text-white'
                      : isCurrent
                        ? 'bg-blue-600 text-white ring-4 ring-blue-100'
                        : 'bg-gray-200 text-gray-500'
                  }
                  ${isClickable ? 'cursor-pointer hover:ring-4 hover:ring-blue-100' : 'cursor-default'}
                `}
              >
                {isCompleted ? (
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                ) : (
                  index + 1
                )}
              </button>

              {/* Step Label */}
              <span
                className={`ml-3 text-sm font-medium hidden sm:block ${
                  isCurrent ? 'text-blue-600' : isCompleted ? 'text-gray-700' : 'text-gray-400'
                }`}
              >
                {step.label}
              </span>

              {/* Connector Line */}
              {index < WIZARD_STEPS.length - 1 && (
                <div className="flex-1 mx-4 h-0.5 bg-gray-200">
                  <div
                    className={`h-full bg-blue-600 transition-all duration-300 ${
                      isCompleted ? 'w-full' : 'w-0'
                    }`}
                  />
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
