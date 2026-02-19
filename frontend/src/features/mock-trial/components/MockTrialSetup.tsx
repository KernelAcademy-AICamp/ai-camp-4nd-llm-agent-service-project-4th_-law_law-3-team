'use client'

import { useState } from 'react'
import type {
  CaseType,
  CaseCategory,
  UserRole,
  CriminalCategory,
  CivilCategory,
} from '../types'
import {
  CASE_TYPE_OPTIONS,
  CRIMINAL_CATEGORIES,
  CIVIL_CATEGORIES,
} from '../types'
import { DEMO_SCENARIOS } from '../demo/demo-scenarios'

interface MockTrialSetupProps {
  onComplete: (setup: {
    caseType: CaseType
    caseCategory: CaseCategory
    userRole: UserRole
    caseSummary: string
  }) => void
  onDemoStart?: (scenario: import('../demo/demo-scenarios').DemoScenario) => void
}

const MAX_SUMMARY_LENGTH = 500

export function MockTrialSetup({ onComplete, onDemoStart }: MockTrialSetupProps) {
  const [caseType, setCaseType] = useState<CaseType | null>(null)
  const [caseCategory, setCaseCategory] = useState<CaseCategory | null>(null)
  const [userRole, setUserRole] = useState<UserRole | null>(null)
  const [caseSummary, setCaseSummary] = useState('')

  const categories = caseType === 'criminal' ? CRIMINAL_CATEGORIES : CIVIL_CATEGORIES

  const criminalRoles: { id: UserRole; name: string; description: string }[] = [
    { id: 'prosecutor', name: '검사', description: '공소 유지 및 범죄 입증' },
    { id: 'attorney', name: '변호사', description: '피고인 방어 및 무죄/감형 논증' },
  ]

  const civilRoles: { id: UserRole; name: string; description: string }[] = [
    { id: 'plaintiff', name: '원고측 대리인', description: '청구원인 입증 및 손해배상 논증' },
    { id: 'defendant', name: '피고측 대리인', description: '청구 기각 및 항변' },
  ]

  const roles = caseType === 'criminal' ? criminalRoles : civilRoles

  const isValid =
    caseType !== null &&
    caseCategory !== null &&
    userRole !== null &&
    caseSummary.trim().length > 0

  const handleSubmit = (): void => {
    if (!isValid || !caseType || !caseCategory || !userRole) return
    onComplete({ caseType, caseCategory, userRole, caseSummary: caseSummary.trim() })
  }

  return (
    <div className="space-y-6">
      {/* 데모 시나리오 */}
      {onDemoStart && (
        <div>
          <h3 className="text-sm font-semibold text-amber-700 mb-2">
            데모 시나리오로 시작
          </h3>
          <p className="text-xs text-gray-500 mb-3">
            사전 작성된 시나리오로 모의 법정을 체험합니다. 각 단계에서 &quot;자동 입력&quot;
            버튼을 누르면 발언이 자동으로 채워집니다.
          </p>
          <div className="space-y-2">
            {DEMO_SCENARIOS.map((scenario) => (
              <button
                key={scenario.id}
                onClick={() => onDemoStart(scenario)}
                className="w-full p-3 rounded-lg border-2 border-amber-300 bg-amber-50 text-left hover:border-amber-500 hover:bg-amber-100 transition-colors"
              >
                <p className="text-sm font-semibold text-amber-800">
                  {scenario.name}
                </p>
                <p className="text-xs text-amber-600 mt-1">
                  {scenario.description}
                </p>
              </button>
            ))}
          </div>
          <div className="border-t border-gray-200 mt-4 pt-4">
            <p className="text-xs text-gray-400 text-center">
              또는 직접 설정하여 시작할 수 있습니다
            </p>
          </div>
        </div>
      )}

      {/* Step 1: 사건 유형 */}
      <div>
        <h3 className="text-sm font-semibold text-gray-700 mb-2">Step 1. 사건 유형</h3>
        <div className="grid grid-cols-2 gap-3">
          {CASE_TYPE_OPTIONS.map((option) => (
            <button
              key={option.id}
              onClick={() => {
                setCaseType(option.id)
                setCaseCategory(null)
                setUserRole(null)
              }}
              className={`p-3 rounded-lg border-2 text-center transition-colors ${
                caseType === option.id
                  ? 'border-blue-500 bg-blue-50 text-blue-700'
                  : 'border-gray-200 hover:border-gray-300 text-gray-700'
              }`}
            >
              <span className="text-xl">{option.icon}</span>
              <p className="text-sm font-medium mt-1">{option.name}</p>
            </button>
          ))}
        </div>
      </div>

      {/* Step 2: 세부 유형 */}
      {caseType && (
        <div>
          <h3 className="text-sm font-semibold text-gray-700 mb-2">Step 2. 세부 유형</h3>
          <div className="flex flex-wrap gap-2">
            {categories.map((category) => (
              <button
                key={category.id}
                onClick={() =>
                  setCaseCategory(
                    category.id as CriminalCategory | CivilCategory
                  )
                }
                className={`px-3 py-2 rounded-lg border text-sm transition-colors ${
                  caseCategory === category.id
                    ? 'border-blue-500 bg-blue-50 text-blue-700'
                    : 'border-gray-200 hover:border-gray-300 text-gray-600'
                }`}
                title={category.description}
              >
                {category.name}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Step 3: 역할 선택 */}
      {caseCategory && (
        <div>
          <h3 className="text-sm font-semibold text-gray-700 mb-2">Step 3. 역할 선택</h3>
          <div className="grid grid-cols-2 gap-3">
            {roles.map((role) => (
              <button
                key={role.id}
                onClick={() => setUserRole(role.id)}
                className={`p-3 rounded-lg border-2 text-left transition-colors ${
                  userRole === role.id
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
              >
                <p className="text-sm font-semibold text-gray-800">{role.name}</p>
                <p className="text-xs text-gray-500 mt-1">{role.description}</p>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Step 4: 사건 개요 */}
      {userRole && (
        <div>
          <h3 className="text-sm font-semibold text-gray-700 mb-2">Step 4. 사건 개요</h3>
          <textarea
            value={caseSummary}
            onChange={(e) =>
              setCaseSummary(e.target.value.slice(0, MAX_SUMMARY_LENGTH))
            }
            placeholder="사건 개요를 입력하세요 (예: 피고인 A가 피해자 B에게 투자금을 약속하고 5천만원을 편취한 사건)"
            className="w-full h-24 p-3 border border-gray-200 rounded-lg text-sm resize-none focus:outline-none focus:ring-2 focus:ring-blue-400"
          />
          <p className="text-xs text-gray-400 text-right mt-1">
            {caseSummary.length}/{MAX_SUMMARY_LENGTH}
          </p>
        </div>
      )}

      {/* 재판 시작 버튼 */}
      {userRole && (
        <button
          onClick={handleSubmit}
          disabled={!isValid}
          className={`w-full py-3 rounded-lg text-sm font-semibold transition-colors ${
            isValid
              ? 'bg-blue-600 text-white hover:bg-blue-700'
              : 'bg-gray-200 text-gray-400 cursor-not-allowed'
          }`}
        >
          재판 시작
        </button>
      )}
    </div>
  )
}
