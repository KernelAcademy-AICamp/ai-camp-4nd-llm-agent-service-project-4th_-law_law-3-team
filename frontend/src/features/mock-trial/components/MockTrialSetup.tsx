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
import { SCENARIOS, type Scenario } from '../scenarios/scenarios'

interface MockTrialSetupProps {
  onComplete: (setup: {
    caseType: CaseType
    caseCategory: CaseCategory
    userRole: UserRole
    caseSummary: string
  }) => void
  onDemoStart?: (scenario: import('../demo/demo-scenarios').DemoScenario) => void
  clarificationQuestion?: string | null
  isClarifying?: boolean
  onClarificationSubmit?: (answer: string) => void
}

const MAX_SUMMARY_LENGTH = 500

export function MockTrialSetup({
  onComplete,
  onDemoStart,
  clarificationQuestion,
  isClarifying,
  onClarificationSubmit,
}: MockTrialSetupProps) {
  const [caseType, setCaseType] = useState<CaseType | null>(null)
  const [caseCategory, setCaseCategory] = useState<CaseCategory | null>(null)
  const [userRole, setUserRole] = useState<UserRole | null>(null)
  const [caseSummary, setCaseSummary] = useState('')
  const [selectedScenarioId, setSelectedScenarioId] = useState<string | null>(null)
  const [clarificationAnswer, setClarificationAnswer] = useState('')

  const categories = caseType === 'criminal' ? CRIMINAL_CATEGORIES : CIVIL_CATEGORIES

  const criminalRoles: { id: UserRole; name: string; description: string }[] = [
    { id: 'prosecutor', name: '검사', description: '공소 유지 및 범죄 입증' },
    { id: 'attorney', name: '변호사', description: '피고인 방어 및 무죄/감형 논증' },
  ]

  const civilRoles: { id: UserRole; name: string; description: string }[] = [
    { id: 'prosecutor', name: '원고측 대리인', description: '청구원인 입증 및 손해배상 논증' },
    { id: 'attorney', name: '피고측 대리인', description: '청구 기각 및 항변' },
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

  const handleScenarioSelect = (scenario: Scenario): void => {
    setSelectedScenarioId(scenario.id)
    setCaseType(scenario.caseType)
    setCaseCategory(scenario.caseCategory)
    setUserRole(scenario.userRole)
    setCaseSummary(scenario.caseSummary)
  }

  const handleClarificationSubmit = (): void => {
    if (!clarificationAnswer.trim() || !onClarificationSubmit) return
    onClarificationSubmit(clarificationAnswer.trim())
    setClarificationAnswer('')
  }

  return (
    <div className="space-y-6">
      {/* 구체화 질문 UI */}
      {clarificationQuestion && (
        <div className="rounded-lg border-2 border-blue-300 bg-blue-50 p-4">
          <div className="flex items-center gap-2 mb-3">
            <span className="text-lg">🔨</span>
            <h3 className="text-sm font-semibold text-blue-800">
              재판장의 추가 질문
            </h3>
          </div>
          <div className="bg-white rounded-lg p-3 mb-3 text-sm text-gray-700 whitespace-pre-wrap">
            {clarificationQuestion}
          </div>
          <textarea
            value={clarificationAnswer}
            onChange={(e) => setClarificationAnswer(e.target.value)}
            placeholder="위 질문에 답변해주세요. 답변을 바탕으로 사건 개요가 구체화됩니다."
            className="w-full h-24 p-3 border border-blue-200 rounded-lg text-sm resize-none focus:outline-none focus:ring-2 focus:ring-blue-400"
          />
          <button
            onClick={handleClarificationSubmit}
            disabled={!clarificationAnswer.trim()}
            className={`mt-2 w-full py-2.5 rounded-lg text-sm font-semibold transition-colors ${
              clarificationAnswer.trim()
                ? 'bg-blue-600 text-white hover:bg-blue-700'
                : 'bg-gray-200 text-gray-400 cursor-not-allowed'
            }`}
          >
            답변 제출
          </button>
        </div>
      )}

      {/* 구체화 진행 중 로딩 */}
      {isClarifying && (
        <div className="rounded-lg border-2 border-blue-200 bg-blue-50 p-4 text-center">
          <div className="animate-pulse text-sm text-blue-700">
            사건 개요를 구체화하고 있습니다...
          </div>
        </div>
      )}

      {/* 구체화 중이면 나머지 설정 숨김 */}
      {!clarificationQuestion && !isClarifying && (
        <>
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
            </div>
          )}

          {/* 시나리오 선택 */}
          <div>
            <div className="border-t border-gray-200 mb-4 pt-4">
              <p className="text-xs text-gray-400 text-center mb-4">
                또는 시나리오를 선택하세요
              </p>
            </div>
            <h3 className="text-sm font-semibold text-emerald-700 mb-2">
              시나리오 선택
            </h3>
            <p className="text-xs text-gray-500 mb-3">
              사전 정의된 시나리오를 선택하면 설정이 자동으로 채워집니다.
              선택 후 내용을 수정할 수도 있습니다.
            </p>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {SCENARIOS.map((scenario) => (
                <button
                  key={scenario.id}
                  onClick={() => handleScenarioSelect(scenario)}
                  className={`w-full p-3 rounded-lg border-2 text-left transition-colors ${
                    selectedScenarioId === scenario.id
                      ? 'border-emerald-500 bg-emerald-50'
                      : 'border-emerald-200 bg-emerald-50/30 hover:border-emerald-400 hover:bg-emerald-50'
                  }`}
                >
                  <p className={`text-sm font-semibold ${
                    selectedScenarioId === scenario.id
                      ? 'text-emerald-800'
                      : 'text-emerald-700'
                  }`}>
                    {scenario.name}
                  </p>
                  <p className="text-xs text-emerald-600 mt-1">
                    {scenario.description}
                  </p>
                </button>
              ))}
            </div>
          </div>

          {/* 직접 설정 구분선 */}
          <div className="border-t border-gray-200 pt-4">
            <p className="text-xs text-gray-400 text-center">
              또는 직접 설정
            </p>
          </div>

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
                    setSelectedScenarioId(null)
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
                    onClick={() => {
                      setCaseCategory(
                        category.id as CriminalCategory | CivilCategory
                      )
                      setSelectedScenarioId(null)
                    }}
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
                    onClick={() => {
                      setUserRole(role.id)
                      setSelectedScenarioId(null)
                    }}
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
                onChange={(e) => {
                  setCaseSummary(e.target.value.slice(0, MAX_SUMMARY_LENGTH))
                  setSelectedScenarioId(null)
                }}
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
        </>
      )}
    </div>
  )
}
