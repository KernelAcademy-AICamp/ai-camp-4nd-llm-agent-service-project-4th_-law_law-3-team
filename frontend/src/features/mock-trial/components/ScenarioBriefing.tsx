'use client'

import type { DemoScenario } from '../demo/demo-scenarios'
import type { GeneratedScenario } from '../types'
import { PHYSICAL_EVIDENCE_TYPE_LABEL } from '../types'

interface ScenarioBriefingProps {
  scenario?: DemoScenario | null
  generatedScenario?: GeneratedScenario | null
  onStart: () => void
  onRegenerate?: () => void
  isRegenerating?: boolean
}

const ROLE_LABEL: Record<string, { icon: string; color: string }> = {
  judge: { icon: '👨‍⚖️', color: 'bg-purple-100 text-purple-700' },
  prosecutor: { icon: '🏛️', color: 'bg-red-100 text-red-700' },
  attorney: { icon: '📋', color: 'bg-blue-100 text-blue-700' },
  defendant: { icon: '🧑', color: 'bg-gray-100 text-gray-700' },
  clerk: { icon: '✏️', color: 'bg-green-100 text-green-700' },
}

const EVIDENCE_HINT_ICON: Record<string, string> = {
  document: '📄',
  video: '📹',
  financial: '🏦',
  photo: '📷',
  testimony: '🗣️',
  other: '📎',
}

export function ScenarioBriefing({
  scenario,
  generatedScenario,
  onStart,
  onRegenerate,
  isRegenerating,
}: ScenarioBriefingProps) {
  // 데모 모드 vs 일반 모드 데이터 선택
  const isDemo = !!scenario
  const title = isDemo ? scenario.name : generatedScenario?.title ?? '시나리오'
  const description = isDemo ? scenario.description : undefined
  const background = isDemo
    ? scenario.setup.caseSummary
    : generatedScenario?.background ?? ''
  const caseTypeLabel = isDemo
    ? scenario.setup.caseType === 'criminal' ? '형사' : '민사'
    : undefined
  const characters = isDemo
    ? scenario.characters ?? []
    : generatedScenario?.characters ?? []
  const objectives = isDemo
    ? scenario.objectives ?? []
    : generatedScenario?.objectives ?? []
  const evidence = isDemo ? scenario.evidence ?? [] : []
  const evidenceHints = isDemo ? [] : generatedScenario?.evidence_hints ?? []
  const issues = isDemo ? [] : generatedScenario?.issues ?? []

  return (
    <div className="space-y-5">
      {/* 헤더 */}
      <div>
        <div className="flex items-center gap-2 mb-2">
          {caseTypeLabel && (
            <span className="text-xs px-2 py-0.5 rounded-full bg-blue-100 text-blue-700 font-medium">
              {caseTypeLabel}
            </span>
          )}
          <span
            className={`text-xs px-2 py-0.5 rounded-full font-medium ${
              isDemo
                ? 'bg-amber-100 text-amber-700'
                : 'bg-emerald-100 text-emerald-700'
            }`}
          >
            {isDemo ? '데모' : 'AI 생성'}
          </span>
        </div>
        <h2 className="text-lg font-bold text-gray-900">{title}</h2>
        {description && (
          <p className="text-sm text-gray-500 mt-1">{description}</p>
        )}
      </div>

      {/* 사건 개요 */}
      <div>
        <h3 className="text-sm font-semibold text-gray-700 mb-2">사건 개요</h3>
        <div className="p-3 rounded-lg bg-gray-50 border border-gray-200">
          <p className="text-sm text-gray-700 leading-relaxed">{background}</p>
        </div>
      </div>

      {/* 쟁점 (LLM 생성 시나리오에만) */}
      {issues.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold text-gray-700 mb-2">
            핵심 쟁점
          </h3>
          <div className="p-3 rounded-lg bg-orange-50 border border-orange-200">
            <ul className="space-y-1.5">
              {issues.map((issue, index) => (
                <li
                  key={`issue-${index}`}
                  className="flex items-start gap-2 text-sm text-orange-800"
                >
                  <span className="text-orange-500 mt-0.5 shrink-0">
                    {index + 1}.
                  </span>
                  <span>{issue}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {/* 사용자 역할 & 목표 */}
      {objectives.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold text-gray-700 mb-2">
            당신의 목표
          </h3>
          <div className="p-3 rounded-lg bg-blue-50 border border-blue-200">
            <ul className="space-y-1.5">
              {objectives.map((objective, index) => (
                <li
                  key={`obj-${index}`}
                  className="flex items-start gap-2 text-sm text-blue-800"
                >
                  <span className="text-blue-500 mt-0.5 shrink-0">
                    {index + 1}.
                  </span>
                  <span>{objective}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {/* 등장인물 */}
      {characters.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold text-gray-700 mb-2">
            등장인물
          </h3>
          <div className="space-y-2">
            {characters.map((character) => {
              const roleInfo = ROLE_LABEL[character.role] ?? {
                icon: '👤',
                color: 'bg-gray-100 text-gray-700',
              }
              const isUser = character.description.includes('사용자 역할')
              return (
                <div
                  key={character.role}
                  className={`flex items-center gap-3 p-2.5 rounded-lg border ${
                    isUser
                      ? 'border-blue-300 bg-blue-50'
                      : 'border-gray-200 bg-white'
                  }`}
                >
                  <span className="text-lg">{roleInfo.icon}</span>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-1.5">
                      <span className="text-sm font-medium text-gray-800">
                        {character.name}
                      </span>
                      {isUser && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-blue-500 text-white font-medium">
                          YOU
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-gray-500">
                      {character.description}
                    </p>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* 물적 증거 미리보기 (데모 모드) */}
      {evidence.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold text-gray-700 mb-2">
            증거 목록
          </h3>
          <div className="space-y-2">
            {evidence.map((item) => {
              const typeInfo = PHYSICAL_EVIDENCE_TYPE_LABEL[item.type]
              const favorableColor =
                item.favorable_to === 'prosecutor'
                  ? 'text-red-500'
                  : item.favorable_to === 'attorney'
                    ? 'text-blue-500'
                    : 'text-gray-400'
              const favorableLabel =
                item.favorable_to === 'prosecutor'
                  ? '검찰측 유리'
                  : item.favorable_to === 'attorney'
                    ? '변호측 유리'
                    : '중립'
              return (
                <div
                  key={item.id}
                  className="p-2.5 rounded-lg border border-gray-200 bg-white"
                >
                  <div className="flex items-center gap-1.5">
                    <span className="text-sm">{typeInfo.icon}</span>
                    <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-indigo-100 text-indigo-700 font-medium">
                      {typeInfo.label}
                    </span>
                    <span className="text-sm font-medium text-gray-800 flex-1 truncate">
                      {item.title}
                    </span>
                    <span
                      className={`text-[10px] font-medium ${favorableColor}`}
                    >
                      {favorableLabel}
                    </span>
                  </div>
                  <p className="text-xs text-gray-500 mt-1">
                    {item.description}
                  </p>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* 증거 힌트 (LLM 생성 시나리오) */}
      {evidenceHints.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold text-gray-700 mb-2">
            예상 증거
          </h3>
          <div className="space-y-2">
            {evidenceHints.map((hint, index) => {
              const icon = EVIDENCE_HINT_ICON[hint.type] ?? '📎'
              const favorableColor =
                hint.favorable_to === 'prosecutor'
                  ? 'text-red-500'
                  : hint.favorable_to === 'attorney'
                    ? 'text-blue-500'
                    : 'text-gray-400'
              const favorableLabel =
                hint.favorable_to === 'prosecutor'
                  ? '검찰측 유리'
                  : hint.favorable_to === 'attorney'
                    ? '변호측 유리'
                    : '중립'
              return (
                <div
                  key={`hint-${index}`}
                  className="p-2.5 rounded-lg border border-gray-200 bg-white"
                >
                  <div className="flex items-center gap-1.5">
                    <span className="text-sm">{icon}</span>
                    <span className="text-sm font-medium text-gray-800 flex-1 truncate">
                      {hint.title}
                    </span>
                    <span
                      className={`text-[10px] font-medium ${favorableColor}`}
                    >
                      {favorableLabel}
                    </span>
                  </div>
                  <p className="text-xs text-gray-500 mt-1">
                    {hint.description}
                  </p>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* 버튼 */}
      <div className="space-y-2">
        <button
          onClick={onStart}
          disabled={isRegenerating}
          className="w-full py-3 rounded-lg text-sm font-semibold bg-blue-600 text-white hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          재판 시작
        </button>
        {onRegenerate && (
          <button
            onClick={onRegenerate}
            disabled={isRegenerating}
            className="w-full py-2.5 rounded-lg text-sm font-medium border border-gray-300 text-gray-700 hover:bg-gray-50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isRegenerating ? '시나리오 재생성 중...' : '시나리오 재생성'}
          </button>
        )}
      </div>
    </div>
  )
}
