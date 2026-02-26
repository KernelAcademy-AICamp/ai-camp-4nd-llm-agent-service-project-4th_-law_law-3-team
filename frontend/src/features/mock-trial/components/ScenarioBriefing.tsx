'use client'

import type { DemoScenario } from '../demo/demo-scenarios'
import { PHYSICAL_EVIDENCE_TYPE_LABEL } from '../types'

interface ScenarioBriefingProps {
  scenario: DemoScenario
  onStart: () => void
}

const ROLE_LABEL: Record<string, { icon: string; color: string }> = {
  judge: { icon: '👨‍⚖️', color: 'bg-purple-100 text-purple-700' },
  prosecutor: { icon: '🏛️', color: 'bg-red-100 text-red-700' },
  attorney: { icon: '📋', color: 'bg-blue-100 text-blue-700' },
  defendant: { icon: '🧑', color: 'bg-gray-100 text-gray-700' },
  clerk: { icon: '✏️', color: 'bg-green-100 text-green-700' },
}

export function ScenarioBriefing({ scenario, onStart }: ScenarioBriefingProps) {
  const { setup, characters = [], objectives = [], evidence = [] } = scenario
  const caseTypeLabel = setup.caseType === 'criminal' ? '형사' : '민사'

  return (
    <div className="space-y-5">
      {/* 헤더 */}
      <div>
        <div className="flex items-center gap-2 mb-2">
          <span className="text-xs px-2 py-0.5 rounded-full bg-blue-100 text-blue-700 font-medium">
            {caseTypeLabel}
          </span>
          <span className="text-xs px-2 py-0.5 rounded-full bg-amber-100 text-amber-700 font-medium">
            데모
          </span>
        </div>
        <h2 className="text-lg font-bold text-gray-900">{scenario.name}</h2>
        <p className="text-sm text-gray-500 mt-1">{scenario.description}</p>
      </div>

      {/* 사건 개요 */}
      <div>
        <h3 className="text-sm font-semibold text-gray-700 mb-2">사건 개요</h3>
        <div className="p-3 rounded-lg bg-gray-50 border border-gray-200">
          <p className="text-sm text-gray-700 leading-relaxed">
            {setup.caseSummary}
          </p>
        </div>
      </div>

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
                    <p className="text-xs text-gray-500">{character.description}</p>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* 물적 증거 미리보기 */}
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
                    <span className={`text-[10px] font-medium ${favorableColor}`}>
                      {favorableLabel}
                    </span>
                  </div>
                  <p className="text-xs text-gray-500 mt-1">{item.description}</p>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* 시작 버튼 */}
      <button
        onClick={onStart}
        className="w-full py-3 rounded-lg text-sm font-semibold bg-blue-600 text-white hover:bg-blue-700 transition-colors"
      >
        재판 시작
      </button>
    </div>
  )
}
