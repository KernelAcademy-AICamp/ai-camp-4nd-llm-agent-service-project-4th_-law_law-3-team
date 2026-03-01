'use client'

import type { TimelineItem, EvidenceFile } from '../types'
import { PARTICIPANT_ROLE_CONFIG } from '../types'

interface GanttDetailPanelProps {
  selectedItem: TimelineItem | null
  evidenceFiles: EvidenceFile[]
  onClose: () => void
  onEvidenceClick: (evidenceId: string) => void
}

const CONFIDENCE_BADGE: Record<'high' | 'medium' | 'low', { label: string; className: string }> = {
  high: { label: '높음', className: 'bg-green-100 text-green-700' },
  medium: { label: '보통', className: 'bg-amber-100 text-amber-700' },
  low: { label: '낮음', className: 'bg-red-100 text-red-700' },
}

const EVIDENCE_TYPE_ICONS: Record<string, string> = {
  kakao_txt: '📱',
  messenger_screenshot: '💬',
  voice_recording: '🎤',
  document: '📄',
  photo: '📷',
  text_input: '✏️',
  other: '📎',
}

function getConfidenceLevel(confidence: number | undefined): 'high' | 'medium' | 'low' {
  if (confidence === undefined || confidence >= 0.8) return 'high'
  if (confidence >= 0.5) return 'medium'
  return 'low'
}

function formatDateRange(item: TimelineItem): string {
  if (item.dateStart && item.dateEnd && item.dateStart !== item.dateEnd) {
    return `${item.dateStart} ~ ${item.dateEnd}`
  }
  return item.dateStart ?? item.date
}

export function GanttDetailPanel({
  selectedItem,
  evidenceFiles,
  onClose,
  onEvidenceClick,
}: GanttDetailPanelProps) {
  if (!selectedItem) return null

  const confidenceLevel = getConfidenceLevel(selectedItem.confidence)
  const badge = CONFIDENCE_BADGE[confidenceLevel]

  const linkedEvidence = evidenceFiles.filter(
    (file) => selectedItem.evidenceIds?.includes(file.evidenceId),
  )

  return (
    <div className="fixed right-0 top-0 h-full w-96 bg-white border-l border-gray-200 shadow-xl z-50 flex flex-col overflow-hidden">
      {/* 헤더 */}
      <div className="flex items-center justify-between px-5 py-4 border-b border-gray-100">
        <h2 className="text-base font-bold text-gray-900 truncate pr-4">
          {selectedItem.title}
        </h2>
        <button
          type="button"
          onClick={onClose}
          aria-label="닫기"
          className="flex-shrink-0 p-1.5 rounded-lg hover:bg-gray-100 text-gray-500 hover:text-gray-800 transition-colors"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* 내용 */}
      <div className="flex-1 overflow-y-auto px-5 py-4 space-y-5">

        {/* 기본 정보 */}
        <section>
          <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">기본 정보</h3>
          <div className="space-y-2 text-sm">
            <div className="flex items-start gap-2">
              <span className="text-gray-400 w-14 flex-shrink-0">날짜</span>
              <span className="text-gray-800 font-medium">{formatDateRange(selectedItem)}</span>
            </div>
            {selectedItem.topic && (
              <div className="flex items-start gap-2">
                <span className="text-gray-400 w-14 flex-shrink-0">주제</span>
                <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-semibold bg-blue-100 text-blue-700">
                  {selectedItem.topic}
                </span>
              </div>
            )}
            {selectedItem.location && (
              <div className="flex items-start gap-2">
                <span className="text-gray-400 w-14 flex-shrink-0">장소</span>
                <span className="text-gray-800">{selectedItem.location}</span>
              </div>
            )}
            {selectedItem.time && (
              <div className="flex items-start gap-2">
                <span className="text-gray-400 w-14 flex-shrink-0">시간</span>
                <span className="text-gray-800">{selectedItem.time}</span>
              </div>
            )}
          </div>
        </section>

        {/* 신뢰도 배지 */}
        {selectedItem.confidence !== undefined && (
          <section>
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">추출 신뢰도</h3>
            <div className="flex items-center gap-2">
              <span className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-bold ${badge.className}`}>
                {badge.label}
              </span>
              <span className="text-xs text-gray-500">
                {(selectedItem.confidence * 100).toFixed(0)}%
              </span>
            </div>
          </section>
        )}

        {/* 상세 설명 */}
        <section>
          <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">설명</h3>
          <p className="text-sm text-gray-700 leading-relaxed">
            {selectedItem.descriptionDetailed ?? selectedItem.description}
          </p>
        </section>

        {/* 참여자 */}
        {(selectedItem.participantsDetailed && selectedItem.participantsDetailed.length > 0) && (
          <section>
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">참여자</h3>
            <div className="flex flex-wrap gap-2">
              {selectedItem.participantsDetailed.map((p, idx) => {
                const roleConfig = PARTICIPANT_ROLE_CONFIG[p.role]
                return (
                  <span
                    key={idx}
                    className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-semibold ${roleConfig.bgColor} ${roleConfig.color}`}
                  >
                    {p.name}
                    <span className="opacity-70">({roleConfig.label})</span>
                  </span>
                )
              })}
            </div>
          </section>
        )}

        {/* 참여자 (레거시) */}
        {(!selectedItem.participantsDetailed || selectedItem.participantsDetailed.length === 0) &&
          selectedItem.participants.length > 0 && (
          <section>
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">참여자</h3>
            <div className="flex flex-wrap gap-2">
              {selectedItem.participants.map((name, idx) => (
                <span key={idx} className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold bg-slate-100 text-slate-600">
                  {name}
                </span>
              ))}
            </div>
          </section>
        )}

        {/* 법적 의미 */}
        {selectedItem.legalSignificance && (
          <section>
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">법적 의미</h3>
            <p className="text-sm text-gray-700 leading-relaxed bg-amber-50 border border-amber-200 rounded-lg px-3 py-2">
              {selectedItem.legalSignificance}
            </p>
          </section>
        )}

        {/* 핵심 대사 */}
        {selectedItem.keyDialogue && (
          <section>
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">핵심 발언</h3>
            <blockquote className="border-l-4 border-blue-400 pl-3 text-sm text-gray-700 italic leading-relaxed">
              {selectedItem.keyDialogue}
            </blockquote>
          </section>
        )}

        {/* 연결된 증거 */}
        {linkedEvidence.length > 0 && (
          <section>
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
              연결된 증거 ({linkedEvidence.length})
            </h3>
            <ul className="space-y-2">
              {linkedEvidence.map((file) => (
                <li key={file.evidenceId}>
                  <button
                    type="button"
                    onClick={() => onEvidenceClick(file.evidenceId)}
                    className="w-full flex items-center gap-3 px-3 py-2 rounded-lg bg-gray-50 hover:bg-blue-50 hover:border-blue-200 border border-transparent transition-colors text-left"
                  >
                    <span className="text-xl flex-shrink-0">
                      {EVIDENCE_TYPE_ICONS[file.evidenceType] ?? '📎'}
                    </span>
                    <div className="min-w-0">
                      <p className="text-sm font-medium text-gray-800 truncate">{file.filename}</p>
                      <p className="text-xs text-gray-400">{file.fileSizeKb} KB</p>
                    </div>
                  </button>
                </li>
              ))}
            </ul>
          </section>
        )}
      </div>
    </div>
  )
}
