'use client'

import { useState } from 'react'
import type { DocumentResponse, DocumentType } from '../types'

interface DocumentStepProps {
  generatedDocument: DocumentResponse | null
  isGenerating: boolean
  error: string | null
  onGenerate: (type: DocumentType) => void
  onPrevious: () => void
  onReset: () => void
}

const DOCUMENT_TYPES: { type: DocumentType; name: string; description: string }[] = [
  {
    type: 'demand_letter',
    name: '내용증명',
    description: '상대방에게 공식적으로 이행을 요구하는 서류',
  },
  {
    type: 'payment_order',
    name: '지급명령신청서',
    description: '법원에 지급명령을 신청하는 서류',
  },
  {
    type: 'complaint',
    name: '소액심판청구서',
    description: '소액사건심판을 청구하는 서류',
  },
]

export function DocumentStep({
  generatedDocument,
  isGenerating,
  error,
  onGenerate,
  onPrevious,
  onReset,
}: DocumentStepProps) {
  const [selectedType, setSelectedType] = useState<DocumentType>('demand_letter')
  const [isEditing, setIsEditing] = useState(false)
  const [editedContent, setEditedContent] = useState('')

  const handleGenerate = () => {
    onGenerate(selectedType)
    setIsEditing(false)
  }

  const handleEdit = () => {
    if (generatedDocument) {
      setEditedContent(generatedDocument.content)
      setIsEditing(true)
    }
  }

  const handleDownload = () => {
    const content = isEditing ? editedContent : generatedDocument?.content
    if (!content) return

    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${generatedDocument?.title || '서류'}.txt`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  return (
    <div className="max-w-3xl mx-auto">
      <div className="text-center mb-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">서류 생성</h2>
        <p className="text-gray-600">AI가 입력하신 정보를 바탕으로 법률 서류를 작성합니다</p>
      </div>

      {/* Document Type Selection */}
      {!generatedDocument && (
        <div className="bg-white rounded-xl border border-gray-200 p-6 mb-6">
          <h3 className="font-semibold text-gray-900 mb-4">생성할 서류 선택</h3>
          <div className="space-y-3">
            {DOCUMENT_TYPES.map((doc) => (
              <label
                key={doc.type}
                className={`flex items-start gap-3 p-4 rounded-lg border cursor-pointer transition ${
                  selectedType === doc.type
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
              >
                <input
                  type="radio"
                  name="documentType"
                  value={doc.type}
                  checked={selectedType === doc.type}
                  onChange={(e) => setSelectedType(e.target.value as DocumentType)}
                  className="mt-1 h-4 w-4 text-blue-600 focus:ring-blue-500"
                />
                <div>
                  <span className="font-medium text-gray-900">{doc.name}</span>
                  <p className="text-sm text-gray-500 mt-0.5">{doc.description}</p>
                </div>
              </label>
            ))}
          </div>

          <button
            onClick={handleGenerate}
            disabled={isGenerating}
            className="w-full mt-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition flex items-center justify-center gap-2"
          >
            {isGenerating ? (
              <>
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white" />
                AI가 서류를 작성 중입니다...
              </>
            ) : (
              <>
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                  />
                </svg>
                AI로 서류 생성하기
              </>
            )}
          </button>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
          <div className="flex items-start gap-3">
            <svg
              className="w-5 h-5 text-red-500 shrink-0 mt-0.5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            <div>
              <p className="text-sm font-medium text-red-800">오류가 발생했습니다</p>
              <p className="text-sm text-red-700 mt-1">{error}</p>
            </div>
          </div>
        </div>
      )}

      {/* Generated Document */}
      {generatedDocument && (
        <div className="bg-white rounded-xl border border-gray-200 overflow-hidden mb-6">
          <div className="px-6 py-4 border-b border-gray-200 bg-gray-50 flex items-center justify-between">
            <div>
              <h3 className="font-semibold text-gray-900">{generatedDocument.title}</h3>
              <p className="text-sm text-gray-500">AI가 생성한 서류입니다. 검토 후 수정할 수 있습니다.</p>
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => {
                  setIsEditing(false)
                  onGenerate(selectedType)
                }}
                disabled={isGenerating}
                className="px-3 py-1.5 text-sm border border-gray-300 text-gray-600 rounded-lg hover:bg-gray-100 transition"
              >
                다시 생성
              </button>
              <button
                onClick={isEditing ? () => setIsEditing(false) : handleEdit}
                className="px-3 py-1.5 text-sm border border-gray-300 text-gray-600 rounded-lg hover:bg-gray-100 transition"
              >
                {isEditing ? '미리보기' : '수정하기'}
              </button>
            </div>
          </div>

          <div className="p-6">
            {isEditing ? (
              <textarea
                value={editedContent}
                onChange={(e) => setEditedContent(e.target.value)}
                className="w-full h-96 p-4 font-mono text-sm border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
              />
            ) : (
              <div className="bg-gray-50 rounded-lg p-6 font-mono text-sm whitespace-pre-wrap leading-relaxed max-h-96 overflow-y-auto">
                {isEditing ? editedContent : generatedDocument.content}
              </div>
            )}
          </div>

          <div className="px-6 py-4 border-t border-gray-200 bg-gray-50 flex justify-end">
            <button
              onClick={handleDownload}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition flex items-center gap-2"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                />
              </svg>
              다운로드 (.txt)
            </button>
          </div>
        </div>
      )}

      {/* Tips */}
      <div className="bg-blue-50 rounded-lg p-4 mb-6">
        <div className="flex items-start gap-3">
          <svg
            className="w-5 h-5 text-blue-600 shrink-0 mt-0.5"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          <div>
            <p className="text-sm font-medium text-blue-800">서류 활용 안내</p>
            <ul className="text-sm text-blue-700 mt-1 space-y-1">
              <li>• 내용증명: 우체국에서 내용증명 양식에 옮겨 발송하세요</li>
              <li>• 지급명령/소액심판: 법원 전자소송(ecourt.scourt.go.kr)에서 제출하세요</li>
              <li>• 생성된 서류는 참고용이며, 법률 전문가 검토를 권장합니다</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <div className="flex justify-between">
        <button
          onClick={onPrevious}
          className="px-6 py-3 border border-gray-300 text-gray-700 rounded-lg font-medium hover:bg-gray-50 transition"
        >
          이전
        </button>
        <button
          onClick={onReset}
          className="px-6 py-3 border border-blue-600 text-blue-600 rounded-lg font-medium hover:bg-blue-50 transition"
        >
          처음부터 다시 시작
        </button>
      </div>
    </div>
  )
}
