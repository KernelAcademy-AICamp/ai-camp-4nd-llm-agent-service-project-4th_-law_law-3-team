'use client'

import { useState, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import { useChat } from '@/context/ChatContext'
import type { ChatSource } from '../types'

// doc_type을 한글로 변환
const docTypeLabels: Record<string, string> = {
  precedent: '판례',
  constitutional: '헌재결정',
  administration: '행정심판',
  legislation: '법령해석',
  committee: '위원회결정',
  law: '법령',
}

const getDocTypeLabel = (docType: string): string => {
  return docTypeLabels[docType] || docType
}

// doc_type별 배지 색상
const getDocTypeBadgeColor = (docType: string): string => {
  switch (docType) {
    case 'law':
      return 'bg-green-50 text-green-600'
    case 'precedent':
      return 'bg-blue-50 text-blue-600'
    case 'constitutional':
      return 'bg-purple-50 text-purple-600'
    case 'committee':
      return 'bg-orange-50 text-orange-600'
    case 'administration':
      return 'bg-yellow-50 text-yellow-700'
    case 'legislation':
      return 'bg-teal-50 text-teal-600'
    default:
      return 'bg-gray-50 text-gray-600'
  }
}

export function UserView() {
  const { sessionData } = useChat()
  const [references, setReferences] = useState<ChatSource[]>([])
  const [selectedRef, setSelectedRef] = useState<ChatSource | null>(null)

  useEffect(() => {
    // 세션 데이터에서 챗봇 참조 자료를 로드합니다.
    if (sessionData.aiReferences && Array.isArray(sessionData.aiReferences)) {
      setReferences(sessionData.aiReferences as ChatSource[])
      // UserView에서는 명시적으로 선택하기 전까지 리스트를 보여줍니다.
      // setSelectedRef(null) 
    }
  }, [sessionData.aiReferences])

  if (references.length === 0) {
    return (
      <div className="h-full flex flex-col items-center justify-center bg-gray-50 text-center p-8 animate-in fade-in duration-500">
        <div className="bg-white p-8 rounded-3xl shadow-sm border border-gray-100 max-w-md">
          <span className="text-6xl mb-6 block">🤖</span>
          <h2 className="text-2xl font-bold text-gray-900 mb-3">
            챗봇에게 질문해보세요!
          </h2>
          <p className="text-gray-500 leading-relaxed">
            &quot;사기죄 성립 요건이 뭐야?&quot;<br />
            &quot;야간 주거침입 시 정당방위는?&quot;
            <br /><br />
            오른쪽 챗봇에게 법률 문제를 물어보면,<br />
            참고한 <strong>판례와 법령 상세 정보</strong>가 이곳에 표시됩니다.
          </p>
        </div>
      </div>
    )
  }

  // Detail View
  if (selectedRef) {
    const isLaw = selectedRef.doc_type === 'law'
    const title = isLaw ? selectedRef.law_name : selectedRef.case_name
    const subtitle = isLaw ? selectedRef.law_type : selectedRef.case_number

    return (
      <div className="h-full flex flex-col bg-white animate-in slide-in-from-right duration-300">
        <div className="p-4 border-b border-gray-100 flex items-center gap-3">
          <button
            onClick={() => setSelectedRef(null)}
            className="p-2 hover:bg-gray-100 rounded-full transition-colors text-gray-500"
          >
            ←
          </button>
          <span className="font-bold text-gray-900">상세 내용</span>
        </div>

        <div className="flex-1 overflow-y-auto p-6 md:p-8">
          <div className="max-w-3xl mx-auto">
            <div className="mb-8 pb-6 border-b border-gray-100">
              <span className={`inline-block px-3 py-1 rounded-full text-sm font-medium mb-4 ${getDocTypeBadgeColor(selectedRef.doc_type)}`}>
                {getDocTypeLabel(selectedRef.doc_type)}
              </span>
              <h1 className="text-2xl font-bold text-gray-900 leading-tight mb-4">
                {title || '상세 정보'}
              </h1>
              {subtitle && (
                <div className="text-gray-500 font-mono text-sm bg-gray-50 px-3 py-1 rounded inline-block">
                  {subtitle}
                </div>
              )}
            </div>

            <div className="prose prose-lg max-w-none text-gray-700">
              <div className="bg-gray-50 p-6 rounded-2xl border border-gray-100 leading-relaxed">
                <ReactMarkdown>
                    {selectedRef.content || "상세 내용을 불러올 수 없습니다."}
                </ReactMarkdown>
              </div>
            </div>
          </div>
        </div>
      </div>
    )
  }

  // List View
  return (
    <div className="h-full flex flex-col bg-gray-50 animate-in slide-in-from-left duration-300">
      <div className="p-6 border-b border-gray-100 bg-white">
        <h2 className="text-xl font-bold text-gray-900 flex items-center gap-2">
          <span>📚</span> 챗봇 참조 자료
        </h2>
        <p className="text-sm text-gray-500 mt-2">
          챗봇이 답변을 생성할 때 참고한 근거 자료들입니다.<br/>
          자세히 보려면 항목을 클릭하세요.
        </p>
      </div>
      
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {references.map((ref, idx) => {
          const isLaw = ref.doc_type === 'law'
          const title = isLaw ? ref.law_name : ref.case_name
          const subtitle = isLaw ? ref.law_type : ref.case_number

          return (
            <button
              key={`${ref.case_number || ref.law_name}-${idx}`}
              onClick={() => setSelectedRef(ref)}
              className="w-full text-left p-5 rounded-xl bg-white border border-gray-100 hover:border-blue-300 hover:shadow-md transition-all duration-200 group"
            >
              <div className="flex items-center gap-2 mb-2">
                <span className={`px-2 py-0.5 rounded text-xs font-bold ${getDocTypeBadgeColor(ref.doc_type)}`}>
                  {getDocTypeLabel(ref.doc_type)}
                </span>
                <span className="text-xs text-gray-400 font-mono group-hover:text-blue-500 transition-colors">
                  {subtitle}
                </span>
              </div>
              <h3 className="font-bold text-lg text-gray-900 mb-2 group-hover:text-blue-700 transition-colors">
                {title || '제목 없음'}
              </h3>
              <p className="text-sm text-gray-500 line-clamp-2">
                {ref.content ? ref.content.slice(0, 150) + '...' : '클릭하여 상세 내용을 확인하세요.'}
              </p>
            </button>
          )
        })}
      </div>
    </div>
  )
}
