import type { WorkspaceCaseDetail } from '@/features/workspace/types'

interface CaseConversationsTabProps {
  caseData: WorkspaceCaseDetail
  onContinue: (conversationId: string) => void
}

export function CaseConversationsTab({ caseData, onContinue }: CaseConversationsTabProps) {
  if (caseData.conversations.length === 0) {
    return (
      <div className="text-center py-12 text-gray-500">
        <p className="text-lg">연결된 대화가 없습니다</p>
        <p className="text-sm mt-1">
          채팅에서 이 사건과 연결된 대화를 시작해보세요
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-3">
      {caseData.conversations.map((conv) => (
        <div
          key={conv.id}
          className="bg-white rounded-lg border border-gray-200 p-4 flex items-center justify-between"
        >
          <div>
            <h4 className="font-medium text-gray-900 text-sm">
              {conv.title || '제목 없는 대화'}
            </h4>
            <p className="text-xs text-gray-500 mt-1">
              {conv.message_count}개 메시지
            </p>
          </div>
          <button
            onClick={() => onContinue(conv.id)}
            className="px-3 py-1.5 text-sm font-medium text-blue-600 bg-blue-50 rounded-lg hover:bg-blue-100 transition-colors shrink-0"
          >
            이어가기
          </button>
        </div>
      ))}
    </div>
  )
}
