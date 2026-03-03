'use client'

interface PersonaWelcomeProps {
  chatHistoryCount: number | null
  error: string | null
  onStartAnalysis: () => void
  onStartOnboarding: () => void
  onSkip: () => void
}

export function PersonaWelcome({
  chatHistoryCount,
  error,
  onStartAnalysis,
  onStartOnboarding,
  onSkip,
}: PersonaWelcomeProps) {
  const hasSufficientHistory = chatHistoryCount !== null && chatHistoryCount >= 5

  return (
    <div className="max-w-xl mx-auto py-10 px-4">
      {/* 에러 배너 (분석 실패 후 복귀 시) */}
      {error && (
        <div className="bg-amber-50 border border-amber-200 text-amber-800 text-sm rounded-lg px-4 py-3 mb-6">
          {error}
        </div>
      )}

      {/* 헤드라인 */}
      <div className="text-center mb-8">
        <h2 className="text-xl font-bold text-slate-800">
          당신의 전문성에 맞춘 콘텐츠를 만들어 보세요
        </h2>
        <p className="text-sm text-gray-500 mt-2">
          AI가 분석하거나, 직접 설정할 수 있습니다.
          <br />약 1분이면 맞춤 콘텐츠 생성을 시작할 수 있어요.
        </p>
      </div>

      {/* 가치 제안: 적용 전/후 비교 */}
      <div className="bg-gray-50 rounded-xl p-4 mb-8">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-xs font-medium text-gray-400 mb-1">적용 전</p>
            <p className="text-sm text-gray-500 italic">&ldquo;법률 콘텐츠 생성&rdquo;</p>
          </div>
          <div>
            <p className="text-xs font-medium text-blue-500 mb-1">적용 후</p>
            <p className="text-sm text-slate-800 font-medium">
              &ldquo;형사법 전문가의 음주운전 실전 가이드&rdquo;
            </p>
          </div>
        </div>
      </div>

      {/* 트랙 선택 버튼 */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-6">
        {/* Track 1: AI 자동 분석 */}
        <button
          onClick={onStartAnalysis}
          className="flex flex-col items-start p-5 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition-all text-left shadow-sm"
        >
          <span className="text-sm font-bold mb-1">AI로 자동 분석하기</span>
          <span className="text-xs text-blue-200">
            기존 대화 이력을 기반으로 AI가 자동 분석합니다
          </span>
          {hasSufficientHistory && (
            <span className="inline-flex items-center mt-2 px-2 py-0.5 text-xs font-medium bg-blue-500 rounded-full">
              {chatHistoryCount}건의 대화 분석 가능
            </span>
          )}
        </button>

        {/* Track 2: 직접 설정 */}
        <button
          onClick={onStartOnboarding}
          className="flex flex-col items-start p-5 bg-white border border-gray-300 text-gray-700 rounded-xl hover:border-blue-300 hover:shadow-sm transition-all text-left"
        >
          <span className="text-sm font-bold mb-1">직접 설정하기</span>
          <span className="text-xs text-gray-500">
            전문분야, 톤, 타겟 독자를 직접 선택합니다
          </span>
          <span className="text-xs text-gray-400 mt-2">약 1분 소요</span>
        </button>
      </div>

      {/* 건너뛰기 */}
      <div className="text-center">
        <button
          onClick={onSkip}
          className="text-xs text-gray-400 hover:text-gray-600 transition-colors"
        >
          나중에 설정하기
        </button>
      </div>
    </div>
  )
}
