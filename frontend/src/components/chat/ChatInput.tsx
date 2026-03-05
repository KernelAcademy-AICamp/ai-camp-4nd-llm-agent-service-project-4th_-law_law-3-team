interface ChatInputProps {
  input: string
  onInputChange: (value: string) => void
  isDisabled: boolean
  chatMode: 'split' | 'floating'
  onSend: () => void
}

export function ChatInput({
  input,
  onInputChange,
  isDisabled,
  chatMode,
  onSend,
}: ChatInputProps) {
  return (
    <div
      className={`p-4 md:p-6 bg-white border-t border-black/[0.06] ${chatMode === 'floating' ? 'rounded-b-2xl' : ''}`}
    >
      <div className="relative flex items-center gap-3">
        <input
          type="text"
          value={input}
          onChange={(e) => onInputChange(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && !isDisabled && onSend()}
          placeholder="법률 질문을 입력하세요..."
          disabled={isDisabled}
          className={`flex-1 rounded-xl px-4 py-3 md:px-6 md:py-4 text-sm md:text-base focus:outline-none transition-all shadow-sm bg-[#F5F5F7] border-black/[0.06] text-[#1D1D1F] placeholder-[#86868B] focus:border-[#007AFF] focus:bg-white ${isDisabled ? 'opacity-50 cursor-not-allowed' : ''}`}
        />
        <button
          onClick={onSend}
          disabled={isDisabled || !input.trim()}
          className={`p-3 md:p-4 bg-[#007AFF] hover:bg-[#0056CC] text-white rounded-xl transition-colors shadow-sm active:scale-95 cursor-pointer ${isDisabled || !input.trim() ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          {isDisabled ? (
            <svg className="w-5 h-5 md:w-6 md:h-6 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
              />
            </svg>
          ) : (
            <svg className="w-5 h-5 md:w-6 md:h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
              />
            </svg>
          )}
        </button>
      </div>
    </div>
  )
}
