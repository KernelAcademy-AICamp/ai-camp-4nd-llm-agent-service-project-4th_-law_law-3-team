'use client'

interface ScoreBarProps {
  score: number
  label: string
  maxScore?: number
}

export function ScoreBar({ score, label, maxScore = 1 }: ScoreBarProps) {
  const percentage = Math.min((score / maxScore) * 100, 100)

  return (
    <div className="flex items-center gap-2 text-sm">
      <span className="w-16 text-gray-500 shrink-0">{label}</span>
      <div className="flex-1 h-2 bg-gray-200 rounded-full overflow-hidden">
        <div
          className="h-full bg-blue-500 rounded-full transition-all duration-300"
          style={{ width: `${percentage}%` }}
        />
      </div>
      <span className="w-10 text-right text-gray-600 shrink-0">
        {score.toFixed(2)}
      </span>
    </div>
  )
}
