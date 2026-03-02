'use client'

interface TagListProps {
  tags: string[] | null
  maxVisible?: number
}

export function TagList({ tags, maxVisible = 5 }: TagListProps) {
  if (!tags || tags.length === 0) return null

  const visible = tags.slice(0, maxVisible)
  const remaining = tags.length - maxVisible

  return (
    <div className="flex flex-wrap gap-1">
      {visible.map((tag) => (
        <span
          key={tag}
          className="inline-flex items-center px-2 py-0.5 rounded-full text-xs bg-gray-100 text-gray-600"
        >
          {tag}
        </span>
      ))}
      {remaining > 0 && (
        <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs bg-gray-50 text-gray-400">
          +{remaining}
        </span>
      )}
    </div>
  )
}
