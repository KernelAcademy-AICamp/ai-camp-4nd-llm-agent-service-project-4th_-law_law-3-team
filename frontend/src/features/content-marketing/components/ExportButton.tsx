'use client'

import { useCallback } from 'react'
import type { ScriptMetadata, SectionType } from '../types'

const SECTION_TITLES: Record<SectionType, string> = {
  hooking: '1. 도입 (Hooking)',
  analysis: '2. 본론 (Legal Analysis)',
  advice_cta: '3. 결론 (Advice & CTA)',
}

interface ExportButtonProps {
  sections: Record<SectionType, string>
  metadata: ScriptMetadata | null
  topic: string
}

function buildFullScript(
  sections: Record<SectionType, string>,
  metadata: ScriptMetadata | null,
  topic: string,
  format: 'txt' | 'md',
): string {
  const sectionOrder: SectionType[] = ['hooking', 'analysis', 'advice_cta']
  const divider = format === 'md' ? '---' : '────────────────────────'
  const heading = (text: string) => (format === 'md' ? `## ${text}` : text)

  let output = format === 'md' ? `# ${topic}\n\n` : `${topic}\n${'═'.repeat(40)}\n\n`

  for (const key of sectionOrder) {
    if (sections[key]) {
      output += `${heading(SECTION_TITLES[key])}\n\n${sections[key]}\n\n`
    }
  }

  if (metadata) {
    output += `${divider}\n\n`
    output += `${heading('메타데이터')}\n\n`
    output += `설명: ${metadata.description}\n\n`
    output += `태그: ${metadata.tags.join(', ')}\n`
    output += `해시태그: ${metadata.hashtags.join(' ')}\n`
    output += `CTA: ${metadata.cta_text}\n`
  }

  output += `\n${divider}\n⚠️ 본 콘텐츠는 AI가 생성한 것으로, 법률 자문이 아닙니다.\n`

  return output
}

export function ExportButton({ sections, metadata, topic }: ExportButtonProps) {
  const hasContent = Object.values(sections).some((s) => s.length > 0)

  const handleCopy = useCallback(async () => {
    const text = buildFullScript(sections, metadata, topic, 'txt')
    await navigator.clipboard.writeText(text)
  }, [sections, metadata, topic])

  const handleDownload = useCallback(
    (format: 'txt' | 'md') => {
      const content = buildFullScript(sections, metadata, topic, format)
      const blob = new Blob([content], { type: 'text/plain;charset=utf-8' })
      const url = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = `script_${Date.now()}.${format}`
      anchor.click()
      URL.revokeObjectURL(url)
    },
    [sections, metadata, topic],
  )

  if (!hasContent) return null

  return (
    <div className="flex items-center gap-2">
      <button
        onClick={handleCopy}
        className="px-3 py-1.5 text-sm font-medium text-gray-700 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
      >
        전체 복사
      </button>
      <button
        onClick={() => handleDownload('txt')}
        className="px-3 py-1.5 text-sm font-medium text-gray-700 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
      >
        TXT 다운로드
      </button>
      <button
        onClick={() => handleDownload('md')}
        className="px-3 py-1.5 text-sm font-medium text-gray-700 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
      >
        MD 다운로드
      </button>
    </div>
  )
}
