'use client'

import { useEffect, useMemo } from 'react'
import { useEditor, EditorContent } from '@tiptap/react'
import StarterKit from '@tiptap/starter-kit'
import { Table } from '@tiptap/extension-table'
import { TableRow } from '@tiptap/extension-table-row'
import { TableCell } from '@tiptap/extension-table-cell'
import { TableHeader } from '@tiptap/extension-table-header'
import { marked } from 'marked'

interface ExamPaperViewProps {
  markdown: string
}

/**
 * 시험 문제 마크다운을 렌더링 전에 전처리.
 * 【섹션명】→ h2, 법조문 패턴 강조, 번호 목록 정리 등.
 */
function preprocessMarkdown(raw: string): string {
  let text = raw

  // 1) 【섹션명】 → ## 섹션명 (커스텀 구분자를 마크다운 헤딩으로)
  text = text.replace(/【([^】]+)】/g, '\n## $1\n')

  // 2) 법조문 패턴: "제N조(제목)" or "제N조의N(제목)" → 볼드 처리
  text = text.replace(
    /^(제\d+조(?:의\d+)?(?:\([^)]*\))?)/gm,
    '**$1**'
  )

  // 3) "제N항", "제N호" 등 항/호 참조 → 볼드
  text = text.replace(
    /(제\d+항|제\d+호|제\d+목)/g,
    '**$1**'
  )

  // 4) 숫자 번호 목록 (1. 2. 3.) 앞에 줄바꿈 보장
  text = text.replace(/\n(\d+)\.\s/g, '\n\n$1. ')

  // 5) "가. 나. 다." 등 한글 번호 목록 → 들여쓰기 목록
  text = text.replace(/^([가-힣])\.\s/gm, '- **$1.** ')

  // 6) 연속 빈 줄 정리 (3줄 이상 → 2줄)
  text = text.replace(/\n{3,}/g, '\n\n')

  return text.trim()
}

export function ExamPaperView({ markdown }: ExamPaperViewProps) {
  const html = useMemo(() => {
    const processed = preprocessMarkdown(markdown)
    return marked.parse(processed, { async: false }) as string
  }, [markdown])

  const editor = useEditor({
    extensions: [
      StarterKit.configure({
        heading: { levels: [1, 2, 3, 4] },
      }),
      Table.configure({ resizable: false }),
      TableRow,
      TableHeader,
      TableCell,
    ],
    content: html,
    editable: false,
    immediatelyRender: false,
    editorProps: {
      attributes: {
        class: 'outline-none focus:outline-none',
      },
    },
  })

  useEffect(() => {
    if (editor && html) {
      editor.commands.setContent(html)
    }
  }, [editor, html])

  if (!editor) return null

  return (
    <div
      className="prose prose-sm max-w-none text-gray-800
        prose-headings:font-bold prose-headings:text-gray-900 prose-headings:tracking-tight
        prose-h1:text-xl prose-h1:mb-4 prose-h1:pb-2 prose-h1:border-b prose-h1:border-gray-200
        prose-h2:text-lg prose-h2:mb-3
        prose-h3:text-base prose-h3:mb-2
        prose-p:leading-7 prose-p:my-2.5
        prose-li:leading-7 prose-li:my-0.5
        prose-strong:text-gray-900
        prose-blockquote:border-l-gray-400 prose-blockquote:text-gray-700 prose-blockquote:bg-gray-50 prose-blockquote:rounded-r-lg prose-blockquote:py-1
        prose-table:border-collapse prose-table:w-full
        prose-th:border prose-th:border-gray-300 prose-th:bg-gray-50 prose-th:px-3 prose-th:py-2 prose-th:text-left prose-th:text-xs prose-th:font-semibold
        prose-td:border prose-td:border-gray-300 prose-td:px-3 prose-td:py-2 prose-td:text-sm
        prose-hr:border-gray-300 prose-hr:my-6
        prose-code:bg-gray-100 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:text-xs
        [&_.ProseMirror]:outline-none"
    >
      <EditorContent editor={editor} />
    </div>
  )
}
