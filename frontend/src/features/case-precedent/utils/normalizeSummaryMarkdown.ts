/** ai_summary 평문/마크다운 텍스트를 ReactMarkdown용으로 정규화 */
export function normalizeSummaryMarkdown(text: string): string {
  let result = text

  // ### 마크다운 헤더가 없으면 "N. 한글제목" → "### N. 한글제목" 변환
  if (!text.includes('### ')) {
    result = result.replace(/(\d{1,2})\.\s+([가-힣])/g, '\n\n### $1. $2')
  }

  // 공통 줄바꿈 보정
  result = result
    .replace(/([^\n])\s*(###\s)/g, '$1\n\n$2')
    .replace(/([^\n])\s*(- )/g, '$1\n$2')
    .replace(/([^\n])(※)/g, '$1\n\n$2')
    .trim()

  // 첫 줄(제목)을 볼드 처리
  const firstHeading = result.indexOf('\n\n###')
  if (firstHeading > 0) {
    const title = result.slice(0, firstHeading).trim()
    if (title && !title.startsWith('#') && !title.startsWith('**')) {
      result = `**${title}**${result.slice(firstHeading)}`
    }
  }

  return result
}
