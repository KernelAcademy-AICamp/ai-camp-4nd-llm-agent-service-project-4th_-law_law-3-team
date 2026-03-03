/** ISO8601 날짜 문자열을 한국어 날짜로 포맷 */
export function formatDate(isoString: string | null): string {
  if (!isoString) return '날짜 없음'

  const date = new Date(isoString)
  if (isNaN(date.getTime())) return '날짜 없음'

  return date.toLocaleDateString('ko-KR', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  })
}

/** ISO8601 날짜 문자열을 YYYY-MM-DD 형식으로 포맷 */
export function formatDateShort(isoString: string | null): string {
  if (!isoString) return '-'

  const date = new Date(isoString)
  if (isNaN(date.getTime())) return '-'

  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const day = String(date.getDate()).padStart(2, '0')
  return `${year}-${month}-${day}`
}
