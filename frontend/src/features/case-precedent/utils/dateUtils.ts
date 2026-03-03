/** "YYYYMMDD" → "YYYY. MM. DD." */
export function formatPromulgationDate(yyyymmdd: string): string {
  if (yyyymmdd.length !== 8) return yyyymmdd
  return `${yyyymmdd.slice(0, 4)}. ${yyyymmdd.slice(4, 6)}. ${yyyymmdd.slice(6, 8)}.`
}

/** "YYYY-MM-DD" → "YYYY. MM. DD." */
export function formatIsoDate(isoDate: string): string {
  const [year, month, day] = isoDate.split('-')
  if (!year || !month || !day) return isoDate
  return `${year}. ${month}. ${day}.`
}
