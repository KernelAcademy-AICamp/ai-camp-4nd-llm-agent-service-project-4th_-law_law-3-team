import type { ReactNode, RefObject } from 'react'
import { createElement } from 'react'

/** 하이라이팅 구간 */
interface HighlightRange {
  start: number
  end: number
}

/** 하이라이팅 최소 문자 길이 (오탐 방지) */
const MIN_MATCH_LENGTH = 30

/**
 * 연속 공백/줄바꿈을 단일 공백으로 정규화
 */
function normalizeWhitespace(text: string): string {
  return text.replace(/\s+/g, ' ').trim()
}

/**
 * 정규화된 텍스트의 인덱스를 원본 텍스트의 인덱스로 매핑하는 테이블 생성
 * normalizedIndex[i] = 원본 텍스트에서의 위치
 */
function buildIndexMap(original: string): number[] {
  const map: number[] = []
  let inWhitespace = false
  let leading = true

  for (let i = 0; i < original.length; i++) {
    const isWs = /\s/.test(original[i])

    if (leading) {
      if (!isWs) {
        leading = false
        map.push(i)
        inWhitespace = false
      }
      continue
    }

    if (isWs) {
      if (!inWhitespace) {
        map.push(i) // 공백 그룹의 첫 번째 → 정규화된 단일 공백에 대응
        inWhitespace = true
      }
      // 연속 공백은 스킵
    } else {
      map.push(i)
      inWhitespace = false
    }
  }

  return map
}

/**
 * 원본 텍스트에서 검색 텍스트의 하이라이팅 구간을 찾는다.
 * 공백/줄바꿈 차이를 무시하고 매칭한다.
 */
export function findHighlightRanges(
  fullText: string,
  searchText: string,
): HighlightRange[] {
  if (!fullText || !searchText) return []

  const normalizedSearch = normalizeWhitespace(searchText)
  if (normalizedSearch.length < MIN_MATCH_LENGTH) return []

  const normalizedFull = normalizeWhitespace(fullText)
  const indexMap = buildIndexMap(fullText)

  // 전체 매칭 시도
  const matchIndex = normalizedFull.indexOf(normalizedSearch)
  if (matchIndex !== -1) {
    const startOrig = indexMap[matchIndex] ?? 0
    const endNormIdx = matchIndex + normalizedSearch.length - 1
    const endOrig = (indexMap[endNormIdx] ?? fullText.length - 1) + 1
    return [{ start: startOrig, end: endOrig }]
  }

  // 전체 매칭 실패 시 문장 단위 분할 매칭
  const sentences = splitIntoSentences(normalizedSearch)
  const ranges: HighlightRange[] = []

  for (const sentence of sentences) {
    if (sentence.length < MIN_MATCH_LENGTH) continue
    const sentIdx = normalizedFull.indexOf(sentence)
    if (sentIdx !== -1) {
      const startOrig = indexMap[sentIdx] ?? 0
      const endNormIdx = sentIdx + sentence.length - 1
      const endOrig = (indexMap[endNormIdx] ?? fullText.length - 1) + 1
      ranges.push({ start: startOrig, end: endOrig })
    }
  }

  return mergeRanges(ranges)
}

/**
 * 텍스트를 문장 단위로 분리
 */
function splitIntoSentences(text: string): string[] {
  // 마침표+공백, 물음표, 느낌표 기준으로 분리
  return text
    .split(/(?<=[.?!])\s+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
}

/**
 * 겹치거나 인접한 구간을 병합
 */
function mergeRanges(ranges: HighlightRange[]): HighlightRange[] {
  if (ranges.length === 0) return []
  const sorted = [...ranges].sort((a, b) => a.start - b.start)
  const merged: HighlightRange[] = [sorted[0]]

  for (let i = 1; i < sorted.length; i++) {
    const last = merged[merged.length - 1]
    if (sorted[i].start <= last.end) {
      last.end = Math.max(last.end, sorted[i].end)
    } else {
      merged.push(sorted[i])
    }
  }

  return merged
}

/**
 * 텍스트를 하이라이트 구간에 따라 React 요소 배열로 분할.
 * 매칭 구간은 <mark> 태그로 감싸고, 첫 번째 매칭에 ref를 부여한다.
 */
export function splitByHighlights(
  text: string,
  ranges: HighlightRange[],
  firstMatchRef?: RefObject<HTMLElement | null>,
): ReactNode[] {
  if (ranges.length === 0) return [text]

  const elements: ReactNode[] = []
  let lastIndex = 0
  let isFirstMatch = true

  for (const range of ranges) {
    // 매칭 전 텍스트
    if (range.start > lastIndex) {
      elements.push(text.slice(lastIndex, range.start))
    }

    // 매칭 텍스트 (<mark>)
    const matchedText = text.slice(range.start, range.end)
    const props: Record<string, unknown> = {
      key: `hl-${range.start}`,
      className: 'bg-yellow-200 rounded-sm px-0.5',
    }

    if (isFirstMatch && firstMatchRef) {
      props.ref = firstMatchRef
      isFirstMatch = false
    }

    elements.push(createElement('mark', props, matchedText))
    lastIndex = range.end
  }

  // 나머지 텍스트
  if (lastIndex < text.length) {
    elements.push(text.slice(lastIndex))
  }

  return elements
}
