/** 데모 모드에서 mock AI 응답 간 딜레이 (ms) */
export const DEMO_RESPONSE_DELAY = 1200

/** 판례번호 패턴 (예: 2023다12345) */
export const CASE_NUMBER_PATTERN = /(\d{2,4}[가-힣]{1,3}\d{1,6})/g

/** 법령 참조 패턴 (예: 형사소송법 제284조, 도로교통법 제50조) */
export const LAW_REFERENCE_PATTERN =
  /([가-힣]{2,}(?:법|규칙|령|조례)(?:시행령|시행규칙)?)\s*(?:제?\s*(\d+)조(?:의\s*\d+)?)/g
