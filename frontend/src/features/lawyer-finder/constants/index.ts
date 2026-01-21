// 서울 구별 좌표 데이터
export const DISTRICT_COORDS: Record<string, { lat: number; lng: number }> = {
  '강남구': { lat: 37.5172, lng: 127.0473 },
  '강동구': { lat: 37.5301, lng: 127.1238 },
  '강북구': { lat: 37.6396, lng: 127.0257 },
  '강서구': { lat: 37.5509, lng: 126.8495 },
  '관악구': { lat: 37.4784, lng: 126.9516 },
  '광진구': { lat: 37.5385, lng: 127.0823 },
  '구로구': { lat: 37.4954, lng: 126.8874 },
  '금천구': { lat: 37.4519, lng: 126.9020 },
  '노원구': { lat: 37.6542, lng: 127.0568 },
  '도봉구': { lat: 37.6688, lng: 127.0471 },
  '동대문구': { lat: 37.5744, lng: 127.0396 },
  '동작구': { lat: 37.5124, lng: 126.9393 },
  '마포구': { lat: 37.5663, lng: 126.9014 },
  '서대문구': { lat: 37.5791, lng: 126.9368 },
  '서초구': { lat: 37.4837, lng: 127.0324 },
  '성동구': { lat: 37.5633, lng: 127.0371 },
  '성북구': { lat: 37.5894, lng: 127.0167 },
  '송파구': { lat: 37.5145, lng: 127.1066 },
  '양천구': { lat: 37.5170, lng: 126.8667 },
  '영등포구': { lat: 37.5264, lng: 126.8962 },
  '용산구': { lat: 37.5324, lng: 126.9906 },
  '은평구': { lat: 37.6027, lng: 126.9291 },
  '종로구': { lat: 37.5735, lng: 126.9790 },
  '중구': { lat: 37.5636, lng: 126.9976 },
  '중랑구': { lat: 37.6063, lng: 127.0925 },
}

// 서울 구 목록 (DISTRICT_COORDS의 키 배열)
export const SEOUL_DISTRICTS = Object.keys(DISTRICT_COORDS)

// 전문분야 12대분류
export interface SpecialtyCategory {
  id: string
  name: string
  icon: string
  description: string
  specialties: string[]
}

export const SPECIALTY_CATEGORIES: SpecialtyCategory[] = [
  {
    id: 'civil-family',
    name: '민사·가사',
    icon: '👨‍👩‍👧',
    description: '개인 간 분쟁 / 가족 관계',
    specialties: ['민사법', '손해배상', '민사집행', '가사법', '이혼', '상속', '성년후견', '소년법'],
  },
  {
    id: 'criminal',
    name: '형사',
    icon: '⚖️',
    description: '범죄, 수사, 재판',
    specialties: ['형사법', '군형법'],
  },
  {
    id: 'real-estate',
    name: '부동산·건설',
    icon: '🏗️',
    description: '부동산 거래·개발·분쟁',
    specialties: ['부동산', '건설', '임대차관련법', '재개발·재건축', '수용 및 보상', '등기·경매'],
  },
  {
    id: 'labor',
    name: '노동·산재',
    icon: '👷',
    description: '근로관계, 산업재해',
    specialties: ['노동법', '산재'],
  },
  {
    id: 'corporate',
    name: '기업·상사',
    icon: '🏢',
    description: '기업 운영·거래·분쟁',
    specialties: ['회사법', '상사법', '인수합병', '영업비밀', '채권추심'],
  },
  {
    id: 'finance',
    name: '금융·자본시장',
    icon: '💰',
    description: '금융 규제, 자본, 구조조정',
    specialties: ['금융', '증권', '보험', '도산'],
  },
  {
    id: 'tax',
    name: '조세·관세',
    icon: '🧾',
    description: '세금·통관',
    specialties: ['조세법', '관세'],
  },
  {
    id: 'public',
    name: '공정·행정·공공',
    icon: '🏛️',
    description: '국가·공공기관 상대 사건',
    specialties: ['공정거래', '국가계약', '행정법'],
  },
  {
    id: 'ip',
    name: '지식재산(IP)',
    icon: '💡',
    description: '기술·콘텐츠 권리 보호',
    specialties: ['특허', '저작권'],
  },
  {
    id: 'it-media',
    name: 'IT·미디어·콘텐츠',
    icon: '📱',
    description: '플랫폼, 데이터, 콘텐츠 산업',
    specialties: ['IT', '언론·방송통신', '엔터테인먼트', '스포츠'],
  },
  {
    id: 'medical',
    name: '의료·바이오·식품',
    icon: '🏥',
    description: '의료 분쟁 + 규제',
    specialties: ['의료', '식품·의약'],
  },
  {
    id: 'international',
    name: '국제·해외',
    icon: '🌐',
    description: '국제 거래·분쟁·이동',
    specialties: ['국제관계법', '국제중재', '중재', '해외투자', '해상', '이주 및 비자'],
  },
]

// 전문분야 → 대분류 매핑 (역방향 조회용)
export const SPECIALTY_TO_CATEGORY: Record<string, string> = {}
SPECIALTY_CATEGORIES.forEach((cat) => {
  cat.specialties.forEach((spec) => {
    SPECIALTY_TO_CATEGORY[spec] = cat.id
  })
})
