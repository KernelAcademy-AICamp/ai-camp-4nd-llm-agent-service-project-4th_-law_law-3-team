/**
 * 법령 유형(law_type)에 따른 로고 파일 맵핑
 * 로고 파일 위치: /public/data/logo/
 */

// law_type → 로고 파일명 맵핑
const LAW_TYPE_LOGO_MAP: Record<string, string> = {
  // 국회
  '법률': 'National_Assembly.png',
  '국회규칙': 'National_Assembly.png',

  // 대통령
  '대통령령': 'president.svg',
  '대통령긴급명령': 'president.svg',

  // 국무총리
  '총리령': 'prime_minister.svg',

  // 헌법재판소
  '헌법재판소규칙': 'cck.svg',

  // 대법원
  '대법원규칙': 'sck.svg',

  // 감사원
  '감사원규칙': 'bai.svg',

  // 중앙선거관리위원회
  '중앙선거관리위원회규칙': 'nec.svg',
  '선거관리위원회규칙': 'nec.svg',

  // 행정안전부 (구: 행정자치부)
  '행정안전부령': 'mois.svg',
  '행정자치부령': 'mois.svg',

  // 국가보훈부
  '국가보훈부령': 'mpva.svg',

  // 국토교통부 (구: 건설교통부, 국토해양부)
  '국토교통부령': 'molit.svg',
  '건설교통부령': 'molit.svg',
  '국토해양부령': 'molit.svg',

  // 고용노동부 (구: 노동부)
  '고용노동부령': 'moel.svg',
  '노동부령': 'moel.svg',

  // 과학기술정보통신부
  '과학기술정보통신부령': 'msit.svg',

  // 기후에너지환경부
  '기후에너지환경부령': 'mcee.webp',

  // 환경부
  '환경부령': 'moee.svg',

  // 성평등가족부 (구: 여성가족부)
  '성평등가족부령': 'mogef.webp',
  '여성가족부령': 'mogef.webp',

  // 농림축산식품부 (구: 농림수산식품부)
  '농림축산식품부령': 'mafra.svg',
  '농림수산식품부령': 'mafra.svg',

  // 보건복지부
  '보건복지부령': 'mohw.svg',

  // 기획재정부 (구: 재정경제부)
  '기획재정부령': 'moef.svg',
  '재정경제부령': 'moef.svg',

  // 통일부
  '통일부령': 'mou.svg',

  // 해양수산부
  '해양수산부령': 'mof.svg',

  // 법무부
  '법무부령': 'moj.svg',

  // 문화체육관광부
  '문화체육관광부령': 'mcst.svg',

  // 국방부
  '국방부령': 'mond.svg',

  // 외교부
  '외교부령': 'mofa.svg',

  // 교육부 (구: 교육인적자원부, 교육과학기술부)
  '교육부령': 'moe.svg',
  '교육인적자원부령': 'moe.svg',
  '교육과학기술부령': 'moe.svg',

  // 경제기획원 (구: 기획예산처)
  '경제기획원령': 'mopb.svg',

  // 중소벤처기업부
  '중소벤처기업부령': 'mosme.svg',

  // 산업통상자원부 (구: 산업통상부)
  '산업통상자원부령': 'motir.webp',
  '산업통상부령': 'motir.webp',
}

// 기관명 맵핑 (로고 alt 텍스트용)
const LAW_TYPE_ORG_NAME: Record<string, string> = {
  '법률': '국회',
  '국회규칙': '국회',
  '대통령령': '대통령',
  '대통령긴급명령': '대통령',
  '총리령': '국무총리',
  '헌법재판소규칙': '헌법재판소',
  '대법원규칙': '대법원',
  '감사원규칙': '감사원',
  '중앙선거관리위원회규칙': '중앙선거관리위원회',
  '선거관리위원회규칙': '중앙선거관리위원회',
  '행정안전부령': '행정안전부',
  '행정자치부령': '행정안전부',
  '국가보훈부령': '국가보훈부',
  '국토교통부령': '국토교통부',
  '건설교통부령': '국토교통부',
  '국토해양부령': '국토교통부',
  '고용노동부령': '고용노동부',
  '노동부령': '고용노동부',
  '과학기술정보통신부령': '과학기술정보통신부',
  '기후에너지환경부령': '기후에너지환경부',
  '환경부령': '환경부',
  '성평등가족부령': '성평등가족부',
  '여성가족부령': '성평등가족부',
  '농림축산식품부령': '농림축산식품부',
  '농림수산식품부령': '농림축산식품부',
  '보건복지부령': '보건복지부',
  '기획재정부령': '기획재정부',
  '재정경제부령': '기획재정부',
  '통일부령': '통일부',
  '해양수산부령': '해양수산부',
  '법무부령': '법무부',
  '문화체육관광부령': '문화체육관광부',
  '국방부령': '국방부',
  '외교부령': '외교부',
  '교육부령': '교육부',
  '교육인적자원부령': '교육부',
  '교육과학기술부령': '교육부',
  '경제기획원령': '기획재정부',
  '중소벤처기업부령': '중소벤처기업부',
  '산업통상자원부령': '산업통상자원부',
  '산업통상부령': '산업통상자원부',
}

/**
 * law_type에 해당하는 로고 경로를 반환
 * @param lawType - 법령 유형 (예: "대통령령", "행정안전부령")
 * @returns 로고 경로 또는 null (매핑 없을 때)
 */
export function getLawTypeLogo(lawType: string | undefined): string | null {
  if (!lawType) return null
  const filename = LAW_TYPE_LOGO_MAP[lawType]
  return filename ? `/data/logo/${filename}` : null
}

/**
 * law_type에 해당하는 기관명을 반환
 * @param lawType - 법령 유형
 * @returns 기관명 또는 법령 유형 그대로
 */
export function getLawTypeOrgName(lawType: string | undefined): string {
  if (!lawType) return ''
  return LAW_TYPE_ORG_NAME[lawType] || lawType
}

/**
 * 기본 정부 로고 경로
 */
export const DEFAULT_GOV_LOGO = '/data/logo/government_of_Korea.svg'

// ============================================
// 법원 (판례용) 로고 맵핑
// ============================================

// court_name → 로고 파일명 맵핑
const COURT_LOGO_MAP: Record<string, string> = {
  // 대법원
  '대법원': 'sck.svg',

  // 헌법재판소
  '헌법재판소': 'cck.svg',

  // 고등법원
  '서울고등법원': 'sck.svg',
  '대전고등법원': 'sck.svg',
  '대구고등법원': 'sck.svg',
  '부산고등법원': 'sck.svg',
  '광주고등법원': 'sck.svg',
  '수원고등법원': 'sck.svg',
  '춘천지방법원': 'sck.svg',
  '특허법원': 'sck.svg',

  // 지방법원 (일반)
  '서울중앙지방법원': 'sck.svg',
  '서울동부지방법원': 'sck.svg',
  '서울서부지방법원': 'sck.svg',
  '서울남부지방법원': 'sck.svg',
  '서울북부지방법원': 'sck.svg',
  '의정부지방법원': 'sck.svg',
  '인천지방법원': 'sck.svg',
  '수원지방법원': 'sck.svg',
  '대전지방법원': 'sck.svg',
  '대구지방법원': 'sck.svg',
  '부산지방법원': 'sck.svg',
  '울산지방법원': 'sck.svg',
  '광주지방법원': 'sck.svg',
  '전주지방법원': 'sck.svg',
  '청주지방법원': 'sck.svg',
  '창원지방법원': 'sck.svg',
  '제주지방법원': 'sck.svg',

  // 행정법원
  '서울행정법원': 'sck.svg',

  // 가정법원
  '서울가정법원': 'sck.svg',
}

/**
 * 법원명에 해당하는 로고 경로를 반환
 * @param courtName - 법원명 (예: "대법원", "헌법재판소", "서울고등법원")
 * @returns 로고 경로 또는 null
 */
export function getCourtLogo(courtName: string | undefined): string | null {
  if (!courtName) return null

  // 정확한 매칭 먼저 시도
  if (COURT_LOGO_MAP[courtName]) {
    return `/data/logo/${COURT_LOGO_MAP[courtName]}`
  }

  // 부분 매칭 (고등법원, 지방법원 등)
  if (courtName.includes('헌법재판소')) {
    return '/data/logo/cck.svg'
  }
  if (courtName.includes('대법원') || courtName.includes('법원')) {
    return '/data/logo/sck.svg'
  }

  return null
}

/**
 * doc_type에 해당하는 기본 로고 경로를 반환
 * @param docType - 문서 유형 (precedent, constitutional 등)
 * @returns 로고 경로 또는 null
 */
export function getDocTypeLogo(docType: string | undefined): string | null {
  if (!docType) return null

  switch (docType) {
    case 'precedent':
      return '/data/logo/sck.svg'  // 판례 기본: 대법원
    case 'constitutional':
      return '/data/logo/cck.svg'  // 헌재결정: 헌법재판소
    default:
      return null
  }
}
