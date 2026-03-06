/**
 * 모의 법정 일반 모드 사전 정의 시나리오
 * 데모 시나리오와 달리 stages/mockResponses 없이 setup 정보만 포함
 */
import type { CaseType, CaseCategory, UserRole } from '../types'

export interface Scenario {
  id: string
  name: string
  description: string
  caseType: CaseType
  caseCategory: CaseCategory
  userRole: UserRole
  caseSummary: string
}

export const SCENARIOS: Scenario[] = [
  // 형사 시나리오
  {
    id: 'criminal-assault-attorney',
    name: '음주 폭행 사건 (변호사)',
    description: '술자리에서 시비가 붙어 상대방을 폭행한 사건. 변호인으로서 피고인을 변호합니다.',
    caseType: 'criminal',
    caseCategory: 'criminal_assault',
    userRole: 'attorney',
    caseSummary:
      '피고인 김모(35세)는 2025년 11월 15일 오후 11시경 서울 강남구 소재 주점에서 피해자 박모(40세)와 술값 문제로 시비가 붙어 주먹으로 피해자의 얼굴을 2회 때려 전치 3주의 상해를 입힌 혐의로 기소되었습니다. 피고인은 피해자가 먼저 시비를 걸었고 자신은 방어 차원이었다고 주장합니다.',
  },
  {
    id: 'criminal-theft-prosecutor',
    name: '편의점 절도 사건 (검사)',
    description: '편의점에서 상습 절도를 한 피의자를 기소. 검사로서 유죄를 입증합니다.',
    caseType: 'criminal',
    caseCategory: 'criminal_theft',
    userRole: 'prosecutor',
    caseSummary:
      '피고인 이모(28세)는 2025년 9월부터 12월까지 서울 마포구 일대 편의점 5곳에서 총 15회에 걸쳐 식료품, 생활용품 등 합계 87만원 상당의 물품을 절취한 혐의로 기소되었습니다. CCTV 영상과 점주 진술이 확보되어 있으며, 피고인은 경제적 어려움을 호소하고 있습니다.',
  },
  {
    id: 'criminal-embezzlement-prosecutor',
    name: '회사 자금 횡령 사건 (검사)',
    description: '회사 경리 직원이 법인 자금을 횡령한 사건. 검사로서 횡령 사실을 입증합니다.',
    caseType: 'criminal',
    caseCategory: 'criminal_embezzlement',
    userRole: 'prosecutor',
    caseSummary:
      '피고인 최모(42세)는 주식회사 A의 경리팀장으로 근무하면서 2024년 3월부터 2025년 8월까지 약 1년 6개월간 회사 법인카드를 사적으로 사용하고 허위 경비를 청구하는 방법으로 총 1억 2천만원을 횡령한 혐의로 기소되었습니다. 회계 감사 과정에서 적발되었으며, 피고인은 일부 금액만 인정하고 있습니다.',
  },
  {
    id: 'criminal-fraud-attorney',
    name: '온라인 중고거래 사기 (변호사)',
    description: '중고거래 플랫폼에서 사기 혐의를 받는 피고인을 변호합니다.',
    caseType: 'criminal',
    caseCategory: 'criminal_fraud',
    userRole: 'attorney',
    caseSummary:
      '피고인 정모(31세)는 2025년 7월경 중고거래 플랫폼에서 노트북 판매 글을 올려 피해자 5명으로부터 총 450만원을 입금받고 물품을 배송하지 않은 혐의로 기소되었습니다. 피고인은 실제로 노트북을 보유하고 있었으나 개인 사정으로 배송이 지연된 것이며 사기 의사는 없었다고 주장합니다.',
  },
  // 민사 시나리오
  {
    id: 'civil-contract-prosecutor',
    name: '임대차 보증금 반환 분쟁 (원고)',
    description: '임대차 계약 종료 후 보증금을 돌려받지 못한 세입자. 원고측으로 보증금 반환을 청구합니다.',
    caseType: 'civil',
    caseCategory: 'civil_contract',
    userRole: 'prosecutor',
    caseSummary:
      '원고 한모(29세)는 서울 송파구 소재 오피스텔에 대해 임대차보증금 5천만원, 월세 80만원의 조건으로 2023년 5월 1일부터 2025년 4월 30일까지 임대차계약을 체결하였습니다. 계약 만료 후 퇴거하였으나 임대인(피고)은 원상복구 비용 1천만원을 공제하겠다며 보증금 반환을 거부하고 있습니다. 원고는 정상적으로 사용한 것이며 원상복구 의무 대상이 아니라고 주장합니다.',
  },
  {
    id: 'civil-property-attorney',
    name: '부동산 매매 하자 분쟁 (피고)',
    description: '아파트 매도인이 하자 미고지로 소송을 당한 사건. 피고측으로 방어합니다.',
    caseType: 'civil',
    caseCategory: 'civil_property',
    userRole: 'attorney',
    caseSummary:
      '원고 윤모(45세)는 2025년 3월 피고 강모(50세)로부터 경기도 성남시 소재 아파트를 매매대금 8억원에 매수하였습니다. 입주 후 욕실 누수, 곰팡이, 배관 노후 등 하자를 발견하고 매도인이 하자를 알면서도 고지하지 않았다며 수리비 3천만원 및 위자료 500만원을 청구하였습니다. 피고는 매매 당시 하자를 인지하지 못했고 매수인의 확인 의무도 있다고 항변합니다.',
  },
  {
    id: 'civil-damages-prosecutor',
    name: '교통사고 손해배상 (원고)',
    description: '교통사고로 부상을 입은 피해자. 원고측으로 손해배상을 청구합니다.',
    caseType: 'civil',
    caseCategory: 'civil_damages',
    userRole: 'prosecutor',
    caseSummary:
      '원고 서모(38세)는 2025년 6월 20일 서울 영등포구 교차로에서 피고 김모(55세)가 운전하는 차량에 의해 교통사고를 당하여 경추 염좌, 요추 디스크 등 전치 8주의 부상을 입었습니다. 원고는 치료비 850만원, 휴업손해 1,200만원, 위자료 500만원 총 2,550만원의 손해배상을 청구합니다. 사고 당시 피고 차량이 신호를 위반한 CCTV 영상이 확보되어 있습니다.',
  },
  {
    id: 'civil-damages-attorney',
    name: '의료과실 손해배상 (피고)',
    description: '수술 후 합병증이 발생한 환자가 병원을 상대로 소송. 피고측 병원을 방어합니다.',
    caseType: 'civil',
    caseCategory: 'civil_damages',
    userRole: 'attorney',
    caseSummary:
      '원고 박모(52세)는 2025년 2월 피고 A병원에서 무릎 관절 수술을 받은 후 감염 합병증이 발생하여 추가 수술 및 장기 입원이 필요하게 되었습니다. 원고는 수술 전 감염 위험에 대한 충분한 설명을 듣지 못했고 수술 과정에서 위생 관리가 미흡했다며 치료비 2천만원, 일실수입 3천만원, 위자료 1천만원을 청구하였습니다. 피고 병원은 의료 가이드라인을 준수했고 합병증은 불가피한 의료 위험이라고 항변합니다.',
  },
  {
    id: 'civil-contract-attorney',
    name: '프리랜서 용역 대금 분쟁 (피고)',
    description: '프리랜서가 용역 대금을 청구한 사건. 피고측으로 납품물 하자를 주장하며 방어합니다.',
    caseType: 'civil',
    caseCategory: 'civil_contract',
    userRole: 'attorney',
    caseSummary:
      '원고 디자이너 A(프리랜서)는 피고 B회사와 2025년 4월 웹사이트 디자인 용역 계약(대금 2천만원)을 체결하고 작업을 완료하였으나, 피고가 잔금 1,200만원을 지급하지 않아 소송을 제기하였습니다. 피고 B회사는 납품물이 계약서에 명시된 요구사항을 충족하지 못하며 수차례 수정 요청에도 불구하고 품질이 개선되지 않아 잔금 지급을 거부하고 있습니다.',
  },
  {
    id: 'criminal-assault-prosecutor',
    name: '가정폭력 상해 사건 (검사)',
    description: '배우자에 대한 상습 폭행 사건. 검사로서 피고인의 유죄를 입증합니다.',
    caseType: 'criminal',
    caseCategory: 'criminal_assault',
    userRole: 'prosecutor',
    caseSummary:
      '피고인 남모(47세)는 2024년 1월부터 2025년 10월까지 배우자 여모(44세)를 수차례 폭행하여 전치 2주에서 6주까지의 상해를 입힌 혐의로 기소되었습니다. 피해자의 진단서 4건, 이웃 목격자 진술 2건, 112 신고 기록 3건이 확보되어 있습니다. 피고인은 부부 싸움 과정에서 우발적으로 발생한 것이라며 상습성을 부인하고 있습니다.',
  },
]
