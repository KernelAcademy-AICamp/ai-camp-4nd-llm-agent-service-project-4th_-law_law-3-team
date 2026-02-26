/**
 * 모의 법정 데모 시나리오 데이터
 * 법률 도메인 지식 없이도 테스트할 수 있도록 사전 작성된 시나리오
 */
import type { CaseType, CaseCategory, UserRole, PhysicalEvidence } from '../types'

interface DemoStage {
  stageId: string
  /** 해당 단계에서 사용자가 입력할 발언 목록 (순서대로 자동 입력) */
  userInputs: string[]
  /** AI 응답 시뮬레이션 (백엔드 미연결 시 사용) */
  mockResponses: { speaker: string; content: string }[]
}

export interface DemoCharacter {
  role: string
  name: string
  description: string
}

export interface DemoScenario {
  id: string
  name: string
  description: string
  setup: {
    caseType: CaseType
    caseCategory: CaseCategory
    userRole: UserRole
    caseSummary: string
  }
  /** 등장인물 정보 */
  characters?: DemoCharacter[]
  /** 사용자의 목표/미션 */
  objectives?: string[]
  /** 시나리오에 포함된 물적 증거물 */
  evidence?: PhysicalEvidence[]
  stages: DemoStage[]
}

/** 형사 사기 사건 - 검사 역할 */
const CRIMINAL_FRAUD_PROSECUTOR: DemoScenario = {
  id: 'criminal-fraud-prosecutor',
  name: '투자 사기 사건 (검사)',
  description:
    '피고인이 허위 투자 수익을 약속하여 5천만원을 편취한 사기 사건. 검사 역할로 공소를 유지합니다.',
  setup: {
    caseType: 'criminal',
    caseCategory: 'criminal_fraud',
    userRole: 'prosecutor',
    caseSummary:
      '피고인 김모씨는 2025년 3월부터 6월까지 피해자 이모씨에게 "해외 부동산 투자로 월 10% 수익을 보장한다"고 허위 사실을 고지하여 총 5천만원을 편취하였습니다. 실제로는 투자 실체가 없었으며, 편취한 금원을 개인 채무 변제에 사용하였습니다.',
  },
  characters: [
    { role: 'judge', name: '재판장', description: '본 사건 담당 판사' },
    { role: 'prosecutor', name: '검사 김민준', description: '공소 유지 담당 (사용자 역할)' },
    { role: 'attorney', name: '변호인 박준호', description: '피고인 측 변호' },
    { role: 'defendant', name: '피고인 김영수', description: '사기 혐의로 기소' },
    { role: 'clerk', name: '서기', description: '재판 진행 보조' },
  ],
  objectives: [
    '피고인의 사기 고의(편취 의도)를 입증하세요',
    '허위 투자 설명서와 계좌 내역을 증거로 활용하세요',
    '피고인 신문에서 진술의 허점을 찾아내세요',
  ],
  evidence: [
    {
      id: 'phys-fraud-1',
      type: 'document',
      title: '허위 투자 설명서',
      description: '피고인이 피해자에게 제시한 해외 부동산 투자 설명서',
      detail: '동남아시아 소재 부동산 프로젝트로 기재되어 있으나, 해당 프로젝트는 실존하지 않는 것으로 수사기관에 의해 확인됨.',
      favorable_to: 'prosecutor',
    },
    {
      id: 'phys-fraud-2',
      type: 'financial',
      title: '피고인 계좌 거래 내역',
      description: '피해자로부터 입금된 5천만원의 사용처를 보여주는 계좌 이체 내역',
      detail: '5천만원 전액이 입금 당일~3일 이내 개인 대출금 상환에 사용. 해외 투자 관련 출금 내역 없음.',
      favorable_to: 'prosecutor',
    },
    {
      id: 'phys-fraud-3',
      type: 'document',
      title: '가짜 수익 보고서',
      description: '피고인이 피해자에게 보낸 월간 수익 보고서',
      detail: '3개월간 월 8~12% 수익률을 기재했으나, 실제 투자 계좌가 존재하지 않음.',
      favorable_to: 'prosecutor',
    },
  ],
  stages: [
    {
      stageId: 'identity',
      userInputs: [],
      mockResponses: [
        {
          speaker: 'clerk',
          content: '2025고단1234호 사기 사건을 개정합니다.',
        },
        {
          speaker: 'judge',
          content:
            '피고인은 앞으로 나오세요. 성명이 김영수이고, 주민등록번호 뒷자리가 1234567이 맞습니까?',
        },
        {
          speaker: 'defendant',
          content: '예, 맞습니다.',
        },
        {
          speaker: 'judge',
          content:
            '피고인에게 진술거부권을 고지합니다. 피고인은 형사소송법 제283조의2에 따라 개개의 질문에 대하여 진술을 거부할 수 있으며, 진술을 하지 않더라도 불이익을 받지 않습니다. 이해하셨습니까?',
        },
        {
          speaker: 'defendant',
          content: '예, 이해했습니다.',
        },
      ],
    },
    {
      stageId: 'opening',
      userInputs: [
        '검사 김민준입니다. 피고인 김영수는 2025년 3월부터 6월까지 피해자 이수진에게 해외 부동산 투자 명목으로 월 10%의 높은 수익률을 보장한다고 기망하여 총 5천만원을 교부받았습니다. 피고인은 투자 실체가 전혀 없음에도 불구하고 허위의 투자 설명서와 가짜 수익 보고서를 제시하여 피해자를 기망하였으며, 편취한 금원을 개인 채무 변제에 사용하였습니다. 이는 형법 제347조 제1항에 해당하는 사기죄에 해당합니다.',
      ],
      mockResponses: [
        {
          speaker: 'judge',
          content: '검사의 모두진술을 잘 들었습니다. 변호인 측 의견을 듣겠습니다.',
        },
        {
          speaker: 'attorney',
          content:
            '변호인 박준호입니다. 피고인은 공소사실을 일부 부인합니다. 피고인은 실제로 투자를 시도했으나 예상치 못한 시장 상황의 변화로 투자에 실패한 것이며, 처음부터 편취의 의도가 있었던 것은 아닙니다. 이 점에 대해 증거조사 단계에서 상세히 소명하겠습니다.',
        },
      ],
    },
    {
      stageId: 'evidence',
      userInputs: [
        '증거 제1호로 피고인이 피해자에게 제시한 허위 투자 설명서를 제출합니다. 해당 문서에 기재된 해외 부동산 프로젝트는 실제로 존재하지 않는 것으로 확인되었습니다. 또한 증거 제2호로 피고인의 계좌 거래 내역을 제출합니다. 피해자로부터 입금된 5천만원이 해외 투자가 아닌 피고인의 개인 대출금 상환에 사용된 것이 확인됩니다.',
      ],
      mockResponses: [
        {
          speaker: 'judge',
          content:
            '증거 제1호, 제2호를 채택합니다. 변호인 측에서 해당 증거에 대한 의견이 있습니까?',
        },
        {
          speaker: 'attorney',
          content:
            '증거 제1호의 투자 설명서는 피고인이 실제 투자 계획을 기반으로 작성한 것이며, 당시에는 프로젝트가 진행 중이었습니다. 증거 제2호의 계좌 이체는 일시적인 자금 운용이었으며 투자금 전용을 위한 것이 아니었습니다.',
        },
      ],
    },
    {
      stageId: 'examination',
      userInputs: [
        '피고인에게 묻겠습니다. 피고인은 피해자에게 월 10%의 수익을 보장한다고 말한 사실이 있습니까? 그리고 피해자에게 보여준 투자 설명서에 기재된 해외 부동산 프로젝트의 구체적인 명칭과 소재지를 말해주시기 바랍니다.',
      ],
      mockResponses: [
        {
          speaker: 'defendant',
          content:
            '수익을 보장한다고 단정적으로 말한 적은 없고, 예상 수익률이라고 설명했습니다. 투자 설명서에 기재된 프로젝트는... 동남아시아 쪽 부동산이었는데, 정확한 명칭은 기억나지 않습니다.',
        },
        {
          speaker: 'judge',
          content: '검사, 추가 질문이 있습니까?',
        },
      ],
    },
    {
      stageId: 'closing',
      userInputs: [
        '존경하는 재판장님, 본건 피고인의 사기 행위는 명백합니다. 첫째, 피고인이 제시한 투자 프로젝트는 실체가 없었으며 이는 검찰 수사로 확인되었습니다. 둘째, 피해자로부터 받은 5천만원은 해외 투자가 아닌 개인 채무 변제에 전액 사용되었습니다. 셋째, 피고인은 법정에서 프로젝트의 구체적 명칭조차 답변하지 못했습니다. 이상의 증거와 정황을 종합하면, 피고인에게 처음부터 편취의 의도가 있었음이 인정됩니다. 검찰은 피고인에 대해 징역 2년을 구형합니다.',
      ],
      mockResponses: [
        {
          speaker: 'attorney',
          content:
            '재판장님, 피고인은 초범이며 반성하고 있습니다. 투자 실패에 대한 민사적 책임은 인정하나, 처음부터 사기의 고의가 있었다고 단정하기는 어렵습니다. 선처를 부탁드립니다.',
        },
        {
          speaker: 'defendant',
          content:
            '피해자분께 진심으로 죄송합니다. 투자가 실패하여 이런 상황이 되었지만, 어떻게든 피해를 변상하겠습니다.',
        },
      ],
    },
    {
      stageId: 'verdict',
      userInputs: [],
      mockResponses: [
        {
          speaker: 'judge',
          content:
            '판결을 선고합니다. 피고인 김영수를 징역 1년 6월에 처합니다. 다만, 이 판결 확정일로부터 3년간 위 형의 집행을 유예합니다.\n\n【이유】\n피고인은 실체 없는 해외 부동산 투자를 빙자하여 피해자로부터 5천만원을 편취하였는바, 이는 형법 제347조 제1항의 사기죄에 해당합니다. 다만, 피고인이 초범인 점, 반성하는 태도를 보이는 점, 피해 변상 의사를 밝힌 점 등을 양형에 참작합니다.',
        },
      ],
    },
  ],
}

/** 민사 손해배상 사건 - 원고측 대리인 역할 */
const CIVIL_DAMAGES_PLAINTIFF: DemoScenario = {
  id: 'civil-damages-plaintiff',
  name: '교통사고 손해배상 (원고)',
  description:
    '교통사고로 인한 치료비 및 위자료 손해배상 청구 사건. 원고측 대리인으로 참여합니다.',
  setup: {
    caseType: 'civil',
    caseCategory: 'civil_damages',
    userRole: 'prosecutor',
    caseSummary:
      '2025년 4월 15일 서울 강남구 역삼동 교차로에서 피고 차량이 신호위반으로 원고 차량을 추돌하여 원고가 경추 염좌 및 요추 추간판 탈출증 진단을 받았습니다. 원고는 치료비 1,200만원, 휴업손해 800만원, 위자료 500만원 등 총 2,500만원의 손해배상을 청구합니다.',
  },
  characters: [
    { role: 'judge', name: '재판장', description: '본 사건 담당 판사' },
    { role: 'prosecutor', name: '원고 대리인', description: '원고측 손해배상 청구 (사용자 역할)' },
    { role: 'defendant', name: '피고 대리인', description: '피고측 과실 비율 다툼' },
    { role: 'clerk', name: '서기', description: '재판 진행 보조' },
  ],
  objectives: [
    '피고의 신호위반 과실을 입증하세요',
    'CCTV 영상과 진단서를 증거로 활용하세요',
    '원고의 과실 상계 비율을 최소화하는 논증을 펼치세요',
  ],
  evidence: [
    {
      id: 'phys-civil-1',
      type: 'video',
      title: '사고 현장 CCTV 영상',
      description: '역삼동 교차로 CCTV에 기록된 사고 장면',
      detail: '피고 차량이 적색 신호에 교차로에 진입하여 원고 차량 좌측면을 추돌하는 장면이 녹화됨.',
      favorable_to: 'prosecutor',
    },
    {
      id: 'phys-civil-2',
      type: 'document',
      title: '진단서 및 의료비 영수증',
      description: '원고의 경추 염좌 및 요추 추간판 탈출증 진단서',
      detail: '담당 전문의: "외부 충격에 의한 급성 발생으로 교통사고와의 인과관계 인정". 치료비 총 1,200만원.',
      favorable_to: 'prosecutor',
    },
    {
      id: 'phys-civil-3',
      type: 'video',
      title: '원고 차량 블랙박스 영상',
      description: '사고 당시 원고 차량의 블랙박스에 기록된 속도계',
      detail: '속도계가 약 75km/h를 표시. 해당 구간 제한속도 60km/h.',
      favorable_to: 'attorney',
    },
  ],
  stages: [
    {
      stageId: 'pretrial',
      userInputs: [],
      mockResponses: [
        {
          speaker: 'clerk',
          content: '2025가단56789호 손해배상(자) 사건을 개정합니다.',
        },
        {
          speaker: 'judge',
          content:
            '양측 대리인은 출석하셨습니까? 본건은 교통사고로 인한 손해배상 청구 사건입니다. 쟁점을 정리하겠습니다. 원고 측은 피고의 과실 및 손해액에 대해, 피고 측은 과실 비율 및 손해액의 적정성에 대해 주장해 주시기 바랍니다.',
        },
      ],
    },
    {
      stageId: 'claims',
      userInputs: [
        '원고 대리인입니다. 2025년 4월 15일 피고는 역삼동 교차로에서 적색 신호를 위반하여 직진 중이던 원고 차량의 좌측면을 추돌하였습니다. 이 사고로 원고는 경추 염좌 및 요추 추간판 탈출증 진단을 받아 3개월간 입원 치료를 받았으며, 현재도 통원 치료 중입니다. 원고는 민법 제750조 불법행위 및 자동차손해배상보장법 제3조에 근거하여 치료비 1,200만원, 휴업손해 800만원, 위자료 500만원 등 총 2,500만원을 청구합니다.',
      ],
      mockResponses: [
        {
          speaker: 'judge',
          content: '원고 측 주장을 확인했습니다. 피고 측 답변을 듣겠습니다.',
        },
        {
          speaker: 'defendant',
          content:
            '피고 대리인입니다. 사고 발생 사실 자체는 인정하나, 피고의 과실 비율에 대해 다툽니다. 사고 당시 원고도 제한속도를 초과하여 운전하고 있었으며, 블랙박스 영상에서 확인 가능합니다. 또한 원고가 청구하는 손해액 중 위자료 500만원은 과다하며, 요추 추간판 탈출증은 사고와의 인과관계가 불분명합니다.',
        },
      ],
    },
    {
      stageId: 'evidence',
      userInputs: [
        '증거 제1호로 사고 현장 CCTV 영상을 제출합니다. 영상에서 피고 차량이 적색 신호임에도 교차로에 진입하는 장면이 명확히 확인됩니다. 증거 제2호로 원고의 진단서 및 의료비 영수증을 제출합니다. 담당 전문의 소견서에 따르면 요추 추간판 탈출증은 외부 충격에 의한 것으로 사고와의 인과관계가 인정됩니다.',
      ],
      mockResponses: [
        {
          speaker: 'judge',
          content:
            '증거를 채택합니다. 피고 측에서 제출할 증거가 있습니까?',
        },
        {
          speaker: 'defendant',
          content:
            '증거 제3호로 사고 당시 원고 차량의 블랙박스 영상을 제출합니다. 원고 차량의 속도계가 제한속도 60km/h 구간에서 약 75km/h를 표시하고 있어, 원고에게도 과실이 있습니다.',
        },
      ],
    },
    {
      stageId: 'argument',
      userInputs: [
        '피고 측이 주장하는 원고의 과속은 이 사고의 직접적인 원인이 아닙니다. 본건 사고의 직접적 원인은 피고의 신호위반이며, 대법원 판례(2019다12345)에 따르면 신호위반 사고에서 피해 차량의 경미한 속도 초과는 과실 상계 사유로 보기 어렵습니다. 설령 과실 상계를 인정하더라도 원고의 과실 비율은 10%를 초과할 수 없습니다.',
      ],
      mockResponses: [
        {
          speaker: 'defendant',
          content:
            '원고 측의 주장은 과실 비율에 대한 일반론에 불과합니다. 원고가 제한속도를 25% 초과한 것은 경미하다고 볼 수 없으며, 교통사고 과실비율 인정기준에 따르면 원고의 과실은 최소 20%로 산정되어야 합니다.',
        },
        {
          speaker: 'judge',
          content: '양측의 주장을 충분히 들었습니다. 추가 주장이 있으면 말씀하세요.',
        },
      ],
    },
    {
      stageId: 'closing',
      userInputs: [
        '재판장님, 정리하겠습니다. 본건 사고는 전적으로 피고의 신호위반에 기인합니다. CCTV 영상으로 피고의 적색 신호 위반이 명백히 증명되었으며, 원고의 상해와 사고 사이의 인과관계도 전문의 소견으로 입증되었습니다. 원고의 경미한 속도 초과가 있었으나 이는 사고의 직접 원인이 아니므로 과실 상계 비율은 최소한으로 산정되어야 합니다. 원고가 청구하는 총 2,500만원의 손해배상을 인용하여 주시기 바랍니다.',
      ],
      mockResponses: [
        {
          speaker: 'defendant',
          content:
            '원고의 과실을 고려하여 적정한 과실 상계를 적용해 주시고, 요추 추간판 탈출증에 대한 인과관계 인정 여부를 신중히 판단하여 주시기 바랍니다.',
        },
      ],
    },
    {
      stageId: 'verdict',
      userInputs: [],
      mockResponses: [
        {
          speaker: 'judge',
          content:
            '판결을 선고합니다. 피고는 원고에게 2,125만원 및 이에 대하여 2025년 4월 15일부터 다 갚는 날까지 연 5%의 비율에 의한 금원을 지급하라.\n\n【이유】\n피고의 신호위반 과실이 사고의 주된 원인으로 인정됩니다. 다만, 원고의 속도 초과(15km/h)를 과실 상계 사유로 인정하여 원고의 과실을 15%로 산정합니다. 치료비 1,200만원, 휴업손해 800만원은 전액 인정하고, 위자료는 제반 사정을 고려하여 500만원을 인정합니다. 총 2,500만원에서 15% 과실 상계를 적용하여 2,125만원을 인용합니다.',
        },
      ],
    },
  ],
}

export const DEMO_SCENARIOS: DemoScenario[] = [
  CRIMINAL_FRAUD_PROSECUTOR,
  CIVIL_DAMAGES_PLAINTIFF,
]
