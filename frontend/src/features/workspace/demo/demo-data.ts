/**
 * 워크스페이스 개발용 더미 데이터
 * NODE_ENV === 'development' 에서만 사용
 */

import type {
  WorkspaceCase,
  WorkspaceCaseDetail,
  PaginatedResponse,
} from '../types'

const DEMO_CASES: WorkspaceCase[] = [
  {
    id: 'demo-001',
    case_name: '김영수 투자 사기 사건',
    case_type: '형사 - 사기',
    status: 'open',
    tagged_items: [
      { type: 'party', value: '김영수', label: '피고인 김영수' },
      { type: 'party', value: '박지현', label: '피해자 박지현' },
      { type: 'amount', value: '50000000', label: '5,000만원' },
      { type: 'date', value: '2025-03-15', label: '범행일 2025.3.15' },
      { type: 'legal_term', value: '사기죄', label: '형법 제347조 사기죄' },
      { type: 'evidence', value: '계좌이체내역', label: '계좌이체 내역서' },
    ],
    created_at: '2025-11-20T09:30:00Z',
  },
  {
    id: 'demo-002',
    case_name: '강남 교차로 교통사고 손해배상',
    case_type: '민사 - 손해배상',
    status: 'open',
    tagged_items: [
      { type: 'party', value: '이민호', label: '원고 이민호' },
      { type: 'party', value: '최대현', label: '피고 최대현' },
      { type: 'amount', value: '25000000', label: '2,500만원' },
      { type: 'date', value: '2025-06-10', label: '사고일 2025.6.10' },
      { type: 'location', value: '강남구 삼성동 교차로', label: '강남구 삼성동 교차로' },
      { type: 'evidence', value: 'CCTV', label: 'CCTV 영상' },
      { type: 'evidence', value: '진단서', label: '진단서 (전치 6주)' },
      { type: 'legal_term', value: '불법행위', label: '민법 제750조 불법행위' },
    ],
    created_at: '2025-12-05T14:20:00Z',
  },
  {
    id: 'demo-003',
    case_name: '(주)테크스타 임금체불 사건',
    case_type: '민사 - 임금',
    status: 'closed',
    tagged_items: [
      { type: 'party', value: '정수빈', label: '원고 정수빈 외 3인' },
      { type: 'party', value: '(주)테크스타', label: '피고 (주)테크스타' },
      { type: 'amount', value: '18500000', label: '1,850만원' },
      { type: 'date', value: '2025-01-31', label: '퇴직일 2025.1.31' },
      { type: 'legal_term', value: '근로기준법', label: '근로기준법 제36조' },
    ],
    created_at: '2025-08-15T11:00:00Z',
  },
  {
    id: 'demo-004',
    case_name: '서초동 오피스텔 보증금 반환',
    case_type: '민사 - 임대차',
    status: 'open',
    tagged_items: [
      { type: 'party', value: '한소영', label: '원고 한소영' },
      { type: 'party', value: '박건우', label: '피고 박건우' },
      { type: 'amount', value: '30000000', label: '3,000만원' },
      { type: 'location', value: '서초구 서초동', label: '서초구 서초동 오피스텔' },
      { type: 'date', value: '2025-09-30', label: '계약만료 2025.9.30' },
      { type: 'legal_term', value: '임대차보호법', label: '주택임대차보호법 제3조의2' },
    ],
    created_at: '2026-01-10T16:45:00Z',
  },
  {
    id: 'demo-005',
    case_name: '온라인 명예훼손 고소 사건',
    case_type: '형사 - 명예훼손',
    status: 'archived',
    tagged_items: [
      { type: 'party', value: '윤서진', label: '고소인 윤서진' },
      { type: 'party', value: '익명 ID blueSky22', label: '피고소인 blueSky22' },
      { type: 'date', value: '2025-07-20', label: '게시일 2025.7.20' },
      { type: 'legal_term', value: '정보통신망법', label: '정보통신망법 제70조' },
      { type: 'evidence', value: '게시물 캡처', label: '게시물 캡처 화면' },
    ],
    created_at: '2025-09-01T10:15:00Z',
  },
]

export function getDemoCaseList(): PaginatedResponse<WorkspaceCase> {
  return {
    items: DEMO_CASES,
    total: DEMO_CASES.length,
    page: 1,
    page_size: 20,
  }
}

export function getDemoCaseDetail(caseId: string): WorkspaceCaseDetail | null {
  const base = DEMO_CASES.find((c) => c.id === caseId)
  if (!base) return null

  const DETAILS: Record<string, Omit<WorkspaceCaseDetail, keyof WorkspaceCase>> = {
    'demo-001': {
      timeline: [
        {
          id: 'tl-001',
          title: '피해자 박지현, 피고인 김영수와 첫 접촉',
          description: '온라인 투자 카페에서 고수익 투자 권유를 받음',
          date_text: '2025년 1월 10일',
          date_normalized: '2025-01-10',
          category: '사건 경위',
          source_type: 'ai_extracted',
          confidence: 0.92,
          source_conversation_id: null,
          created_at: '2025-11-20T10:00:00Z',
        },
        {
          id: 'tl-002',
          title: '1차 투자금 2,000만원 송금',
          description: '김영수 명의 계좌로 2,000만원 이체',
          date_text: '2025년 2월 5일',
          date_normalized: '2025-02-05',
          category: '금전 거래',
          source_type: 'ai_extracted',
          confidence: 0.95,
          source_conversation_id: null,
          created_at: '2025-11-20T10:00:00Z',
        },
        {
          id: 'tl-003',
          title: '2차 투자금 3,000만원 송금',
          description: '추가 수익 약속 후 3,000만원 추가 이체',
          date_text: '2025년 3월 1일',
          date_normalized: '2025-03-01',
          category: '금전 거래',
          source_type: 'ai_extracted',
          confidence: 0.88,
          source_conversation_id: null,
          created_at: '2025-11-20T10:00:00Z',
        },
        {
          id: 'tl-004',
          title: '김영수 연락 두절',
          description: '전화번호 변경, SNS 계정 삭제',
          date_text: '2025년 3월 15일',
          date_normalized: '2025-03-15',
          category: '사건 경위',
          source_type: 'ai_extracted',
          confidence: 0.90,
          source_conversation_id: null,
          created_at: '2025-11-20T10:00:00Z',
        },
        {
          id: 'tl-005',
          title: '경찰 고소장 접수',
          description: '강남경찰서에 사기 혐의 고소장 제출',
          date_text: '2025년 4월 2일',
          date_normalized: '2025-04-02',
          category: '법적 절차',
          source_type: 'manual',
          confidence: 1.0,
          source_conversation_id: null,
          created_at: '2025-11-20T10:00:00Z',
        },
      ],
      conversations: [
        { id: 'conv-001', title: '사기 사건 법률 상담', message_count: 12 },
        { id: 'conv-002', title: '고소장 작성 도움', message_count: 8 },
      ],
    },
    'demo-002': {
      timeline: [
        {
          id: 'tl-010',
          title: '교통사고 발생',
          description: '강남구 삼성동 교차로에서 신호위반 추돌 사고',
          date_text: '2025년 6월 10일',
          date_normalized: '2025-06-10',
          category: '사건 경위',
          source_type: 'ai_extracted',
          confidence: 0.95,
          source_conversation_id: null,
          created_at: '2025-12-05T15:00:00Z',
        },
        {
          id: 'tl-011',
          title: '응급실 이송 및 입원',
          description: '삼성서울병원 응급실, 전치 6주 진단',
          date_text: '2025년 6월 10일',
          date_normalized: '2025-06-10',
          category: '의료',
          source_type: 'ai_extracted',
          confidence: 0.93,
          source_conversation_id: null,
          created_at: '2025-12-05T15:00:00Z',
        },
        {
          id: 'tl-012',
          title: '보험사 합의 제안 (800만원)',
          description: '피고측 보험사에서 800만원 합의 제안, 원고 거절',
          date_text: '2025년 8월 20일',
          date_normalized: '2025-08-20',
          category: '협상',
          source_type: 'ai_extracted',
          confidence: 0.85,
          source_conversation_id: null,
          created_at: '2025-12-05T15:00:00Z',
        },
        {
          id: 'tl-013',
          title: '소장 접수',
          description: '서울중앙지방법원에 손해배상 소장 접수',
          date_text: '2025년 10월 15일',
          date_normalized: '2025-10-15',
          category: '법적 절차',
          source_type: 'manual',
          confidence: 1.0,
          source_conversation_id: null,
          created_at: '2025-12-05T15:00:00Z',
        },
      ],
      conversations: [
        { id: 'conv-010', title: '교통사고 손해배상 상담', message_count: 15 },
        { id: 'conv-011', title: '소장 작성 및 증거 정리', message_count: 10 },
        { id: 'conv-012', title: '합의금 적정성 검토', message_count: 6 },
      ],
    },
    'demo-003': {
      timeline: [
        {
          id: 'tl-020',
          title: '퇴직',
          description: '정수빈 외 3인 퇴직',
          date_text: '2025년 1월 31일',
          date_normalized: '2025-01-31',
          category: '사건 경위',
          source_type: 'ai_extracted',
          confidence: 0.90,
          source_conversation_id: null,
          created_at: '2025-08-15T12:00:00Z',
        },
        {
          id: 'tl-021',
          title: '임금 미지급 확인',
          description: '퇴직 후 14일 경과, 퇴직금 및 미지급 임금 미수령',
          date_text: '2025년 2월 14일',
          date_normalized: '2025-02-14',
          category: '금전 거래',
          source_type: 'ai_extracted',
          confidence: 0.88,
          source_conversation_id: null,
          created_at: '2025-08-15T12:00:00Z',
        },
      ],
      conversations: [
        { id: 'conv-020', title: '임금체불 법률 상담', message_count: 9 },
      ],
    },
    'demo-004': {
      timeline: [
        {
          id: 'tl-030',
          title: '임대차 계약 체결',
          description: '서초동 오피스텔 보증금 3,000만원, 월세 50만원',
          date_text: '2023년 10월 1일',
          date_normalized: '2023-10-01',
          category: '계약',
          source_type: 'ai_extracted',
          confidence: 0.92,
          source_conversation_id: null,
          created_at: '2026-01-10T17:00:00Z',
        },
        {
          id: 'tl-031',
          title: '계약 만료 및 퇴거 통보',
          description: '계약 만료일에 퇴거 통보, 보증금 반환 요청',
          date_text: '2025년 9월 30일',
          date_normalized: '2025-09-30',
          category: '계약',
          source_type: 'ai_extracted',
          confidence: 0.90,
          source_conversation_id: null,
          created_at: '2026-01-10T17:00:00Z',
        },
        {
          id: 'tl-032',
          title: '임대인 보증금 반환 거부',
          description: '수리비 공제 주장하며 보증금 반환 거부',
          date_text: '2025년 10월 15일',
          date_normalized: '2025-10-15',
          category: '분쟁',
          source_type: 'ai_extracted',
          confidence: 0.87,
          source_conversation_id: null,
          created_at: '2026-01-10T17:00:00Z',
        },
      ],
      conversations: [
        { id: 'conv-030', title: '보증금 반환 청구 상담', message_count: 11 },
        { id: 'conv-031', title: '내용증명 작성 도움', message_count: 5 },
      ],
    },
    'demo-005': {
      timeline: [
        {
          id: 'tl-040',
          title: '명예훼손 게시물 작성',
          description: '인터넷 커뮤니티에 허위사실 적시 게시물 게재',
          date_text: '2025년 7월 20일',
          date_normalized: '2025-07-20',
          category: '사건 경위',
          source_type: 'ai_extracted',
          confidence: 0.91,
          source_conversation_id: null,
          created_at: '2025-09-01T11:00:00Z',
        },
      ],
      conversations: [
        { id: 'conv-040', title: '명예훼손 고소 상담', message_count: 7 },
      ],
    },
  }

  const detail = DETAILS[caseId]
  if (!detail) {
    return { ...base, timeline: [], conversations: [] }
  }

  return { ...base, ...detail }
}
