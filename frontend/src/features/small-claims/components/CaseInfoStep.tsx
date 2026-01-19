'use client'

import type { CaseInfo } from '../types'

interface CaseInfoStepProps {
  caseInfo: Partial<CaseInfo>
  onUpdate: (info: Partial<CaseInfo>) => void
  onNext: () => void
  onPrevious: () => void
}

export function CaseInfoStep({ caseInfo, onUpdate, onNext, onPrevious }: CaseInfoStepProps) {
  const isValid =
    caseInfo.plaintiff_name &&
    caseInfo.plaintiff_address &&
    caseInfo.defendant_name &&
    caseInfo.amount &&
    caseInfo.description

  const handleAmountChange = (value: string) => {
    const numericValue = value.replace(/[^0-9]/g, '')
    onUpdate({ amount: numericValue ? parseInt(numericValue, 10) : undefined })
  }

  const formatAmount = (amount: number | undefined) => {
    if (!amount) return ''
    return amount.toLocaleString('ko-KR')
  }

  return (
    <div className="max-w-2xl mx-auto">
      <div className="text-center mb-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">사건 정보 입력</h2>
        <p className="text-gray-600">소송에 필요한 기본 정보를 입력해주세요</p>
      </div>

      <div className="bg-white rounded-xl border border-gray-200 p-6 space-y-6">
        {/* 원고(본인) 정보 */}
        <div>
          <h3 className="font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <span className="w-6 h-6 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-sm">
              1
            </span>
            원고(본인) 정보
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                이름 <span className="text-red-500">*</span>
              </label>
              <input
                type="text"
                value={caseInfo.plaintiff_name || ''}
                onChange={(e) => onUpdate({ plaintiff_name: e.target.value })}
                placeholder="홍길동"
                className="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">연락처</label>
              <input
                type="tel"
                value={caseInfo.plaintiff_phone || ''}
                onChange={(e) => onUpdate({ plaintiff_phone: e.target.value })}
                placeholder="010-0000-0000"
                className="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div className="md:col-span-2">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                주소 <span className="text-red-500">*</span>
              </label>
              <input
                type="text"
                value={caseInfo.plaintiff_address || ''}
                onChange={(e) => onUpdate({ plaintiff_address: e.target.value })}
                placeholder="서울시 강남구 테헤란로 123"
                className="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>
        </div>

        {/* 피고(상대방) 정보 */}
        <div>
          <h3 className="font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <span className="w-6 h-6 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-sm">
              2
            </span>
            피고(상대방) 정보
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                이름/상호명 <span className="text-red-500">*</span>
              </label>
              <input
                type="text"
                value={caseInfo.defendant_name || ''}
                onChange={(e) => onUpdate({ defendant_name: e.target.value })}
                placeholder="김철수 또는 (주)회사명"
                className="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">연락처</label>
              <input
                type="tel"
                value={caseInfo.defendant_phone || ''}
                onChange={(e) => onUpdate({ defendant_phone: e.target.value })}
                placeholder="010-0000-0000"
                className="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div className="md:col-span-2">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                주소 <span className="text-gray-400">(알고 있는 경우)</span>
              </label>
              <input
                type="text"
                value={caseInfo.defendant_address || ''}
                onChange={(e) => onUpdate({ defendant_address: e.target.value })}
                placeholder="서울시 서초구 서초대로 456"
                className="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>
        </div>

        {/* 청구 내용 */}
        <div>
          <h3 className="font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <span className="w-6 h-6 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-sm">
              3
            </span>
            청구 내용
          </h3>
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  청구 금액 <span className="text-red-500">*</span>
                </label>
                <div className="relative">
                  <input
                    type="text"
                    value={formatAmount(caseInfo.amount)}
                    onChange={(e) => handleAmountChange(e.target.value)}
                    placeholder="1,000,000"
                    className="w-full px-4 py-2.5 pr-8 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                  <span className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-500">
                    원
                  </span>
                </div>
                {caseInfo.amount && caseInfo.amount > 30000000 && (
                  <p className="mt-1 text-xs text-amber-600">
                    * 3,000만원 초과 시 소액사건이 아닌 일반 민사소송입니다
                  </p>
                )}
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">발생일</label>
                <input
                  type="date"
                  value={caseInfo.incident_date || ''}
                  onChange={(e) => onUpdate({ incident_date: e.target.value })}
                  className="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                분쟁 경위 <span className="text-red-500">*</span>
              </label>
              <textarea
                value={caseInfo.description || ''}
                onChange={(e) => onUpdate({ description: e.target.value })}
                placeholder="어떤 일이 있었는지 자세히 설명해주세요. (예: 2024년 1월 15일 중고나라에서 아이폰을 50만원에 구매했으나, 입금 후 연락이 두절되었습니다.)"
                rows={5}
                className="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
              />
            </div>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <div className="flex justify-between mt-6">
        <button
          onClick={onPrevious}
          className="px-6 py-3 border border-gray-300 text-gray-700 rounded-lg font-medium hover:bg-gray-50 transition"
        >
          이전
        </button>
        <button
          onClick={onNext}
          disabled={!isValid}
          className="px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition"
        >
          다음 단계
        </button>
      </div>
    </div>
  )
}
