'use client'

import type { CourtCostBreakdown } from '../types'

interface CourtCostCalculatorProps {
  claimAmount: number | undefined
}

const DELIVERY_COST_PER_TIME = 5200
const DEFAULT_DELIVERY_COUNT = 15
const DEFAULT_PARTY_COUNT = 2

function calculateStampFee(amount: number): number {
  if (amount <= 0) return 0

  if (amount <= 10_000_000) {
    return Math.max(Math.round(amount * 0.005), 1000)
  }
  if (amount <= 100_000_000) {
    return 50_000 + Math.round((amount - 10_000_000) * 0.0045)
  }
  return 455_000 + Math.round((amount - 100_000_000) * 0.004)
}

function calculateCourtCost(claimAmount: number): CourtCostBreakdown {
  const stampFee = calculateStampFee(claimAmount)
  const deliveryFee = DEFAULT_PARTY_COUNT * DELIVERY_COST_PER_TIME * DEFAULT_DELIVERY_COUNT
  const totalCost = stampFee + deliveryFee

  return {
    claimAmount,
    stampFee,
    deliveryFee,
    totalCost,
    partyCount: DEFAULT_PARTY_COUNT,
  }
}

function formatWon(amount: number): string {
  return amount.toLocaleString('ko-KR')
}

export function CourtCostCalculator({ claimAmount }: CourtCostCalculatorProps) {
  if (!claimAmount || claimAmount <= 0) {
    return null
  }

  const cost = calculateCourtCost(claimAmount)

  return (
    <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl border border-blue-200 p-5">
      <div className="flex items-center gap-2 mb-4">
        <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z"
          />
        </svg>
        <h3 className="font-semibold text-gray-900">예상 소송 비용</h3>
      </div>

      <div className="space-y-3">
        {/* 청구 금액 */}
        <div className="flex items-center justify-between text-sm">
          <span className="text-gray-600">청구 금액</span>
          <span className="font-medium text-gray-900">{formatWon(cost.claimAmount)}원</span>
        </div>

        <div className="border-t border-blue-200" />

        {/* 인지대 */}
        <div className="flex items-center justify-between text-sm">
          <div className="flex items-center gap-1">
            <span className="text-gray-600">인지대</span>
            <span className="text-xs text-gray-400" title="소가에 따른 법정 요율">
              (법정 요율)
            </span>
          </div>
          <span className="font-medium text-gray-900">{formatWon(cost.stampFee)}원</span>
        </div>

        {/* 송달료 */}
        <div className="flex items-center justify-between text-sm">
          <div className="flex items-center gap-1">
            <span className="text-gray-600">송달료</span>
            <span className="text-xs text-gray-400">
              ({cost.partyCount}명 x {DEFAULT_DELIVERY_COUNT}회)
            </span>
          </div>
          <span className="font-medium text-gray-900">{formatWon(cost.deliveryFee)}원</span>
        </div>

        <div className="border-t border-blue-300" />

        {/* 합계 */}
        <div className="flex items-center justify-between">
          <span className="font-semibold text-gray-900">합계</span>
          <span className="text-lg font-bold text-blue-700">{formatWon(cost.totalCost)}원</span>
        </div>
      </div>

      <p className="mt-3 text-xs text-gray-500 leading-relaxed">
        * 인지대는 소가 기준 법정 요율로 계산됩니다. 송달료는 당사자 {cost.partyCount}명, {DEFAULT_DELIVERY_COUNT}회
        기준 산정이며 실제와 다를 수 있습니다. 승소 시 상대방에게 소송 비용 청구가 가능합니다.
      </p>
    </div>
  )
}
