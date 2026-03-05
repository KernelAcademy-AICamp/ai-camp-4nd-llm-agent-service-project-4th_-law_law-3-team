'use client'

import { useEffect, useState } from 'react'
import type { TrendDetailResponse, TrendIssue, TrendSource } from '../types'
import { ScoreBar } from './ScoreBar'

const SOURCE_LABELS: Record<TrendSource, string> = {
  tavily: 'Tavily',
  naver: 'Naver',
  perplexity: 'Perplexity',
  google_trends: 'Google',
  youtube: 'YouTube',
  newsdata: 'NewsData',
  newsapi: 'NewsAPI',
}

interface TrendDetailViewProps {
  issue: TrendIssue
  onClose: () => void
  onGenerateScript: (issue: TrendIssue) => void
  loadDetail: (trendId: string) => Promise<TrendDetailResponse | null>
}

export function TrendDetailView({
  issue,
  onClose,
  onGenerateScript,
  loadDetail,
}: TrendDetailViewProps) {
  const [detail, setDetail] = useState<TrendDetailResponse | null>(null)
  const [detailError, setDetailError] = useState(false)

  useEffect(() => {
    setDetailError(false)
    loadDetail(issue.id).then((result) => {
      setDetail(result)
      if (!result) setDetailError(true)
    })
  }, [issue.id, loadDetail])

  const articles = detail?.source_articles ?? issue.source_articles

  const relatedLaws = detail?.related_laws_detail?.length
    ? detail.related_laws_detail
    : (detail?.issue?.related_laws ?? issue.related_laws)

  const relatedCases = detail?.related_cases_detail?.length
    ? detail.related_cases_detail
    : (detail?.issue?.related_cases ?? issue.related_cases)

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
      <div className="bg-white rounded-2xl shadow-xl w-full max-w-2xl max-h-[85vh] overflow-y-auto m-4">
        {/* 헤더 */}
        <div className="sticky top-0 bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between">
          <button
            onClick={onClose}
            className="text-sm text-gray-500 hover:text-gray-700"
          >
            ← 뒤로
          </button>
          <h2 className="text-lg font-bold text-gray-900 truncate mx-4">
            {issue.title}
          </h2>
          <div className="w-10" />
        </div>

        <div className="px-6 py-5 space-y-6">
          {/* 요약 */}
          <section>
            <h3 className="text-sm font-semibold text-gray-500 mb-2">요약</h3>
            <p className="text-sm text-gray-700 bg-gray-50 rounded-lg p-3">
              {issue.summary}
            </p>
          </section>

          {/* 점수 */}
          <section>
            <h3 className="text-sm font-semibold text-gray-500 mb-2">
              종합점수: {issue.score.toFixed(1)} / 100
            </h3>

            {/* v2.0 Legal Gate 상세 */}
            {issue.score_detail ? (
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <span
                    className={`px-2 py-0.5 text-xs font-medium rounded-full ${
                      issue.score_detail.legal_gate_passed
                        ? 'bg-green-100 text-green-700'
                        : 'bg-red-100 text-red-600'
                    }`}
                  >
                    Legal Gate {issue.score_detail.legal_gate_passed ? '통과' : '미달'}
                  </span>
                  {issue.fitness_label && (
                    <span className="px-2 py-0.5 text-xs bg-purple-100 text-purple-700 rounded-full">
                      {issue.fitness_label}
                    </span>
                  )}
                  {!issue.score_detail.legal_gate_passed && issue.score_detail.gate_rejection_reason && (
                    <span className="text-xs text-red-500">
                      ({issue.score_detail.gate_rejection_reason})
                    </span>
                  )}
                </div>
                <div className="space-y-1.5">
                  <ScoreBar score={issue.score_detail.mention_score} label="언급량" />
                  <ScoreBar score={issue.score_detail.legal_score} label="법적쟁점화" />
                  <ScoreBar score={issue.score_detail.controversy_score} label="논란도" />
                  <ScoreBar score={issue.score_detail.spread_score} label="확산도" />
                  <ScoreBar score={issue.score_detail.fitness_score} label="채널적합도" />
                </div>
              </div>
            ) : (
              <div className="space-y-1.5">
                <ScoreBar score={issue.mention_score} label="언급량" />
                <ScoreBar score={issue.legal_relevance_score} label="법적해석" />
              </div>
            )}
          </section>

          {/* 핵심 쟁점 */}
          <section>
            <h3 className="text-sm font-semibold text-gray-500 mb-2">핵심 쟁점</h3>
            <ol className="space-y-1.5">
              {issue.key_points.map((point, index) => (
                <li key={`point-${index}`} className="text-sm text-gray-700 flex gap-2">
                  <span className="text-blue-500 font-medium shrink-0">{index + 1}.</span>
                  {point}
                </li>
              ))}
            </ol>
          </section>

          {/* 관련 법령 */}
          <section>
            <h3 className="text-sm font-semibold text-gray-500 mb-2">관련 법령</h3>
            {relatedLaws.length > 0 ? (
              <div className="space-y-1.5">
                {relatedLaws.map((law) => (
                  <div
                    key={law.law_id}
                    className="flex items-center justify-between text-sm bg-blue-50 rounded-lg px-3 py-2"
                  >
                    <span className="text-gray-700 truncate mr-2">📜 {law.law_name}</span>
                    <a
                      href={`/law-search?id=${law.law_id}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="shrink-0 text-xs text-blue-600 hover:underline"
                    >
                      원본 보기
                    </a>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-xs text-gray-400">
                {detailError
                  ? '상세 데이터를 불러올 수 없습니다. 트렌드를 다시 조회해 주세요.'
                  : !detail
                    ? '로딩 중...'
                    : '관련 법령이 없습니다.'}
              </p>
            )}
          </section>

          {/* 관련 판례 */}
          <section>
            <h3 className="text-sm font-semibold text-gray-500 mb-2">관련 판례</h3>
            {relatedCases.length > 0 ? (
              <div className="space-y-1.5">
                {relatedCases.map((caseItem) => (
                  <div
                    key={caseItem.case_id}
                    className="flex items-center justify-between text-sm bg-purple-50 rounded-lg px-3 py-2"
                  >
                    <span className="text-gray-700 truncate mr-2">
                      ⚖️ {caseItem.case_number} ({caseItem.case_name})
                    </span>
                    <a
                      href={`/case-precedent?id=${caseItem.case_id}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="shrink-0 text-xs text-purple-600 hover:underline"
                    >
                      원본 보기
                    </a>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-xs text-gray-400">
                {detailError
                  ? '상세 데이터를 불러올 수 없습니다. 트렌드를 다시 조회해 주세요.'
                  : !detail
                    ? '로딩 중...'
                    : '관련 판례가 없습니다.'}
              </p>
            )}
          </section>

          {/* 원본 기사 */}
          {articles.length > 0 && (
            <section>
              <h3 className="text-sm font-semibold text-gray-500 mb-2">
                원본 기사 ({articles.length}건)
              </h3>
              <div className="space-y-2">
                {articles.map((article, index) => (
                  <a
                    key={`article-${index}`}
                    href={article.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-start gap-2 text-sm bg-gray-50 rounded-lg px-3 py-2 hover:bg-gray-100 transition-colors"
                  >
                    <span className="px-1.5 py-0.5 text-xs bg-gray-200 text-gray-600 rounded shrink-0">
                      {SOURCE_LABELS[article.source]}
                    </span>
                    <span className="text-blue-600 hover:underline line-clamp-1">
                      {article.title}
                    </span>
                  </a>
                ))}
              </div>
            </section>
          )}

          {/* 대본 생성 버튼 */}
          <button
            onClick={() => onGenerateScript(issue)}
            className="w-full py-3 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 transition-colors"
          >
            이 주제로 대본 생성하기
          </button>
        </div>
      </div>
    </div>
  )
}
