'use client'

import { useState, useCallback } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { lawStudyService } from '../services'
import type {
  ExamFile,
  ReferenceSearchResult,
} from '../types'

export function useLawStudy() {
  // 선택 상태
  const [selectedCategory, setSelectedCategory] = useState<string>('CIVIL')
  const [selectedExam, setSelectedExam] = useState<ExamFile | null>(null)
  const [answerText, setAnswerText] = useState('')

  // 패널 토글
  const [isReferencePanelOpen, setReferencePanelOpen] = useState(false)
  const [isFeedbackPanelOpen, setFeedbackPanelOpen] = useState(false)

  // 참조 검색 상태
  const [referenceResults, setReferenceResults] = useState<ReferenceSearchResult[]>([])
  const [isSearchingReference, setIsSearchingReference] = useState(false)

  // 시험 목록 조회
  const examListQuery = useQuery({
    queryKey: ['law-study', 'exams'],
    queryFn: () => lawStudyService.getExamList(),
    staleTime: Infinity,
  })

  // 시험 내용 조회
  const examContentQuery = useQuery({
    queryKey: ['law-study', 'content', selectedExam?.category, selectedExam?.session],
    queryFn: () =>
      lawStudyService.getExamContent(selectedExam!.category, selectedExam!.session),
    enabled: !!selectedExam,
  })

  // AI 피드백
  const feedbackMutation = useMutation({
    mutationFn: () =>
      lawStudyService.getAnswerFeedback(
        selectedExam!.category,
        selectedExam!.session,
        { answer_text: answerText },
      ),
    onSuccess: () => {
      setFeedbackPanelOpen(true)
    },
  })

  // 문제 선택
  const selectExam = useCallback((exam: ExamFile) => {
    setSelectedExam(exam)
    setAnswerText('')
    setFeedbackPanelOpen(false)
    feedbackMutation.reset()
  }, [feedbackMutation])

  // 참조 검색
  const searchReference = useCallback(
    async (query: string, docType?: string | null) => {
      if (!query.trim()) return
      setIsSearchingReference(true)
      try {
        const response = await lawStudyService.searchReferences({
          query,
          doc_type: docType,
          n_results: 10,
        })
        setReferenceResults(response.results)
      } finally {
        setIsSearchingReference(false)
      }
    },
    [],
  )

  // 카테고리별 필터링된 시험 목록
  const filteredExams = (examListQuery.data?.exams ?? []).filter(
    (e) => e.category === selectedCategory,
  )

  return {
    // 상태
    selectedCategory,
    setSelectedCategory,
    selectedExam,
    selectExam,
    answerText,
    setAnswerText,
    isReferencePanelOpen,
    setReferencePanelOpen,
    isFeedbackPanelOpen,
    setFeedbackPanelOpen,

    // 시험 데이터
    filteredExams,
    examListQuery,
    examContentQuery,

    // 피드백
    feedbackMutation,

    // 참조 검색
    referenceResults,
    isSearchingReference,
    searchReference,
  }
}
