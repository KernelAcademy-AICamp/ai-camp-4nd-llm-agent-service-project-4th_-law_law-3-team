'use client'

import { useState, useCallback, useMemo, useRef } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { lawStudyService } from '../services'
import type { ExamFile, ReferenceSearchRequest } from '../types'

export function useLawStudy() {
  // 선택 상태
  const [selectedCategory, setSelectedCategory] = useState<string>('CIVIL')
  const [selectedExam, setSelectedExam] = useState<ExamFile | null>(null)
  const [answerText, setAnswerText] = useState('')

  // 패널 토글
  const [isReferencePanelOpen, setReferencePanelOpen] = useState(false)
  const [isFeedbackPanelOpen, setFeedbackPanelOpen] = useState(false)

  // 시험 목록 조회
  const examListQuery = useQuery({
    queryKey: ['law-study', 'exams'],
    queryFn: () => lawStudyService.getExamList(),
    staleTime: Infinity,
  })

  // 시험 내용 조회 (staleTime 추가 - 시험 문제는 자주 변경되지 않음)
  const examContentQuery = useQuery({
    queryKey: ['law-study', 'content', selectedExam?.category, selectedExam?.session],
    queryFn: () =>
      lawStudyService.getExamContent(selectedExam!.category, selectedExam!.session),
    enabled: !!selectedExam,
    staleTime: 5 * 60 * 1000,
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

  // 판례/법령 참조 검색 (useMutation으로 에러 핸들링 개선)
  const referenceMutation = useMutation({
    mutationFn: (params: ReferenceSearchRequest) =>
      lawStudyService.searchReferences(params),
  })

  // 안정적인 콜백 참조를 위한 refs (rerender-dependencies 해결)
  const feedbackResetRef = useRef(feedbackMutation.reset)
  feedbackResetRef.current = feedbackMutation.reset

  const referenceMutateRef = useRef(referenceMutation.mutate)
  referenceMutateRef.current = referenceMutation.mutate

  // 문제 선택 (안정적 참조 - 불필요한 자식 리렌더 방지)
  const selectExam = useCallback((exam: ExamFile) => {
    setSelectedExam(exam)
    setAnswerText('')
    setFeedbackPanelOpen(false)
    feedbackResetRef.current()
  }, [])

  // 참조 검색 (안정적 참조)
  const searchReference = useCallback(
    (query: string, docType?: string | null) => {
      if (!query.trim()) return
      referenceMutateRef.current({ query: query.trim(), doc_type: docType, n_results: 10 })
    },
    [],
  )

  // 카테고리별 필터링된 시험 목록 (메모이제이션)
  const filteredExams = useMemo(
    () =>
      (examListQuery.data?.exams ?? []).filter(
        (e) => e.category === selectedCategory,
      ),
    [examListQuery.data?.exams, selectedCategory],
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
    referenceResults: referenceMutation.data?.results ?? [],
    isSearchingReference: referenceMutation.isPending,
    searchReference,
  }
}
