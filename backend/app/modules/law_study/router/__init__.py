"""변호사시험 기록형 연습 - API 엔드포인트"""

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.rate_limit import AI_RATE_LIMIT, limiter
from app.modules.law_study.schema import (
    AnswerFeedbackRequest,
    AnswerFeedbackResponse,
    ExamContentResponse,
    ExamListResponse,
    ReferenceSearchRequest,
    ReferenceSearchResponse,
    ReferenceSearchResult,
)
from app.modules.law_study.service import (
    CATEGORY_LABELS,
    get_exam_content,
    list_exam_files,
)

router = APIRouter()

FEEDBACK_SYSTEM_PROMPT = """당신은 변호사시험 기록형 채점 전문가입니다.
학생의 답안을 아래 구조로 피드백해주세요:

## 전반적 평가
(답안의 전반적 완성도를 간략히 평가)

## 잘된 부분
- (구체적으로 잘 작성된 부분)

## 개선이 필요한 부분
- (부족하거나 오류가 있는 부분과 개선 방향)

## 법리 검토
- (답안에서 다룬 법리의 정확성 검토)

## 핵심 조언
(가장 중요한 개선 포인트 1-2가지)"""

# 문제 컨텍스트로 포함할 최대 줄 수
_PROBLEM_CONTEXT_MAX_LINES = 200


@router.get("/exams")
async def get_exam_list(
    category: str | None = Query(default=None, max_length=20),
) -> ExamListResponse:
    """시험 문제 목록 조회 (카테고리 필터 가능)"""
    exams = list_exam_files()
    if category:
        upper = category.upper()
        exams = [e for e in exams if e["category"] == upper]
    return ExamListResponse(
        exams=exams,  # type: ignore[arg-type]
        total=len(exams),
    )


@router.get("/exams/{category}/{session}")
async def get_exam_detail(category: str, session: int) -> ExamContentResponse:
    """시험 문제 전문 조회"""
    content = get_exam_content(category, session)
    if content is None:
        raise HTTPException(status_code=404, detail="해당 시험 문제를 찾을 수 없습니다.")

    year = 2011 + session
    label = CATEGORY_LABELS.get(category.upper(), category)
    return ExamContentResponse(
        category=category.upper(),
        year=year,
        session=session,
        title=f"제{session}회 변호사시험 {label} 기록형 ({year}년)",
        content=content,
        total_chars=len(content),
    )


@router.post("/reference/search")
async def search_references(
    body: ReferenceSearchRequest,
) -> ReferenceSearchResponse:
    """판례/법령 오픈북 RAG 검색"""
    from app.services.rag.retrieval import search_relevant_documents_async

    raw_results: list[dict[str, Any]] = await search_relevant_documents_async(
        query=body.query,
        n_results=body.n_results,
        doc_type=body.doc_type,
    )

    results: list[ReferenceSearchResult] = []
    for doc in raw_results:
        results.append(
            ReferenceSearchResult(
                id=doc.get("id", ""),
                doc_type=doc.get("doc_type", doc.get("data_type", "")),
                title=doc.get("title", ""),
                case_number=doc.get("case_number"),
                summary=doc.get("summary", doc.get("content", ""))[:500],
                similarity=round(doc.get("similarity", doc.get("score", 0.0)), 4),
            )
        )

    return ReferenceSearchResponse(query=body.query, results=results)


@router.post("/exams/{category}/{session}/feedback")
@limiter.limit(AI_RATE_LIMIT)
async def get_answer_feedback(
    request: Request,
    category: str,
    session: int,
    body: AnswerFeedbackRequest,
) -> AnswerFeedbackResponse:
    """AI 답안 피드백 생성"""
    content = get_exam_content(category, session)
    if content is None:
        raise HTTPException(status_code=404, detail="해당 시험 문제를 찾을 수 없습니다.")

    # 문제 앞부분만 LLM 컨텍스트에 포함 (토큰 절약)
    lines = content.split("\n")
    problem_context = "\n".join(lines[:_PROBLEM_CONTEXT_MAX_LINES])

    from app.tools.llm import get_chat_model

    llm = get_chat_model(temperature=0.3, max_tokens=4096)

    year = 2011 + session
    label = CATEGORY_LABELS.get(category.upper(), category)
    user_prompt = (
        f"## 시험 정보\n제{session}회 변호사시험 {label} 기록형 ({year}년)\n\n"
        f"## 문제 (발췌)\n{problem_context}\n\n"
        f"## 학생 답안\n{body.answer_text}"
    )

    messages = [
        SystemMessage(content=FEEDBACK_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    response = await llm.ainvoke(messages)
    feedback_text = str(response.content)

    return AnswerFeedbackResponse(feedback=feedback_text)
