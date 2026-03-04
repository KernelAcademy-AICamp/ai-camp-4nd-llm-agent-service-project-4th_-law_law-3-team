"""
모의 법정 모듈 라우터

Design 문서 Section 4.2-4.3 기반 엔드포인트 구현
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from app.core.rate_limit import AI_RATE_LIMIT, limiter
from app.modules.mock_trial.schema import (
    CASE_TYPES,
    CIVIL_ROLES,
    CIVIL_STAGES,
    CRIMINAL_ROLES,
    CRIMINAL_STAGES,
    CaseTypeInfo,
    CaseTypesResponse,
    EvidenceArticleItem,
    EvidenceCaseItem,
    EvidenceSearchRequest,
    EvidenceSearchResponse,
    RoleInfo,
    RolesResponse,
    StageInfo,
    StageInfoResponse,
)
from app.services.service_function.mock_trial_service import (
    get_evidence_searcher,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/case-types", response_model=CaseTypesResponse)
async def get_case_types() -> CaseTypesResponse:
    """사건 유형 목록 반환"""
    return CaseTypesResponse(
        case_types=[CaseTypeInfo(**ct) for ct in CASE_TYPES]
    )


@router.get("/roles/{case_type}", response_model=RolesResponse)
async def get_roles(case_type: str) -> RolesResponse:
    """사건 유형별 선택 가능 역할 반환

    Args:
        case_type: "criminal" 또는 "civil"
    """
    if case_type == "criminal":
        roles = CRIMINAL_ROLES
    elif case_type == "civil":
        roles = CIVIL_ROLES
    else:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 사건 유형: {case_type}. "
            "criminal 또는 civil만 가능합니다.",
        )

    return RolesResponse(
        case_type=case_type,
        roles=[RoleInfo(id=r["id"], name=r["name"], description=r["description"]) for r in roles],
    )


@router.post("/search-evidence", response_model=EvidenceSearchResponse)
@limiter.limit(AI_RATE_LIMIT)
async def search_evidence(
    request: Request,
    body: EvidenceSearchRequest,
) -> EvidenceSearchResponse:
    """모의재판 전용 판례/법령 검색"""
    searcher = get_evidence_searcher()
    cases: list[dict[str, Any]] = []
    articles: list[dict[str, Any]] = []

    if body.search_type in ("all", "cases"):
        cases = await searcher.search_cases(body.query, body.limit)

    if body.search_type in ("all", "articles"):
        articles = await searcher.search_articles(body.query, body.limit)

    return EvidenceSearchResponse(
        cases=[
            EvidenceCaseItem(
                id=c.get("id", ""),
                title=c.get("title", ""),
                summary=c.get("summary", ""),
                relevance_score=c.get("relevance_score", 0.0),
                source=c.get("source", "lancedb"),
            )
            for c in cases
        ],
        articles=[
            EvidenceArticleItem(
                id=a.get("id", ""),
                title=a.get("title", ""),
                content=a.get("content", ""),
                relevance_score=a.get("relevance_score", 0.0),
                source=a.get("source", "lancedb"),
            )
            for a in articles
        ],
    )


@router.get("/stage-info/{case_type}", response_model=StageInfoResponse)
async def get_stage_info(case_type: str) -> StageInfoResponse:
    """사건 유형별 단계 정보 반환

    Args:
        case_type: "criminal" 또는 "civil"
    """
    if case_type == "criminal":
        stages = CRIMINAL_STAGES
    elif case_type == "civil":
        stages = CIVIL_STAGES
    else:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 사건 유형: {case_type}. "
            "criminal 또는 civil만 가능합니다.",
        )

    return StageInfoResponse(
        case_type=case_type,
        stages=[
            StageInfo(
                id=s["id"],
                name=s["name"],
                order=int(s["order"]),
                legal_basis=s["legal_basis"],
                description=s["description"],
                user_action=s["user_action"],
                duration_hint=s["duration_hint"],
            )
            for s in stages
        ],
    )
