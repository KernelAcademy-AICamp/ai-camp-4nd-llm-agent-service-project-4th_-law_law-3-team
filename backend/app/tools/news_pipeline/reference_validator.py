"""요약 내 법령/판례 Cross-Reference 검증"""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.law_document import LawDocument
from app.models.precedent_document import PrecedentDocument
from app.tools.news_pipeline.models import ArticleSummary

logger = logging.getLogger(__name__)


class ReferenceValidator:
    """요약 내 법령/판례 번호가 DB에 존재하는지 검증

    존재하지 않는 법령/판례는 요약에서 제거하여 환각(hallucination) 방지.
    v0.2.0: 검증된 참조의 DB ID도 함께 반환하여 프론트엔드 직접 링크 지원.
    """

    async def validate_and_filter(
        self,
        summary: ArticleSummary,
        db: AsyncSession,
    ) -> ArticleSummary:
        """검증 후 유효한 참조만 남긴 요약 반환 (DB ID 포함)"""
        validated_laws, law_ids = await self._validate_laws(summary.laws, db)
        validated_cases, case_ids = await self._validate_cases(summary.cases, db)

        removed_laws = set(summary.laws) - set(validated_laws)
        removed_cases = set(summary.cases) - set(validated_cases)

        if removed_laws:
            logger.info("환각 법령 제거: %s", removed_laws)
        if removed_cases:
            logger.info("환각 판례 제거: %s", removed_cases)

        return ArticleSummary(
            one_liner=summary.one_liner,
            issues=summary.issues,
            laws=validated_laws,
            cases=validated_cases,
            institutions=summary.institutions,
            implications=summary.implications,
            law_ids=law_ids,
            case_ids=case_ids,
        )

    async def _validate_laws(
        self, laws: list[str], db: AsyncSession,
    ) -> tuple[list[str], list[int]]:
        """법령명이 DB에 존재하는지 확인 (부분 매칭 허용). ID도 반환."""
        valid: list[str] = []
        valid_ids: list[int] = []
        for law_name in laws:
            base_name = law_name.split(" 제")[0].strip()
            result = await db.execute(
                select(LawDocument.id).where(
                    LawDocument.law_name.contains(base_name),
                ).limit(1),
            )
            row = result.scalar_one_or_none()
            if row is not None:
                valid.append(law_name)
                valid_ids.append(row)
        return valid, valid_ids

    async def _validate_cases(
        self, cases: list[str], db: AsyncSession,
    ) -> tuple[list[str], list[int]]:
        """판례 번호가 DB에 존재하는지 확인. ID도 반환."""
        valid: list[str] = []
        valid_ids: list[int] = []
        for case_ref in cases:
            result = await db.execute(
                select(PrecedentDocument.id).where(
                    PrecedentDocument.case_number.contains(case_ref),
                ).limit(1),
            )
            row = result.scalar_one_or_none()
            if row is not None:
                valid.append(case_ref)
                valid_ids.append(row)
        return valid, valid_ids
