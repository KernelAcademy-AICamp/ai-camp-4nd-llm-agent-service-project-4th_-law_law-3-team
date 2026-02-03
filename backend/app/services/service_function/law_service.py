"""
법령 서비스

법령 조회 및 참조조문 처리
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set

from sqlalchemy import or_, select

from app.core.database import sync_session_factory
from app.models.law import Law
from app.models.legal_document import LegalDocument

logger = logging.getLogger(__name__)

# 상수
MAX_CONTENT_TRUNCATE_LENGTH = 1000


def extract_law_names(reference_articles: str) -> Set[str]:
    """
    참조조문에서 법령명 추출

    Args:
        reference_articles: 참조조문 문자열
            예: "민사소송법 제704조 제2항", "근로기준법 제34조"

    Returns:
        법령명 집합 (예: {"민사소송법", "근로기준법"})
    """
    if not reference_articles:
        return set()

    law_names: Set[str] = set()

    # 패턴 1: "구 법령명(날짜...)" 형식에서 법령명 추출
    pattern_old = r'구\s+([가-힣]+(?:법|령|규칙|규정))'
    for match in re.finditer(pattern_old, reference_articles):
        law_names.add(match.group(1))

    # 패턴 2: 일반 법령명 추출
    pattern_normal = r'([가-힣]+(?:법|령|규칙|규정))(?:\s|$|제|\(|,)'
    for match in re.finditer(pattern_normal, reference_articles):
        name = match.group(1)
        if not name.startswith("구") and len(name) >= 2:
            law_names.add(name)

    return law_names


def fetch_laws_by_names(law_names: Set[str], limit: int = 5) -> List[Dict[str, Any]]:
    """
    법령명으로 laws 테이블에서 법령 조회

    Args:
        law_names: 법령명 집합
        limit: 최대 조회 건수

    Returns:
        법령 정보 목록
    """
    if not law_names:
        return []

    try:
        with sync_session_factory() as session:
            conditions = []
            for name in law_names:
                conditions.append(Law.law_name.ilike(f"%{name}%"))

            if not conditions:
                return []

            result = session.execute(
                select(Law)
                .where(or_(*conditions))
                .limit(limit)
            )
            laws = result.scalars().all()

            return [
                {
                    "law_id": law.law_id,
                    "law_name": law.law_name,
                    "law_type": law.law_type,
                    "content": (
                        law.content[:MAX_CONTENT_TRUNCATE_LENGTH] if law.content else ""
                    ),
                }
                for law in laws
            ]
    except Exception as e:
        logger.warning("법령 조회 실패: %s", e)
        return []


def fetch_reference_articles_from_docs(doc_ids: List[int]) -> str:
    """
    문서 ID 목록에서 reference_articles 수집

    Args:
        doc_ids: 문서 ID 목록

    Returns:
        모든 reference_articles를 합친 문자열
    """
    if not doc_ids:
        return ""

    try:
        with sync_session_factory() as session:
            result = session.execute(
                select(LegalDocument.reference_articles)
                .where(LegalDocument.id.in_(doc_ids))
                .where(LegalDocument.reference_articles.isnot(None))
            )
            articles = [row[0] for row in result.fetchall() if row[0]]

        return " ".join(articles)
    except Exception as e:
        logger.warning("참조조문 조회 실패: %s", e)
        return ""


class LawService:
    """법령 서비스 클래스"""

    def extract_law_names(self, reference_text: str) -> Set[str]:
        """참조조문에서 법령명 추출"""
        return extract_law_names(reference_text)

    def get_laws_by_names(
        self,
        names: Set[str],
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """법령명으로 법령 조회"""
        return fetch_laws_by_names(names, limit)

    def get_reference_articles(self, doc_ids: List[int]) -> str:
        """문서 ID 목록에서 참조조문 수집"""
        return fetch_reference_articles_from_docs(doc_ids)

    def get_law_by_id(self, law_id: str) -> Optional[Dict[str, Any]]:
        """
        법령 ID로 단일 법령 조회

        Args:
            law_id: 법령 ID

        Returns:
            법령 정보 또는 None
        """
        try:
            with sync_session_factory() as session:
                result = session.execute(
                    select(Law).where(Law.law_id == law_id)
                )
                law = result.scalar_one_or_none()

                if law:
                    return {
                        "law_id": law.law_id,
                        "law_name": law.law_name,
                        "law_type": law.law_type,
                        "content": law.content,
                    }
                return None
        except Exception as e:
            logger.warning("법령 조회 실패: %s", e)
            return None


_law_service: Optional[LawService] = None


def get_law_service() -> LawService:
    """LawService 싱글톤 인스턴스 반환"""
    global _law_service
    if _law_service is None:
        _law_service = LawService()
    return _law_service
