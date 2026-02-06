"""
법률 용어 메모리 사전 (MeCab 토크나이저 보강용)

frozenset 기반 O(1) lookup으로 법률 복합명사를 빠르게 검색.
DB 또는 JSON에서 용어를 로드하여 메모리에 캐싱.

Usage:
    from app.tools.vectorstore.legal_term_dict import LegalTermDictionary

    # JSON에서 로드
    d = LegalTermDictionary()
    count = d.load_from_json("data/law_data/lawterms_full.json")

    # 텍스트에서 법률 용어 탐지
    terms = d.find_terms_in_text("손해배상청구권의 소멸시효")
    # → ["손해배상", "손해배상청구", "손해배상청구권", "소멸시효"]
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# 한글 전용 판별 패턴
_KOREAN_ONLY_PATTERN = re.compile(r"^[가-힣\s]+$")

# 기본 필터: 토크나이저에 로드할 용어 조건
DEFAULT_MIN_LENGTH = 2
DEFAULT_MAX_LENGTH = 10


class LegalTermDictionary:
    """
    법률 용어 메모리 사전

    frozenset 기반 O(1) 존재 확인.
    sliding window로 텍스트에서 법률 복합명사를 탐지.
    """

    def __init__(self) -> None:
        self._terms: frozenset[str] = frozenset()
        self._max_len: int = 0
        self._min_len: int = DEFAULT_MIN_LENGTH

    @property
    def is_loaded(self) -> bool:
        """사전 로드 여부"""
        return len(self._terms) > 0

    @property
    def term_count(self) -> int:
        """로드된 용어 수"""
        return len(self._terms)

    def contains(self, term: str) -> bool:
        """용어 존재 여부 확인 (O(1))"""
        return term in self._terms

    def load_from_json(
        self,
        json_path: str | Path,
        source_code: Optional[str] = None,
        min_length: int = DEFAULT_MIN_LENGTH,
        max_length: int = DEFAULT_MAX_LENGTH,
        korean_only: bool = True,
    ) -> int:
        """
        JSON 파일에서 법률 용어 로드

        Args:
            json_path: lawterms_full.json 경로
            source_code: 필터링할 사전유형 (None이면 전체)
            min_length: 최소 글자 수
            max_length: 최대 글자 수
            korean_only: 한글 전용 필터

        Returns:
            로드된 용어 수
        """
        path = Path(json_path)
        if not path.exists():
            logger.error("법률 용어 JSON 파일이 없습니다: %s", path)
            return 0

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            logger.error("JSON 형식 오류: 리스트가 아닙니다")
            return 0

        terms: set[str] = set()
        for item in data:
            term = item.get("법령용어명_한글", "").strip()
            if not term:
                continue

            # 사전유형 필터
            if source_code and item.get("법령용어코드명", "") != source_code:
                continue

            term_len = len(term)

            # 길이 필터
            if term_len < min_length or term_len > max_length:
                continue

            # 한글 전용 필터
            if korean_only and not _KOREAN_ONLY_PATTERN.match(term):
                continue

            terms.add(term)

        self._terms = frozenset(terms)
        self._max_len = max(len(t) for t in terms) if terms else 0
        self._min_len = min_length

        logger.info(
            "법률 용어 사전 로드 완료: %d개 (max_len=%d)",
            len(self._terms), self._max_len,
        )
        return len(self._terms)

    async def load_from_db(
        self,
        session: "AsyncSession",  # type: ignore[name-defined]  # noqa: F821
        min_length: int = DEFAULT_MIN_LENGTH,
        max_length: int = DEFAULT_MAX_LENGTH,
        korean_only: bool = True,
    ) -> int:
        """
        PostgreSQL에서 법률 용어 로드

        Args:
            session: SQLAlchemy AsyncSession
            min_length: 최소 글자 수
            max_length: 최대 글자 수
            korean_only: 한글 전용 필터

        Returns:
            로드된 용어 수
        """
        from sqlalchemy import select

        from app.models.legal_term import LegalTerm

        query = select(LegalTerm.term).where(
            LegalTerm.term_length >= min_length,
            LegalTerm.term_length <= max_length,
        )

        if korean_only:
            query = query.where(LegalTerm.is_korean_only.is_(True))

        result = await session.execute(query)
        rows = result.scalars().all()

        terms = frozenset(rows)
        self._terms = terms
        self._max_len = max(len(t) for t in terms) if terms else 0
        self._min_len = min_length

        logger.info(
            "법률 용어 사전 DB 로드 완료: %d개 (max_len=%d)",
            len(self._terms), self._max_len,
        )
        return len(self._terms)

    def load_from_terms(self, terms: set[str]) -> int:
        """
        용어 집합에서 직접 로드 (테스트용)

        Args:
            terms: 용어 문자열 집합

        Returns:
            로드된 용어 수
        """
        self._terms = frozenset(terms)
        self._max_len = max(len(t) for t in terms) if terms else 0
        self._min_len = min(len(t) for t in terms) if terms else DEFAULT_MIN_LENGTH
        return len(self._terms)

    def find_terms_in_text(self, text: str) -> list[str]:
        """
        텍스트에서 사전에 존재하는 법률 용어 탐지

        sliding window: 각 위치에서 max_len → min_len 순서로 부분문자열 검사.
        longest match 우선, 중복 제거.

        성능: O(n * max_term_length) ≈ O(n * 10)

        Args:
            text: 원본 텍스트

        Returns:
            발견된 법률 용어 리스트 (중복 제거, 발견 순서)
        """
        if not self._terms or not text:
            return []

        found: list[str] = []
        seen: set[str] = set()
        text_len = len(text)

        for i in range(text_len):
            # max_len → min_len 순서로 검사 (longest match 우선)
            end_max = min(i + self._max_len, text_len)
            for j in range(end_max, i + self._min_len - 1, -1):
                substr = text[i:j]
                if substr in self._terms and substr not in seen:
                    found.append(substr)
                    seen.add(substr)

        return found


# 모듈 레벨 싱글톤 (선택적 사용)
_global_dict: Optional[LegalTermDictionary] = None


def get_legal_term_dict() -> Optional[LegalTermDictionary]:
    """글로벌 법률 용어 사전 반환 (초기화되지 않았으면 None)"""
    return _global_dict


def init_legal_term_dict(json_path: str | Path) -> LegalTermDictionary:
    """글로벌 법률 용어 사전 초기화"""
    global _global_dict  # noqa: PLW0603
    _global_dict = LegalTermDictionary()
    _global_dict.load_from_json(json_path)
    return _global_dict


async def init_legal_term_dict_from_db(
    min_length: int = DEFAULT_MIN_LENGTH,
    max_length: int = DEFAULT_MAX_LENGTH,
    korean_only: bool = True,
) -> int:
    """
    PostgreSQL에서 글로벌 법률 용어 사전 초기화

    앱 시작(lifespan) 시 호출.

    Returns:
        로드된 용어 수
    """
    global _global_dict  # noqa: PLW0603
    from app.core.database import async_session_factory

    _global_dict = LegalTermDictionary()
    async with async_session_factory() as session:
        count = await _global_dict.load_from_db(
            session,
            min_length=min_length,
            max_length=max_length,
            korean_only=korean_only,
        )
    return count


def reset_legal_term_dict() -> None:
    """글로벌 법률 용어 사전 초기화 (테스트용)"""
    global _global_dict  # noqa: PLW0603
    _global_dict = None
