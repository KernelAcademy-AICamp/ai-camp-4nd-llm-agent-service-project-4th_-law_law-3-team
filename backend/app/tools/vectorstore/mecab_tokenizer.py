"""
MeCab 기반 한국어 형태소 분석기 (LanceDB FTS 사전 토크나이징용)

LanceDB는 한국어 네이티브 FTS 토크나이저를 미지원(PR #2855 미머지)하므로,
MeCab으로 사전 토크나이징한 텍스트를 FTS 인덱싱하는 전략을 채택한다.

Usage:
    from app.tools.vectorstore.mecab_tokenizer import MeCabTokenizer

    tokenizer = MeCabTokenizer()
    tokenized = tokenizer.tokenize("손해배상청구")
    # → "손해 배상 청구"

    # FTS 검색 쿼리 사전 토크나이징
    query = tokenizer.tokenize_query("불법행위 손해배상")
    # → "불법 행위 손해 배상"
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# MeCab 설치 여부 확인
_MECAB_AVAILABLE = False
try:
    import MeCab as _MeCab  # noqa: N811

    _MECAB_AVAILABLE = True
except ImportError:
    _MeCab = None  # type: ignore[assignment]


def is_mecab_available() -> bool:
    """MeCab 설치 여부 확인"""
    return _MECAB_AVAILABLE


class MeCabTokenizer:
    """
    MeCab 한국어 형태소 분석기

    사전 토크나이징 전략:
    1. content → MeCab 형태소 분석 → 공백 구분 토큰 문자열
    2. content_tokenized 컬럼에 저장
    3. LanceDB FTS 인덱스는 content_tokenized에 생성
    4. 검색 쿼리도 동일하게 토크나이징 후 FTS 검색

    MeCab 미설치 시 공백 분리 fallback 동작.
    """

    def __init__(self) -> None:
        self._tagger: Optional[object] = None
        if _MECAB_AVAILABLE and _MeCab is not None:
            self._tagger = _MeCab.Tagger()

    @property
    def is_available(self) -> bool:
        """MeCab 토크나이저 사용 가능 여부"""
        return self._tagger is not None

    def morphs(self, text: str) -> list[str]:
        """
        형태소 분석 결과를 리스트로 반환

        Args:
            text: 분석할 한국어 텍스트

        Returns:
            형태소 리스트 (예: ["손해", "배상", "청구"])
            MeCab 미설치 시 공백 분리 결과 반환
        """
        if not text or not text.strip():
            return []

        if self._tagger is None:
            logger.warning("MeCab 미설치: 공백 분리 fallback 사용")
            return text.strip().split()

        # MeCab 형태소 분석
        parsed: str = self._tagger.parse(text)  # type: ignore[union-attr]
        morphs_list: list[str] = []
        for line in parsed.strip().split("\n"):
            if line == "EOS" or line == "":
                continue
            token = line.split("\t")[0]
            if token.strip():
                morphs_list.append(token)

        return morphs_list

    def tokenize(self, text: str) -> str:
        """
        텍스트를 형태소 분석하여 공백 구분 문자열로 반환

        Args:
            text: 원본 텍스트

        Returns:
            공백 구분 형태소 문자열 (예: "손해 배상 청구")
        """
        return " ".join(self.morphs(text))

    def tokenize_query(self, query: str) -> str:
        """
        검색 쿼리를 형태소 분석 (tokenize의 별칭, 의미 구분용)

        Args:
            query: 검색 쿼리 문자열

        Returns:
            공백 구분 형태소 문자열
        """
        return self.tokenize(query)
