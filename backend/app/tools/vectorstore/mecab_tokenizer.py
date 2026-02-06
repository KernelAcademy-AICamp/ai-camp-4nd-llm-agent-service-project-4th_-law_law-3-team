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
    _MeCab = None  # type: ignore[assignment,unused-ignore]


def is_mecab_available() -> bool:
    """MeCab 설치 여부 확인 (Python 패키지 + 시스템 라이브러리 모두 필요)"""
    if not _MECAB_AVAILABLE:
        return False
    # Python 패키지만 설치된 경우 (시스템 라이브러리 미설치) 체크
    try:
        _MeCab.Tagger()  # type: ignore[union-attr,unused-ignore]
        return True
    except (RuntimeError, AttributeError):
        return False


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
            try:
                self._tagger = _MeCab.Tagger()
            except RuntimeError:
                logger.warning("MeCab Python 패키지는 설치되었으나 시스템 라이브러리 미설치")
                self._tagger = None

    @property
    def is_available(self) -> bool:
        """MeCab 토크나이저 사용 가능 여부"""
        return self._tagger is not None

    def morphs(self, text: str) -> list[str]:
        """
        형태소 분석 결과를 리스트로 반환 (복합명사 분해 포함)

        mecab-ko-dic은 문맥에 따라 복합명사를 하나의 NNP로 묶는다:
          "손해배상 청구" → ["손해배상", "청구"] (Compound 타입)
        이때 features[4]=="Compound"이면 features[7]에 분해 정보가 있다:
          "손해/NNG/*+배상/NNG/*" → ["손해", "배상"]
        FTS 검색에서 부분어 매칭을 위해 복합어를 구성 형태소로 분해한다.

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
        parsed: str = self._tagger.parse(text)  # type: ignore[union-attr,attr-defined,unused-ignore]
        morphs_list: list[str] = []
        for line in parsed.strip().split("\n"):
            if line == "EOS" or line == "":
                continue
            parts = line.split("\t")
            surface = parts[0].strip()
            if not surface:
                continue

            # 피처 문자열 분석하여 Compound 분해
            if len(parts) > 1:
                features = parts[1].split(",")
                morph_type = features[4] if len(features) > 4 else "*"
                if morph_type == "Compound" and len(features) > 7:
                    decomposed = self._decompose_compound(features[7])
                    if decomposed:
                        morphs_list.extend(decomposed)
                        continue

            morphs_list.append(surface)

        return morphs_list

    @staticmethod
    def _decompose_compound(decomp_str: str) -> list[str]:
        """
        MeCab Compound 분해 문자열에서 구성 형태소 추출

        Args:
            decomp_str: "손해/NNG/*+배상/NNG/*" 형식 문자열

        Returns:
            구성 형태소 리스트 (예: ["손해", "배상"])
            파싱 실패 시 빈 리스트 반환
        """
        if not decomp_str or decomp_str == "*":
            return []
        try:
            components = decomp_str.split("+")
            result: list[str] = []
            for comp in components:
                morph = comp.split("/")[0]
                if morph.strip():
                    result.append(morph)
            return result
        except (IndexError, ValueError):
            return []

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
