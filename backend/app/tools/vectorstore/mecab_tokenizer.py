"""
MeCab 기반 한국어 형태소 분석기 (LanceDB FTS 사전 토크나이징용)

LanceDB는 한국어 네이티브 FTS 토크나이저를 미지원(PR #2855 미머지)하므로,
MeCab으로 사전 토크나이징한 텍스트를 FTS 인덱싱하는 전략을 채택한다.

법률 용어 사전(LegalTermDictionary)이 주입되면, MeCab 분해 결과에
법률 복합명사를 추가 토큰으로 삽입하여 recall을 높인다.

Usage:
    from app.tools.vectorstore.mecab_tokenizer import MeCabTokenizer

    tokenizer = MeCabTokenizer()
    tokenized = tokenizer.tokenize("손해배상청구")
    # → "손해 배상 청구"

    # 법률 용어 사전 보강
    from app.tools.vectorstore.legal_term_dict import LegalTermDictionary
    d = LegalTermDictionary()
    d.load_from_json("data/law_data/lawterms_full.json")
    tokenizer = MeCabTokenizer(legal_dict=d)
    tokenized = tokenizer.tokenize("손해배상청구")
    # → "손해 배상 손해배상 청구 손해배상청구"
"""

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from app.tools.vectorstore.legal_term_dict import LegalTermDictionary

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

    법률 용어 사전 주입 시:
    - MeCab 분해 결과 + 법률 복합명사 추가 토큰
    - recall 향상 (부분어 + 복합어 모두 매칭)

    MeCab 미설치 시 공백 분리 fallback 동작.
    """

    def __init__(
        self,
        legal_dict: Optional["LegalTermDictionary"] = None,
    ) -> None:
        self._tagger: Optional[object] = None
        self._legal_dict = legal_dict

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
        형태소 분석 결과를 리스트로 반환 (복합명사 분해 + 법률 용어 보강)

        mecab-ko-dic은 문맥에 따라 복합명사를 하나의 NNP로 묶는다:
          "손해배상 청구" → ["손해배상", "청구"] (Compound 타입)
        이때 features[4]=="Compound"이면 features[7]에 분해 정보가 있다:
          "손해/NNG/*+배상/NNG/*" → ["손해", "배상"]
        FTS 검색에서 부분어 매칭을 위해 복합어를 구성 형태소로 분해한다.

        법률 용어 사전이 주입된 경우, 원본 텍스트에서 법률 복합명사를 탐지하여
        MeCab 분해 결과 뒤에 추가 토큰으로 삽입한다.

        Args:
            text: 분석할 한국어 텍스트

        Returns:
            형태소 리스트 (예: ["손해", "배상", "손해배상", "청구", "손해배상청구"])
            MeCab 미설치 시 공백 분리 결과 반환
        """
        base_morphs = self._mecab_morphs(text)

        if self._legal_dict and self._legal_dict.is_loaded:
            return self._augment_with_legal_terms(text, base_morphs)

        return base_morphs

    def _mecab_morphs(self, text: str) -> list[str]:
        """
        MeCab 형태소 분석 (기존 로직, Compound 분해 포함)

        Args:
            text: 분석할 한국어 텍스트

        Returns:
            형태소 리스트
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

    def _augment_with_legal_terms(
        self,
        text: str,
        base_morphs: list[str],
    ) -> list[str]:
        """
        MeCab 결과에 법률 복합명사 추가 토큰 삽입

        전략:
        1. 원본 텍스트에서 법률 용어 사전 매칭
        2. MeCab이 이미 분해한 형태소에 없는 복합명사만 추가
        3. base_morphs 뒤에 추가 토큰 append

        Args:
            text: 원본 텍스트
            base_morphs: MeCab 분해 결과

        Returns:
            base_morphs + 법률 복합명사 추가 토큰
        """
        assert self._legal_dict is not None  # noqa: S101

        legal_terms = self._legal_dict.find_terms_in_text(text)
        if not legal_terms:
            return base_morphs

        # MeCab 결과에 이미 있는 토큰은 제외
        existing = set(base_morphs)
        additional = [t for t in legal_terms if t not in existing]

        if not additional:
            return base_morphs

        return base_morphs + additional

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
