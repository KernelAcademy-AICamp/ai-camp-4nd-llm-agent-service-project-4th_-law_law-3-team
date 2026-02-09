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
import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from app.tools.vectorstore.legal_term_dict import LegalTermDictionary

logger = logging.getLogger(__name__)

# MeCab이 UNKNOWN으로 처리하여 오분석을 유발하는 문자 → 공백 정규화
# ㆍ (U+318D, 한글 가운뎃점): "시ㆍ도" → MeCab이 "시"+"ㆍ도"로 합쳐 오분석
# · (U+00B7, 가운데점): MeCab이 SC(특수문자)로 정상 처리하지만 통일성 위해 포함
_MIDDOT_PATTERN = re.compile(r"[ㆍ·]")

# MeCab 설치 여부 확인
_MECAB_AVAILABLE = False
try:
    import MeCab as _MeCab  # noqa: N811

    _MECAB_AVAILABLE = True
except ImportError:
    _MeCab = None  # type: ignore[assignment,unused-ignore]


# MeCab 시스템 사전 경로 후보
_MECAB_SYS_DICT_CANDIDATES = [
    "/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ko-dic",
    "/usr/local/lib/mecab/dic/mecab-ko-dic",
    "/usr/lib/mecab/dic/mecab-ko-dic",
    "/opt/homebrew/lib/mecab/dic/mecab-ko-dic",
]


def _find_mecab_sys_dict() -> Optional[str]:
    """MeCab 시스템 사전 경로 자동 탐지"""
    for candidate in _MECAB_SYS_DICT_CANDIDATES:
        if Path(candidate).is_dir():
            return candidate
    return None


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
        userdic_path: Optional[str] = None,
    ) -> None:
        self._tagger: Optional[object] = None
        self._legal_dict = legal_dict
        self._userdic_active: bool = False

        if _MECAB_AVAILABLE and _MeCab is not None:
            try:
                if userdic_path:
                    sys_dict = _find_mecab_sys_dict()
                    if sys_dict:
                        self._tagger = _MeCab.Tagger(
                            f"-d {sys_dict} -u {userdic_path}"
                        )
                        self._userdic_active = True
                        logger.info("MeCab userdic 활성화: %s", userdic_path)
                    else:
                        logger.warning("MeCab 시스템 사전 경로를 찾을 수 없어 기본 사전 사용")
                        self._tagger = _MeCab.Tagger()
                else:
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

        if self._userdic_active and self._legal_dict:
            # userdic 모드: MeCab이 복합명사를 정확히 인식 + 분해 토큰 추가
            return self._decompose_compounds(base_morphs)
        elif self._legal_dict and self._legal_dict.is_loaded:
            # 기존 모드: 사후 복원
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

        # 전처리: 가운뎃점(ㆍ/·) → 공백 (MeCab UNKNOWN 오분석 방지)
        text = _MIDDOT_PATTERN.sub(" ", text)

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
        text: str,  # noqa: ARG002
        base_morphs: list[str],
    ) -> list[str]:
        """
        MeCab 결과에 법률 복합명사 추가 토큰 삽입

        전략:
        1. 형태소 경계 기반으로 연속 형태소를 결합하여 법률 용어 매칭
        2. MeCab이 이미 분해한 형태소에 없는 복합명사만 추가
        3. base_morphs 뒤에 추가 토큰 append

        형태소 경계를 존중하므로 "매수인" 안에서 "수인"을 잘못
        매칭하는 오탐을 방지한다.

        Args:
            text: 원본 텍스트 (미사용, 인터페이스 호환용)
            base_morphs: MeCab 분해 결과

        Returns:
            base_morphs + 법률 복합명사 추가 토큰
        """
        assert self._legal_dict is not None  # noqa: S101

        legal_terms = self._legal_dict.find_terms_in_morphs(base_morphs)
        if not legal_terms:
            return base_morphs

        # MeCab 결과에 이미 있는 토큰은 제외
        existing = set(base_morphs)
        additional = [t for t in legal_terms if t not in existing]

        if not additional:
            return base_morphs

        return base_morphs + additional

    def _decompose_compounds(self, morphs: list[str]) -> list[str]:
        """
        userdic 인식된 복합어에 분해 토큰 추가 (FTS 부분검색용)

        userdic이 "법정이율"을 단일 NNG로 인식한 후,
        분해맵에서 ["법정", "이율"]을 찾아 추가 토큰으로 삽입.

        Args:
            morphs: MeCab(userdic) 형태소 리스트

        Returns:
            원본 morphs + 분해 토큰 (중복 제거)
        """
        if not self._legal_dict:
            return morphs

        result = list(morphs)
        additional: list[str] = []
        existing = set(morphs)

        for morph in morphs:
            sub_tokens = self._legal_dict.get_sub_tokens(morph)
            for st in sub_tokens:
                if st not in existing:
                    additional.append(st)
                    existing.add(st)

        return result + additional if additional else result

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
