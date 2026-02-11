"""
MeCab 기반 한국어 형태소 분석기 (FTS 사전 토크나이징용)

MeCab userdic으로 법률 복합명사를 단일 NNG로 직접 인식하고,
decomposition_map으로 FTS 부분검색용 서브 토큰을 추가한다.

Usage:
    from app.tools.vectorstore.mecab_tokenizer import MeCabTokenizer

    tokenizer = MeCabTokenizer()
    tokenized = tokenizer.tokenize("손해배상청구")
    # → "손해 배상 청구"

    # userdic + 분해맵
    tokenizer = MeCabTokenizer(
        userdic_path="data/mecab_userdic/legal_terms.dic",
        decomposition_map={"소멸시효": ["소멸", "시효"]},
    )
    tokenized = tokenizer.tokenize("소멸시효")
    # → "소멸시효 소멸 시효"
"""

import logging
import re
from pathlib import Path
from typing import Optional

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

    userdic 모드:
    1. MeCab userdic이 법률 복합명사를 단일 NNG로 직접 인식
    2. decomposition_map으로 복합어의 서브 토큰을 추가 (FTS 부분검색용)
       예: "소멸시효" → ["소멸시효"] + ["소멸", "시효"]

    MeCab 미설치 시 공백 분리 fallback 동작.
    """

    def __init__(
        self,
        userdic_path: Optional[str] = None,
        decomposition_map: Optional[dict[str, list[str]]] = None,
    ) -> None:
        self._tagger: Optional[object] = None
        self._decomposition_map: dict[str, list[str]] = decomposition_map or {}
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
        형태소 분석 결과를 리스트로 반환

        userdic이 활성화되고 decomposition_map이 있으면,
        MeCab이 인식한 복합어에 서브 토큰을 추가한다.

        Args:
            text: 분석할 한국어 텍스트

        Returns:
            형태소 리스트 (예: ["소멸시효", "소멸", "시효"])
            MeCab 미설치 시 공백 분리 결과 반환
        """
        base_morphs = self._mecab_morphs(text)

        if self._userdic_active and self._decomposition_map:
            return self._decompose_compounds(base_morphs)

        return base_morphs

    def _mecab_morphs(self, text: str) -> list[str]:
        """
        MeCab 형태소 분석 (Compound 분해 포함)

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

    def _decompose_compounds(self, morphs: list[str]) -> list[str]:
        """
        userdic 인식된 복합어에 분해 토큰 추가 (FTS 부분검색용)

        userdic이 "법정이율"을 단일 NNG로 인식한 후,
        decomposition_map에서 ["법정", "이율"]을 찾아 추가 토큰으로 삽입.

        Args:
            morphs: MeCab(userdic) 형태소 리스트

        Returns:
            원본 morphs + 분해 토큰 (중복 제거)
        """
        result = list(morphs)
        additional: list[str] = []
        existing = set(morphs)

        for morph in morphs:
            sub_tokens = self._decomposition_map.get(morph, [])
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
