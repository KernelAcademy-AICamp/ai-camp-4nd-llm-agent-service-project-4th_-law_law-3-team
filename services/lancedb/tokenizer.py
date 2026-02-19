"""
MeCab 기반 한국어 형태소 분석기 (마이크로서비스 독립 버전)

backend/app/tools/vectorstore/mecab_tokenizer.py 로직을 추출.
app 패키지 의존 없이 독립 동작.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# 한글 가운뎃점 → 공백 정규화 (MeCab UNKNOWN 오분석 방지)
_MIDDOT_PATTERN = re.compile(r"[ㆍ·]")

# 한글 전용 판별
_KOREAN_ONLY_PATTERN = re.compile(r"^[가-힣]+$")

# MeCab 설치 여부
_MECAB_AVAILABLE = False
try:
    import MeCab as _MeCab  # noqa: N811

    _MECAB_AVAILABLE = True
except ImportError:
    _MeCab = None  # type: ignore[assignment,unused-ignore]


class LegalTermDictionary:
    """법률 용어 메모리 사전 (JSON 기반, DB 의존 없음)"""

    def __init__(self) -> None:
        self._terms: frozenset[str] = frozenset()
        self._max_len: int = 0
        self._min_len: int = 2
        self._decomposition_map: dict[str, list[str]] = {}

    @property
    def is_loaded(self) -> bool:
        return len(self._terms) > 0

    @property
    def term_count(self) -> int:
        return len(self._terms)

    def load_from_json(
        self,
        json_path: str | Path,
        min_length: int = 2,
        max_length: int = 15,
        korean_only: bool = True,
    ) -> int:
        """JSON 파일에서 법률 용어 로드"""
        path = Path(json_path)
        if not path.exists():
            logger.warning("법률 용어 JSON 없음: %s", path)
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
            term_len = len(term)
            if term_len < min_length or term_len > max_length:
                continue
            if korean_only and not _KOREAN_ONLY_PATTERN.match(term):
                continue
            terms.add(term)

        self._terms = frozenset(terms)
        self._max_len = max(len(t) for t in terms) if terms else 0
        self._min_len = min_length
        logger.info("법률 용어 사전 로드: %d개", len(self._terms))
        return len(self._terms)

    def load_from_terms(self, terms: set[str]) -> int:
        """용어 집합에서 직접 로드"""
        self._terms = frozenset(terms)
        self._max_len = max(len(t) for t in terms) if terms else 0
        self._min_len = min(len(t) for t in terms) if terms else 2
        return len(self._terms)

    def load_decomposition_map(self, path: str | Path) -> int:
        """decomposition_map.json 로드"""
        p = Path(path)
        if not p.exists():
            logger.warning("분해맵 파일 없음: %s", p)
            return 0

        with open(p, encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return 0

        self._decomposition_map = {
            k: v for k, v in data.items() if isinstance(v, list)
        }
        logger.info("분해맵 로드: %d개", len(self._decomposition_map))
        return len(self._decomposition_map)

    def find_terms_in_morphs(
        self,
        morphs: list[str],
        min_combine_len: int = 3,
    ) -> list[str]:
        """형태소 경계 기반 법률 용어 탐지"""
        if not self._terms or not morphs:
            return []

        found: list[str] = []
        seen: set[str] = set()
        n = len(morphs)

        for i in range(n):
            combined = ""
            for j in range(i, n):
                combined += morphs[j]
                if len(combined) > self._max_len:
                    break
                if (
                    len(combined) >= min_combine_len
                    and combined in self._terms
                    and combined not in seen
                ):
                    found.append(combined)
                    seen.add(combined)

        return found

    def get_sub_tokens(self, term: str) -> list[str]:
        """복합어의 분해 토큰 반환"""
        return self._decomposition_map.get(term, [])

    def contains(self, term: str) -> bool:
        return term in self._terms


class MeCabTokenizer:
    """MeCab 한국어 형태소 분석기 (독립 버전)"""

    def __init__(
        self,
        legal_dict: Optional[LegalTermDictionary] = None,
    ) -> None:
        self._tagger: Optional[object] = None
        self._legal_dict = legal_dict

        if _MECAB_AVAILABLE and _MeCab is not None:
            try:
                self._tagger = _MeCab.Tagger()
            except RuntimeError:
                logger.warning("MeCab 시스템 라이브러리 미설치")
                self._tagger = None

    @property
    def is_available(self) -> bool:
        return self._tagger is not None

    def morphs(self, text: str) -> list[str]:
        """형태소 분석 결과 리스트 (복합명사 분해 + 법률 용어 보강)"""
        base_morphs = self._mecab_morphs(text)

        if self._legal_dict and self._legal_dict.is_loaded:
            return self._augment_with_legal_terms(base_morphs)

        return base_morphs

    def _mecab_morphs(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        if self._tagger is None:
            return text.strip().split()

        text = _MIDDOT_PATTERN.sub(" ", text)
        parsed: str = self._tagger.parse(text)  # type: ignore[union-attr,attr-defined]
        morphs_list: list[str] = []
        for line in parsed.strip().split("\n"):
            if line == "EOS" or line == "":
                continue
            parts = line.split("\t")
            surface = parts[0].strip()
            if not surface:
                continue

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
        base_morphs: list[str],
    ) -> list[str]:
        assert self._legal_dict is not None  # noqa: S101

        legal_terms = self._legal_dict.find_terms_in_morphs(base_morphs)
        if not legal_terms:
            return base_morphs

        existing = set(base_morphs)
        additional = [t for t in legal_terms if t not in existing]

        if not additional:
            return base_morphs

        return base_morphs + additional

    @staticmethod
    def _decompose_compound(decomp_str: str) -> list[str]:
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
        return " ".join(self.morphs(text))

    def tokenize_query(self, query: str) -> str:
        return self.tokenize(query)
