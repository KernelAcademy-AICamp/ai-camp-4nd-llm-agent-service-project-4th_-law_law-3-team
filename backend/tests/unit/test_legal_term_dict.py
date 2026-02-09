"""
법률 용어 메모리 사전 단위 테스트

대상:
- LegalTermDictionary: load_from_terms, find_terms_in_text, contains
- 빈 사전, 경계 케이스
"""

import json
from pathlib import Path

import pytest

from app.tools.vectorstore.legal_term_dict import (
    LegalTermDictionary,
    get_legal_term_dict,
    init_legal_term_dict,
    reset_legal_term_dict,
)

# ============================================================================
# 기본 동작 테스트
# ============================================================================


class TestLegalTermDictionaryBasic:
    """LegalTermDictionary 기본 동작"""

    def test_empty_dict(self) -> None:
        """초기 상태는 비어 있음"""
        d = LegalTermDictionary()
        assert d.is_loaded is False
        assert d.term_count == 0
        assert d.contains("손해배상") is False

    def test_load_from_terms(self) -> None:
        """용어 집합에서 직접 로드"""
        d = LegalTermDictionary()
        count = d.load_from_terms({"손해배상", "소멸시효", "불법행위"})
        assert count == 3
        assert d.is_loaded is True
        assert d.term_count == 3

    def test_contains(self) -> None:
        """용어 존재 확인"""
        d = LegalTermDictionary()
        d.load_from_terms({"손해배상", "소멸시효"})
        assert d.contains("손해배상") is True
        assert d.contains("소멸시효") is True
        assert d.contains("없는용어") is False

    def test_empty_terms_set(self) -> None:
        """빈 집합 로드 시 is_loaded는 False"""
        d = LegalTermDictionary()
        d.load_from_terms(set())
        assert d.is_loaded is False
        assert d.term_count == 0


# ============================================================================
# find_terms_in_text 테스트
# ============================================================================


class TestFindTermsInText:
    """텍스트에서 법률 용어 탐지"""

    @pytest.fixture
    def legal_dict(self) -> LegalTermDictionary:
        """테스트용 법률 용어 사전"""
        d = LegalTermDictionary()
        d.load_from_terms({
            "손해배상",
            "손해배상청구",
            "손해배상청구권",
            "소멸시효",
            "불법행위",
            "위자료",
        })
        return d

    def test_find_simple(self, legal_dict: LegalTermDictionary) -> None:
        """단순 용어 탐지"""
        result = legal_dict.find_terms_in_text("불법행위로 인한 소멸시효")
        assert "불법행위" in result
        assert "소멸시효" in result

    def test_find_overlapping(self, legal_dict: LegalTermDictionary) -> None:
        """겹치는 용어 (longest match 포함)"""
        result = legal_dict.find_terms_in_text("손해배상청구권의 소멸시효")
        # 긴 것 먼저 + 짧은 것도 발견
        assert "손해배상청구권" in result
        assert "손해배상청구" in result
        assert "손해배상" in result
        assert "소멸시효" in result

    def test_find_no_match(self, legal_dict: LegalTermDictionary) -> None:
        """매칭 없는 텍스트"""
        result = legal_dict.find_terms_in_text("오늘 날씨가 좋습니다")
        assert result == []

    def test_find_empty_text(self, legal_dict: LegalTermDictionary) -> None:
        """빈 텍스트"""
        assert legal_dict.find_terms_in_text("") == []

    def test_find_empty_dict(self) -> None:
        """빈 사전에서 검색"""
        d = LegalTermDictionary()
        assert d.find_terms_in_text("손해배상청구") == []

    def test_no_duplicates(self, legal_dict: LegalTermDictionary) -> None:
        """같은 용어 반복 시 중복 제거"""
        result = legal_dict.find_terms_in_text("위자료 청구와 위자료 산정")
        assert result.count("위자료") == 1

    def test_single_char_not_matched(self) -> None:
        """1글자 용어는 기본 min_length에 의해 무시"""
        d = LegalTermDictionary()
        d.load_from_terms({"법", "손해배상"})
        # "법"은 1글자이므로 min_len=1으로 설정됨
        result = d.find_terms_in_text("민법에서 손해배상")
        assert "손해배상" in result
        # "법"은 로드되었으므로 발견될 수 있음
        assert "법" in result


# ============================================================================
# find_terms_in_morphs 테스트 (형태소 경계 기반)
# ============================================================================


class TestFindTermsInMorphs:
    """형태소 경계 기반 법률 용어 탐지"""

    @pytest.fixture
    def legal_dict(self) -> LegalTermDictionary:
        d = LegalTermDictionary()
        d.load_from_terms({
            "손해배상",
            "손해배상청구",
            "배상청구",
            "소멸시효",
            "불법행위",
            "상의",     # 오탐 유발 가능 용어
            "수인",     # 오탐 유발 가능 용어
            "고도",     # 오탐 유발 가능 용어
            "중도금",
            "지급의무",
        })
        return d

    def test_combines_consecutive_morphs(
        self, legal_dict: LegalTermDictionary,
    ) -> None:
        """연속 형태소 결합으로 복합명사 탐지"""
        morphs = ["손해", "배상", "청구"]
        result = legal_dict.find_terms_in_morphs(morphs)
        assert "손해배상" in result
        assert "손해배상청구" in result
        assert "배상청구" in result

    def test_no_false_positive_substring(
        self, legal_dict: LegalTermDictionary,
    ) -> None:
        """형태소 경계를 존중하여 substring 오탐 방지"""
        # MeCab: "매수인" → 단일 형태소, "수인"은 추출 불가
        morphs = ["매수", "인", "의", "중도", "금"]
        result = legal_dict.find_terms_in_morphs(morphs)
        assert "수인" not in result
        assert "중도금" in result

    def test_no_short_particle_combination(
        self, legal_dict: LegalTermDictionary,
    ) -> None:
        """1글자 조사끼리 결합된 2글자 오탐 방지 (min_combine_len=3)"""
        # "관리"+"상"+"의" → "상의" 오탐 방지
        morphs = ["관리", "상", "의", "잘못"]
        result = legal_dict.find_terms_in_morphs(morphs)
        assert "상의" not in result

        # "하"+"고"+"도" → "고도" 오탐 방지
        morphs2 = ["중대", "하", "고", "도"]
        result2 = legal_dict.find_terms_in_morphs(morphs2)
        assert "고도" not in result2

    def test_empty_morphs(self, legal_dict: LegalTermDictionary) -> None:
        """빈 형태소 리스트"""
        assert legal_dict.find_terms_in_morphs([]) == []

    def test_empty_dict(self) -> None:
        """빈 사전에서 검색"""
        d = LegalTermDictionary()
        assert d.find_terms_in_morphs(["손해", "배상"]) == []

    def test_no_duplicates(self, legal_dict: LegalTermDictionary) -> None:
        """동일 용어 중복 방지"""
        morphs = ["손해", "배상", "과", "손해", "배상"]
        result = legal_dict.find_terms_in_morphs(morphs)
        assert result.count("손해배상") == 1

    def test_custom_min_combine_len(
        self, legal_dict: LegalTermDictionary,
    ) -> None:
        """min_combine_len 파라미터로 최소 길이 조절"""
        morphs = ["관리", "상", "의"]
        # min_combine_len=2면 "상의" 매칭됨
        result = legal_dict.find_terms_in_morphs(morphs, min_combine_len=2)
        assert "상의" in result
        # min_combine_len=3이면 "상의" 매칭 안 됨 (기본값)
        result2 = legal_dict.find_terms_in_morphs(morphs, min_combine_len=3)
        assert "상의" not in result2


# ============================================================================
# JSON 로드 테스트
# ============================================================================


class TestLoadFromJson:
    """JSON 파일에서 로드"""

    def test_load_valid_json(self, tmp_path: Path) -> None:
        """유효한 JSON 파일 로드"""
        data = [
            {
                "법령용어 일련번호": "1",
                "법령용어명_한글": "손해배상",
                "법령용어명_한자": "損害賠償",
                "법령용어코드": "011402",
                "법령용어코드명": "법령정의사전",
                "출처": "민법",
                "법령용어정의": "손해를 배상하는 것",
            },
            {
                "법령용어 일련번호": "2",
                "법령용어명_한글": "소멸시효",
                "법령용어명_한자": "消滅時效",
                "법령용어코드": "011402",
                "법령용어코드명": "법령정의사전",
                "출처": "민법",
                "법령용어정의": "시효 소멸",
            },
            {
                "법령용어 일련번호": "3",
                "법령용어명_한글": "ABC Corp",
                "법령용어명_한자": "",
                "법령용어코드": "011402",
                "법령용어코드명": "법령정의사전",
                "출처": "상법",
                "법령용어정의": "회사",
            },
        ]

        json_path = tmp_path / "lawterms.json"
        json_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        d = LegalTermDictionary()
        count = d.load_from_json(json_path)

        # korean_only 필터로 "ABC Corp"는 제외
        assert count == 2
        assert d.contains("손해배상") is True
        assert d.contains("소멸시효") is True
        assert d.contains("ABC Corp") is False

    def test_load_space_term_excluded(self, tmp_path: Path) -> None:
        """공백 포함 한글 용어는 korean_only 필터에서 제외"""
        data = [
            {"법령용어명_한글": "상속 승인", "법령용어코드명": "법령정의사전"},
            {"법령용어명_한글": "공동 상속인", "법령용어코드명": "법령정의사전"},
            {"법령용어명_한글": "손해배상", "법령용어코드명": "법령정의사전"},
        ]

        json_path = tmp_path / "lawterms.json"
        json_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        d = LegalTermDictionary()
        count = d.load_from_json(json_path)

        # 공백 포함 용어는 제외됨 (형태소 결합 시 매칭 불가하므로)
        assert count == 1
        assert d.contains("손해배상") is True
        assert d.contains("상속 승인") is False
        assert d.contains("공동 상속인") is False

    def test_load_with_source_filter(self, tmp_path: Path) -> None:
        """사전유형 필터링"""
        data = [
            {
                "법령용어 일련번호": "1",
                "법령용어명_한글": "손해배상",
                "법령용어코드명": "법령정의사전",
            },
            {
                "법령용어 일련번호": "2",
                "법령용어명_한글": "소멸시효",
                "법령용어코드명": "법령한영사전",
            },
        ]

        json_path = tmp_path / "lawterms.json"
        json_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        d = LegalTermDictionary()
        count = d.load_from_json(json_path, source_code="법령정의사전")
        assert count == 1
        assert d.contains("손해배상") is True
        assert d.contains("소멸시효") is False

    def test_load_nonexistent_file(self) -> None:
        """존재하지 않는 파일"""
        d = LegalTermDictionary()
        count = d.load_from_json("/nonexistent/path.json")
        assert count == 0
        assert d.is_loaded is False

    def test_load_length_filter(self, tmp_path: Path) -> None:
        """길이 필터"""
        data = [
            {"법령용어명_한글": "법", "법령용어코드명": "법령정의사전"},  # 1글자
            {"법령용어명_한글": "손해", "법령용어코드명": "법령정의사전"},  # 2글자
            {"법령용어명_한글": "이것은아주긴법률용어입니다열두글자", "법령용어코드명": "법령정의사전"},  # 17글자
        ]

        json_path = tmp_path / "lawterms.json"
        json_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        d = LegalTermDictionary()
        count = d.load_from_json(json_path, min_length=2, max_length=10)
        assert count == 1  # "손해"만
        assert d.contains("손해") is True


# ============================================================================
# 글로벌 사전 관리
# ============================================================================


class TestGlobalDict:
    """글로벌 싱글톤 사전"""

    def test_get_before_init(self) -> None:
        """초기화 전 get은 None"""
        reset_legal_term_dict()
        assert get_legal_term_dict() is None

    def test_init_and_get(self, tmp_path: Path) -> None:
        """초기화 후 get"""
        data = [
            {"법령용어명_한글": "손해배상", "법령용어코드명": "법령정의사전"},
        ]
        json_path = tmp_path / "lawterms.json"
        json_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        try:
            d = init_legal_term_dict(json_path)
            assert d.is_loaded is True
            assert get_legal_term_dict() is d
        finally:
            reset_legal_term_dict()

    def test_reset(self, tmp_path: Path) -> None:
        """리셋 후 None"""
        data = [
            {"법령용어명_한글": "손해배상", "법령용어코드명": "법령정의사전"},
        ]
        json_path = tmp_path / "lawterms.json"
        json_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        init_legal_term_dict(json_path)
        reset_legal_term_dict()
        assert get_legal_term_dict() is None


# ============================================================================
# 분해맵 (decomposition map) 테스트
# ============================================================================


class TestDecompositionMap:
    """분해맵 로드 및 조회 테스트"""

    @pytest.fixture
    def decomp_dict(self, tmp_path: Path) -> LegalTermDictionary:
        """분해맵이 로드된 법률 용어 사전"""
        d = LegalTermDictionary()
        d.load_from_terms({"소멸시효", "손해배상", "법정이율", "소멸", "시효"})

        decomp_data = {
            "소멸시효": ["소멸", "시효"],
            "손해배상": ["손해", "배상"],
            "법정이율": ["법정", "이율"],
        }
        decomp_path = tmp_path / "decomposition_map.json"
        decomp_path.write_text(
            json.dumps(decomp_data, ensure_ascii=False), encoding="utf-8",
        )
        d.load_decomposition_map(decomp_path)
        return d

    def test_has_decomposition_map_initial(self) -> None:
        """초기 상태는 분해맵 없음"""
        d = LegalTermDictionary()
        assert d.has_decomposition_map is False

    def test_load_decomposition_map(self, decomp_dict: LegalTermDictionary) -> None:
        """분해맵 로드 성공"""
        assert decomp_dict.has_decomposition_map is True

    def test_get_sub_tokens_exists(self, decomp_dict: LegalTermDictionary) -> None:
        """분해맵에 존재하는 용어"""
        result = decomp_dict.get_sub_tokens("소멸시효")
        assert result == ["소멸", "시효"]

    def test_get_sub_tokens_not_exists(
        self, decomp_dict: LegalTermDictionary,
    ) -> None:
        """분해맵에 없는 용어는 빈 리스트"""
        result = decomp_dict.get_sub_tokens("없는용어")
        assert result == []

    def test_load_nonexistent_decomp_file(self) -> None:
        """존재하지 않는 분해맵 파일"""
        d = LegalTermDictionary()
        count = d.load_decomposition_map("/nonexistent/path.json")
        assert count == 0
        assert d.has_decomposition_map is False

    def test_load_invalid_decomp_format(self, tmp_path: Path) -> None:
        """잘못된 형식의 분해맵 파일 (리스트)"""
        bad_path = tmp_path / "bad_decomp.json"
        bad_path.write_text("[]", encoding="utf-8")

        d = LegalTermDictionary()
        count = d.load_decomposition_map(bad_path)
        assert count == 0
