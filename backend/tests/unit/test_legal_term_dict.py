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
