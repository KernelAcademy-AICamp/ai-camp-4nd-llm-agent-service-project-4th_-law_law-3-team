"""
MeCab 한국어 형태소 분석기 단위 테스트

대상:
- MeCabTokenizer: 초기화, morphs(), tokenize(), tokenize_query()
- is_mecab_available(): 설치 여부 확인
- Fallback: MeCab 미설치 시 공백 분리
"""

import pytest

from app.tools.vectorstore.mecab_tokenizer import MeCabTokenizer, is_mecab_available

# ============================================================================
# MeCab 설치 필요 테스트 (requires_mecab 마커)
# ============================================================================


@pytest.mark.requires_mecab
class TestMeCabWithInstallation:
    """MeCab 설치 환경에서의 테스트"""

    def test_mecab_initialization(self, mecab_tokenizer: MeCabTokenizer) -> None:
        """MeCab 토크나이저 초기화 성공"""
        assert mecab_tokenizer.is_available is True

    def test_mecab_morphs_basic(self, mecab_tokenizer: MeCabTokenizer) -> None:
        """기본 한국어 문장 형태소 분석"""
        result = mecab_tokenizer.morphs("손해배상")
        assert len(result) > 0
        assert "손해" in result

    def test_mecab_legal_terms_tokenization(
        self, mecab_tokenizer: MeCabTokenizer
    ) -> None:
        """법률 용어 토크나이징 정확성"""
        result = mecab_tokenizer.morphs("손해배상청구")
        assert "손해" in result
        assert "배상" in result
        assert "청구" in result

    def test_mecab_article_reference(
        self, mecab_tokenizer: MeCabTokenizer
    ) -> None:
        """조문 참조 토크나이징"""
        result = mecab_tokenizer.morphs("민법 제750조")
        assert "민법" in result
        assert len(result) > 1

    def test_mecab_case_number(self, mecab_tokenizer: MeCabTokenizer) -> None:
        """사건번호 토크나이징 (에러 없이 처리)"""
        result = mecab_tokenizer.morphs("2023다12345")
        assert len(result) > 0

    def test_pretokenize_content_for_fts(
        self, mecab_tokenizer: MeCabTokenizer
    ) -> None:
        """원본 content → 토크나이징된 문자열 변환"""
        result = mecab_tokenizer.tokenize("불법행위로 인한 손해배상")
        assert isinstance(result, str)
        assert " " in result  # 공백 구분

    def test_pretokenize_preserves_searchability(
        self, mecab_tokenizer: MeCabTokenizer
    ) -> None:
        """토크나이징 결과에 핵심 형태소가 포함되는지 검증"""
        result = mecab_tokenizer.tokenize("불법행위로 인한 손해배상청구")
        assert "손해" in result
        assert "배상" in result

    def test_pretokenize_query(self, mecab_tokenizer: MeCabTokenizer) -> None:
        """tokenize_query()가 tokenize()와 동일 결과"""
        text = "손해배상"
        assert mecab_tokenizer.tokenize_query(text) == mecab_tokenizer.tokenize(text)

    def test_mecab_mixed_korean_english(
        self, mecab_tokenizer: MeCabTokenizer
    ) -> None:
        """한영 혼합 텍스트 처리"""
        result = mecab_tokenizer.tokenize("OWASP 보안 취약점")
        assert "OWASP" in result


# ============================================================================
# MeCab 미설치 환경에서도 동작하는 테스트 (마커 없음)
# ============================================================================


class TestMeCabFallback:
    """MeCab 미설치 시 fallback 동작 테스트"""

    def test_mecab_not_installed_fallback(self) -> None:
        """MeCab 미설치 시 공백 분리 fallback"""
        tokenizer = MeCabTokenizer()
        # _tagger를 None으로 강제 설정하여 fallback 테스트
        tokenizer._tagger = None

        result = tokenizer.morphs("손해 배상 청구")
        assert result == ["손해", "배상", "청구"]

        result_str = tokenizer.tokenize("손해 배상 청구")
        assert result_str == "손해 배상 청구"

    def test_mecab_empty_string(self) -> None:
        """빈 문자열 입력 시 에러 없이 처리"""
        tokenizer = MeCabTokenizer()
        assert tokenizer.morphs("") == []
        assert tokenizer.morphs("   ") == []
        assert tokenizer.tokenize("") == ""

    def test_is_mecab_available_returns_bool(self) -> None:
        """is_mecab_available()가 bool을 반환하는지 확인"""
        result = is_mecab_available()
        assert isinstance(result, bool)

    def test_mecab_tokenizer_is_available_property(self) -> None:
        """MeCabTokenizer.is_available 프로퍼티 동작 확인"""
        tokenizer = MeCabTokenizer()
        assert isinstance(tokenizer.is_available, bool)
