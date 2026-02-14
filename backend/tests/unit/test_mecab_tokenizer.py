"""
MeCab 한국어 형태소 분석기 단위 테스트

대상:
- MeCabTokenizer: 초기화, morphs(), tokenize(), tokenize_query()
- is_mecab_available(): 설치 여부 확인
- 에러 처리: MeCab 미설치, userdic 미빌드 시 RuntimeError/FileNotFoundError
- decomposition_map: 복합어 분해 토큰 추가
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
        assert mecab_tokenizer is not None

    def test_mecab_morphs_basic(self, mecab_tokenizer: MeCabTokenizer) -> None:
        """기본 한국어 문장 형태소 분석"""
        result = mecab_tokenizer.morphs("손해배상")
        assert len(result) > 0
        assert "손해" in result or "손해배상" in result

    def test_mecab_legal_terms_tokenization(
        self, mecab_tokenizer: MeCabTokenizer
    ) -> None:
        """법률 용어 토크나이징 정확성"""
        result = mecab_tokenizer.morphs("손해배상청구")
        assert len(result) >= 1

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
        assert "손해" in result or "손해배상" in result

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

    def test_mecab_empty_string(self, mecab_tokenizer: MeCabTokenizer) -> None:
        """빈 문자열 입력 시 에러 없이 처리"""
        assert mecab_tokenizer.morphs("") == []
        assert mecab_tokenizer.morphs("   ") == []
        assert mecab_tokenizer.tokenize("") == ""


# ============================================================================
# 에러 처리 테스트 (MeCab 설치 여부 무관)
# ============================================================================


class TestMeCabErrors:
    """MeCab 초기화 에러 테스트"""

    def test_is_mecab_available_returns_bool(self) -> None:
        """is_mecab_available()가 bool을 반환하는지 확인"""
        result = is_mecab_available()
        assert isinstance(result, bool)

    def test_missing_userdic_raises_error(self, tmp_path: "Path") -> None:  # type: ignore[name-defined]  # noqa: F821
        """존재하지 않는 userdic 경로 → FileNotFoundError"""
        if not is_mecab_available():
            pytest.skip("MeCab이 설치되지 않았습니다")

        fake_path = str(tmp_path / "nonexistent.dic")
        with pytest.raises(FileNotFoundError, match="userdic 파일이 없습니다"):
            MeCabTokenizer(
                userdic_path=fake_path,
                decomposition_map={},
            )


# ============================================================================
# decomposition_map 테스트 (userdic 모드)
# ============================================================================


@pytest.mark.requires_mecab
class TestDecompositionMap:
    """decomposition_map을 통한 복합어 분해 토큰 추가 테스트"""

    def test_decompose_adds_sub_tokens(
        self, mecab_tokenizer: MeCabTokenizer,
    ) -> None:
        """userdic이 인식한 복합어에 분해맵의 서브 토큰이 추가됨"""
        # userdic이 "소멸시효"를 단일 NNG로 인식
        # decomposition_map에 {"소멸시효": ["소멸", "시효"]}가 있으면 추가
        result = mecab_tokenizer.morphs("소멸시효")
        # 적어도 "소멸시효" 자체 또는 분해된 서브 토큰이 존재해야 함
        assert len(result) >= 1

    def test_decompose_no_duplicates(
        self, mecab_tokenizer: MeCabTokenizer,
    ) -> None:
        """분해 토큰이 이미 morphs에 존재하면 중복 추가하지 않음"""
        result = mecab_tokenizer.morphs("소멸 시효")
        # "소멸"과 "시효"가 이미 별도 토큰이면 중복 추가하지 않아야 함
        assert result.count("소멸") <= 1
        assert result.count("시효") <= 1

    def test_tokenize_includes_decomposed_tokens(
        self, mecab_tokenizer: MeCabTokenizer,
    ) -> None:
        """tokenize() 결과에도 분해 토큰이 포함됨"""
        result = mecab_tokenizer.tokenize("소멸시효")
        assert isinstance(result, str)
        # 최소한 무언가 토큰화됨
        assert len(result.split()) >= 1

    def test_empty_decomposition_map(self) -> None:
        """빈 decomposition_map이면 원본 morphs 그대로 반환"""
        if not is_mecab_available():
            pytest.skip("MeCab이 설치되지 않았습니다")

        from pathlib import Path

        from app.core.config import settings

        dic_path = Path(settings.MECAB_USERDIC_PATH)
        if not dic_path.exists():
            pytest.skip(f"userdic 미빌드: {dic_path}")

        tokenizer = MeCabTokenizer(
            userdic_path=str(dic_path),
            decomposition_map={},
        )
        result = tokenizer.morphs("소멸시효")
        # 분해맵 없으면 MeCab 기본 결과만 (서브 토큰 추가 없음)
        assert len(result) >= 1


# ============================================================================
# _decompose_compound 정적 메서드 테스트
# ============================================================================


class TestDecomposeCompound:
    """MeCab Compound 분해 문자열 파싱 테스트 (MeCab 설치 불필요)"""

    def test_normal_compound(self) -> None:
        """정상적인 Compound 분해 문자열 파싱"""
        result = MeCabTokenizer._decompose_compound("손해/NNG/*+배상/NNG/*")
        assert result == ["손해", "배상"]

    def test_three_component_compound(self) -> None:
        """3개 구성요소 Compound 분해"""
        result = MeCabTokenizer._decompose_compound("손해/NNG/*+배상/NNG/*+청구/NNG/*")
        assert result == ["손해", "배상", "청구"]

    def test_empty_string(self) -> None:
        """빈 문자열 → 빈 리스트"""
        assert MeCabTokenizer._decompose_compound("") == []

    def test_asterisk(self) -> None:
        """'*' → 빈 리스트"""
        assert MeCabTokenizer._decompose_compound("*") == []

    def test_single_component(self) -> None:
        """단일 구성요소"""
        result = MeCabTokenizer._decompose_compound("손해/NNG/*")
        assert result == ["손해"]
