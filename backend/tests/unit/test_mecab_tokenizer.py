"""
MeCab 한국어 형태소 분석기 단위 테스트

대상:
- MeCabTokenizer: 초기화, morphs(), tokenize(), tokenize_query()
- is_mecab_available(): 설치 여부 확인
- Fallback: MeCab 미설치 시 공백 분리
- 법률 용어 사전 보강 (LegalTermDictionary 연동)
"""

import pytest

from app.tools.vectorstore.legal_term_dict import LegalTermDictionary
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
        # MeCab-ko-dic은 "손해배상"을 복합명사로 인식하여 한 토큰으로 처리
        assert "청구" in result
        assert len(result) >= 2

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


# ============================================================================
# 법률 용어 사전 보강 테스트
# ============================================================================


class TestLegalTermAugmentation:
    """법률 용어 사전 보강 동작 테스트 (MeCab 설치 불필요)"""

    @pytest.fixture
    def legal_dict(self) -> LegalTermDictionary:
        """테스트용 법률 용어 사전"""
        d = LegalTermDictionary()
        d.load_from_terms({
            "손해배상",
            "손해배상청구",
            "소멸시효",
            "불법행위",
        })
        return d

    def test_no_dict_returns_base_morphs(self) -> None:
        """사전 없으면 기존 동작 (하위 호환)"""
        tokenizer = MeCabTokenizer(legal_dict=None)
        tokenizer._tagger = None  # fallback 모드
        result = tokenizer.morphs("손해 배상 청구")
        assert result == ["손해", "배상", "청구"]

    def test_empty_dict_returns_base_morphs(self) -> None:
        """빈 사전이면 기존 동작"""
        d = LegalTermDictionary()  # 로드 안 함
        tokenizer = MeCabTokenizer(legal_dict=d)
        tokenizer._tagger = None
        result = tokenizer.morphs("손해 배상 청구")
        assert result == ["손해", "배상", "청구"]

    def test_augmentation_adds_legal_terms(
        self, legal_dict: LegalTermDictionary,
    ) -> None:
        """법률 용어 보강: 연속 형태소 결합으로 복합명사 추가"""
        tokenizer = MeCabTokenizer(legal_dict=legal_dict)
        tokenizer._tagger = None  # fallback으로 공백 분리

        # fallback에서는 형태소 경계가 공백이므로
        # "손해배상청구"는 하나의 토큰 → 내부 substring 매칭 안 함
        result = tokenizer.morphs("손해배상청구")
        assert "손해배상청구" in result

        # 공백으로 분리된 토큰 결합은 가능
        result2 = tokenizer.morphs("손해 배상 청구")
        assert "손해" in result2
        assert "배상" in result2
        # "손해"+"배상" 결합 → "손해배상" (사전에 있으면 추가)
        if legal_dict.contains("손해배상"):
            assert "손해배상" in result2

    def test_augmentation_no_duplicates(
        self, legal_dict: LegalTermDictionary,
    ) -> None:
        """사전 보강 시 기존 토큰과 중복 안 됨"""
        tokenizer = MeCabTokenizer(legal_dict=legal_dict)
        tokenizer._tagger = None

        result = tokenizer.morphs("불법행위")
        # base: ["불법행위"]
        # 사전에 "불법행위" 있지만 base에 이미 존재 → 추가 안 함
        assert result.count("불법행위") == 1

    def test_augmentation_tokenize_output(
        self, legal_dict: LegalTermDictionary,
    ) -> None:
        """tokenize()가 보강된 결과를 공백 구분 문자열로 반환"""
        tokenizer = MeCabTokenizer(legal_dict=legal_dict)
        tokenizer._tagger = None

        # fallback: "손해배상청구의 소멸시효" → ["손해배상청구의", "소멸시효"]
        result = tokenizer.tokenize("손해배상청구의 소멸시효")
        assert isinstance(result, str)
        tokens = result.split()
        # fallback에서는 공백 기준 토큰만 존재
        assert "손해배상청구의" in tokens
        assert "소멸시효" in tokens

    @pytest.mark.requires_mecab
    def test_augmentation_with_real_mecab(
        self, legal_dict: LegalTermDictionary,
    ) -> None:
        """MeCab + 법률 용어 사전 통합"""
        tokenizer = MeCabTokenizer(legal_dict=legal_dict)
        if not tokenizer.is_available:
            pytest.skip("MeCab이 설치되지 않았습니다")

        result = tokenizer.morphs("손해배상청구권의 소멸시효")
        # MeCab이 "손해", "배상", "청구", "권", "의", "소멸", "시효" 등으로 분해
        # 사전 보강으로 "손해배상", "손해배상청구", "소멸시효" 등 추가
        assert "손해" in result or "손해배상" in result
        # 법률 복합명사가 추가되었는지 확인
        assert "손해배상" in result


class TestReverseExtractionTerms:
    """한영사전 역추출 용어 보강 테스트"""

    @pytest.fixture
    def reverse_dict(self) -> LegalTermDictionary:
        """역추출 핵심 용어를 포함한 사전"""
        d = LegalTermDictionary()
        d.load_from_terms({
            "손해배상",
            "불법행위",
            "채무불이행",
            "소멸시효",
            "부당이득",
        })
        return d

    def test_reverse_terms_in_augmentation(
        self, reverse_dict: LegalTermDictionary,
    ) -> None:
        """역추출 핵심 용어가 형태소 보강에 활용됨"""
        tokenizer = MeCabTokenizer(legal_dict=reverse_dict)
        tokenizer._tagger = None  # fallback 모드

        # fallback: 공백 기준 분리 → 결합 매칭
        result = tokenizer.morphs("채무 불이행 의")
        assert "채무불이행" in result

    @pytest.mark.requires_mecab
    def test_reverse_terms_with_real_mecab(
        self, reverse_dict: LegalTermDictionary,
    ) -> None:
        """MeCab + 역추출 용어 통합"""
        tokenizer = MeCabTokenizer(legal_dict=reverse_dict)
        if not tokenizer.is_available:
            pytest.skip("MeCab이 설치되지 않았습니다")

        result = tokenizer.morphs("불법행위로 인한 손해배상")
        assert "불법행위" in result
        assert "손해배상" in result


# ============================================================================
# MeCab userdic 모드 테스트
# ============================================================================


class TestMeCabUserdic:
    """userdic 활성화 시 _decompose_compounds 동작 테스트"""

    @pytest.fixture
    def decomp_dict(self) -> LegalTermDictionary:
        """분해맵이 로드된 법률 용어 사전"""
        import json
        import tempfile
        from pathlib import Path

        d = LegalTermDictionary()
        d.load_from_terms({
            "소멸시효",
            "손해배상",
            "법정이율",
            "소멸",
            "시효",
            "손해",
            "배상",
        })

        decomp_data = {
            "소멸시효": ["소멸", "시효"],
            "손해배상": ["손해", "배상"],
            "법정이율": ["법정", "이율"],
        }

        # 임시 파일에 분해맵 저장
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8",
        )
        json.dump(decomp_data, tmp, ensure_ascii=False)
        tmp.close()
        d.load_decomposition_map(Path(tmp.name))
        return d

    def test_decompose_compounds_adds_sub_tokens(
        self, decomp_dict: LegalTermDictionary,
    ) -> None:
        """_decompose_compounds가 분해 토큰을 추가"""
        tokenizer = MeCabTokenizer(legal_dict=decomp_dict)
        tokenizer._tagger = None  # fallback 모드

        # userdic_active를 수동 설정하여 _decompose_compounds 경로 테스트
        tokenizer._userdic_active = True

        # fallback: "소멸시효" → ["소멸시효"] (단일 토큰)
        result = tokenizer.morphs("소멸시효")
        # _decompose_compounds가 분해맵에서 ["소멸", "시효"]를 추가
        assert "소멸시효" in result
        assert "소멸" in result
        assert "시효" in result

    def test_decompose_no_duplicates(
        self, decomp_dict: LegalTermDictionary,
    ) -> None:
        """이미 존재하는 토큰은 추가하지 않음"""
        tokenizer = MeCabTokenizer(legal_dict=decomp_dict)
        tokenizer._tagger = None
        tokenizer._userdic_active = True

        # fallback: "소멸 시효" → ["소멸", "시효"]
        # _decompose_compounds에서 "소멸", "시효"가 이미 morphs에 있으므로 추가 안 함
        result = tokenizer.morphs("소멸 시효")
        assert result.count("소멸") == 1
        assert result.count("시효") == 1

    def test_decompose_no_map_returns_base(self) -> None:
        """분해맵 없으면 원본 형태소 그대로 반환"""
        d = LegalTermDictionary()
        d.load_from_terms({"소멸시효"})
        # 분해맵 미로드
        tokenizer = MeCabTokenizer(legal_dict=d)
        tokenizer._tagger = None
        tokenizer._userdic_active = True

        result = tokenizer.morphs("소멸시효")
        # get_sub_tokens("소멸시효") → [] (분해맵 없음)
        assert result == ["소멸시효"]

    def test_userdic_active_false_uses_augment(self) -> None:
        """userdic_active=False이면 기존 augment 방식 사용"""
        d = LegalTermDictionary()
        d.load_from_terms({"손해배상"})

        tokenizer = MeCabTokenizer(legal_dict=d)
        tokenizer._tagger = None
        tokenizer._userdic_active = False  # 기존 방식

        result = tokenizer.morphs("손해 배상 청구")
        # 기존 방식: find_terms_in_morphs로 "손해배상" 추가
        assert "손해배상" in result

    def test_tokenize_with_userdic_mode(
        self, decomp_dict: LegalTermDictionary,
    ) -> None:
        """tokenize()도 userdic 모드에서 분해 토큰 포함"""
        tokenizer = MeCabTokenizer(legal_dict=decomp_dict)
        tokenizer._tagger = None
        tokenizer._userdic_active = True

        result = tokenizer.tokenize("법정이율")
        tokens = result.split()
        assert "법정이율" in tokens
        assert "법정" in tokens
        assert "이율" in tokens
