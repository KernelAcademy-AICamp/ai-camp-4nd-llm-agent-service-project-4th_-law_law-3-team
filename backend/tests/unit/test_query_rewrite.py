"""
대화형 쿼리 리라이팅 테스트

_is_followup_query(), rewrite_conversational_query() 함수 테스트
"""

import pytest

from app.services.rag.query_rewrite import (
    _is_followup_query,
    rewrite_conversational_query,
)


class TestIsFollowupQuery:
    """_is_followup_query() 키워드 기반 감지 테스트"""

    def test_short_message_without_legal_keyword(self) -> None:
        """짧은 메시지 + 법률 키워드 없음 → follow-up"""
        assert _is_followup_query("더 알려줘") is True
        assert _is_followup_query("계속해줘") is True
        assert _is_followup_query("그거 뭐야") is True

    def test_short_message_with_legal_keyword(self) -> None:
        """짧은 메시지 + 법률 키워드 있음 → 독립 질문"""
        assert _is_followup_query("손해배상") is False
        assert _is_followup_query("계약 해제") is False

    def test_long_standalone_query(self) -> None:
        """긴 독립 질문 → follow-up 아님"""
        assert _is_followup_query("임대차 보증금 반환 소송 절차 알려주세요") is False
        assert _is_followup_query("교통사고 손해배상 판례를 검색해주세요") is False

    def test_multiple_followup_keywords(self) -> None:
        """follow-up 키워드 2개 이상 → follow-up"""
        assert _is_followup_query("그거 더 자세히 알려줘") is True
        assert _is_followup_query("아까 말한 그것 설명해줘") is True

    def test_followup_patterns(self) -> None:
        """특정 패턴 매칭"""
        assert _is_followup_query("더 자세히 설명해줘") is True
        assert _is_followup_query("계속 알려줘") is True
        assert _is_followup_query("그것 좀 더 알려줘") is True

    def test_empty_and_whitespace(self) -> None:
        """빈 문자열, 공백"""
        assert _is_followup_query("") is True
        assert _is_followup_query("   ") is True

    def test_borderline_messages(self) -> None:
        """경계 케이스"""
        # 법률 키워드(LEGAL_KEYWORDS)가 포함된 짧은 메시지 → 독립 질문
        assert _is_followup_query("손해배상 소송") is False
        # "판례"는 LEGAL_KEYWORDS에 없으므로 짧은 메시지는 follow-up 처리
        assert _is_followup_query("판례 검색해줘") is True
        # 9자 이상 + follow-up 키워드 1개만 → 독립 질문
        assert _is_followup_query("손해배상 판결 사례 검색") is False


@pytest.mark.asyncio
class TestRewriteConversationalQuery:
    """rewrite_conversational_query() async 함수 테스트"""

    async def test_non_followup_returns_original(self) -> None:
        """follow-up이 아닌 독립 질문 → 원본 반환 (LLM 호출 없음)"""
        result = await rewrite_conversational_query(
            "임대차 보증금 반환 소송 절차",
            history=[{"role": "user", "content": "안녕하세요"}],
        )
        assert result == "임대차 보증금 반환 소송 절차"

    async def test_followup_without_history_returns_original(self) -> None:
        """follow-up + 히스토리 없음 → 원본 반환"""
        result = await rewrite_conversational_query("더 알려줘", history=None)
        assert result == "더 알려줘"

        result2 = await rewrite_conversational_query("더 알려줘", history=[])
        assert result2 == "더 알려줘"

    async def test_followup_with_history_calls_llm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """follow-up + 히스토리 있음 → LLM 호출하여 리라이팅"""

        class MockResponse:
            content = "손해배상 청구 요건과 절차"

        class MockModel:
            def invoke(self, messages: list) -> MockResponse:
                return MockResponse()

        monkeypatch.setattr(
            "app.services.rag.query_rewrite.get_chat_model",
            lambda **kwargs: MockModel(),
        )

        result = await rewrite_conversational_query(
            "더 자세히 알려줘",
            history=[
                {"role": "user", "content": "손해배상 청구 요건이 뭐야?"},
                {"role": "assistant", "content": "손해배상 청구 요건은..."},
            ],
        )
        assert result == "손해배상 청구 요건과 절차"

    async def test_llm_failure_falls_back_to_history(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """LLM 실패 → 히스토리에서 이전 사용자 메시지 폴백"""

        class MockModel:
            def invoke(self, messages: list) -> None:
                raise RuntimeError("LLM unavailable")

        monkeypatch.setattr(
            "app.services.rag.query_rewrite.get_chat_model",
            lambda **kwargs: MockModel(),
        )

        result = await rewrite_conversational_query(
            "더 알려줘",
            history=[
                {"role": "user", "content": "임대차 보증금 분쟁"},
                {"role": "assistant", "content": "임대차 보증금은..."},
            ],
        )
        assert result == "임대차 보증금 분쟁"

    async def test_llm_returns_empty_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """LLM이 빈 문자열 반환 → 폴백"""

        class MockResponse:
            content = "   "

        class MockModel:
            def invoke(self, messages: list) -> MockResponse:
                return MockResponse()

        monkeypatch.setattr(
            "app.services.rag.query_rewrite.get_chat_model",
            lambda **kwargs: MockModel(),
        )

        result = await rewrite_conversational_query(
            "더 알려줘",
            history=[
                {"role": "user", "content": "형사 사기죄 구성요건"},
                {"role": "assistant", "content": "사기죄는..."},
            ],
        )
        assert result == "형사 사기죄 구성요건"
