"""
LangGraph 상태 변환 단위 테스트

ChatRequest <-> ChatState <-> ChatResponse 변환 검증
"""

from app.multi_agent.schemas.messages import ChatMessage, ChatRequest, ChatResponse
from app.multi_agent.state import request_to_state, state_to_response


class TestRequestToState:
    """ChatRequest -> ChatState 변환 테스트"""

    def test_basic_conversion(self) -> None:
        """기본 변환 검증"""
        request = ChatRequest(
            message="손해배상 판례 알려줘",
            history=[
                ChatMessage(role="user", content="안녕"),
                ChatMessage(role="assistant", content="안녕하세요"),
            ],
            session_data={"key": "value"},
        )
        state = request_to_state(request)

        assert state["message"] == "손해배상 판례 알려줘"
        assert state["history"] == [
            {"role": "user", "content": "안녕"},
            {"role": "assistant", "content": "안녕하세요"},
        ]
        assert state["session_data"] == {"key": "value"}

    def test_empty_history(self) -> None:
        """빈 히스토리 처리"""
        request = ChatRequest(message="테스트", history=[])
        state = request_to_state(request)

        assert state["history"] == []

    def test_agent_override(self) -> None:
        """agent 직접 지정 시 agent_override 설정"""
        request = ChatRequest(message="테스트", agent="small_claims")
        state = request_to_state(request)

        assert state["agent_override"] == "small_claims"

    def test_no_agent_override(self) -> None:
        """agent 미지정 시 None"""
        request = ChatRequest(message="테스트")
        state = request_to_state(request)

        assert state["agent_override"] is None

    def test_user_location(self) -> None:
        """사용자 위치 전달"""
        request = ChatRequest(
            message="변호사 찾아줘",
            user_location={"latitude": 37.5665, "longitude": 126.978},
        )
        state = request_to_state(request)

        assert state["user_location"] == {
            "latitude": 37.5665,
            "longitude": 126.978,
        }

    def test_default_values_initialized(self) -> None:
        """기본값 초기화 확인"""
        request = ChatRequest(message="테스트")
        state = request_to_state(request)

        assert state["selected_agent"] == ""
        assert state["search_focus"] == ""
        assert state["routing_confidence"] == 0.0
        assert state["routing_reason"] == ""
        assert state["response"] == ""
        assert state["sources"] == []
        assert state["actions"] == []
        assert state["output_session_data"] == {}
        assert state["agent_used"] == ""

    def test_session_data_with_thread_id(self) -> None:
        """thread_id가 있는 session_data 전달"""
        request = ChatRequest(
            message="중고거래 사기",
            session_data={"thread_id": "abc-123"},
        )
        state = request_to_state(request)

        assert state["session_data"]["thread_id"] == "abc-123"


class TestStateToResponse:
    """ChatState -> ChatResponse 변환 테스트"""

    def test_basic_conversion(self) -> None:
        """기본 변환 검증"""
        state = {
            "response": "법률 답변입니다.",
            "agent_used": "legal_answer",
            "sources": [{"title": "판례1"}],
            "actions": [{"label": "더보기"}],
            "output_session_data": {"key": "value"},
            "routing_confidence": 0.95,
        }
        response = state_to_response(state)

        assert response.response == "법률 답변입니다."
        assert response.agent_used == "legal_answer"
        assert response.sources == [{"title": "판례1"}]
        assert response.actions == [{"label": "더보기"}]
        assert response.session_data == {"key": "value"}
        assert response.confidence == 0.95

    def test_missing_fields_use_defaults(self) -> None:
        """누락된 필드는 기본값 사용"""
        state: dict = {}
        response = state_to_response(state)

        assert response.response == ""
        assert response.agent_used == "unknown"
        assert response.sources == []
        assert response.actions == []
        assert response.session_data == {}
        assert response.confidence == 1.0

    def test_response_is_chat_response(self) -> None:
        """반환 타입이 ChatResponse인지 확인"""
        state = {"response": "테스트", "agent_used": "general"}
        response = state_to_response(state)

        assert isinstance(response, ChatResponse)
