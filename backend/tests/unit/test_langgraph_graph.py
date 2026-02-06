"""
LangGraph 그래프 빌드 및 라우팅 단위 테스트

그래프 컴파일, 라우팅 결정, 노드 매핑 검증
"""

from app.multi_agent.graph import build_graph, get_graph
from app.multi_agent.nodes import AGENT_NODE_MAP, router_node
from app.multi_agent.router import detect_search_type
from app.multi_agent.state import ChatState


class TestGraphBuild:
    """그래프 빌드 검증"""

    def test_build_graph_returns_state_graph(self) -> None:
        """build_graph가 StateGraph를 반환"""
        from langgraph.graph import StateGraph

        builder = build_graph()
        assert isinstance(builder, StateGraph)

    def test_graph_compiles_successfully(self) -> None:
        """그래프가 에러 없이 컴파일"""
        builder = build_graph()
        compiled = builder.compile()
        assert compiled is not None

    def test_graph_has_all_nodes(self) -> None:
        """8개 노드가 모두 등록되어 있는지 확인"""
        builder = build_graph()
        compiled = builder.compile()

        node_names = set(compiled.nodes.keys())
        expected_nodes = {
            "router_node",
            "legal_search_node",
            "lawyer_finder_node",
            "small_claims_subgraph",
            "storyboard_node",
            "lawyer_stats_node",
            "law_study_node",
            "simple_chat_node",
        }
        assert expected_nodes.issubset(node_names)

    def test_get_graph_singleton(self) -> None:
        """get_graph가 싱글톤 인스턴스 반환"""
        import app.multi_agent.graph as graph_module

        graph_module._compiled_graph = None

        graph1 = get_graph()
        graph2 = get_graph()
        assert graph1 is graph2

        # cleanup
        graph_module._compiled_graph = None


class TestAgentNodeMap:
    """에이전트 노드 매핑 검증"""

    def test_legal_search_mapped(self) -> None:
        assert AGENT_NODE_MAP["legal_search"] == "legal_search_node"

    def test_case_search_mapped_to_legal_search(self) -> None:
        assert AGENT_NODE_MAP["case_search"] == "legal_search_node"

    def test_law_search_mapped_to_legal_search(self) -> None:
        assert AGENT_NODE_MAP["law_search"] == "legal_search_node"

    def test_lawyer_finder_mapped(self) -> None:
        assert AGENT_NODE_MAP["lawyer_finder"] == "lawyer_finder_node"

    def test_small_claims_mapped(self) -> None:
        assert AGENT_NODE_MAP["small_claims"] == "small_claims_subgraph"

    def test_storyboard_mapped(self) -> None:
        assert AGENT_NODE_MAP["storyboard"] == "storyboard_node"

    def test_lawyer_stats_mapped(self) -> None:
        assert AGENT_NODE_MAP["lawyer_stats"] == "lawyer_stats_node"

    def test_law_study_mapped(self) -> None:
        assert AGENT_NODE_MAP["law_study"] == "law_study_node"

    def test_general_mapped_to_simple_chat(self) -> None:
        assert AGENT_NODE_MAP["general"] == "simple_chat_node"

    def test_backward_compat_legal_answer(self) -> None:
        """기존 legal_answer 키가 legal_search_node로 매핑 (하위호환)"""
        assert AGENT_NODE_MAP["legal_answer"] == "legal_search_node"


class TestDetectSearchType:
    """search_focus 분류 검증"""

    def test_precedent_by_default(self) -> None:
        assert detect_search_type("손해배상 관련 질문") == "precedent"

    def test_law_keyword_statute(self) -> None:
        assert detect_search_type("법령 조문 확인") == "law"

    def test_law_keyword_legislation(self) -> None:
        assert detect_search_type("민법 법률 내용") == "law"

    def test_law_keyword_article(self) -> None:
        assert detect_search_type("제750조 조문 해석") == "law"

    def test_law_keyword_enforcement_decree(self) -> None:
        assert detect_search_type("시행령 개정 사항") == "law"

    def test_law_keyword_enforcement_rule(self) -> None:
        assert detect_search_type("시행규칙 확인") == "law"

    def test_precedent_without_law_keywords(self) -> None:
        assert detect_search_type("판례 검색") == "precedent"


class TestRouterNode:
    """라우터 노드 라우팅 결정 검증"""

    def _make_state(
        self,
        message: str = "테스트",
        user_role: str = "user",
        agent_override: str | None = None,
    ) -> ChatState:
        """테스트용 상태 생성 헬퍼"""
        state: ChatState = {
            "message": message,
            "user_role": user_role,
            "history": [],
            "session_data": {},
            "user_location": None,
            "agent_override": agent_override,
            "selected_agent": "",
            "search_focus": "",
            "routing_confidence": 0.0,
            "routing_reason": "",
            "response": "",
            "sources": [],
            "actions": [],
            "output_session_data": {},
            "agent_used": "",
        }
        return state

    def test_agent_override_routes_directly(self) -> None:
        """agent 직접 지정 시 해당 노드로 라우팅"""
        state = self._make_state(
            message="아무말", agent_override="small_claims"
        )
        result = router_node(state)

        assert result.goto == "small_claims_subgraph"
        assert result.update["selected_agent"] == "small_claims"
        assert result.update["routing_confidence"] == 1.0

    def test_agent_override_case_search_sets_focus(self) -> None:
        """agent_override=case_search → search_focus=precedent"""
        state = self._make_state(
            message="아무말", agent_override="case_search"
        )
        result = router_node(state)

        assert result.goto == "legal_search_node"
        assert result.update["search_focus"] == "precedent"

    def test_agent_override_law_search_sets_focus(self) -> None:
        """agent_override=law_search → search_focus=law"""
        state = self._make_state(
            message="아무말", agent_override="law_search"
        )
        result = router_node(state)

        assert result.goto == "legal_search_node"
        assert result.update["search_focus"] == "law"

    def test_legal_keyword_routes_to_legal_search(self) -> None:
        """'판례' 키워드 → legal_search_node"""
        state = self._make_state(message="손해배상 판례 검색해줘")
        result = router_node(state)

        assert result.goto in (
            "legal_search_node",
            "simple_chat_node",
        )

    def test_lawyer_keyword_routes_to_lawyer_finder(self) -> None:
        """'변호사 찾' 키워드 → lawyer_finder_node"""
        state = self._make_state(message="근처 변호사 찾아줘")
        result = router_node(state)

        assert result.goto == "lawyer_finder_node"

    def test_small_claims_keyword(self) -> None:
        """'소액소송' 키워드 → small_claims_subgraph"""
        state = self._make_state(message="소액소송 하고 싶어요")
        result = router_node(state)

        assert result.goto == "small_claims_subgraph"

    def test_storyboard_keyword(self) -> None:
        """'타임라인' 키워드 → storyboard_node"""
        state = self._make_state(message="사건 타임라인 정리해줘")
        result = router_node(state)

        assert result.goto == "storyboard_node"

    def test_lawyer_stats_keyword(self) -> None:
        """'변호사 통계' 키워드 → lawyer_stats_node (lawyer 역할)"""
        state = self._make_state(
            message="변호사 통계 보여줘", user_role="lawyer"
        )
        result = router_node(state)

        assert result.goto == "lawyer_stats_node"

    def test_law_study_keyword(self) -> None:
        """'로스쿨' 키워드 → law_study_node (lawyer 역할)"""
        state = self._make_state(
            message="로스쿨 공부 도와줘", user_role="lawyer"
        )
        result = router_node(state)

        assert result.goto == "law_study_node"

    def test_default_fallback(self) -> None:
        """매칭 없는 일반 메시지 → legal_search_node 또는 simple_chat"""
        state = self._make_state(message="안녕하세요")
        result = router_node(state)

        assert result.goto in (
            "legal_search_node",
            "simple_chat_node",
        )

    def test_router_returns_command(self) -> None:
        """라우터가 Command 객체를 반환"""
        from langgraph.types import Command

        state = self._make_state(message="테스트")
        result = router_node(state)

        assert isinstance(result, Command)

    def test_router_sets_routing_metadata(self) -> None:
        """라우터가 라우팅 메타데이터를 설정"""
        state = self._make_state(message="변호사 찾아줘")
        result = router_node(state)

        assert "selected_agent" in result.update
        assert "search_focus" in result.update
        assert "routing_confidence" in result.update
        assert "routing_reason" in result.update
        assert result.update["routing_confidence"] > 0

    def test_agent_override_blocked_by_role(self) -> None:
        """user 역할에서 lawyer 전용 에이전트 agent_override 시 폴백"""
        state = self._make_state(
            message="변호사 통계 보여줘",
            user_role="user",
            agent_override="lawyer_stats",
        )
        result = router_node(state)

        # lawyer_stats는 user에게 허용되지 않으므로 lawyer_stats_node가 아님
        assert result.goto != "lawyer_stats_node"

    def test_agent_override_allowed_for_correct_role(self) -> None:
        """lawyer 역할에서 lawyer 전용 에이전트 agent_override 허용"""
        state = self._make_state(
            message="아무말",
            user_role="lawyer",
            agent_override="lawyer_stats",
        )
        result = router_node(state)

        assert result.goto == "lawyer_stats_node"
        assert result.update["selected_agent"] == "lawyer_stats"
