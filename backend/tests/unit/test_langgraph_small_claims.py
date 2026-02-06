"""
SmallClaims subgraph 단위 테스트

노드 함수, subgraph 빌드, 상태 전환 검증
"""

from typing import Any

from app.multi_agent.agents.small_claims_agent import (
    SMALL_CLAIMS_LIMIT,
    SmallClaimsStep,
    detect_dispute_type,
    extract_amount,
)
from app.multi_agent.subgraphs.small_claims import (
    SmallClaimsState,
    _court_actions,
    _dispute_type_actions,
    _evidence_actions,
    build_small_claims_subgraph,
    court_node,
)


class TestSmallClaimsHelpers:
    """헬퍼 함수 검증 (small_claims_agent.py에서 재사용)"""

    def test_detect_dispute_type_fraud(self) -> None:
        """중고거래 사기 감지"""
        assert detect_dispute_type("중고거래 사기 당했어요") is not None

    def test_detect_dispute_type_deposit(self) -> None:
        """보증금 분쟁 감지"""
        assert detect_dispute_type("보증금 안 돌려줘요") is not None

    def test_detect_dispute_type_goods(self) -> None:
        """물품 대금 감지"""
        assert detect_dispute_type("물건 대금을 안 줘요") is not None

    def test_detect_dispute_type_none(self) -> None:
        """매칭 없으면 None"""
        assert detect_dispute_type("안녕하세요") is None

    def test_extract_amount_man_won(self) -> None:
        """만원 단위 추출"""
        assert extract_amount("500만원입니다") == 5_000_000

    def test_extract_amount_won(self) -> None:
        """원 단위 추출 (1~3자리)"""
        assert extract_amount("100원 입니다") == 100

    def test_extract_amount_comma(self) -> None:
        """콤마 포함 금액 추출"""
        assert extract_amount("3,000,000원") == 3_000_000

    def test_extract_amount_none(self) -> None:
        """금액 없으면 None"""
        assert extract_amount("소송 하고 싶어요") is None

    def test_small_claims_limit(self) -> None:
        """소액소송 한도 3천만원"""
        assert SMALL_CLAIMS_LIMIT == 30_000_000


class TestActionBuilders:
    """액션 버튼 빌더 검증"""

    def test_dispute_type_actions(self) -> None:
        """분쟁 유형 선택 버튼 3개"""
        actions = _dispute_type_actions()
        assert len(actions) == 3
        labels = [a["label"] for a in actions]
        assert "물품 대금" in labels
        assert "중고거래 사기" in labels
        assert "임대차 보증금" in labels

    def test_evidence_actions(self) -> None:
        """증거 단계 액션 버튼 2개"""
        actions = _evidence_actions()
        assert len(actions) == 2
        labels = [a["label"] for a in actions]
        assert "내용증명 작성 도움" in labels
        assert "바로 소송 진행" in labels

    def test_court_actions(self) -> None:
        """소송 단계 액션 버튼 3개"""
        actions = _court_actions()
        assert len(actions) == 3
        labels = [a["label"] for a in actions]
        assert "전자소송 바로가기" in labels
        assert "소장 작성 도움" in labels
        assert "처음부터 다시" in labels

    def test_court_actions_has_link(self) -> None:
        """전자소송 버튼이 링크 타입"""
        actions = _court_actions()
        link_action = next(a for a in actions if a["label"] == "전자소송 바로가기")
        assert link_action["type"] == "link"
        assert "ecfs.scourt.go.kr" in link_action["url"]


class TestCourtNode:
    """court_node (최종 단계, interrupt 없음) 검증"""

    def _make_state(self, **kwargs: Any) -> SmallClaimsState:
        """테스트용 SmallClaimsState 생성"""
        base: SmallClaimsState = {
            "message": "소송 진행할게요",
            "history": [],
            "session_data": {},
            "user_location": None,
            "dispute_type": "중고거래",
            "claim_amount": 1_000_000,
            "step": SmallClaimsStep.DEMAND_LETTER,
            "is_complete": False,
            "response": "",
            "sources": [],
            "actions": [],
            "output_session_data": {},
            "agent_used": "",
        }
        base.update(kwargs)  # type: ignore[typeddict-item]
        return base

    def test_court_node_returns_dict(self) -> None:
        """court_node가 dict 반환 (Command 아님)"""
        state = self._make_state()
        result = court_node(state)
        assert isinstance(result, dict)

    def test_court_node_is_complete(self) -> None:
        """court_node가 is_complete=True 설정"""
        state = self._make_state()
        result = court_node(state)
        assert result["is_complete"] is True

    def test_court_node_step_complete(self) -> None:
        """court_node가 step=complete 설정"""
        state = self._make_state()
        result = court_node(state)
        assert result["step"] == SmallClaimsStep.COMPLETE

    def test_court_node_agent_used(self) -> None:
        """court_node가 agent_used 설정"""
        state = self._make_state()
        result = court_node(state)
        assert result["agent_used"] == "small_claims"

    def test_court_node_has_actions(self) -> None:
        """court_node가 액션 버튼 포함"""
        state = self._make_state()
        result = court_node(state)
        assert len(result["actions"]) == 3

    def test_court_node_response_not_empty(self) -> None:
        """court_node가 응답 메시지 포함"""
        state = self._make_state()
        result = court_node(state)
        assert result["response"] != ""
        assert "소장" in result["response"] or "소액소송" in result["response"]


class TestInitNodeSync:
    """init_node의 동기 분기 (dispute_type 감지 시 interrupt 없이 진행) 검증"""

    def test_init_node_with_dispute_type_detected(self) -> None:
        """분쟁 유형이 감지되면 Command(goto=gather_info_node) 반환"""
        from langgraph.types import Command

        from app.multi_agent.subgraphs.small_claims import init_node

        state: SmallClaimsState = {
            "message": "중고거래 사기 당했어요",
            "history": [],
            "session_data": {},
            "user_location": None,
            "dispute_type": "",
            "claim_amount": 0,
            "step": SmallClaimsStep.INIT,
            "is_complete": False,
            "response": "",
            "sources": [],
            "actions": [],
            "output_session_data": {},
            "agent_used": "",
        }

        result = init_node(state)

        assert isinstance(result, Command)
        assert result.goto == "gather_info_node"
        assert "중고거래" in result.update["dispute_type"]
        assert result.update["step"] == SmallClaimsStep.GATHER_INFO
        assert result.update["agent_used"] == "small_claims"


class TestSubgraphBuild:
    """subgraph 빌드 및 컴파일 검증"""

    def test_build_returns_compiled_graph(self) -> None:
        """build_small_claims_subgraph가 컴파일된 그래프 반환"""
        from langgraph.graph.state import CompiledStateGraph

        graph = build_small_claims_subgraph()
        assert isinstance(graph, CompiledStateGraph)

    def test_subgraph_has_all_nodes(self) -> None:
        """모든 노드가 등록되어 있는지 확인"""
        graph = build_small_claims_subgraph()
        node_names = set(graph.nodes.keys())
        expected = {
            "init_node",
            "gather_info_node",
            "evidence_node",
            "demand_letter_node",
            "court_node",
        }
        assert expected.issubset(node_names)

    def test_subgraph_compiles_without_error(self) -> None:
        """subgraph 컴파일이 에러 없이 완료"""
        graph = build_small_claims_subgraph()
        assert graph is not None
