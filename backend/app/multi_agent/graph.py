"""
LangGraph StateGraph 빌드 및 컴파일

메인 그래프: router_node -> (agent nodes | small_claims_subgraph) -> END
"""

from __future__ import annotations

import logging

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.multi_agent.nodes import (
    law_study_node,
    lawyer_finder_node,
    lawyer_stats_node,
    legal_search_node,
    router_node,
    simple_chat_node,
    storyboard_node,
)
from app.multi_agent.state import ChatState
from app.multi_agent.subgraphs.small_claims import build_small_claims_subgraph

logger = logging.getLogger(__name__)


def build_graph() -> StateGraph:
    """메인 채팅 그래프 빌드

    Returns:
        빌드된 StateGraph (미컴파일)
    """
    builder = StateGraph(ChatState)

    # 노드 등록
    builder.add_node("router_node", router_node)
    builder.add_node("legal_search_node", legal_search_node)
    builder.add_node("lawyer_finder_node", lawyer_finder_node)
    builder.add_node("small_claims_subgraph", build_small_claims_subgraph())
    builder.add_node("storyboard_node", storyboard_node)
    builder.add_node("lawyer_stats_node", lawyer_stats_node)
    builder.add_node("law_study_node", law_study_node)
    builder.add_node("simple_chat_node", simple_chat_node)

    # 엣지: START -> router_node
    builder.add_edge(START, "router_node")
    # router_node는 Command(goto=...)로 라우팅하므로 conditional edge 불필요

    # 에이전트 노드 -> END
    for node_name in (
        "legal_search_node",
        "lawyer_finder_node",
        "small_claims_subgraph",
        "storyboard_node",
        "lawyer_stats_node",
        "law_study_node",
        "simple_chat_node",
    ):
        builder.add_edge(node_name, END)

    return builder


# 싱글톤 인스턴스
_compiled_graph: CompiledStateGraph | None = None


def get_graph() -> CompiledStateGraph:
    """컴파일된 그래프 싱글톤 반환

    InMemorySaver 체크포인터를 사용하여 SmallClaims interrupt 상태를 관리합니다.

    Returns:
        컴파일된 StateGraph

    TODO: 프로덕션 배포 시 InMemorySaver → 영속 체크포인터(PostgreSQL 등)로 교체
    """
    global _compiled_graph
    if _compiled_graph is None:
        builder = build_graph()
        checkpointer = InMemorySaver()
        _compiled_graph = builder.compile(checkpointer=checkpointer)
        logger.info("LangGraph 채팅 그래프 컴파일 완료")
    return _compiled_graph
