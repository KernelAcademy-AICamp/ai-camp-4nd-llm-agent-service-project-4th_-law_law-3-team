"""
LangGraph StateGraph 빌드 및 컴파일

메인 그래프: router_node -> (agent nodes | small_claims_subgraph) -> END

체크포인터:
- USE_PERSISTENT_CHECKPOINTER=true (기본): AsyncPostgresSaver (PostgreSQL)
- USE_PERSISTENT_CHECKPOINTER=false: InMemorySaver (개발/테스트용)
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.multi_agent.nodes import (
    content_marketing_node,
    law_study_node,
    lawyer_finder_node,
    lawyer_stats_node,
    legal_search_node,
    router_node,
    simple_chat_node,
    storyboard_node,
)
from app.multi_agent.state import ChatState
from app.multi_agent.subgraphs.mock_trial import build_mock_trial_subgraph
from app.multi_agent.subgraphs.small_claims import build_small_claims_subgraph

logger = logging.getLogger(__name__)

# 체크포인터 및 컴파일된 그래프 싱글톤
_checkpointer: BaseCheckpointSaver[Any] | None = None
_checkpointer_context: Any = None  # async context manager 참조 유지
_compiled_graph: CompiledStateGraph | None = None


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
    builder.add_node("mock_trial_subgraph", build_mock_trial_subgraph())
    builder.add_node("storyboard_node", storyboard_node)
    builder.add_node("lawyer_stats_node", lawyer_stats_node)
    builder.add_node("law_study_node", law_study_node)
    builder.add_node("content_marketing_node", content_marketing_node)
    builder.add_node("simple_chat_node", simple_chat_node)

    # 엣지: START -> router_node
    builder.add_edge(START, "router_node")
    # router_node는 Command(goto=...)로 라우팅하므로 conditional edge 불필요

    # 에이전트 노드 -> END
    for node_name in (
        "legal_search_node",
        "lawyer_finder_node",
        "small_claims_subgraph",
        "mock_trial_subgraph",
        "storyboard_node",
        "lawyer_stats_node",
        "law_study_node",
        "content_marketing_node",
        "simple_chat_node",
    ):
        builder.add_edge(node_name, END)

    return builder


async def init_checkpointer(conn_string: str) -> None:
    """PostgreSQL 체크포인터 초기화 (lifespan에서 호출)

    AsyncPostgresSaver를 생성하고 테이블을 자동 생성합니다.
    실패 시 InMemorySaver로 폴백합니다.

    Args:
        conn_string: PostgreSQL 연결 문자열 (psycopg 형식)
    """
    global _checkpointer, _checkpointer_context, _compiled_graph

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        ctx = AsyncPostgresSaver.from_conn_string(conn_string)
        checkpointer = await ctx.__aenter__()
        await checkpointer.setup()

        _checkpointer = checkpointer
        _checkpointer_context = ctx
        _compiled_graph = None  # 재컴파일 필요
        logger.info("PostgreSQL 체크포인터 초기화 완료")
    except Exception as e:
        logger.warning("PostgreSQL 체크포인터 초기화 실패, InMemorySaver 사용: %s", e)
        _checkpointer = InMemorySaver()
        _checkpointer_context = None
        _compiled_graph = None


async def shutdown_checkpointer() -> None:
    """PostgreSQL 체크포인터 종료 (lifespan에서 호출)"""
    global _checkpointer, _checkpointer_context, _compiled_graph

    if _checkpointer_context is not None:
        try:
            await _checkpointer_context.__aexit__(None, None, None)
            logger.info("PostgreSQL 체크포인터 종료 완료")
        except Exception as e:
            logger.warning("PostgreSQL 체크포인터 종료 중 오류: %s", e)
        finally:
            _checkpointer = None
            _checkpointer_context = None
            _compiled_graph = None


def get_graph() -> CompiledStateGraph:
    """컴파일된 그래프 싱글톤 반환

    체크포인터가 init_checkpointer()로 초기화되었으면 사용하고,
    없으면 InMemorySaver 폴백.

    Returns:
        컴파일된 StateGraph
    """
    global _compiled_graph
    if _compiled_graph is None:
        builder = build_graph()
        checkpointer = _checkpointer or InMemorySaver()
        _compiled_graph = builder.compile(checkpointer=checkpointer)
        logger.info("LangGraph 채팅 그래프 컴파일 완료")
    return _compiled_graph
