"""
RAG 트레이스 API 엔드포인트

디버깅/모니터링용 트레이스 조회 API.
Gradio 뷰어 및 외부 도구에서 사용.
"""

from typing import Any

from fastapi import APIRouter, HTTPException

from app.services.rag.trace_store import trace_store

router = APIRouter(prefix="/rag-traces", tags=["rag-traces"])


@router.get("")
async def list_traces() -> list[dict[str, Any]]:
    """최근 트레이스 목록 (요약)."""
    traces = trace_store.list_all()
    return [t.to_summary() for t in traces]


@router.get("/{trace_id}")
async def get_trace(trace_id: str) -> dict[str, Any]:
    """트레이스 상세 조회."""
    trace = trace_store.get(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="트레이스를 찾을 수 없습니다")
    return trace.to_dict()


@router.delete("")
async def clear_traces() -> dict[str, str]:
    """전체 트레이스 삭제."""
    trace_store.clear()
    return {"status": "cleared"}
