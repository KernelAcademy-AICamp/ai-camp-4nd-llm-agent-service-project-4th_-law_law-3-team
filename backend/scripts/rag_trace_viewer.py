"""
RAG 질의 트레이스 뷰어 (Gradio)

trace_store 싱글턴을 직접 import하여 데이터 표시.

- FastAPI 마운트: http://localhost:8000/trace-viewer (서버와 같이 실행)
- 독립 실행: uv run python scripts/rag_trace_viewer.py → http://localhost:7860
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import gradio as gr

# 프로젝트 루트를 path에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.rag.trace_store import trace_store  # noqa: E402


def fetch_traces() -> list[dict[str, Any]]:
    """trace_store에서 직접 트레이스 목록 조회."""
    return [t.to_summary() for t in trace_store.list_all()]


def fetch_trace_detail(trace_id: str) -> dict[str, Any]:
    """trace_store에서 직접 트레이스 상세 조회."""
    trace = trace_store.get(trace_id)
    if trace is None:
        return {"error": "트레이스를 찾을 수 없습니다"}
    return trace.to_dict()


def clear_all_traces() -> str:
    """전체 트레이스 삭제."""
    trace_store.clear()
    return "트레이스가 모두 삭제되었습니다."


def format_trace_list(traces: list[dict[str, Any]]) -> list[list[str]]:
    """트레이스 목록을 테이블 형식으로 변환."""
    if not traces or "error" in traces[0]:
        return []
    rows: list[list[str]] = []
    for t in traces:
        ts = t.get("timestamp", "")[:19].replace("T", " ")
        rows.append([
            t.get("trace_id", ""),
            t.get("query", "")[:40],
            t.get("agent_used", ""),
            f"{t.get('total_time_ms', 0):.0f}ms",
            str(t.get("step_count", 0)),
            ts,
        ])
    return rows


# ---------------------------------------------------------------------------
# 단계별 맞춤 테이블 포맷
# ---------------------------------------------------------------------------


def _is_vector_step(name: str) -> bool:
    return "vector_search" in name


def _is_keyword_step(name: str) -> bool:
    return "keyword_search" in name


def _is_rerank_step(name: str) -> bool:
    return name.endswith("rerank")


def _is_rrf_step(name: str) -> bool:
    return "rrf_fusion" in name


def _is_final_step(name: str) -> bool:
    return "final_context" in name


def _truncate(text: str, length: int = 80) -> str:
    """텍스트를 지정 길이로 자르고 말줄임 처리."""
    if not text:
        return ""
    text = text.replace("\n", " ").strip()
    if len(text) <= length:
        return text
    return text[:length] + "..."


def format_step_detail(step: dict[str, Any]) -> str:
    """단계별 맞춤 컬럼으로 Markdown 테이블 생성."""
    name = step.get("step_name", "")
    count = step.get("count", 0)
    time_ms = step.get("time_ms", 0)
    metadata = step.get("metadata", {})
    docs = step.get("documents", [])

    header = f"### {name} ({count}건, {time_ms:.0f}ms)\n\n"

    # 메타데이터 (쿼리 리라이팅 등)
    if metadata:
        header += "**메타데이터:**\n```json\n"
        header += json.dumps(metadata, ensure_ascii=False, indent=2)
        header += "\n```\n\n"

    # RRF 융합: 건수만 표시
    if _is_rrf_step(name):
        header += f"RRF 융합 결과: **{count}건**\n"
        return header

    # 문서가 없으면 종료
    if not docs:
        return header

    # --- 벡터 검색 ---
    if _is_vector_step(name):
        header += "| # | source_id | title | content | similarity |\n"
        header += "|---|-----------|-------|---------|------------|\n"
        for i, doc in enumerate(docs[:20], 1):
            sid = doc.get("source_id", "")[:12]
            title = _truncate(doc.get("title", ""), 25)
            content = _truncate(doc.get("content", ""), 60)
            sim = doc.get("similarity", 0)
            sim_str = f"{sim:.4f}" if isinstance(sim, (int, float)) else str(sim)
            header += f"| {i} | {sid} | {title} | {content} | {sim_str} |\n"

    # --- 키워드 검색 (FTS rank 기반, content 없음) ---
    elif _is_keyword_step(name):
        header += "| # | source_id | title | data_type | fts_rank (정규화) |\n"
        header += "|---|-----------|-------|-----------|------------------|\n"
        for i, doc in enumerate(docs[:20], 1):
            sid = doc.get("source_id", "")[:12]
            title = _truncate(doc.get("title", ""), 30)
            dtype = doc.get("data_type", "")
            sim = doc.get("similarity", 0)
            sim_str = f"{sim:.4f}" if isinstance(sim, (int, float)) else str(sim)
            header += f"| {i} | {sid} | {title} | {dtype} | {sim_str} |\n"

    # --- 리랭킹 ---
    elif _is_rerank_step(name):
        header += "| # | source_id | title | content | rerank_score |\n"
        header += "|---|-----------|-------|---------|-------------|\n"
        for i, doc in enumerate(docs[:20], 1):
            sid = doc.get("source_id", "")[:12]
            title = _truncate(doc.get("title", ""), 25)
            content = _truncate(doc.get("content", ""), 60)
            rscore = doc.get("rerank_score", 0)
            rs_str = f"{rscore:.4f}" if isinstance(rscore, (int, float)) else str(rscore)
            header += f"| {i} | {sid} | {title} | {content} | {rs_str} |\n"

    # --- final_context ---
    elif _is_final_step(name):
        header += "| # | source_id | title | similarity |\n"
        header += "|---|-----------|-------|------------|\n"
        for i, doc in enumerate(docs[:20], 1):
            sid = doc.get("source_id", "")[:12]
            title = _truncate(doc.get("title", ""), 30)
            sim = doc.get("similarity", 0)
            sim_str = f"{sim:.4f}" if isinstance(sim, (int, float)) else str(sim)
            header += f"| {i} | {sid} | {title} | {sim_str} |\n"

    # --- 기타 (fallback) ---
    else:
        header += "| # | source_id | title |\n"
        header += "|---|-----------|-------|\n"
        for i, doc in enumerate(docs[:20], 1):
            sid = doc.get("source_id", "")[:12]
            title = _truncate(doc.get("title", ""), 40)
            header += f"| {i} | {sid} | {title} |\n"

    if len(docs) > 20:
        header += f"\n*... 외 {len(docs) - 20}건*\n"

    return header


def build_detail_view(trace_id: str) -> str:
    """트레이스 상세 뷰 생성."""
    if not trace_id:
        return "트레이스를 선택하세요."

    detail = fetch_trace_detail(trace_id)
    if "error" in detail:
        return f"오류: {detail['error']}"

    md = f"# 트레이스: {detail.get('trace_id', '')}\n\n"
    md += f"**시간:** {detail.get('timestamp', '')[:19].replace('T', ' ')}\n\n"
    md += f"**원본 질문:** {detail.get('query', '')}\n\n"
    md += f"**검색 쿼리:** {detail.get('search_query', '')}\n\n"
    md += f"**에이전트:** {detail.get('agent_used', '')}\n\n"
    md += f"**총 소요시간:** {detail.get('total_time_ms', 0):.0f}ms\n\n"

    # 파이프라인 설정
    config = detail.get("pipeline_config", {})
    if config:
        md += "**파이프라인 설정:**\n```json\n"
        md += json.dumps(config, ensure_ascii=False, indent=2)
        md += "\n```\n\n"

    md += "---\n\n## 파이프라인 단계\n\n"

    # 각 단계 표시
    steps = detail.get("steps", [])
    for i, step in enumerate(steps, 1):
        md += f"## Step {i}: "
        md += format_step_detail(step)
        md += "\n---\n\n"

    # LLM 응답 전문
    response = detail.get("response_full", "")
    if response:
        md += "## LLM 응답\n\n"
        md += response + "\n"

    return md


def create_gradio_app() -> gr.Blocks:
    """Gradio 앱 생성. FastAPI 마운트 및 독립 실행 모두 지원."""
    with gr.Blocks(
        title="RAG 질의 트레이스 뷰어",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown("# RAG 질의 트레이스 뷰어")
        gr.Markdown("챗봇 질의의 RAG 파이프라인 중간 결과를 확인합니다.")

        with gr.Row():
            refresh_button = gr.Button("새로고침", variant="primary", scale=1)
            clear_button = gr.Button("전체 삭제", variant="stop", scale=1)
            status_text = gr.Textbox(
                label="상태",
                interactive=False,
                scale=3,
            )

        with gr.Row():
            # 왼쪽: 트레이스 목록
            with gr.Column(scale=1):
                gr.Markdown("### 최근 질의 목록")
                trace_table = gr.Dataframe(
                    headers=["ID", "질문", "에이전트", "소요시간", "단계수", "시각"],
                    datatype=["str", "str", "str", "str", "str", "str"],
                    interactive=False,
                    wrap=True,
                )

            # 오른쪽: 상세 뷰
            with gr.Column(scale=2):
                detail_view = gr.Markdown(
                    "테이블에서 행을 클릭하면 상세 정보가 표시됩니다."
                )

        # 이벤트 핸들러
        def on_refresh() -> tuple[list[list[str]], str]:
            traces = fetch_traces()
            rows = format_trace_list(traces)
            if not rows:
                if traces and "error" in traces[0]:
                    return [], f"백엔드 연결 실패: {traces[0]['error']}"
                return [], "트레이스가 없습니다."
            return rows, f"{len(rows)}건의 트레이스"

        def on_clear() -> tuple[list[list[str]], str, str]:
            msg = clear_all_traces()
            return [], msg, "테이블에서 행을 클릭하면 상세 정보가 표시됩니다."

        def on_select(evt: gr.SelectData, data: Any) -> str:
            """테이블 행 클릭 → 오른쪽에 상세 뷰 즉시 표시."""
            try:
                if evt.index is None or data is None:
                    return "테이블에서 행을 클릭하면 상세 정보가 표시됩니다."
                row_idx = (
                    evt.index[0]
                    if isinstance(evt.index, (list, tuple))
                    else evt.index
                )
                # pandas DataFrame
                if hasattr(data, "iloc"):
                    trace_id = str(data.iloc[row_idx, 0])
                # list
                elif isinstance(data, list) and row_idx < len(data):
                    trace_id = str(data[row_idx][0])
                else:
                    return "테이블에서 행을 클릭하면 상세 정보가 표시됩니다."
                return build_detail_view(trace_id)
            except (IndexError, KeyError, TypeError):
                return "테이블에서 행을 클릭하면 상세 정보가 표시됩니다."

        refresh_button.click(
            fn=on_refresh,
            outputs=[trace_table, status_text],
        )

        clear_button.click(
            fn=on_clear,
            outputs=[trace_table, status_text, detail_view],
        )

        trace_table.select(
            fn=on_select,
            inputs=[trace_table],
            outputs=[detail_view],
        )

        # 초기 로드
        app.load(
            fn=on_refresh,
            outputs=[trace_table, status_text],
        )

    return app


if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
