"""
실험 결과 뷰어 UI

실험 결과 조회 및 분석:
1. 실험 목록 및 선택
2. 전체 메트릭 대시보드
3. 쿼리 유형/카테고리별 분석
4. 실패 케이스 분석
5. 실험 비교
"""

from pathlib import Path
from typing import Optional

import gradio as gr

from evaluation.runners.experiment_tracker import ExperimentTracker
from evaluation.config import PerformanceTargets


def get_experiment_list() -> list[str]:
    """실험 목록 조회"""
    experiments = ExperimentTracker.list_experiments()
    return [exp["id"] for exp in experiments]


def load_experiment_summary(experiment_id: str) -> tuple[str, str, str, str]:
    """
    실험 결과 로드

    Returns:
        (전체 메트릭, 유형별 메트릭, 카테고리별 메트릭, 실패 분석)
    """
    if not experiment_id:
        return "실험을 선택해주세요.", "", "", ""

    result = ExperimentTracker.load_experiment(experiment_id)
    if not result:
        return "실험 결과를 찾을 수 없습니다.", "", "", ""

    targets = PerformanceTargets.check_metrics({
        "recall_at_5": result.metrics.recall_at_5,
        "recall_at_10": result.metrics.recall_at_10,
        "mrr": result.metrics.mrr,
        "hit_rate": result.metrics.hit_rate,
        "ndcg_at_10": result.metrics.ndcg_at_10,
        "latency_p50_ms": result.metrics.latency_p50_ms,
        "latency_p95_ms": result.metrics.latency_p95_ms,
    })

    overall_lines = [
        f"## 실험: {experiment_id}",
        f"**데이터셋**: {result.config.dataset_path}",
        f"**실행 시간**: {result.started_at} ~ {result.completed_at}",
        "",
        "| 메트릭 | 값 | 목표 | 달성 |",
        "|--------|-----|------|------|",
        f"| Recall@5 | {result.metrics.recall_at_5:.4f} | ≥ {PerformanceTargets.RECALL_AT_5} | {'✓' if targets['recall_at_5'] else '✗'} |",
        f"| Recall@10 | {result.metrics.recall_at_10:.4f} | ≥ {PerformanceTargets.RECALL_AT_10} | {'✓' if targets['recall_at_10'] else '✗'} |",
        f"| MRR | {result.metrics.mrr:.4f} | ≥ {PerformanceTargets.MRR} | {'✓' if targets['mrr'] else '✗'} |",
        f"| Hit Rate | {result.metrics.hit_rate:.4f} | ≥ {PerformanceTargets.HIT_RATE} | {'✓' if targets['hit_rate'] else '✗'} |",
        f"| NDCG@10 | {result.metrics.ndcg_at_10:.4f} | ≥ {PerformanceTargets.NDCG_AT_10} | {'✓' if targets['ndcg_at_10'] else '✗'} |",
        f"| Latency P50 | {result.metrics.latency_p50_ms:.2f}ms | ≤ {PerformanceTargets.LATENCY_P50_MS}ms | {'✓' if targets['latency_p50'] else '✗'} |",
        f"| Latency P95 | {result.metrics.latency_p95_ms:.2f}ms | ≤ {PerformanceTargets.LATENCY_P95_MS}ms | {'✓' if targets['latency_p95'] else '✗'} |",
    ]
    overall_md = "\n".join(overall_lines)

    type_lines = ["## 쿼리 유형별 메트릭", ""]
    if result.metrics_by_type:
        type_lines.extend([
            "| 유형 | Recall@5 | Recall@10 | MRR | Hit Rate | NDCG@10 |",
            "|------|----------|-----------|-----|----------|---------|",
        ])
        for qtype, metrics in result.metrics_by_type.items():
            type_lines.append(
                f"| {qtype} | {metrics.recall_at_5:.4f} | {metrics.recall_at_10:.4f} | "
                f"{metrics.mrr:.4f} | {metrics.hit_rate:.4f} | {metrics.ndcg_at_10:.4f} |"
            )
    else:
        type_lines.append("데이터 없음")
    type_md = "\n".join(type_lines)

    cat_lines = ["## 카테고리별 메트릭", ""]
    if result.metrics_by_category:
        cat_lines.extend([
            "| 카테고리 | Recall@5 | Recall@10 | MRR | Hit Rate | NDCG@10 |",
            "|----------|----------|-----------|-----|----------|---------|",
        ])
        for cat, metrics in result.metrics_by_category.items():
            cat_lines.append(
                f"| {cat} | {metrics.recall_at_5:.4f} | {metrics.recall_at_10:.4f} | "
                f"{metrics.mrr:.4f} | {metrics.hit_rate:.4f} | {metrics.ndcg_at_10:.4f} |"
            )
    else:
        cat_lines.append("데이터 없음")
    cat_md = "\n".join(cat_lines)

    fail_lines = ["## 실패/저성능 분석", ""]

    if result.failed_queries:
        fail_lines.append("### 실패한 쿼리")
        for failed in result.failed_queries[:10]:
            fail_lines.append(f"- **{failed['query_id']}**: {failed['error']}")
        fail_lines.append("")

    low_performers = [
        qr for qr in result.query_results
        if qr.recall_at_10 < 0.5
    ]
    if low_performers:
        fail_lines.append("### 저성능 쿼리 (Recall@10 < 0.5)")
        for qr in low_performers[:10]:
            fail_lines.append(f"- **{qr.query_id}**: {qr.question[:60]}...")
            fail_lines.append(f"  - Recall@10: {qr.recall_at_10:.4f}, MRR: {qr.mrr:.4f}")
        fail_lines.append("")

    if not result.failed_queries and not low_performers:
        fail_lines.append("실패하거나 저성능인 쿼리가 없습니다.")

    fail_md = "\n".join(fail_lines)

    return overall_md, type_md, cat_md, fail_md


def get_query_detail(experiment_id: str, query_id: str) -> str:
    """개별 쿼리 결과 상세"""
    if not experiment_id or not query_id:
        return "실험과 쿼리 ID를 입력해주세요."

    result = ExperimentTracker.load_experiment(experiment_id)
    if not result:
        return "실험 결과를 찾을 수 없습니다."

    query_result = next(
        (qr for qr in result.query_results if qr.query_id == query_id),
        None,
    )
    if not query_result:
        return f"쿼리를 찾을 수 없습니다: {query_id}"

    lines = [
        f"## 쿼리: {query_id}",
        "",
        f"**질문**: {query_result.question}",
        "",
        "### 메트릭",
        f"- Recall@5: {query_result.recall_at_5:.4f}",
        f"- Recall@10: {query_result.recall_at_10:.4f}",
        f"- MRR: {query_result.mrr:.4f}",
        f"- Hit: {'✓' if query_result.hit else '✗'}",
        f"- NDCG@10: {query_result.ndcg_at_10:.4f}",
        "",
        "### 검색 결과",
        f"**Latency**: {query_result.retrieval.latency_ms:.2f}ms",
        "",
    ]

    for sr in query_result.retrieval.results[:10]:
        lines.append(f"#### #{sr.rank} [{sr.doc_id}] - Score: {sr.score:.4f}")
        lines.append(f"**{sr.title}** ({sr.doc_type.value})")
        lines.append(f"```\n{sr.content[:200]}...\n```")
        lines.append("")

    return "\n".join(lines)


def compare_experiments(exp_ids: str) -> str:
    """실험 비교"""
    ids = [id.strip() for id in exp_ids.split(",") if id.strip()]
    if len(ids) < 2:
        return "비교할 실험 ID를 2개 이상 입력해주세요 (쉼표로 구분)"

    lines = ["## 실험 비교", ""]
    headers = ["메트릭"] + ids
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    results = {}
    for exp_id in ids:
        result = ExperimentTracker.load_experiment(exp_id)
        if result:
            results[exp_id] = result.metrics

    if not results:
        return "비교할 실험 결과가 없습니다."

    metrics_names = [
        ("Recall@5", "recall_at_5"),
        ("Recall@10", "recall_at_10"),
        ("MRR", "mrr"),
        ("Hit Rate", "hit_rate"),
        ("NDCG@10", "ndcg_at_10"),
        ("Latency P50", "latency_p50_ms"),
        ("Latency P95", "latency_p95_ms"),
    ]

    for display_name, attr_name in metrics_names:
        row = [display_name]
        for exp_id in ids:
            if exp_id in results:
                value = getattr(results[exp_id], attr_name, 0)
                if "latency" in attr_name:
                    row.append(f"{value:.2f}ms")
                else:
                    row.append(f"{value:.4f}")
            else:
                row.append("N/A")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def create_experiment_viewer_tab():
    """실험 결과 뷰어 탭 생성"""
    with gr.Column():
        gr.Markdown("## 실험 선택")

        with gr.Row():
            experiment_select = gr.Dropdown(
                choices=get_experiment_list(),
                label="실험 선택",
                scale=3,
            )
            refresh_list_btn = gr.Button("목록 새로고침", scale=1)
            load_btn = gr.Button("결과 로드", variant="primary", scale=1)

        with gr.Tabs():
            with gr.TabItem("전체 메트릭"):
                overall_metrics = gr.Markdown()

            with gr.TabItem("유형별 분석"):
                type_metrics = gr.Markdown()

            with gr.TabItem("카테고리별 분석"):
                category_metrics = gr.Markdown()

            with gr.TabItem("실패 분석"):
                failure_analysis = gr.Markdown()

        gr.Markdown("---")
        gr.Markdown("## 개별 쿼리 상세")

        with gr.Row():
            query_id_input = gr.Textbox(
                label="쿼리 ID",
                placeholder="Q-001",
                scale=3,
            )
            query_detail_btn = gr.Button("상세 보기", scale=1)

        query_detail = gr.Markdown()

        gr.Markdown("---")
        gr.Markdown("## 실험 비교")

        with gr.Row():
            compare_input = gr.Textbox(
                label="비교할 실험 ID (쉼표로 구분)",
                placeholder="EXP-20260129-001, EXP-20260129-002",
                scale=3,
            )
            compare_btn = gr.Button("비교", scale=1)

        comparison_result = gr.Markdown()

    def refresh_list():
        return gr.Dropdown(choices=get_experiment_list())

    refresh_list_btn.click(
        fn=refresh_list,
        outputs=[experiment_select],
    )

    load_btn.click(
        fn=load_experiment_summary,
        inputs=[experiment_select],
        outputs=[overall_metrics, type_metrics, category_metrics, failure_analysis],
    )

    query_detail_btn.click(
        fn=get_query_detail,
        inputs=[experiment_select, query_id_input],
        outputs=[query_detail],
    )

    compare_btn.click(
        fn=compare_experiments,
        inputs=[compare_input],
        outputs=[comparison_result],
    )
