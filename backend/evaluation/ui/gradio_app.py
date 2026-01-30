"""
Gradio 메인 앱

3개의 탭으로 구성:
1. 데이터셋 빌더 (역추적 방식)
2. 검색 분석
3. 실험 결과
"""

import gradio as gr

from evaluation.ui.dataset_editor import create_dataset_editor_tab
from evaluation.ui.search_analyzer import create_search_analyzer_tab
from evaluation.ui.experiment_viewer import create_experiment_viewer_tab
from evaluation.config import eval_settings


def create_app() -> gr.Blocks:
    """Gradio 앱 생성"""
    with gr.Blocks(
        title="RAG 평가 시스템",
        theme=gr.themes.Soft(),
        css="""
        .container { max-width: 1400px; margin: auto; }
        .result-box { border: 1px solid #ddd; padding: 10px; margin: 5px 0; }
        .highlight { background-color: #fffacd; }
        .score-high { color: #28a745; font-weight: bold; }
        .score-low { color: #dc3545; font-weight: bold; }
        """,
    ) as app:
        gr.Markdown(
            """
            # RAG 평가 시스템

            PostgreSQL(판례/법령 원본)과 LanceDB(벡터 청크)를 기반으로
            RAG 챗봇 평가 데이터셋 생성 및 분석
            """
        )

        with gr.Tabs():
            with gr.TabItem("📝 데이터셋 빌더"):
                create_dataset_editor_tab()

            with gr.TabItem("🔍 검색 분석"):
                create_search_analyzer_tab()

            with gr.TabItem("📊 실험 결과"):
                create_experiment_viewer_tab()

    return app


def launch_app(
    host: str = None,
    port: int = None,
    share: bool = False,
) -> None:
    """
    앱 실행

    Args:
        host: 호스트 (기본: 설정값)
        port: 포트 (기본: 설정값)
        share: 공유 링크 생성 여부
    """
    app = create_app()
    app.launch(
        server_name=host or eval_settings.gradio_host,
        server_port=port or eval_settings.gradio_port,
        share=share,
    )


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    parser = argparse.ArgumentParser(description="RAG 평가 Gradio UI")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="호스트")
    parser.add_argument("--port", type=int, default=7860, help="포트")
    parser.add_argument("--share", action="store_true", help="공유 링크 생성")

    args = parser.parse_args()
    launch_app(host=args.host, port=args.port, share=args.share)
