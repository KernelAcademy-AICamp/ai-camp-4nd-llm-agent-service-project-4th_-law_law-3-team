"""
RAG 평가 시스템

PostgreSQL(판례/법령 원본)과 LanceDB(벡터 청크)를 기반으로
RAG 챗봇 평가 데이터셋 생성 및 Gradio 분석 UI 제공

모듈 구조:
- schemas: Pydantic 데이터 모델
- config: 실험 설정 관리
- metrics: 평가 지표 (Recall, MRR, NDCG 등)
- tools: 데이터셋 빌더, Solar 자동 생성
- runners: 평가 실행기, 실험 추적기
- ui: Gradio 분석 UI

사용법:
    # Gradio UI 실행
    uv run python -m evaluation

    # 평가 실행
    from evaluation.runners import EvaluationRunner
    runner = EvaluationRunner("EXP-001", "dataset.json")
    result = await runner.run()
"""

__version__ = "1.0.0"

# Lazy imports to avoid circular dependencies
def __getattr__(name: str):
    if name in (
        "EvalQuery",
        "GroundTruth",
        "SourceDocument",
        "EvalDataset",
        "SearchResult",
        "ExperimentConfig",
        "ExperimentResult",
    ):
        from evaluation import schemas
        return getattr(schemas, name)

    if name in ("DatasetBuilder", "SolarGenerator", "DatasetValidator"):
        from evaluation import tools
        return getattr(tools, name)

    if name in ("EvaluationRunner", "ExperimentTracker"):
        from evaluation import runners
        return getattr(runners, name)

    raise AttributeError(f"module 'evaluation' has no attribute '{name}'")
