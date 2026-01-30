"""
RAG 평가 시스템 설정

실험 설정, 디렉토리 경로, 성능 목표 등을 관리
모든 설정은 app.core.config.settings에서 가져옴 (.env 중앙 관리)
"""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings

# 중앙 설정에서 기본값 가져오기
try:
    from app.core.config import settings as app_settings
    _DEFAULT_EMBEDDING_MODEL = app_settings.LOCAL_EMBEDDING_MODEL
    _DEFAULT_LANCEDB_URI = app_settings.LANCEDB_URI
    _DEFAULT_LANCEDB_TABLE = app_settings.LANCEDB_TABLE_NAME
except ImportError:
    # 독립 실행 시 환경변수에서 직접 로드
    import os
    _DEFAULT_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "nlpai-lab/KURE-v1")
    _DEFAULT_LANCEDB_URI = os.getenv("LANCEDB_URI", "./lancedb_data")
    _DEFAULT_LANCEDB_TABLE = os.getenv("LANCEDB_TABLE_NAME", "legal_chunks")


class EvaluationSettings(BaseSettings):
    """평가 시스템 설정 (app.core.config.settings 기반)"""

    # 디렉토리 경로
    evaluation_dir: Path = Field(
        default=Path(__file__).parent,
        description="evaluation 모듈 루트 경로",
    )

    @property
    def datasets_dir(self) -> Path:
        """데이터셋 디렉토리"""
        path = self.evaluation_dir / "datasets"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def experiments_dir(self) -> Path:
        """실험 결과 디렉토리"""
        path = self.evaluation_dir / "experiments"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def reports_dir(self) -> Path:
        """리포트 디렉토리"""
        path = self.evaluation_dir / "reports"
        path.mkdir(parents=True, exist_ok=True)
        return path

    # 임베딩 설정 (app.core.config에서 기본값)
    embedding_model: str = Field(
        default=_DEFAULT_EMBEDDING_MODEL,
        description="임베딩 모델 (.env LOCAL_EMBEDDING_MODEL)",
    )
    embedding_dimension: int = Field(
        default=1024,
        description="임베딩 차원",
    )

    # LanceDB 설정 (app.core.config에서 기본값)
    lancedb_uri: str = Field(
        default=_DEFAULT_LANCEDB_URI,
        description="LanceDB 경로 (.env LANCEDB_URI)",
    )
    lancedb_table: str = Field(
        default=_DEFAULT_LANCEDB_TABLE,
        description="LanceDB 테이블명 (.env LANCEDB_TABLE_NAME)",
    )
    distance_metric: str = Field(
        default="cosine",
        description="거리 메트릭",
    )

    # 검색 설정
    default_top_k: int = Field(
        default=10,
        description="기본 검색 결과 수",
    )

    # Upstage API 설정
    upstage_api_key: Optional[str] = Field(
        default=None,
        description="Upstage API 키",
    )
    upstage_model: str = Field(
        default="solar-pro",
        description="Upstage 모델",
    )

    # Gradio 설정
    gradio_host: str = Field(
        default="0.0.0.0",
        description="Gradio 호스트",
    )
    gradio_port: int = Field(
        default=7860,
        description="Gradio 포트",
    )

    class Config:
        env_prefix = "EVAL_"
        env_file = ".env"
        extra = "ignore"


class PerformanceTargets:
    """성능 목표치"""

    # 검색 품질 목표
    RECALL_AT_5: float = 0.7
    RECALL_AT_10: float = 0.8
    MRR: float = 0.7
    HIT_RATE: float = 0.9
    NDCG_AT_10: float = 0.75

    # 시스템 성능 목표
    LATENCY_P50_MS: float = 200.0
    LATENCY_P95_MS: float = 500.0

    @classmethod
    def check_metrics(cls, metrics: dict) -> dict[str, bool]:
        """메트릭이 목표를 달성했는지 확인"""
        return {
            "recall_at_5": metrics.get("recall_at_5", 0) >= cls.RECALL_AT_5,
            "recall_at_10": metrics.get("recall_at_10", 0) >= cls.RECALL_AT_10,
            "mrr": metrics.get("mrr", 0) >= cls.MRR,
            "hit_rate": metrics.get("hit_rate", 0) >= cls.HIT_RATE,
            "ndcg_at_10": metrics.get("ndcg_at_10", 0) >= cls.NDCG_AT_10,
            "latency_p50": metrics.get("latency_p50_ms", float("inf")) <= cls.LATENCY_P50_MS,
            "latency_p95": metrics.get("latency_p95_ms", float("inf")) <= cls.LATENCY_P95_MS,
        }


# 쿼리 유형별 목표 비율
QUERY_TYPE_DISTRIBUTION = {
    "단순조회": 0.20,
    "개념검색": 0.30,
    "비교검색": 0.15,
    "참조추적": 0.15,
    "시간검색": 0.10,
    "복합검색": 0.10,
}


# 전역 설정 인스턴스
eval_settings = EvaluationSettings()
