"""
RAG 평가 시스템 스키마 정의

평가 데이터셋, 실험 설정, 결과 등의 Pydantic 모델
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """문서 유형"""
    PRECEDENT = "precedent"
    LAW = "law"


class QueryType(str, Enum):
    """쿼리 유형"""
    SIMPLE_LOOKUP = "단순조회"
    CONCEPT_SEARCH = "개념검색"
    COMPARISON = "비교검색"
    REFERENCE_TRACE = "참조추적"
    TEMPORAL = "시간검색"
    COMPLEX = "복합검색"


class Difficulty(str, Enum):
    """난이도"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Relevance(str, Enum):
    """관련성 수준"""
    HIGHLY_RELEVANT = "highly_relevant"
    RELEVANT = "relevant"
    PARTIALLY_RELEVANT = "partially_relevant"


class Category(str, Enum):
    """법률 카테고리"""
    CIVIL = "민사"
    CRIMINAL = "형사"
    ADMINISTRATIVE = "행정"
    CONSTITUTIONAL = "헌법"
    LABOR = "노동"
    COMMERCIAL = "상사"
    TAX = "조세"
    OTHER = "기타"


class SourceDocument(BaseModel):
    """Ground Truth 소스 문서"""
    doc_id: str = Field(description="문서 ID (PostgreSQL)")
    doc_type: DocumentType = Field(description="문서 유형 (precedent/law)")
    relevance: Relevance = Field(
        default=Relevance.HIGHLY_RELEVANT,
        description="관련성 수준",
    )
    reason: Optional[str] = Field(
        default=None,
        description="관련성 이유",
    )
    article: Optional[str] = Field(
        default=None,
        description="조문 번호 (법령인 경우)",
    )


class GroundTruth(BaseModel):
    """Ground Truth 정보"""
    source_documents: list[SourceDocument] = Field(
        default_factory=list,
        description="정답 문서 목록",
    )
    key_points: list[str] = Field(
        default_factory=list,
        description="핵심 포인트",
    )
    required_citations: list[str] = Field(
        default_factory=list,
        description="필수 인용 조문/판례",
    )


class QueryMetadata(BaseModel):
    """쿼리 메타데이터"""
    category: Category = Field(description="법률 카테고리")
    subcategory: Optional[str] = Field(
        default=None,
        description="하위 카테고리",
    )
    query_type: QueryType = Field(description="쿼리 유형")
    difficulty: Difficulty = Field(
        default=Difficulty.MEDIUM,
        description="난이도",
    )


class EvalQuery(BaseModel):
    """평가 쿼리"""
    id: str = Field(description="쿼리 ID (Q-001 형식)")
    question: str = Field(description="질문 텍스트")
    metadata: QueryMetadata = Field(description="쿼리 메타데이터")
    ground_truth: GroundTruth = Field(description="Ground Truth 정보")
    source: str = Field(
        default="manual",
        description="생성 방식 (manual/solar)",
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="생성 일시",
    )


class EvalDataset(BaseModel):
    """평가 데이터셋"""
    version: str = Field(
        default="1.0",
        description="데이터셋 버전",
    )
    name: str = Field(description="데이터셋 이름")
    description: Optional[str] = Field(
        default=None,
        description="데이터셋 설명",
    )
    queries: list[EvalQuery] = Field(
        default_factory=list,
        description="평가 쿼리 목록",
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="생성 일시",
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        description="수정 일시",
    )

    def add_query(self, query: EvalQuery) -> None:
        """쿼리 추가"""
        self.queries.append(query)
        self.updated_at = datetime.now()

    def get_next_id(self) -> str:
        """다음 쿼리 ID 반환"""
        if not self.queries:
            return "Q-001"
        max_id = max(int(q.id.split("-")[1]) for q in self.queries)
        return f"Q-{max_id + 1:03d}"


class SearchResult(BaseModel):
    """검색 결과 (단일 문서)"""
    doc_id: str = Field(description="문서 ID")
    chunk_id: str = Field(description="청크 ID")
    doc_type: DocumentType = Field(description="문서 유형")
    title: str = Field(description="문서 제목")
    content: str = Field(description="청크 내용")
    score: float = Field(description="유사도 점수")
    rank: int = Field(description="순위")


class RetrievalResult(BaseModel):
    """검색 결과 전체"""
    query: str = Field(description="검색 쿼리")
    results: list[SearchResult] = Field(
        default_factory=list,
        description="검색 결과 목록",
    )
    latency_ms: float = Field(description="검색 소요 시간 (ms)")
    retrieved_at: datetime = Field(
        default_factory=datetime.now,
        description="검색 시각",
    )


class MetricsResult(BaseModel):
    """평가 메트릭 결과"""
    recall_at_5: float = Field(description="Recall@5")
    recall_at_10: float = Field(description="Recall@10")
    mrr: float = Field(description="Mean Reciprocal Rank")
    hit_rate: float = Field(description="Hit Rate")
    ndcg_at_10: float = Field(description="NDCG@10")
    latency_p50_ms: float = Field(description="Latency P50 (ms)")
    latency_p95_ms: float = Field(description="Latency P95 (ms)")


class QueryResult(BaseModel):
    """개별 쿼리 평가 결과"""
    query_id: str = Field(description="쿼리 ID")
    question: str = Field(description="질문 텍스트")
    retrieval: RetrievalResult = Field(description="검색 결과")
    recall_at_5: float = Field(description="Recall@5")
    recall_at_10: float = Field(description="Recall@10")
    mrr: float = Field(description="MRR")
    hit: bool = Field(description="Hit 여부")
    ndcg_at_10: float = Field(description="NDCG@10")


class ExperimentConfig(BaseModel):
    """실험 설정"""
    experiment_id: str = Field(description="실험 ID (EXP-YYYYMMDD-NNN 형식)")
    name: str = Field(description="실험 이름")
    description: Optional[str] = Field(
        default=None,
        description="실험 설명",
    )
    dataset_path: str = Field(description="평가 데이터셋 경로")
    embedding_model: str = Field(
        default="nlpai-lab/KURE-v1",
        description="임베딩 모델",
    )
    distance_metric: str = Field(
        default="cosine",
        description="거리 메트릭",
    )
    top_k: int = Field(
        default=10,
        description="검색 결과 수",
    )
    filters: Optional[dict] = Field(
        default=None,
        description="검색 필터",
    )


class ExperimentResult(BaseModel):
    """실험 결과"""
    config: ExperimentConfig = Field(description="실험 설정")
    metrics: MetricsResult = Field(description="전체 메트릭")
    metrics_by_type: dict[str, MetricsResult] = Field(
        default_factory=dict,
        description="쿼리 유형별 메트릭",
    )
    metrics_by_category: dict[str, MetricsResult] = Field(
        default_factory=dict,
        description="카테고리별 메트릭",
    )
    query_results: list[QueryResult] = Field(
        default_factory=list,
        description="개별 쿼리 결과",
    )
    failed_queries: list[dict] = Field(
        default_factory=list,
        description="실패 분석",
    )
    started_at: datetime = Field(description="실험 시작 시각")
    completed_at: Optional[datetime] = Field(
        default=None,
        description="실험 완료 시각",
    )
