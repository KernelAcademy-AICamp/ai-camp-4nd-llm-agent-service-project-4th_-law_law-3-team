"""
공통 예외 클래스 정의

애플리케이션 전체에서 사용되는 커스텀 예외
"""

from typing import Any, Dict, Optional


class AppError(Exception):
    """애플리케이션 기본 예외"""

    def __init__(
        self,
        message: str,
        code: str = "APP_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}


class RAGError(AppError):
    """RAG 파이프라인 관련 예외"""

    def __init__(
        self,
        message: str,
        code: str = "RAG_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, code, details)


class RoutingError(AppError):
    """에이전트 라우팅 관련 예외"""

    def __init__(
        self,
        message: str,
        code: str = "ROUTING_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, code, details)


class ServiceError(AppError):
    """서비스 레이어 예외"""

    def __init__(
        self,
        message: str,
        code: str = "SERVICE_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, code, details)


class EmbeddingModelNotFoundError(RAGError):
    """임베딩 모델이 캐시되지 않았을 때 발생"""

    def __init__(self, model_name: str) -> None:
        super().__init__(
            message=f"임베딩 모델 '{model_name}'이 캐시되지 않았습니다. "
                    "'uv run python scripts/download_models.py'를 실행해주세요.",
            code="EMBEDDING_MODEL_NOT_FOUND",
            details={"model_name": model_name},
        )
        self.model_name = model_name


class VectorStoreError(RAGError):
    """벡터 저장소 관련 예외"""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, "VECTOR_STORE_ERROR", details)


class GraphServiceError(ServiceError):
    """Neo4j 그래프 서비스 관련 예외"""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, "GRAPH_SERVICE_ERROR", details)
