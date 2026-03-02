"""뉴스 파이프라인 예외 클래스"""


class NewsPipelineError(Exception):
    """파이프라인 기본 예외"""


class SourceFetchError(NewsPipelineError):
    """소스 수집 실패"""

    def __init__(self, source: str, message: str) -> None:
        self.source = source
        super().__init__(f"[{source}] 수집 실패: {message}")


class ArticleCleanError(NewsPipelineError):
    """기사 정제 실패"""


class SummaryGenerationError(NewsPipelineError):
    """요약 생성 실패"""

    def __init__(self, url: str, message: str) -> None:
        self.url = url
        super().__init__(f"요약 실패 [{url}]: {message}")


class StorageError(NewsPipelineError):
    """저장 실패"""


class ChunkingError(NewsPipelineError):
    """청킹 실패"""
