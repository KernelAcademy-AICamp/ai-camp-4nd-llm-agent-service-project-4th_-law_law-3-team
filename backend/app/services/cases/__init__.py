"""
판례 및 문서 서비스 모듈
"""

from app.services.cases.precedent_service import (
    PrecedentService,
    fetch_precedent_details,
    get_precedent_service,
)
from app.services.cases.document_service import (
    DocumentService,
    extract_law_names,
    fetch_laws_by_names,
    fetch_reference_articles_from_docs,
    get_document_service,
)

__all__ = [
    "PrecedentService",
    "fetch_precedent_details",
    "get_precedent_service",
    "DocumentService",
    "extract_law_names",
    "fetch_laws_by_names",
    "fetch_reference_articles_from_docs",
    "get_document_service",
]
