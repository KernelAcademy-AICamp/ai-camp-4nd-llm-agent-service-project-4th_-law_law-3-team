"""
RAG 검색 결과 포맷팅 유틸리티

검색된 문서를 LLM 컨텍스트 또는 프론트엔드 소스 정보로 변환.
에이전트에서 공통으로 사용하여 중복을 제거한다.
"""

from typing import Any

# ---------------------------------------------------------------------------
# 공통 헬퍼
# ---------------------------------------------------------------------------


def _format_fields(doc: dict[str, Any]) -> str:
    """content_fields를 [컬럼명] 값 형태로 포맷.

    content_fields가 없으면 content로 fallback.
    """
    fields: dict[str, str] = doc.get("content_fields", {})
    if not fields:
        return str(doc.get("content", ""))
    return "\n".join(f"[{col}] {val}" for col, val in fields.items())


def _get_title(metadata: dict[str, Any]) -> str:
    """metadata에서 문서 제목 추출."""
    return str(metadata.get("case_name", "") or metadata.get("title", ""))


# ---------------------------------------------------------------------------
# 컨텍스트 구성 (LLM에 주입할 텍스트)
# ---------------------------------------------------------------------------


def format_precedent_context(
    documents: list[dict[str, Any]],
) -> str:
    """판례 문서 → LLM 컨텍스트 문자열.

    content_fields(DOCUMENT_TABLE_REGISTRY 기반)로 LLM context를 구성.
    """
    if not documents:
        return ""

    parts: list[str] = ["## 관련 판례"]

    for i, doc in enumerate(documents, 1):
        metadata = doc.get("metadata", {})
        doc_id = metadata.get("doc_id", "")
        case_name = _get_title(metadata)
        case_number = metadata.get("case_number", "")

        header = f"[판례 {i}] {case_name}"
        if case_number:
            header += f" ({case_number})"
        header += f" (id: {doc_id})"

        parts.append(f"{header}\n{_format_fields(doc)}")

    return "\n\n".join(parts)


def format_law_context(documents: list[dict[str, Any]]) -> str:
    """법령 문서 → LLM 컨텍스트 문자열."""
    if not documents:
        return ""

    parts: list[str] = ["## 관련 법령"]
    for i, doc in enumerate(documents, 1):
        metadata = doc.get("metadata", {})
        doc_id = metadata.get("doc_id", "")
        law_name = _get_title(metadata)

        parts.append(f"[법령 {i}] {law_name} (id: {doc_id})\n{_format_fields(doc)}")

    return "\n\n".join(parts)


def format_supplementary_context(documents: list[dict[str, Any]]) -> str:
    """보충 문서 (다양한 data_type) → LLM 컨텍스트 문자열."""
    if not documents:
        return ""

    parts: list[str] = ["## 관련 법률 자료 (보충)"]
    for i, doc in enumerate(documents, 1):
        metadata = doc.get("metadata", {})
        data_type = metadata.get("data_type", "")
        doc_id = metadata.get("doc_id", "")
        title = _get_title(metadata)

        parts.append(
            f"[{data_type} {i}] {title} (id: {doc_id})\n{_format_fields(doc)}"
        )

    return "\n\n".join(parts)


def format_generic_context(documents: list[dict[str, Any]]) -> str:
    """범용 문서 → LLM 컨텍스트 문자열 (data_type 기반 자동 포맷)."""
    if not documents:
        return ""

    parts: list[str] = ["## 관련 자료"]
    for i, doc in enumerate(documents, 1):
        metadata = doc.get("metadata", {})
        data_type = metadata.get("data_type", "")
        doc_id = metadata.get("doc_id", "")
        title = _get_title(metadata)

        label = f"{data_type} " if data_type else ""
        parts.append(
            f"[{label}{i}] {title} (id: {doc_id})\n{_format_fields(doc)}"
        )

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# 소스 포맷팅 (프론트엔드 전달용)
# ---------------------------------------------------------------------------


def format_precedent_sources(
    documents: list[dict[str, Any]],
    details: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """판례 문서 → 프론트엔드 소스 정보.

    Args:
        documents: 파이프라인 검색 결과
        details: PrecedentService.get_details()로 조회한 판례 상세 정보
    """
    details = details or {}
    sources: list[dict[str, Any]] = []

    for doc in documents:
        metadata = doc.get("metadata", {})
        doc_id = metadata.get("doc_id", "")
        case_number = metadata.get("case_number", "")

        source_item: dict[str, Any] = {
            "doc_id": doc_id,
            "doc_type": "precedent",
            "case_name": metadata.get("case_name", ""),
            "case_number": case_number,
            "court_name": metadata.get("court_name", ""),
            "similarity": round(doc.get("similarity", 0), 3),
            "content": doc.get("content", ""),
        }

        if doc_id in details:
            detail = details[doc_id]
            if not case_number and detail.get("case_number"):
                source_item["case_number"] = detail["case_number"]
            if not source_item["case_name"] and detail.get("case_name"):
                source_item["case_name"] = detail["case_name"]
            source_item["ruling"] = detail.get("ruling", "")
            source_item["claim"] = detail.get("claim", "")
            source_item["reasoning"] = detail.get("reasoning", "")
            source_item["decision_date"] = detail.get("decision_date", "")
            source_item["case_type"] = detail.get("case_type", "")
            source_item["summary"] = detail.get("summary", "")
            source_item["full_reason"] = detail.get("full_reason", "")
            source_item["full_text"] = detail.get("full_text", "")
            source_item["reference_provisions"] = detail.get(
                "reference_provisions", ""
            )
            source_item["reference_cases"] = detail.get(
                "reference_cases", ""
            )

        sources.append(source_item)

    return sources


def format_law_sources(
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """법령 문서 → 프론트엔드 소스 정보."""
    sources: list[dict[str, Any]] = []
    for doc in documents:
        metadata = doc.get("metadata", {})
        sources.append({
            "doc_id": metadata.get("doc_id", ""),
            "doc_type": "law",
            "law_name": (
                metadata.get("case_name", "") or metadata.get("title", "")
            ),
            "similarity": round(doc.get("similarity", 0), 3),
            "content": doc.get("content", ""),
        })
    return sources


def format_supplementary_sources(
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """보충 문서 → 프론트엔드 소스 정보."""
    sources: list[dict[str, Any]] = []
    for doc in documents:
        metadata = doc.get("metadata", {})
        sources.append({
            "doc_id": metadata.get("doc_id", ""),
            "doc_type": metadata.get("data_type", ""),
            "title": (
                metadata.get("case_name", "") or metadata.get("title", "")
            ),
            "source_name": metadata.get("court_name", ""),
            "date": metadata.get("date", ""),
            "similarity": round(doc.get("similarity", 0), 3),
            "content": doc.get("content", ""),
        })
    return sources
