"""
입력 JSON 구조 검증 및 벡터 기대 수 예측.

법령/자치법규는 조문 요약 기준 다중 벡터 생성 특성이 있어
기존 1문서=1벡터 전제 검증과 차별됩니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class SourceValidationSummary:
    """입력 소스 단위 검증 결과"""

    config_name: str
    source_path: str
    total_documents: int
    expected_vectors: int
    skipped_no_summary_documents: int
    invalid_documents: int
    invalid_article_shapes: int
    missing_id_documents: int
    sample_invalid_ids: list[str]
    sample_skipped_ids: list[str]

    @property
    def has_issues(self) -> bool:
        return (
            self.invalid_documents > 0
            or self.invalid_article_shapes > 0
            or self.missing_id_documents > 0
        )


def _iter_json_items(source_path: Path) -> Iterator[dict[str, Any]]:
    """JSON 스트리밍 로드 (dict 항목만 전달)"""
    import ijson

    with open(source_path, "rb") as f:
        for item in ijson.items(f, "item"):
            if isinstance(item, dict):
                yield item


def _first_non_empty_text(item: dict[str, Any], keys: list[str]) -> str:
    """여러 후보 키 중 첫 번째 비어있지 않은 문자열 반환"""
    for key in keys:
        value = item.get(key)
        if isinstance(value, str):
            value = value.strip()
            if value:
                return value
    return ""


def _first_non_empty_text_or_value(item: dict[str, Any], keys: list[str]) -> str:
    """여러 후보 키 중 첫 번째 값 반환(문자열/숫자 지원)."""
    for key in keys:
        value = item.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            value = value.strip()
            if value:
                return value
        else:
            text = str(value).strip()
            if text:
                return text
    return ""


def _validate_multi_vector_source(
    config_name: str,
    source_path: Path,
    id_fields: list[str],
    summary_fields: list[str],
    articles_key: str,
    article_summary_fields: list[str],
    sample_limit: int = 20,
) -> SourceValidationSummary:
    """
    다중 벡터 타입(1문서=요약 N개) 기대 벡터 수 + 기본 구조 검증.

    Returns:
        SourceValidationSummary
    """
    total_documents = 0
    expected_vectors = 0
    skipped_no_summary = 0
    invalid_documents = 0
    invalid_article_shapes = 0
    missing_id = 0
    sample_invalid_ids: list[str] = []
    sample_skipped_ids: list[str] = []

    for item in _iter_json_items(source_path):
        total_documents += 1
        doc_id = _first_non_empty_text_or_value(item, id_fields).strip()
        if not doc_id:
            missing_id += 1
            doc_id = f"index-{total_documents}"

        vectors_for_doc = 0

        overall = _first_non_empty_text(item, summary_fields)
        if overall:
            vectors_for_doc += 1

        articles = item.get(articles_key, [])
        if articles is None:
            articles = []
        elif not isinstance(articles, list):
            if len(sample_invalid_ids) < sample_limit and doc_id not in sample_invalid_ids:
                sample_invalid_ids.append(doc_id)
            invalid_article_shapes += 1
            articles = []

        for article in articles:
            if isinstance(article, dict):
                article_summary = _first_non_empty_text(
                    article, article_summary_fields
                )
                if article_summary:
                    vectors_for_doc += 1
            else:
                invalid_article_shapes += 1

        expected_vectors += vectors_for_doc
        if vectors_for_doc == 0:
            skipped_no_summary += 1
            if len(sample_skipped_ids) < sample_limit:
                sample_skipped_ids.append(doc_id)

    return SourceValidationSummary(
        config_name=config_name,
        source_path=str(source_path),
        total_documents=total_documents,
        expected_vectors=expected_vectors,
        skipped_no_summary_documents=skipped_no_summary,
        invalid_documents=invalid_documents,
        invalid_article_shapes=invalid_article_shapes,
        missing_id_documents=missing_id,
        sample_invalid_ids=sample_invalid_ids,
        sample_skipped_ids=sample_skipped_ids,
    )


def validate_law_source(source_path: Path) -> SourceValidationSummary:
    """법령 입력 구조 + 예상 벡터 수 검증"""
    return _validate_multi_vector_source(
        config_name="law",
        source_path=source_path,
        id_fields=["법령ID", "law_id"],
        summary_fields=["법령 요약", "ai_summary"],
        articles_key="조문",
        article_summary_fields=["조문요약"],
    )


def validate_local_ordinance_source(source_path: Path) -> SourceValidationSummary:
    """자치법규 입력 구조 + 예상 벡터 수 검증"""
    return _validate_multi_vector_source(
        config_name="local_ordinance",
        source_path=source_path,
        id_fields=["자치법규ID", "ordinance_id"],
        summary_fields=["전체요약"],
        articles_key="조",
        article_summary_fields=["조문요약"],
    )


def validate_source(
    config_name: str,
    source_path: Path,
    id_field: str,
    summary_fields: list[str],
) -> SourceValidationSummary:
    """
    타입별 소스 입력 검증.

    현재는 법령/자치법규의 다중벡터 산출만 정밀 지원.
    """
    if config_name == "law":
        return validate_law_source(source_path)
    if config_name == "local_ordinance":
        return validate_local_ordinance_source(source_path)

    return _validate_single_vector_source(
        config_name=config_name,
        source_path=source_path,
        id_fields=[id_field],
        summary_fields=summary_fields,
    )


def _validate_single_vector_source(
    config_name: str,
    source_path: Path,
    id_fields: list[str],
    summary_fields: list[str],
    sample_limit: int = 20,
) -> SourceValidationSummary:
    """일반 1문서=1벡터 타입의 기본 검증"""
    total_documents = 0
    expected_vectors = 0
    skipped_no_summary = 0
    missing_id = 0
    sample_skipped_ids: list[str] = []

    for item in _iter_json_items(source_path):
        total_documents += 1
        doc_id = _first_non_empty_text_or_value(item, id_fields).strip()
        if not doc_id:
            missing_id += 1
            doc_id = f"index-{total_documents}"

        if _first_non_empty_text(item, summary_fields):
            expected_vectors += 1
        else:
            skipped_no_summary += 1
            if len(sample_skipped_ids) < sample_limit:
                sample_skipped_ids.append(doc_id)

    return SourceValidationSummary(
        config_name=config_name,
        source_path=str(source_path),
        total_documents=total_documents,
        expected_vectors=expected_vectors,
        skipped_no_summary_documents=skipped_no_summary,
        invalid_documents=0,
        invalid_article_shapes=0,
        missing_id_documents=missing_id,
        sample_invalid_ids=[],
        sample_skipped_ids=sample_skipped_ids,
    )


def format_validation_summary(summary: SourceValidationSummary) -> str:
    """CLI 출력용 요약 문자열 생성"""
    lines = [
        f"Type: {summary.config_name}",
        f"Source: {summary.source_path}",
        f"문서 수: {summary.total_documents:,}",
        f"예상 벡터 수: {summary.expected_vectors:,}",
        f"요약 없음(예상 스킵): {summary.skipped_no_summary_documents:,}",
    ]

    if summary.has_issues or summary.sample_invalid_ids or summary.sample_skipped_ids:
        lines.append(
            f"문서 ID 미스: {summary.missing_id_documents}, "
            f"잘못된 조문 구조: {summary.invalid_article_shapes}"
        )
        if summary.sample_invalid_ids:
            lines.append(f"  샘플 이상치 ID: {', '.join(summary.sample_invalid_ids)}")
        if summary.sample_skipped_ids:
            lines.append(
                f"  샘플 요약 없음 ID: {', '.join(summary.sample_skipped_ids)}"
            )

    return "\n".join(lines)
