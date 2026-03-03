"""OpenTelemetry 기반 파이프라인 관측성 (v0.3.0)

Consultant 피드백: 단계별 통계만으로는 운영 가시성 부족.
run/article 단위 Trace ID 연결로 병목 구간 즉시 파악 가능.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from app.core.config import settings

logger = logging.getLogger(__name__)

# 전역 트레이서 (lazy init)
_tracer = None


def _init_tracer() -> Any:
    """OpenTelemetry TracerProvider 초기화 (최초 1회)"""
    global _tracer  # noqa: PLW0603

    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create({
        "service.name": "news-pipeline",
        "service.version": "0.3.0",
    })

    provider = TracerProvider(resource=resource)

    from opentelemetry.sdk.trace.export import SpanExporter

    otel_endpoint = getattr(settings, "NEWS_PIPELINE_OTEL_ENDPOINT", "")
    span_exporter: SpanExporter
    if otel_endpoint:
        # OTLP gRPC exporter
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        span_exporter = OTLPSpanExporter(endpoint=otel_endpoint)
    else:
        # 콘솔 출력 (개발/로컬용)
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter
        span_exporter = ConsoleSpanExporter()

    provider.add_span_processor(BatchSpanProcessor(span_exporter))
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer("news_pipeline", "0.3.0")

    logger.info("OpenTelemetry 초기화 완료 (endpoint=%s)", otel_endpoint or "console")
    return _tracer


def get_tracer() -> Any:
    """트레이서 인스턴스 반환 (lazy init)"""
    global _tracer  # noqa: PLW0603
    if _tracer is None:
        return _init_tracer()
    return _tracer


@asynccontextmanager
async def pipeline_span(
    name: str,
    *,
    run_id: str = "",
    article_url: str = "",
    attributes: dict[str, str] | None = None,
) -> AsyncIterator[Any]:
    """파이프라인 단계별 span 컨텍스트 매니저

    사용 예:
        async with pipeline_span("collect", run_id=run_id):
            articles = await source.fetch(target_date)
    """
    from opentelemetry import trace

    tracer = get_tracer()
    span_attrs: dict[str, str] = {
        "pipeline.run_id": run_id,
    }
    if article_url:
        span_attrs["pipeline.article_url"] = article_url
    if attributes:
        span_attrs.update(attributes)

    with tracer.start_as_current_span(name, attributes=span_attrs) as span:
        try:
            yield span
        except Exception as exc:
            span.set_status(trace.StatusCode.ERROR, str(exc))
            span.record_exception(exc)
            raise


def record_metric(span: Any, key: str, value: int | float) -> None:
    """span에 커스텀 메트릭 기록"""
    if span is not None:
        span.set_attribute(f"pipeline.metric.{key}", value)
