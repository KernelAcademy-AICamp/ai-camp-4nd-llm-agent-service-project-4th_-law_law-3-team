"""
리포트 생성기

실험 결과를 마크다운 또는 HTML 리포트로 생성

Usage:
    generator = ReportGenerator()
    report = generator.generate(experiment_id="EXP-20260129-001")
    generator.save(report, "report.md")
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

from evaluation.schemas import ExperimentResult
from evaluation.config import eval_settings, PerformanceTargets
from evaluation.runners.experiment_tracker import ExperimentTracker


class ReportGenerator:
    """
    실험 리포트 생성기

    마크다운 및 HTML 형식의 상세 리포트 생성
    """

    def __init__(self):
        self.reports_dir = eval_settings.reports_dir

    def generate(
        self,
        experiment_id: str,
        format: str = "markdown",
    ) -> str:
        """
        리포트 생성

        Args:
            experiment_id: 실험 ID
            format: 출력 형식 (markdown, html)

        Returns:
            생성된 리포트 텍스트
        """
        result = ExperimentTracker.load_experiment(experiment_id)
        if not result:
            raise ValueError(f"실험을 찾을 수 없습니다: {experiment_id}")

        if format == "markdown":
            return self._generate_markdown(result)
        elif format == "html":
            return self._generate_html(result)
        else:
            raise ValueError(f"지원하지 않는 형식: {format}")

    def _generate_markdown(self, result: ExperimentResult) -> str:
        """마크다운 리포트 생성"""
        targets = PerformanceTargets.check_metrics({
            "recall_at_5": result.metrics.recall_at_5,
            "recall_at_10": result.metrics.recall_at_10,
            "mrr": result.metrics.mrr,
            "hit_rate": result.metrics.hit_rate,
            "ndcg_at_10": result.metrics.ndcg_at_10,
            "latency_p50_ms": result.metrics.latency_p50_ms,
            "latency_p95_ms": result.metrics.latency_p95_ms,
        })

        achieved = sum(1 for v in targets.values() if v)
        total = len(targets)

        lines = [
            f"# RAG 평가 리포트: {result.config.experiment_id}",
            "",
            f"**생성일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            "",
            "## 1. 실험 개요",
            "",
            f"| 항목 | 값 |",
            f"|------|-----|",
            f"| 실험 ID | {result.config.experiment_id} |",
            f"| 실험명 | {result.config.name} |",
            f"| 데이터셋 | {result.config.dataset_path} |",
            f"| 임베딩 모델 | {result.config.embedding_model} |",
            f"| 거리 메트릭 | {result.config.distance_metric} |",
            f"| Top K | {result.config.top_k} |",
            f"| 시작 시간 | {result.started_at} |",
            f"| 완료 시간 | {result.completed_at} |",
            "",
            "---",
            "",
            "## 2. 전체 성능 메트릭",
            "",
            "| 메트릭 | 값 | 목표 | 달성 |",
            "|--------|-----|------|------|",
            f"| Recall@5 | {result.metrics.recall_at_5:.4f} | ≥ {PerformanceTargets.RECALL_AT_5} | {'✅' if targets['recall_at_5'] else '❌'} |",
            f"| Recall@10 | {result.metrics.recall_at_10:.4f} | ≥ {PerformanceTargets.RECALL_AT_10} | {'✅' if targets['recall_at_10'] else '❌'} |",
            f"| MRR | {result.metrics.mrr:.4f} | ≥ {PerformanceTargets.MRR} | {'✅' if targets['mrr'] else '❌'} |",
            f"| Hit Rate | {result.metrics.hit_rate:.4f} | ≥ {PerformanceTargets.HIT_RATE} | {'✅' if targets['hit_rate'] else '❌'} |",
            f"| NDCG@10 | {result.metrics.ndcg_at_10:.4f} | ≥ {PerformanceTargets.NDCG_AT_10} | {'✅' if targets['ndcg_at_10'] else '❌'} |",
            f"| Latency P50 | {result.metrics.latency_p50_ms:.2f}ms | ≤ {PerformanceTargets.LATENCY_P50_MS}ms | {'✅' if targets['latency_p50'] else '❌'} |",
            f"| Latency P95 | {result.metrics.latency_p95_ms:.2f}ms | ≤ {PerformanceTargets.LATENCY_P95_MS}ms | {'✅' if targets['latency_p95'] else '❌'} |",
            "",
            f"**목표 달성률**: {achieved}/{total} ({100*achieved/total:.1f}%)",
            "",
            "---",
            "",
            "## 3. 쿼리 유형별 분석",
            "",
        ]

        if result.metrics_by_type:
            lines.extend([
                "| 유형 | 쿼리 수 | Recall@5 | Recall@10 | MRR | Hit Rate | NDCG@10 |",
                "|------|---------|----------|-----------|-----|----------|---------|",
            ])
            for qtype, metrics in result.metrics_by_type.items():
                count = sum(
                    1 for qr in result.query_results
                    if any(qr.query_id in q.id for q in result.query_results)
                )
                lines.append(
                    f"| {qtype} | - | {metrics.recall_at_5:.4f} | {metrics.recall_at_10:.4f} | "
                    f"{metrics.mrr:.4f} | {metrics.hit_rate:.4f} | {metrics.ndcg_at_10:.4f} |"
                )
        else:
            lines.append("데이터 없음")

        lines.extend([
            "",
            "---",
            "",
            "## 4. 카테고리별 분석",
            "",
        ])

        if result.metrics_by_category:
            lines.extend([
                "| 카테고리 | Recall@5 | Recall@10 | MRR | Hit Rate | NDCG@10 |",
                "|----------|----------|-----------|-----|----------|---------|",
            ])
            for cat, metrics in result.metrics_by_category.items():
                lines.append(
                    f"| {cat} | {metrics.recall_at_5:.4f} | {metrics.recall_at_10:.4f} | "
                    f"{metrics.mrr:.4f} | {metrics.hit_rate:.4f} | {metrics.ndcg_at_10:.4f} |"
                )
        else:
            lines.append("데이터 없음")

        lines.extend([
            "",
            "---",
            "",
            "## 5. 저성능 쿼리 분석",
            "",
        ])

        low_performers = [
            qr for qr in result.query_results
            if qr.recall_at_10 < 0.5
        ]

        if low_performers:
            lines.append(f"**저성능 쿼리 수**: {len(low_performers)}개 (Recall@10 < 0.5)")
            lines.append("")
            for qr in low_performers[:10]:
                lines.append(f"### {qr.query_id}")
                lines.append(f"**질문**: {qr.question}")
                lines.append(f"- Recall@10: {qr.recall_at_10:.4f}")
                lines.append(f"- MRR: {qr.mrr:.4f}")
                lines.append(f"- Hit: {'Yes' if qr.hit else 'No'}")
                lines.append("")
        else:
            lines.append("저성능 쿼리 없음")

        if result.failed_queries:
            lines.extend([
                "",
                "---",
                "",
                "## 6. 실패한 쿼리",
                "",
            ])
            for failed in result.failed_queries:
                lines.append(f"- **{failed['query_id']}**: {failed['error']}")

        lines.extend([
            "",
            "---",
            "",
            "## 7. 권장 사항",
            "",
        ])

        recommendations = []
        if not targets["recall_at_5"]:
            recommendations.append("- Recall@5 개선 필요: 청킹 전략 검토 또는 임베딩 모델 튜닝 고려")
        if not targets["mrr"]:
            recommendations.append("- MRR 개선 필요: 검색 결과 재순위화(reranking) 도입 고려")
        if not targets["latency_p95"]:
            recommendations.append("- 응답 시간 개선 필요: 인덱스 최적화 또는 캐싱 도입 고려")

        if recommendations:
            lines.extend(recommendations)
        else:
            lines.append("모든 목표 달성! 현재 설정 유지 권장")

        lines.extend([
            "",
            "---",
            "",
            "*이 리포트는 RAG 평가 시스템에 의해 자동 생성되었습니다.*",
        ])

        return "\n".join(lines)

    def _generate_html(self, result: ExperimentResult) -> str:
        """HTML 리포트 생성"""
        md_content = self._generate_markdown(result)

        html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG 평가 리포트: {result.config.experiment_id}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f4f4f4; }}
        code {{ background-color: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
        pre {{ background-color: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        h1 {{ color: #333; border-bottom: 2px solid #333; padding-bottom: 10px; }}
        h2 {{ color: #555; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
        hr {{ border: none; border-top: 1px solid #ddd; margin: 30px 0; }}
    </style>
</head>
<body>
    <pre>{md_content}</pre>
</body>
</html>"""
        return html

    def save(
        self,
        content: str,
        filename: str,
        path: Optional[Path] = None,
    ) -> Path:
        """
        리포트 저장

        Args:
            content: 리포트 내용
            filename: 파일명
            path: 저장 경로 (None이면 기본 경로)

        Returns:
            저장된 파일 경로
        """
        if path is None:
            path = self.reports_dir / filename

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        return path

    def generate_and_save(
        self,
        experiment_id: str,
        format: str = "markdown",
    ) -> Path:
        """
        리포트 생성 및 저장

        Args:
            experiment_id: 실험 ID
            format: 출력 형식 (markdown, html)

        Returns:
            저장된 파일 경로
        """
        content = self.generate(experiment_id, format)
        ext = "md" if format == "markdown" else "html"
        filename = f"{experiment_id}_report.{ext}"
        return self.save(content, filename)
