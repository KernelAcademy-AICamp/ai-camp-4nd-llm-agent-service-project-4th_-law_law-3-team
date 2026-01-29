"""
실험 추적기

실험 결과를 저장하고 분석하는 기능 제공

디렉토리 구조:
    experiments/
    └── EXP-YYYYMMDD-NNN/
        ├── config.yaml
        ├── results.yaml
        └── analysis.md
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from evaluation.schemas import ExperimentConfig, ExperimentResult
from evaluation.config import eval_settings, PerformanceTargets


class ExperimentTracker:
    """
    실험 추적기

    실험 설정, 결과, 로그를 관리

    Usage:
        tracker = ExperimentTracker("EXP-20260129-001")
        tracker.log_start(config)
        tracker.log_query_result(query_id, metrics)
        tracker.log_completion(result)
        tracker.save_result(result)
    """

    def __init__(self, experiment_id: str):
        """
        Args:
            experiment_id: 실험 ID (EXP-YYYYMMDD-NNN 형식)
        """
        self.experiment_id = experiment_id
        self.experiment_dir = eval_settings.experiments_dir / experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self._logs: list[dict] = []

    def log_start(self, config: ExperimentConfig) -> None:
        """실험 시작 로그"""
        self._logs.append({
            "event": "start",
            "timestamp": datetime.now().isoformat(),
            "config": config.model_dump(mode="json"),
        })

        config_path = self.experiment_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(
                config.model_dump(mode="json"),
                f,
                allow_unicode=True,
                default_flow_style=False,
            )

    def log_query_result(self, query_id: str, metrics: dict) -> None:
        """쿼리 결과 로그"""
        self._logs.append({
            "event": "query_result",
            "timestamp": datetime.now().isoformat(),
            "query_id": query_id,
            "metrics": metrics,
        })

    def log_error(self, query_id: str, error: str) -> None:
        """에러 로그"""
        self._logs.append({
            "event": "error",
            "timestamp": datetime.now().isoformat(),
            "query_id": query_id,
            "error": error,
        })

    def log_completion(self, result: ExperimentResult) -> None:
        """실험 완료 로그"""
        self._logs.append({
            "event": "completion",
            "timestamp": datetime.now().isoformat(),
            "metrics": result.metrics.model_dump(),
        })

    def save_result(self, result: ExperimentResult) -> Path:
        """
        결과 저장

        Args:
            result: ExperimentResult

        Returns:
            저장된 파일 경로
        """
        results_path = self.experiment_dir / "results.yaml"
        with open(results_path, "w", encoding="utf-8") as f:
            yaml.dump(
                result.model_dump(mode="json"),
                f,
                allow_unicode=True,
                default_flow_style=False,
                default=str,
            )

        analysis_path = self.experiment_dir / "analysis.md"
        self._generate_analysis(result, analysis_path)

        logs_path = self.experiment_dir / "logs.json"
        with open(logs_path, "w", encoding="utf-8") as f:
            json.dump(self._logs, f, ensure_ascii=False, indent=2, default=str)

        return results_path

    def _generate_analysis(self, result: ExperimentResult, path: Path) -> None:
        """분석 마크다운 생성"""
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
            f"# 실험 분석: {self.experiment_id}",
            "",
            f"**실행 시간**: {result.started_at} ~ {result.completed_at}",
            "",
            "## 전체 메트릭",
            "",
            "| 메트릭 | 값 | 목표 | 달성 |",
            "|--------|-----|------|------|",
            f"| Recall@5 | {result.metrics.recall_at_5:.4f} | ≥ {PerformanceTargets.RECALL_AT_5} | {'✓' if targets['recall_at_5'] else '✗'} |",
            f"| Recall@10 | {result.metrics.recall_at_10:.4f} | ≥ {PerformanceTargets.RECALL_AT_10} | {'✓' if targets['recall_at_10'] else '✗'} |",
            f"| MRR | {result.metrics.mrr:.4f} | ≥ {PerformanceTargets.MRR} | {'✓' if targets['mrr'] else '✗'} |",
            f"| Hit Rate | {result.metrics.hit_rate:.4f} | ≥ {PerformanceTargets.HIT_RATE} | {'✓' if targets['hit_rate'] else '✗'} |",
            f"| NDCG@10 | {result.metrics.ndcg_at_10:.4f} | ≥ {PerformanceTargets.NDCG_AT_10} | {'✓' if targets['ndcg_at_10'] else '✗'} |",
            f"| Latency P50 | {result.metrics.latency_p50_ms:.2f}ms | ≤ {PerformanceTargets.LATENCY_P50_MS}ms | {'✓' if targets['latency_p50'] else '✗'} |",
            f"| Latency P95 | {result.metrics.latency_p95_ms:.2f}ms | ≤ {PerformanceTargets.LATENCY_P95_MS}ms | {'✓' if targets['latency_p95'] else '✗'} |",
            "",
            f"**목표 달성률**: {achieved}/{total} ({100*achieved/total:.1f}%)",
            "",
        ]

        if result.metrics_by_type:
            lines.extend([
                "## 쿼리 유형별 메트릭",
                "",
                "| 유형 | Recall@5 | Recall@10 | MRR | Hit Rate |",
                "|------|----------|-----------|-----|----------|",
            ])
            for qtype, metrics in result.metrics_by_type.items():
                lines.append(
                    f"| {qtype} | {metrics.recall_at_5:.4f} | {metrics.recall_at_10:.4f} | "
                    f"{metrics.mrr:.4f} | {metrics.hit_rate:.4f} |"
                )
            lines.append("")

        if result.metrics_by_category:
            lines.extend([
                "## 카테고리별 메트릭",
                "",
                "| 카테고리 | Recall@5 | Recall@10 | MRR | Hit Rate |",
                "|----------|----------|-----------|-----|----------|",
            ])
            for cat, metrics in result.metrics_by_category.items():
                lines.append(
                    f"| {cat} | {metrics.recall_at_5:.4f} | {metrics.recall_at_10:.4f} | "
                    f"{metrics.mrr:.4f} | {metrics.hit_rate:.4f} |"
                )
            lines.append("")

        if result.failed_queries:
            lines.extend([
                "## 실패 분석",
                "",
            ])
            for failed in result.failed_queries[:10]:
                lines.append(f"- **{failed['query_id']}**: {failed['error']}")
            lines.append("")

        low_performers = [
            qr for qr in result.query_results
            if qr.recall_at_10 < 0.5
        ]
        if low_performers:
            lines.extend([
                "## 저성능 쿼리 (Recall@10 < 0.5)",
                "",
            ])
            for qr in low_performers[:10]:
                lines.append(f"- **{qr.query_id}**: {qr.question[:50]}...")
                lines.append(f"  - Recall@10: {qr.recall_at_10:.4f}, MRR: {qr.mrr:.4f}")
            lines.append("")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    @classmethod
    def list_experiments(cls) -> list[dict]:
        """모든 실험 목록"""
        experiments = []
        for exp_dir in eval_settings.experiments_dir.iterdir():
            if exp_dir.is_dir() and exp_dir.name.startswith("EXP-"):
                config_path = exp_dir / "config.yaml"
                results_path = exp_dir / "results.yaml"

                exp_info = {
                    "id": exp_dir.name,
                    "path": str(exp_dir),
                    "has_config": config_path.exists(),
                    "has_results": results_path.exists(),
                }

                if results_path.exists():
                    with open(results_path, "r", encoding="utf-8") as f:
                        results = yaml.safe_load(f)
                        if results and "metrics" in results:
                            exp_info["metrics"] = results["metrics"]
                        if results and "completed_at" in results:
                            exp_info["completed_at"] = results["completed_at"]

                experiments.append(exp_info)

        experiments.sort(key=lambda x: x["id"], reverse=True)
        return experiments

    @classmethod
    def load_experiment(cls, experiment_id: str) -> Optional[ExperimentResult]:
        """실험 결과 로드"""
        results_path = eval_settings.experiments_dir / experiment_id / "results.yaml"
        if not results_path.exists():
            return None

        with open(results_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return ExperimentResult.model_validate(data)

    @classmethod
    def generate_next_id(cls) -> str:
        """다음 실험 ID 생성"""
        today = datetime.now().strftime("%Y%m%d")
        prefix = f"EXP-{today}-"

        existing = [
            exp["id"] for exp in cls.list_experiments()
            if exp["id"].startswith(prefix)
        ]

        if not existing:
            return f"{prefix}001"

        max_num = max(int(exp_id.split("-")[-1]) for exp_id in existing)
        return f"{prefix}{max_num + 1:03d}"
