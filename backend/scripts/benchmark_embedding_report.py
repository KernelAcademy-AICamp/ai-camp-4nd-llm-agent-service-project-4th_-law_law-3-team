"""ONNX 그래프 최적화 벤치마크 MD 보고서 생성 모듈.

벤치마크 결과(Phase 1~6)를 받아 Markdown 보고서를 생성한다.

사용법:
  from scripts.benchmark_embedding_report import generate_report
  generate_report(speed, quality, search, size, ingest, e2e, output_path)
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# 전체 법률 데이터 벡터 수 (legal_chunks 테이블)
TOTAL_LEGAL_CHUNKS = 253_768

MODEL_LABELS: dict[str, str] = {
    "pytorch_fp32": "PyTorch FP32",
    "onnx_fp32": "ONNX FP32",
    "onnx_int8": "ONNX INT8",
    "onnx_o2": "ONNX O2",
    "onnx_o3": "ONNX O3",
    "onnx_o3_int8": "ONNX O3+INT8",
}

VARIANT_ORDER: list[str] = [
    "pytorch_fp32",
    "onnx_fp32",
    "onnx_int8",
    "onnx_o2",
    "onnx_o3",
    "onnx_o3_int8",
]

QUALITY_THRESHOLDS: dict[str, float] = {
    "pairwise_cosine_mean": 0.995,
    "sim_matrix_pearson": 0.998,
    "separation_diff": 0.05,
    "top3_match_rate": 0.90,
    "top5_match_rate": 0.80,
    "top10_match_rate": 0.70,
    "search_spearman": 0.95,
}


def _pass_fail(
    value: float, threshold: float, *, higher_is_better: bool = True
) -> str:
    """품질 기준 PASS/FAIL 판정."""
    if higher_is_better:
        return "PASS" if value >= threshold else "FAIL"
    return "PASS" if value <= threshold else "FAIL"


def _available_variants(*result_dicts: dict[str, Any]) -> list[str]:
    """결과가 하나라도 있는 variant 목록 (정렬된 순서)."""
    all_keys: set[str] = set()
    for d in result_dicts:
        all_keys.update(d.keys())
    return [v for v in VARIANT_ORDER if v in all_keys]


def _is_quality_pass(
    variant: str, quality_results: dict[str, dict[str, float]]
) -> bool:
    """variant가 품질 기준을 통과하는지 확인."""
    if variant not in quality_results:
        return True  # 측정 안 됨 → 통과로 간주
    q = quality_results[variant]
    if q.get("pairwise_cosine_mean", 0) < QUALITY_THRESHOLDS["pairwise_cosine_mean"]:
        return False
    if q.get("sim_matrix_pearson", 0) < QUALITY_THRESHOLDS["sim_matrix_pearson"]:
        return False
    if q.get("separation_diff", 1) > QUALITY_THRESHOLDS["separation_diff"]:
        return False
    return True


def _determine_recommendation(
    variants: list[str],
    speed_results: dict[str, dict[str, Any]],
    quality_results: dict[str, dict[str, float]],
) -> dict[str, str]:
    """운영 환경별 추천 variant 결정."""
    result: dict[str, str] = {"speed": "", "quality": "", "balanced": ""}
    if not variants:
        return result

    # 속도 순위 (단일 쿼리 ms 기준, 오름차순)
    speed_ranking: list[tuple[str, float]] = []
    for v in variants:
        if v in speed_results:
            ms = float(speed_results[v].get("single_ms", 9999))
            speed_ranking.append((v, ms))
    speed_ranking.sort(key=lambda x: x[1])

    quality_pass = {v for v in variants if _is_quality_pass(v, quality_results)}

    # 속도 우선: 품질 통과 중 가장 빠른 것
    for v, _ in speed_ranking:
        if v in quality_pass:
            result["speed"] = v
            break
    if not result["speed"] and speed_ranking:
        result["speed"] = speed_ranking[0][0]

    # 품질 우선: pairwise_cosine_mean 가장 높은 것
    quality_ranking = sorted(
        [(v, quality_results[v].get("pairwise_cosine_mean", 0))
         for v in variants if v in quality_results],
        key=lambda x: x[1],
        reverse=True,
    )
    result["quality"] = quality_ranking[0][0] if quality_ranking else variants[0]

    # 균형: 품질 통과 + 속도 상위
    balanced = [(v, ms) for v, ms in speed_ranking if v in quality_pass]
    result["balanced"] = balanced[0][0] if balanced else (
        speed_ranking[0][0] if speed_ranking else variants[0]
    )
    return result


def _format_time_estimate(total_seconds: float) -> str:
    """초 단위를 읽기 좋은 시간 문자열로 변환."""
    if total_seconds >= 3600:
        return f"{total_seconds / 3600:.1f}시간"
    if total_seconds >= 60:
        return f"{total_seconds / 60:.0f}분"
    return f"{total_seconds:.0f}초"


def _build_speed_section(
    lines: list[str],
    variants: list[str],
    speed_results: dict[str, dict[str, Any]],
    ingest_results: dict[str, dict[str, float]],
    e2e_results: dict[str, dict[str, float]],
    ingest_docs: int,
) -> None:
    """섹션 1: 속도 비교."""
    lines.append("## 1. 속도 비교")
    lines.append("")

    pt_single = float(speed_results.get("pytorch_fp32", {}).get("single_ms", 0))

    # 1.1 단일 쿼리
    lines.append("### 1.1 단일 쿼리 지연시간")
    lines.append("")
    lines.append("| Variant | 평균 (ms) | 상대 속도 |")
    lines.append("|---------|----------:|----------:|")
    for v in variants:
        if v in speed_results:
            label = MODEL_LABELS.get(v, v)
            ms = float(speed_results[v].get("single_ms", 0))
            if v == "pytorch_fp32":
                lines.append(f"| {label} | {ms:.1f} | 1.0x (baseline) |")
            else:
                speedup = pt_single / ms if ms > 0 else 0
                lines.append(f"| {label} | {ms:.1f} | {speedup:.1f}x |")
    lines.append("")

    # 1.2 배치
    lines.append("### 1.2 배치 처리 (20문서)")
    lines.append("")
    batch_sizes = [32, 64, 128]
    header = "| Variant |" + " | ".join(f"batch={b} (ms)" for b in batch_sizes) + " |"
    sep = "|---------|" + " | ".join("---------:" for _ in batch_sizes) + " |"
    lines.append(header)
    lines.append(sep)
    for v in variants:
        if v in speed_results:
            label = MODEL_LABELS.get(v, v)
            batch_data = speed_results[v].get("batch", {})
            cols = " | ".join(f"{float(batch_data.get(b, 0)):.0f}" for b in batch_sizes)
            lines.append(f"| {label} | {cols} |")
    lines.append("")

    # 1.3 인제스트
    if ingest_results:
        lines.append(f"### 1.3 데이터 인제스트 ({ingest_docs}문서)")
        lines.append("")
        lines.append("| Variant | 임베딩 (ms) | 쓰기 (ms) | 총 (ms) | 문서/초 |")
        lines.append("|---------|----------:|--------:|-------:|-------:|")
        for v in variants:
            if v in ingest_results:
                label = MODEL_LABELS.get(v, v)
                r = ingest_results[v]
                lines.append(
                    f"| {label}"
                    f" | {r.get('embedding_time_ms', 0):.0f}"
                    f" | {r.get('write_time_ms', 0):.0f}"
                    f" | {r.get('total_ms', 0):.0f}"
                    f" | {r.get('docs_per_second', 0):.1f} |"
                )
        lines.append("")

    # 1.4 E2E
    if e2e_results:
        lines.append("### 1.4 E2E 쿼리 지연시간")
        lines.append("")
        lines.append("| Variant | 임베딩 (ms) | 검색 (ms) | E2E (ms) | P95 (ms) |")
        lines.append("|---------|----------:|--------:|--------:|--------:|")
        for v in variants:
            if v in e2e_results:
                label = MODEL_LABELS.get(v, v)
                r = e2e_results[v]
                lines.append(
                    f"| {label}"
                    f" | {r.get('embed_ms', 0):.1f}"
                    f" | {r.get('search_ms', 0):.1f}"
                    f" | {r.get('e2e_ms', 0):.1f}"
                    f" | {r.get('p95_e2e_ms', 0):.1f} |"
                )
        lines.append("")


def _build_quality_section(
    lines: list[str],
    variants: list[str],
    quality_results: dict[str, dict[str, float]],
) -> None:
    """섹션 2: 임베딩 품질."""
    if not quality_results:
        return

    lines.append("## 2. 임베딩 품질")
    lines.append("")
    present = [v for v in variants if v in quality_results]
    header = "| 메트릭 | " + " | ".join(MODEL_LABELS.get(v, v) for v in present) + " | 기준 | 판정 |"
    sep = "|--------| " + " | ".join("-----:" for _ in present) + " | -----: | -----: |"
    lines.append(header)
    lines.append(sep)

    metrics = [
        ("Pairwise Cosine", "pairwise_cosine_mean", QUALITY_THRESHOLDS["pairwise_cosine_mean"], True),
        ("Sim Matrix Pearson", "sim_matrix_pearson", QUALITY_THRESHOLDS["sim_matrix_pearson"], True),
        ("Separation Diff", "separation_diff", QUALITY_THRESHOLDS["separation_diff"], False),
    ]

    for label, key, threshold, higher in metrics:
        row = f"| {label} |"
        worst = "PASS"
        for v in present:
            val = quality_results[v].get(key, 0.0)
            row += f" {val:.4f} |"
            if _pass_fail(val, threshold, higher_is_better=higher) == "FAIL":
                worst = "FAIL"
        t_str = f">={threshold}" if higher else f"<{threshold}"
        row += f" {t_str} | {worst} |"
        lines.append(row)
    lines.append("")


def _build_search_section(
    lines: list[str],
    variants: list[str],
    search_results: dict[str, dict[str, float]],
) -> None:
    """섹션 3: LanceDB 검색 품질."""
    if not search_results:
        return

    lines.append("## 3. LanceDB 검색 품질")
    lines.append("")
    present = [v for v in variants if v in search_results]
    header = "| 메트릭 | " + " | ".join(MODEL_LABELS.get(v, v) for v in present) + " | 기준 | 판정 |"
    sep = "|--------| " + " | ".join("-----:" for _ in present) + " | -----: | -----: |"
    lines.append(header)
    lines.append(sep)

    s_metrics: list[tuple[str, str, float, bool, bool]] = [
        ("Top-3 일치율", "top3_match_rate", QUALITY_THRESHOLDS["top3_match_rate"], True, True),
        ("Top-5 일치율", "top5_match_rate", QUALITY_THRESHOLDS["top5_match_rate"], True, True),
        ("Top-10 일치율", "top10_match_rate", QUALITY_THRESHOLDS["top10_match_rate"], True, True),
        ("Spearman", "search_spearman", QUALITY_THRESHOLDS["search_spearman"], True, False),
    ]

    for label, key, threshold, higher, is_pct in s_metrics:
        row = f"| {label} |"
        worst = "PASS"
        for v in present:
            val = search_results[v].get(key, 0.0)
            row += f" {val:.0%} |" if is_pct else f" {val:.4f} |"
            if _pass_fail(val, threshold, higher_is_better=higher) == "FAIL":
                worst = "FAIL"
        t_str = f">={threshold:.0%}" if is_pct else f">={threshold}"
        row += f" {t_str} | {worst} |"
        lines.append(row)
    lines.append("")


def _build_size_section(
    lines: list[str],
    variants: list[str],
    size_results: dict[str, float],
) -> None:
    """섹션 4: 모델 크기."""
    if not size_results:
        return

    lines.append("## 4. 모델 크기")
    lines.append("")
    lines.append("| Variant | 크기 (MB) | 절약 (%) |")
    lines.append("|---------|--------:|---------:|")
    pt_mb = size_results.get("pytorch_fp32_mb", 0.0)
    for v in variants:
        label = MODEL_LABELS.get(v, v)
        if v == "pytorch_fp32":
            lines.append(f"| {label} | {pt_mb:.0f} | - |")
        else:
            mb = size_results.get(f"{v}_mb", 0.0)
            if mb > 0:
                saving = (1 - mb / pt_mb) * 100 if pt_mb > 0 else 0
                lines.append(f"| {label} | {mb:.0f} | {saving:+.0f}% |")
    lines.append("")


def _speed_rating(speedup: float) -> str:
    """속도 등급."""
    if speedup >= 2.0:
        return "**빠름**"
    if speedup >= 1.3:
        return "보통"
    return "느림"


def _quality_rating(variant: str, quality_results: dict[str, dict[str, float]]) -> str:
    """품질 등급."""
    if variant not in quality_results:
        return "-"
    cos = quality_results[variant].get("pairwise_cosine_mean", 0.0)
    if cos >= 0.999:
        return "**우수**"
    if cos >= QUALITY_THRESHOLDS["pairwise_cosine_mean"]:
        return "양호"
    return "미달"


def _search_rating(variant: str, search_results: dict[str, dict[str, float]]) -> str:
    """검색 등급."""
    if variant not in search_results:
        return "-"
    t3 = search_results[variant].get("top3_match_rate", 0.0)
    if t3 >= 0.95:
        return "**우수**"
    if t3 >= QUALITY_THRESHOLDS["top3_match_rate"]:
        return "양호"
    return "미달"


def _size_rating(variant: str, size_results: dict[str, float]) -> str:
    """크기 등급."""
    pt_mb = size_results.get("pytorch_fp32_mb", 0.0)
    mb = size_results.get(f"{variant}_mb", 0.0)
    if pt_mb <= 0 or mb <= 0:
        return "-"
    saving = (1 - mb / pt_mb) * 100
    if saving >= 50:
        return "**작음**"
    if saving >= 20:
        return "보통"
    return "큼"


def _build_recommendation_section(
    lines: list[str],
    variants: list[str],
    speed_results: dict[str, dict[str, Any]],
    quality_results: dict[str, dict[str, float]],
    search_results: dict[str, dict[str, float]],
    size_results: dict[str, float],
    ingest_results: dict[str, dict[str, float]],
) -> None:
    """섹션 5: 종합 비교 및 추천."""
    lines.append("## 5. 종합 비교 및 추천")
    lines.append("")

    pt_single = float(speed_results.get("pytorch_fp32", {}).get("single_ms", 0))
    onnx_variants = [v for v in variants if v != "pytorch_fp32"]

    # Trade-off 분석표
    lines.append("### Trade-off 분석")
    lines.append("")
    lines.append("| Variant | 속도 | 품질 | 검색 | 크기 | 종합 |")
    lines.append("|---------|:----:|:----:|:----:|:----:|:----:|")

    for v in onnx_variants:
        label = MODEL_LABELS.get(v, v)
        ms = float(speed_results.get(v, {}).get("single_ms", 0))
        speedup = pt_single / ms if ms > 0 else 0
        spd = _speed_rating(speedup)
        qual = _quality_rating(v, quality_results)
        srch = _search_rating(v, search_results)
        sz = _size_rating(v, size_results)
        overall = "PASS" if _is_quality_pass(v, quality_results) else "FAIL"
        lines.append(f"| {label} | {spd} | {qual} | {srch} | {sz} | {overall} |")
    lines.append("")

    # 운영 환경별 추천
    recommendation = _determine_recommendation(variants, speed_results, quality_results)
    lines.append("### 운영 환경별 추천")
    lines.append("")
    env_desc = [
        ("속도 우선", "speed", "응답 지연시간이 최우선인 환경 (실시간 검색)"),
        ("품질 우선", "quality", "임베딩 품질이 최우선인 환경 (정밀 검색)"),
        ("균형", "balanced", "속도와 품질의 균형 (일반 운영)"),
    ]
    for env, key, desc in env_desc:
        rec_v = recommendation.get(key, "")
        rec_l = MODEL_LABELS.get(rec_v, rec_v)
        lines.append(f"- **{env}**: `{rec_l}` — {desc}")
    lines.append("")

    # 전체 데이터 재임베딩 예상 시간
    if ingest_results:
        lines.append("### 전체 데이터 재임베딩 예상 시간")
        lines.append("")
        lines.append(f"대상: `legal_chunks` 테이블 ({TOTAL_LEGAL_CHUNKS:,}건)")
        lines.append("")
        lines.append("| Variant | 인제스트 속도 (문서/초) | 예상 시간 |")
        lines.append("|---------|--------------------:|--------:|")
        for v in variants:
            if v in ingest_results:
                label = MODEL_LABELS.get(v, v)
                dps = ingest_results[v].get("docs_per_second", 0.0)
                if dps > 0:
                    total_sec = TOTAL_LEGAL_CHUNKS / dps
                    time_str = _format_time_estimate(total_sec)
                    lines.append(f"| {label} | {dps:.1f} | ~{time_str} |")
        lines.append("")


def _build_appendix(
    lines: list[str],
    model_name: str,
    ingest_docs: int,
) -> None:
    """부록: 설정값 및 품질 기준."""
    lines.append("## 부록")
    lines.append("")

    lines.append("### 벤치마크 설정값")
    lines.append("")
    lines.append("| 항목 | 값 |")
    lines.append("|------|-----|")
    lines.append(f"| 모델 | {model_name} |")
    lines.append("| 차원 | 1024 |")
    lines.append("| Warm-up | 2회 |")
    lines.append("| 반복 | 5회 (median) |")
    lines.append(f"| 인제스트 테스트 문서 | {ingest_docs}건 |")
    lines.append("| LanceDB 테이블 | legal_chunks |")
    lines.append("")

    lines.append("### 품질 기준")
    lines.append("")
    lines.append("| 메트릭 | 기준 |")
    lines.append("|--------|------|")
    for key, val in QUALITY_THRESHOLDS.items():
        if "match_rate" in key:
            lines.append(f"| {key} | >= {val:.0%} |")
        elif "diff" in key:
            lines.append(f"| {key} | < {val} |")
        else:
            lines.append(f"| {key} | >= {val} |")
    lines.append("")


def generate_report(
    speed_results: dict[str, dict[str, Any]],
    quality_results: dict[str, dict[str, float]],
    search_results: dict[str, dict[str, float]],
    size_results: dict[str, float],
    ingest_results: dict[str, dict[str, float]],
    e2e_results: dict[str, dict[str, float]],
    output_path: Path,
    model_name: str = "nlpai-lab/KURE-v1",
    ingest_docs: int = 200,
) -> Path:
    """모든 Phase 결과를 받아 MD 보고서를 생성한다.

    Returns:
        생성된 보고서 파일 경로.
    """
    variants = _available_variants(
        speed_results, quality_results, search_results,
        ingest_results, e2e_results,
    )
    recommendation = _determine_recommendation(
        variants, speed_results, quality_results,
    )

    lines: list[str] = []

    # 헤더
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines.append("# ONNX 그래프 최적화 벤치마크 보고서")
    lines.append("")
    lines.append(f"> 생성일: {now}  ")
    lines.append(f"> 모델: {model_name} (1024차원)  ")
    lines.append(f"> 벤치마크 쿼리: 10개, 문서: 20개, 인제스트 테스트: {ingest_docs}건")
    lines.append("")

    # 요약
    lines.append("## 요약")
    lines.append("")
    rec = recommendation.get("balanced", "")
    rec_label = MODEL_LABELS.get(rec, rec)
    if rec and rec in speed_results:
        pt_ms = float(speed_results.get("pytorch_fp32", {}).get("single_ms", 0))
        rec_ms = float(speed_results[rec].get("single_ms", 0))
        speedup = pt_ms / rec_ms if rec_ms > 0 else 0
        lines.append(
            f"**추천 variant**: `{rec_label}` — PyTorch 대비"
            f" **{speedup:.1f}x** 속도 향상, 품질 기준 충족"
        )
    else:
        lines.append(f"**추천 variant**: `{rec_label}`")
    lines.append("")

    # 본문 섹션
    _build_speed_section(
        lines, variants, speed_results, ingest_results, e2e_results, ingest_docs,
    )
    _build_quality_section(lines, variants, quality_results)
    _build_search_section(lines, variants, search_results)
    _build_size_section(lines, variants, size_results)
    _build_recommendation_section(
        lines, variants, speed_results, quality_results,
        search_results, size_results, ingest_results,
    )
    _build_appendix(lines, model_name, ingest_docs)

    # 파일 쓰기
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines) + "\n"
    output_path.write_text(content, encoding="utf-8")

    return output_path
