"""리랭커 ONNX 벤치마크 MD 보고서 생성 모듈.

벤치마크 결과(Phase 1~4)를 받아 Markdown 보고서를 생성한다.

사용법:
  from scripts.benchmark_reranker_report import generate_report
  generate_report(speed, quality, size, output_path)
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RERANKER_MODEL = "dragonkue/bge-reranker-v2-m3-ko"
PIPELINE_TOTAL_MS = 24400
NUM_DOCUMENTS = 15

MODEL_LABELS: dict[str, str] = {
    "pytorch_fp32": "PyTorch FP32",
    "onnx_fp32": "ONNX FP32",
    "onnx_int8": "ONNX INT8",
    "onnx_o2": "ONNX O2",
    "onnx_o3": "ONNX O3",
    "onnx_o3_int8": "ONNX O3+INT8",
    "onnx_fp16": "ONNX FP16",
}

VARIANT_ORDER: list[str] = [
    "pytorch_fp32",
    "onnx_fp32",
    "onnx_int8",
    "onnx_o2",
    "onnx_o3",
    "onnx_o3_int8",
    "onnx_fp16",
]

QUALITY_THRESHOLDS: dict[str, float] = {
    "pearson": 0.99,
    "spearman": 0.99,
    "top3_match": 3,
    "top5_match": 4,
    "max_diff": 0.05,
}


def _pass_fail(
    value: float, threshold: float, *, higher_is_better: bool = True
) -> str:
    """PASS/FAIL 판정."""
    if higher_is_better:
        return "PASS" if value >= threshold else "FAIL"
    return "PASS" if value <= threshold else "FAIL"


def _available_variants(
    *result_dicts: dict[str, Any],
) -> list[str]:
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
        return True
    q = quality_results[variant]
    if q.get("pearson", 0) < QUALITY_THRESHOLDS["pearson"]:
        return False
    if q.get("spearman", 0) < QUALITY_THRESHOLDS["spearman"]:
        return False
    if q.get("max_diff", 1) >= QUALITY_THRESHOLDS["max_diff"]:
        return False
    return True


def _determine_recommendation(
    variants: list[str],
    speed_results: dict[str, dict[str, Any]],
    quality_results: dict[str, dict[str, float]],
) -> dict[str, str]:
    """운영 환경별 추천 variant 결정."""
    result: dict[str, str] = {"speed": "", "quality": "", "balanced": ""}
    onnx_variants = [v for v in variants if v != "pytorch_fp32"]
    if not onnx_variants:
        return result

    # 속도 순위 (배치 ms 우선, 없으면 개별 ms)
    speed_ranking: list[tuple[str, float]] = []
    for v in onnx_variants:
        if v in speed_results:
            ms = float(speed_results[v].get("batch_time_ms", 0))
            if ms <= 0:
                ms = float(speed_results[v].get("time_ms", 9999))
            speed_ranking.append((v, ms))
    speed_ranking.sort(key=lambda x: x[1])

    quality_pass = {v for v in onnx_variants if _is_quality_pass(v, quality_results)}

    # 속도 우선: 품질 통과 중 가장 빠른 것
    for v, _ in speed_ranking:
        if v in quality_pass:
            result["speed"] = v
            break
    if not result["speed"] and speed_ranking:
        result["speed"] = speed_ranking[0][0]

    # 품질 우선: pearson 가장 높은 것
    quality_ranking = sorted(
        [(v, quality_results[v].get("pearson", 0))
         for v in onnx_variants if v in quality_results],
        key=lambda x: x[1],
        reverse=True,
    )
    result["quality"] = quality_ranking[0][0] if quality_ranking else onnx_variants[0]

    # 균형: 품질 통과 + 속도 상위
    balanced = [(v, ms) for v, ms in speed_ranking if v in quality_pass]
    result["balanced"] = balanced[0][0] if balanced else (
        speed_ranking[0][0] if speed_ranking else onnx_variants[0]
    )
    return result


# ============================================================
# 섹션 빌더
# ============================================================


def _build_speed_section(
    lines: list[str],
    variants: list[str],
    speed_results: dict[str, dict[str, Any]],
) -> None:
    """섹션 2: 속도 비교 (개별 + 배치)."""
    lines.append("## 2. 속도 비교")
    lines.append("")

    pt = speed_results.get("pytorch_fp32", {})
    pt_ms = float(pt.get("time_ms", 0))

    # 2.1 개별 추론
    lines.append("### 2.1 개별 추론 (15건 순차)")
    lines.append("")
    lines.append("| Variant | 시간 (ms) | 건당 (ms) | 상대 속도 |")
    lines.append("|---------|--------:|---------:|----------:|")
    for v in variants:
        if v not in speed_results:
            continue
        r = speed_results[v]
        label = MODEL_LABELS.get(v, v)
        ms = float(r.get("time_ms", 0))
        per_doc = float(r.get("per_doc_ms", 0))
        if v == "pytorch_fp32":
            lines.append(f"| {label} | {ms:.1f} | {per_doc:.1f} | 1.0x (baseline) |")
        else:
            speedup = pt_ms / ms if ms > 0 else 0
            lines.append(f"| {label} | {ms:.1f} | {per_doc:.1f} | {speedup:.1f}x |")
    lines.append("")

    # 2.2 배치 추론
    lines.append("### 2.2 배치 추론 (15건 일괄)")
    lines.append("")
    lines.append("| Variant | 시간 (ms) | 건당 (ms) | 상대 속도 |")
    lines.append("|---------|--------:|---------:|----------:|")
    for v in variants:
        if v == "pytorch_fp32" or v not in speed_results:
            continue
        r = speed_results[v]
        batch_ms = float(r.get("batch_time_ms", 0))
        if batch_ms <= 0:
            continue
        label = MODEL_LABELS.get(v, v)
        batch_per_doc = float(r.get("batch_per_doc_ms", 0))
        batch_speedup = pt_ms / batch_ms if batch_ms > 0 else 0
        lines.append(f"| {label} | {batch_ms:.1f} | {batch_per_doc:.1f} | {batch_speedup:.1f}x |")
    lines.append("")


def _build_quality_section(
    lines: list[str],
    variants: list[str],
    quality_results: dict[str, dict[str, float]],
) -> None:
    """섹션 3: 품질 비교."""
    if not quality_results:
        return

    lines.append("## 3. 품질 비교 (vs PyTorch FP32 baseline)")
    lines.append("")
    present = [v for v in variants if v in quality_results]
    header = "| 메트릭 | " + " | ".join(MODEL_LABELS.get(v, v) for v in present) + " | 기준 | 판정 |"
    sep = "|--------| " + " | ".join("-----:" for _ in present) + " | -----: | -----: |"
    lines.append(header)
    lines.append(sep)

    metrics: list[tuple[str, str, float, bool, str]] = [
        ("Pearson", "pearson", QUALITY_THRESHOLDS["pearson"], True, ".6f"),
        ("Spearman", "spearman", QUALITY_THRESHOLDS["spearman"], True, ".6f"),
        ("Top-3 일치", "top3_match", QUALITY_THRESHOLDS["top3_match"], True, ".0f"),
        ("Top-5 일치", "top5_match", QUALITY_THRESHOLDS["top5_match"], True, ".0f"),
        ("평균 차이", "avg_diff", QUALITY_THRESHOLDS["max_diff"], False, ".4f"),
        ("최대 차이", "max_diff", QUALITY_THRESHOLDS["max_diff"], False, ".4f"),
    ]

    for label, key, threshold, higher, fmt in metrics:
        row = f"| {label} |"
        worst = "PASS"
        for v in present:
            val = quality_results[v].get(key, 0.0)
            row += f" {val:{fmt}} |"
            if _pass_fail(val, threshold, higher_is_better=higher) == "FAIL":
                worst = "FAIL"
        t_str = f">={threshold:{fmt}}" if higher else f"<{threshold:{fmt}}"
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

    fp32_mb = size_results.get("onnx_fp32", 0.0)
    for v in variants:
        if v == "pytorch_fp32":
            continue
        mb = size_results.get(v, 0.0)
        if mb <= 0:
            continue
        label = MODEL_LABELS.get(v, v)
        if v == "onnx_fp32":
            lines.append(f"| {label} | {mb:.0f} | - (baseline) |")
        else:
            saving = (1 - mb / fp32_mb) * 100 if fp32_mb > 0 else 0
            lines.append(f"| {label} | {mb:.0f} | {saving:+.0f}% |")
    lines.append("")


def _build_comprehensive_section(
    lines: list[str],
    variants: list[str],
    speed_results: dict[str, dict[str, Any]],
    quality_results: dict[str, dict[str, float]],
    size_results: dict[str, float],
) -> None:
    """섹션 5: 종합 비교 (속도 × 품질 × 크기)."""
    lines.append("## 5. 종합 비교")
    lines.append("")

    onnx_variants = [v for v in variants if v != "pytorch_fp32"]
    pt_ms = float(speed_results.get("pytorch_fp32", {}).get("time_ms", 0))

    lines.append("| Variant | 속도 (배치) | 품질 | 크기 | 종합 |")
    lines.append("|---------|:--------:|:----:|:----:|:----:|")

    for v in onnx_variants:
        if v not in speed_results:
            continue
        label = MODEL_LABELS.get(v, v)
        r = speed_results[v]

        # 속도 등급 (배치 기준)
        batch_ms = float(r.get("batch_time_ms", 0))
        batch_speedup = pt_ms / batch_ms if batch_ms > 0 else 0
        if batch_speedup >= 2.0:
            spd = "**빠름**"
        elif batch_speedup >= 1.3:
            spd = "보통"
        else:
            spd = "느림"

        # 품질 등급
        is_pass = _is_quality_pass(v, quality_results)
        q = quality_results.get(v, {})
        pearson = q.get("pearson", 0)
        if pearson >= 0.999:
            qual = "**우수**"
        elif pearson >= QUALITY_THRESHOLDS["pearson"]:
            qual = "양호"
        else:
            qual = "미달"

        # 크기 등급
        fp32_mb = size_results.get("onnx_fp32", 0)
        mb = size_results.get(v, 0)
        if fp32_mb > 0 and mb > 0:
            saving = (1 - mb / fp32_mb) * 100
            if saving >= 50:
                sz = "**작음**"
            elif saving >= 20:
                sz = "보통"
            else:
                sz = "큼"
        else:
            sz = "-"

        overall = "PASS" if is_pass else "FAIL"
        lines.append(f"| {label} | {spd} | {qual} | {sz} | {overall} |")
    lines.append("")


def _build_pipeline_section(
    lines: list[str],
    speed_results: dict[str, dict[str, Any]],
    quality_results: dict[str, dict[str, float]],
) -> None:
    """섹션 6: 파이프라인 영향."""
    lines.append("## 6. 파이프라인 영향")
    lines.append("")
    lines.append(f"현재 전체 파이프라인: **{PIPELINE_TOTAL_MS:,}ms**")
    lines.append("")

    pt_ms = float(speed_results.get("pytorch_fp32", {}).get("time_ms", 0))
    lines.append(f"현재 리랭킹 (PyTorch FP32): **{pt_ms:.0f}ms**")
    lines.append("")

    lines.append("| Variant | 배치 시간 (ms) | 파이프라인 (ms) | 절약 (ms) | 절약 (%) |")
    lines.append("|---------|------------:|-----------:|---------:|---------:|")

    for v in VARIANT_ORDER:
        if v == "pytorch_fp32" or v not in speed_results:
            continue
        r = speed_results[v]
        batch_ms = float(r.get("batch_time_ms", 0))
        if batch_ms <= 0:
            continue
        label = MODEL_LABELS.get(v, v)
        new_total = PIPELINE_TOTAL_MS - pt_ms + batch_ms
        saved = PIPELINE_TOTAL_MS - new_total
        pct = saved / PIPELINE_TOTAL_MS * 100
        is_pass = _is_quality_pass(v, quality_results)
        suffix = "" if is_pass else " (품질 미달)"
        lines.append(f"| {label}{suffix} | {batch_ms:.0f} | {new_total:.0f} | {saved:.0f} | {pct:.1f}% |")
    lines.append("")


def _build_recommendation_section(
    lines: list[str],
    variants: list[str],
    speed_results: dict[str, dict[str, Any]],
    quality_results: dict[str, dict[str, float]],
) -> None:
    """섹션 7: 운영 환경별 추천."""
    lines.append("## 7. 운영 환경별 추천")
    lines.append("")

    recommendation = _determine_recommendation(variants, speed_results, quality_results)

    env_desc = [
        ("속도 우선", "speed", "응답 지연시간이 최우선인 환경 (실시간 검색)"),
        ("품질 우선", "quality", "리랭킹 정확도가 최우선인 환경 (정밀 검색)"),
        ("균형 (권장)", "balanced", "속도와 품질의 균형 (일반 운영)"),
    ]
    for env, key, desc in env_desc:
        rec_v = recommendation.get(key, "")
        rec_l = MODEL_LABELS.get(rec_v, rec_v)
        lines.append(f"- **{env}**: `{rec_l}` — {desc}")
    lines.append("")


def _build_appendix(lines: list[str]) -> None:
    """섹션 8: 부록 (설정값, 품질 기준)."""
    lines.append("## 8. 부록")
    lines.append("")

    lines.append("### 벤치마크 설정값")
    lines.append("")
    lines.append("| 항목 | 값 |")
    lines.append("|------|-----|")
    lines.append(f"| 리랭커 모델 | {RERANKER_MODEL} |")
    lines.append("| 파라미터 | 568M |")
    lines.append(f"| 테스트 문서 | {NUM_DOCUMENTS}건 |")
    lines.append("| 반복 | 5회 (평균) |")
    lines.append(f"| 파이프라인 기준 | {PIPELINE_TOTAL_MS:,}ms |")
    lines.append("")

    lines.append("### 품질 기준")
    lines.append("")
    lines.append("| 메트릭 | 기준 |")
    lines.append("|--------|------|")
    for key, val in QUALITY_THRESHOLDS.items():
        if "match" in key:
            lines.append(f"| {key} | >= {val:.0f} |")
        elif "diff" in key:
            lines.append(f"| {key} | < {val} |")
        else:
            lines.append(f"| {key} | >= {val} |")
    lines.append("")


# ============================================================
# 보고서 생성
# ============================================================


def generate_report(
    speed_results: dict[str, dict[str, Any]],
    quality_results: dict[str, dict[str, float]],
    size_results: dict[str, float],
    output_path: Path,
) -> Path:
    """벤치마크 결과를 받아 MD 보고서를 생성한다.

    Args:
        speed_results: variant → {time_ms, speedup, per_doc_ms, batch_time_ms, ...}
        quality_results: variant → {pearson, spearman, top3_match, top5_match, ...}
        size_results: variant → size_mb
        output_path: 보고서 파일 경로.

    Returns:
        생성된 보고서 파일 경로.
    """
    variants = _available_variants(speed_results, quality_results, size_results)
    recommendation = _determine_recommendation(
        variants, speed_results, quality_results,
    )

    lines: list[str] = []

    # 헤더
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines.append("# 리랭커 ONNX 벤치마크 보고서")
    lines.append("")
    lines.append(f"> 생성일: {now}  ")
    lines.append(f"> 모델: {RERANKER_MODEL} (568M params)  ")
    lines.append(f"> 테스트: 쿼리 1개, 문서 {NUM_DOCUMENTS}건, 5회 반복")
    lines.append("")

    # 1. 요약
    lines.append("## 1. 요약")
    lines.append("")
    rec = recommendation.get("balanced", "")
    rec_label = MODEL_LABELS.get(rec, rec)
    if rec and rec in speed_results:
        pt_ms = float(speed_results.get("pytorch_fp32", {}).get("time_ms", 0))
        rec_batch_ms = float(speed_results[rec].get("batch_time_ms", 0))
        speedup = pt_ms / rec_batch_ms if rec_batch_ms > 0 else 0
        new_total = PIPELINE_TOTAL_MS - pt_ms + rec_batch_ms
        saved = PIPELINE_TOTAL_MS - new_total
        lines.append(
            f"**추천 variant**: `{rec_label}` — PyTorch 대비"
            f" **{speedup:.1f}x** 속도 향상 (배치), 품질 기준 충족"
        )
        lines.append("")
        lines.append(
            f"파이프라인 영향: {PIPELINE_TOTAL_MS:,}ms → {new_total:.0f}ms"
            f" (절약 {saved:.0f}ms, {saved / PIPELINE_TOTAL_MS * 100:.1f}%)"
        )
    else:
        lines.append(f"**추천 variant**: `{rec_label}`")
    lines.append("")

    # 본문 섹션
    _build_speed_section(lines, variants, speed_results)
    _build_quality_section(lines, variants, quality_results)
    _build_size_section(lines, variants, size_results)
    _build_comprehensive_section(
        lines, variants, speed_results, quality_results, size_results,
    )
    _build_pipeline_section(lines, speed_results, quality_results)
    _build_recommendation_section(lines, variants, speed_results, quality_results)
    _build_appendix(lines)

    # 파일 쓰기
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines) + "\n"
    output_path.write_text(content, encoding="utf-8")

    return output_path
