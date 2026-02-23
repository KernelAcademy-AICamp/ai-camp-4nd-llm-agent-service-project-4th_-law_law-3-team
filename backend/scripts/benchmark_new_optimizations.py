"""
ONNX 신규 최적화 벤치마크

리서치 결과 기반으로 미시도 최적화 기법을 벤치마크한다.
테스트 대상:
  1. Session Config 최적화 (ORT_DISABLE_ALL, denormal, spinning, gelu approx)
  2. CoreML ExecutionProvider (ANE/GPU 가속)
  3. Dynamic Quantization (INT8)
  4. INT4 Weight-Only 양자화 (MatMulNBits)
  5. ORT 프로파일링 (병목 분석)
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# 프로젝트 루트 설정
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── 상수 ──────────────────────────────────────────────
MODELS_DIR = PROJECT_ROOT / "data" / "models"
ORT_OPT_DIR = MODELS_DIR / "kure-v1-ort-opt"
QDQ_DIR = MODELS_DIR / "kure-v1-ort-opt-qdq"
RESULTS_FILE = PROJECT_ROOT / "benchmark_new_results.json"

# 법률 도메인 테스트 쿼리 (다양한 길이)
TEST_QUERIES = [
    "손해배상",
    "이혼 재산분할 청구",
    "부동산 매매계약 해제 손해배상 청구 소송",
    "교통사고로 인한 불법행위 손해배상 책임의 성립 요건",
    "임대차보증금 반환 청구에서 임차인의 대항력과 우선변제권의 관계",
    "주식회사 이사의 충실의무 위반에 따른 회사에 대한 손해배상 책임 범위와 제한 사유",
    "상속재산 분할",
    "근로계약 해지 부당해고",
    "특허권 침해 금지 청구",
    "공동불법행위자의 연대책임과 구상권 행사 요건에 관한 대법원 판례 분석",
]

NUM_WARMUP = 2
NUM_ITERATIONS = 10


def _load_tokenizer(model_dir: Path) -> Any:
    """토크나이저 로드."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)


def _get_baseline_embeddings(tokenizer: Any, model_dir: Path) -> np.ndarray:
    """PyTorch FP32 기준 임베딩 생성 (cosine similarity 비교용)."""
    try:
        import torch
        from transformers import AutoModel

        hf_dir = MODELS_DIR / "models--nlpai-lab--KURE-v1"
        if not hf_dir.exists():
            logger.warning("PyTorch 모델 없음, cosine 비교 건너뜀")
            return np.array([])

        model = AutoModel.from_pretrained(str(hf_dir), trust_remote_code=True)
        model.eval()

        inputs = tokenizer(
            TEST_QUERIES, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        with torch.no_grad():
            outputs = model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :].numpy()
        norm = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / np.clip(norm, 1e-9, None)
    except Exception as e:
        logger.warning("PyTorch baseline 생성 실패: %s", e)
        return np.array([])


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """두 임베딩 행렬의 평균 cosine similarity."""
    if a.size == 0 or b.size == 0:
        return -1.0
    sims = []
    for i in range(min(len(a), len(b))):
        cos = np.dot(a[i], b[i]) / (np.linalg.norm(a[i]) * np.linalg.norm(b[i]) + 1e-9)
        sims.append(float(cos))
    return statistics.mean(sims)


def _run_ort_benchmark(
    session: Any,
    tokenizer: Any,
    label: str,
    static_length: int | None = None,
) -> dict[str, Any]:
    """ORT 세션으로 벤치마크 실행."""
    input_names = [inp.name for inp in session.get_inputs()]

    # Warmup
    for _ in range(NUM_WARMUP):
        inputs = tokenizer(
            TEST_QUERIES[0], return_tensors="np",
            padding="max_length" if static_length else True,
            truncation=True, max_length=static_length or 512,
        )
        feed = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
        if "token_type_ids" in input_names:
            feed["token_type_ids"] = inputs.get("token_type_ids", np.zeros_like(inputs["input_ids"]))
        session.run(None, feed)

    # 배치 추론 벤치마크
    latencies = []
    all_embeddings = []

    for _ in range(NUM_ITERATIONS):
        inputs = tokenizer(
            TEST_QUERIES, return_tensors="np", padding=True, truncation=True, max_length=512,
        )
        feed = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
        if "token_type_ids" in input_names:
            feed["token_type_ids"] = inputs.get("token_type_ids", np.zeros_like(inputs["input_ids"]))

        start = time.perf_counter()
        outputs = session.run(None, feed)
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

        if not all_embeddings:
            emb = outputs[0][:, 0, :]
            norm_val = np.linalg.norm(emb, axis=1, keepdims=True)
            all_embeddings.append(emb / np.clip(norm_val, 1e-9, None))

    median_ms = statistics.median(latencies)
    mean_ms = statistics.mean(latencies)
    std_ms = statistics.stdev(latencies) if len(latencies) > 1 else 0.0

    result = {
        "label": label,
        "median_ms": round(median_ms, 1),
        "mean_ms": round(mean_ms, 1),
        "std_ms": round(std_ms, 1),
        "min_ms": round(min(latencies), 1),
        "max_ms": round(max(latencies), 1),
        "embeddings": all_embeddings[0] if all_embeddings else None,
    }
    return result


def benchmark_session_config() -> list[dict[str, Any]]:
    """Session Config 최적화 벤치마크."""
    import onnxruntime as ort

    model_file = str(ORT_OPT_DIR / "model_optimized.onnx")
    tokenizer = _load_tokenizer(ORT_OPT_DIR)
    results = []

    # 1a. 현재 설정 (baseline)
    logger.info("\n=== 1a. ORT-opt 현재 설정 (ORT_ENABLE_ALL) ===")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(model_file, opts, providers=["CPUExecutionProvider"])
    results.append(_run_ort_benchmark(session, tokenizer, "ORT-opt (현재 설정)"))
    logger.info("  → %s: %.1fms", results[-1]["label"], results[-1]["median_ms"])

    # 1b. ORT_DISABLE_ALL (오프라인 최적화 모델용)
    logger.info("\n=== 1b. ORT_DISABLE_ALL ===")
    opts2 = ort.SessionOptions()
    opts2.intra_op_num_threads = 4
    opts2.inter_op_num_threads = 1
    opts2.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts2.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session2 = ort.InferenceSession(model_file, opts2, providers=["CPUExecutionProvider"])
    results.append(_run_ort_benchmark(session2, tokenizer, "ORT-opt (ORT_DISABLE_ALL)"))
    logger.info("  → %s: %.1fms", results[-1]["label"], results[-1]["median_ms"])

    # 1c. 모든 세션 최적화 적용
    logger.info("\n=== 1c. 전체 Session Config 최적화 ===")
    opts3 = ort.SessionOptions()
    opts3.intra_op_num_threads = 4
    opts3.inter_op_num_threads = 1
    opts3.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts3.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    opts3.enable_mem_pattern = True
    opts3.add_session_config_entry("session.set_denormal_as_zero", "1")
    opts3.add_session_config_entry("session.intra_op.allow_spinning", "0")
    opts3.add_session_config_entry("session.force_spinning_stop", "1")
    opts3.add_session_config_entry("optimization.enable_gelu_approximation", "1")
    session3 = ort.InferenceSession(model_file, opts3, providers=["CPUExecutionProvider"])
    results.append(_run_ort_benchmark(session3, tokenizer, "ORT-opt (전체 Config 최적화)"))
    logger.info("  → %s: %.1fms", results[-1]["label"], results[-1]["median_ms"])

    return results


def benchmark_coreml_ep() -> list[dict[str, Any]]:
    """CoreML ExecutionProvider 벤치마크."""
    import onnxruntime as ort

    if "CoreMLExecutionProvider" not in ort.get_available_providers():
        logger.warning("CoreMLExecutionProvider 미지원, 건너뜀")
        return []

    model_file = str(ORT_OPT_DIR / "model_optimized.onnx")
    tokenizer = _load_tokenizer(ORT_OPT_DIR)
    results = []
    cache_dir = str(ORT_OPT_DIR / ".coreml_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # 2a. CoreML EP (ALL compute units)
    logger.info("\n=== 2a. CoreML EP (ALL) ===")
    try:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        providers = [
            ("CoreMLExecutionProvider", {
                "ModelFormat": "MLProgram",
                "MLComputeUnits": "ALL",
                "ModelCacheDirectory": cache_dir,
            }),
            "CPUExecutionProvider",
        ]
        session = ort.InferenceSession(model_file, opts, providers=providers)
        results.append(_run_ort_benchmark(session, tokenizer, "CoreML EP (ALL)"))
        logger.info("  → %s: %.1fms", results[-1]["label"], results[-1]["median_ms"])
    except Exception as e:
        logger.error("CoreML EP (ALL) 실패: %s", e)
        results.append({"label": "CoreML EP (ALL)", "error": str(e)})

    # 2b. CoreML EP (CPUAndNeuralEngine)
    logger.info("\n=== 2b. CoreML EP (CPU+ANE) ===")
    try:
        opts2 = ort.SessionOptions()
        opts2.intra_op_num_threads = 4
        opts2.inter_op_num_threads = 1
        opts2.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        cache_dir2 = str(ORT_OPT_DIR / ".coreml_cache_ane")
        os.makedirs(cache_dir2, exist_ok=True)
        providers2 = [
            ("CoreMLExecutionProvider", {
                "ModelFormat": "MLProgram",
                "MLComputeUnits": "CPUAndNeuralEngine",
                "ModelCacheDirectory": cache_dir2,
            }),
            "CPUExecutionProvider",
        ]
        session2 = ort.InferenceSession(model_file, opts2, providers=providers2)
        results.append(_run_ort_benchmark(session2, tokenizer, "CoreML EP (CPU+ANE)"))
        logger.info("  → %s: %.1fms", results[-1]["label"], results[-1]["median_ms"])
    except Exception as e:
        logger.error("CoreML EP (CPU+ANE) 실패: %s", e)
        results.append({"label": "CoreML EP (CPU+ANE)", "error": str(e)})

    # 2c. CoreML EP (CPUOnly - CoreML CPU 최적화 경로)
    logger.info("\n=== 2c. CoreML EP (CPUOnly) ===")
    try:
        opts3 = ort.SessionOptions()
        opts3.intra_op_num_threads = 4
        opts3.inter_op_num_threads = 1
        opts3.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        cache_dir3 = str(ORT_OPT_DIR / ".coreml_cache_cpu")
        os.makedirs(cache_dir3, exist_ok=True)
        providers3 = [
            ("CoreMLExecutionProvider", {
                "ModelFormat": "MLProgram",
                "MLComputeUnits": "CPUOnly",
                "ModelCacheDirectory": cache_dir3,
            }),
            "CPUExecutionProvider",
        ]
        session3 = ort.InferenceSession(model_file, opts3, providers=providers3)
        results.append(_run_ort_benchmark(session3, tokenizer, "CoreML EP (CPUOnly)"))
        logger.info("  → %s: %.1fms", results[-1]["label"], results[-1]["median_ms"])
    except Exception as e:
        logger.error("CoreML EP (CPUOnly) 실패: %s", e)
        results.append({"label": "CoreML EP (CPUOnly)", "error": str(e)})

    return results


def benchmark_dynamic_quantization() -> list[dict[str, Any]]:
    """Dynamic Quantization 벤치마크."""
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    model_file = ORT_OPT_DIR / "model_optimized.onnx"
    output_dir = MODELS_DIR / "kure-v1-ort-opt-dynamic-int8"
    output_file = output_dir / "model_optimized.onnx"
    tokenizer = _load_tokenizer(ORT_OPT_DIR)
    results = []

    # Dynamic INT8 모델 생성
    if not output_file.exists():
        logger.info("\n=== Dynamic INT8 모델 생성 중... ===")
        os.makedirs(output_dir, exist_ok=True)
        try:
            quantize_dynamic(
                model_input=str(model_file),
                model_output=str(output_file),
                weight_type=QuantType.QInt8,
                op_types_to_quantize=["MatMul", "Attention"],
            )
            logger.info("  → Dynamic INT8 모델 저장: %s", output_file)
        except Exception as e:
            logger.error("Dynamic INT8 생성 실패: %s", e)
            return []
    else:
        logger.info("  → 기존 Dynamic INT8 모델 사용: %s", output_file)

    # 벤치마크
    logger.info("\n=== 3. Dynamic INT8 벤치마크 ===")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    opts.add_session_config_entry("session.set_denormal_as_zero", "1")

    session = ort.InferenceSession(str(output_file), opts, providers=["CPUExecutionProvider"])
    results.append(_run_ort_benchmark(session, tokenizer, "Dynamic INT8"))
    logger.info("  → %s: %.1fms", results[-1]["label"], results[-1]["median_ms"])

    return results


def benchmark_int4_quantization() -> list[dict[str, Any]]:
    """INT4 Weight-Only 양자화 벤치마크."""
    results = []
    try:
        from onnxruntime.quantization import matmul_4bits_quantizer
    except ImportError:
        logger.warning("matmul_4bits_quantizer 미지원 (ORT 1.22+ 필요), 건너뜀")
        return []

    model_file = str(ORT_OPT_DIR / "model_optimized.onnx")
    output_dir = MODELS_DIR / "kure-v1-ort-opt-int4"
    output_file = output_dir / "model_optimized.onnx"
    tokenizer = _load_tokenizer(ORT_OPT_DIR)

    if not output_file.exists():
        logger.info("\n=== INT4 Weight-Only 모델 생성 중... ===")
        os.makedirs(output_dir, exist_ok=True)
        try:
            quantizer = matmul_4bits_quantizer.MatMul4BitsQuantizer(
                model_path=model_file,
                block_size=32,
                is_symmetric=True,
                accuracy_level=4,
            )
            quantizer.process()
            quantizer.model.save(str(output_file))
            logger.info("  → INT4 모델 저장: %s", output_file)
        except Exception as e:
            logger.error("INT4 생성 실패: %s", e)
            return []
    else:
        logger.info("  → 기존 INT4 모델 사용: %s", output_file)

    # 벤치마크
    logger.info("\n=== 4. INT4 Weight-Only 벤치마크 ===")
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.add_session_config_entry("session.set_denormal_as_zero", "1")

    try:
        session = ort.InferenceSession(str(output_file), opts, providers=["CPUExecutionProvider"])
        results.append(_run_ort_benchmark(session, tokenizer, "INT4 Weight-Only (MatMulNBits)"))
        logger.info("  → %s: %.1fms", results[-1]["label"], results[-1]["median_ms"])
    except Exception as e:
        logger.error("INT4 벤치마크 실패: %s", e)
        results.append({"label": "INT4 Weight-Only (MatMulNBits)", "error": str(e)})

    return results


def benchmark_qdq_with_config() -> list[dict[str, Any]]:
    """기존 QDQ INT8 + 새 Session Config 조합 벤치마크."""
    import onnxruntime as ort

    if not QDQ_DIR.exists():
        logger.warning("QDQ 모델 없음, 건너뜀")
        return []

    model_file = str(QDQ_DIR / "model_optimized.onnx")
    tokenizer = _load_tokenizer(QDQ_DIR)
    results = []

    # 5a. QDQ 현재 설정
    logger.info("\n=== 5a. QDQ INT8 (현재 설정) ===")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(model_file, opts, providers=["CPUExecutionProvider"])
    results.append(_run_ort_benchmark(session, tokenizer, "QDQ INT8 (현재 설정)"))
    logger.info("  → %s: %.1fms", results[-1]["label"], results[-1]["median_ms"])

    # 5b. QDQ + 최적화 Config
    logger.info("\n=== 5b. QDQ INT8 + Session Config 최적화 ===")
    opts2 = ort.SessionOptions()
    opts2.intra_op_num_threads = 4
    opts2.inter_op_num_threads = 1
    opts2.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts2.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    opts2.enable_mem_pattern = True
    opts2.add_session_config_entry("session.set_denormal_as_zero", "1")
    opts2.add_session_config_entry("session.intra_op.allow_spinning", "0")
    opts2.add_session_config_entry("session.force_spinning_stop", "1")
    opts2.add_session_config_entry("optimization.enable_gelu_approximation", "1")
    session2 = ort.InferenceSession(model_file, opts2, providers=["CPUExecutionProvider"])
    results.append(_run_ort_benchmark(session2, tokenizer, "QDQ INT8 + Config 최적화"))
    logger.info("  → %s: %.1fms", results[-1]["label"], results[-1]["median_ms"])

    return results


def run_profiling() -> dict[str, Any]:
    """ORT 프로파일링으로 병목 분석."""
    import onnxruntime as ort

    model_file = str(ORT_OPT_DIR / "model_optimized.onnx")
    tokenizer = _load_tokenizer(ORT_OPT_DIR)

    logger.info("\n=== 프로파일링 실행 ===")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.enable_profiling = True
    opts.profile_file_prefix = str(PROJECT_ROOT / "ort_profile")

    session = ort.InferenceSession(model_file, opts, providers=["CPUExecutionProvider"])

    inputs = tokenizer(
        TEST_QUERIES, return_tensors="np", padding=True, truncation=True, max_length=512,
    )
    input_names = [inp.name for inp in session.get_inputs()]
    feed = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
    if "token_type_ids" in input_names:
        feed["token_type_ids"] = inputs.get("token_type_ids", np.zeros_like(inputs["input_ids"]))

    session.run(None, feed)
    profile_file = session.end_profiling()
    logger.info("  → 프로파일 저장: %s", profile_file)

    # 프로파일 분석
    with open(profile_file, encoding="utf-8") as f:
        profile_data = json.load(f)

    # 연산자별 시간 집계
    op_times: dict[str, float] = {}
    for event in profile_data:
        if isinstance(event, dict) and "cat" in event and event.get("cat") == "Node":
            op_type = event.get("args", {}).get("op_name", "unknown")
            dur = event.get("dur", 0)  # microseconds
            op_times[op_type] = op_times.get(op_type, 0) + dur

    # 상위 10개 연산자
    sorted_ops = sorted(op_times.items(), key=lambda x: x[1], reverse=True)[:10]
    total_time = sum(op_times.values())

    logger.info("\n  연산자별 시간 분포 (상위 10):")
    for op, time_us in sorted_ops:
        pct = (time_us / total_time * 100) if total_time > 0 else 0
        logger.info("    %-30s %8.1fms  (%5.1f%%)", op, time_us / 1000, pct)

    return {
        "profile_file": profile_file,
        "top_ops": [{"op": op, "time_ms": round(t / 1000, 1), "pct": round(t / total_time * 100, 1)} for op, t in sorted_ops],
        "total_time_ms": round(total_time / 1000, 1),
    }


def main() -> None:
    """메인 벤치마크 실행."""
    logger.info("=" * 60)
    logger.info("ONNX 신규 최적화 벤치마크")
    logger.info("=" * 60)

    all_results: list[dict[str, Any]] = []
    baseline_emb: np.ndarray | None = None

    # 0. 프로파일링 (병목 분석)
    profile_result = run_profiling()

    # 1. Session Config 최적화
    logger.info("\n" + "─" * 60)
    logger.info("Phase 1: Session Config 최적화")
    logger.info("─" * 60)
    config_results = benchmark_session_config()
    if config_results:
        baseline_emb = config_results[0].get("embeddings")
    all_results.extend(config_results)

    # 2. CoreML EP
    logger.info("\n" + "─" * 60)
    logger.info("Phase 2: CoreML ExecutionProvider")
    logger.info("─" * 60)
    coreml_results = benchmark_coreml_ep()
    all_results.extend(coreml_results)

    # 3. Dynamic Quantization
    logger.info("\n" + "─" * 60)
    logger.info("Phase 3: Dynamic Quantization")
    logger.info("─" * 60)
    dynamic_results = benchmark_dynamic_quantization()
    all_results.extend(dynamic_results)

    # 4. INT4 Weight-Only
    logger.info("\n" + "─" * 60)
    logger.info("Phase 4: INT4 Weight-Only Quantization")
    logger.info("─" * 60)
    int4_results = benchmark_int4_quantization()
    all_results.extend(int4_results)

    # 5. QDQ INT8 + Config 최적화
    logger.info("\n" + "─" * 60)
    logger.info("Phase 5: QDQ INT8 + Session Config")
    logger.info("─" * 60)
    qdq_results = benchmark_qdq_with_config()
    all_results.extend(qdq_results)

    # Cosine similarity 계산
    if baseline_emb is not None:
        for r in all_results:
            emb = r.pop("embeddings", None)
            if emb is not None and baseline_emb is not None:
                r["cosine_vs_baseline"] = round(_cosine_similarity(baseline_emb, emb), 6)
            else:
                r["cosine_vs_baseline"] = None

    # PyTorch baseline과 비교
    pytorch_baseline_ms = 1501.0  # 기존 벤치마크 결과

    # 결과 요약
    logger.info("\n" + "=" * 60)
    logger.info("종합 결과")
    logger.info("=" * 60)
    logger.info(
        "%-35s %10s %8s %8s",
        "Variant", "Median(ms)", "Speedup", "Cosine",
    )
    logger.info("-" * 65)

    for r in all_results:
        if "error" in r:
            logger.info("%-35s %s", r["label"], f"ERROR: {r['error'][:40]}")
            continue
        speedup = pytorch_baseline_ms / r["median_ms"] if r["median_ms"] > 0 else 0
        cosine_str = f"{r.get('cosine_vs_baseline', 'N/A')}"
        logger.info(
            "%-35s %10.1f %7.2fx %8s",
            r["label"], r["median_ms"], speedup, cosine_str,
        )

    # JSON 저장 (embeddings 제외)
    save_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pytorch_baseline_ms": pytorch_baseline_ms,
        "profiling": profile_result,
        "results": [
            {k: v for k, v in r.items() if k != "embeddings"} for r in all_results
        ],
    }
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    logger.info("\n결과 저장: %s", RESULTS_FILE)


if __name__ == "__main__":
    main()
