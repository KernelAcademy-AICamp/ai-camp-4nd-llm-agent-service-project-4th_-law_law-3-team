"""리랭커 ONNX 신규 최적화 벤치마크

기존 benchmark_reranker_quantize.py에서 테스트한 기본 variant(FP32/INT8/O2/O3/O3+INT8/FP16)
이외의 미시도 최적화 기법을 벤치마크한다.

테스트 대상:
  1. Session Config 최적화 (ORT_DISABLE_ALL, denormal, spinning, gelu approx)
  2. CoreML ExecutionProvider (ANE/GPU 가속)
  3. QDQ INT8 + Session Config 조합
  4. IO Binding (CPU EP 오버헤드/이점 측정)
  5. ORT 프로파일링 (병목 분석)

사용법:
  cd backend && uv run python scripts/benchmark_reranker_optimizations.py
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

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 프로젝트 루트 설정
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── 상수 ──────────────────────────────────────────────
RERANKER_MODEL = "dragonkue/bge-reranker-v2-m3-ko"
MODELS_DIR = PROJECT_ROOT / "data" / "models"
ORT_OPT_DIR = MODELS_DIR / "reranker-ort-opt"
QDQ_DIR = MODELS_DIR / "reranker-ort-opt-qdq"
RESULTS_FILE = PROJECT_ROOT / "benchmark_reranker_opt_results.json"

QUERY = "교통사고 손해배상 판례"

# 실제 파이프라인에서 리랭킹에 들어가는 수준의 문서들 (15건)
DOCUMENTS = [
    "피고는 원고에게 금 50,000,000원 및 이에 대한 지연손해금을 지급하라. 교통사고로 인한 손해배상 청구 사건에서 피해자의 과실 비율을 30%로 인정하고 치료비 및 위자료를 산정함.",
    "자동차손해배상 보장법 제3조에 의하면 자기를 위하여 자동차를 운행하는 자는 그 운행으로 다른 사람을 사망하게 하거나 부상하게 한 경우에는 그 손해를 배상할 책임을 진다.",
    "불법행위로 인한 손해배상 청구권의 소멸시효는 피해자나 그 법정대리인이 그 손해 및 가해자를 안 날로부터 3년, 불법행위를 한 날로부터 10년이다.",
    "민법 제750조에 의한 불법행위 손해배상 책임이 성립하려면 가해행위의 위법성, 가해자의 고의 또는 과실, 손해의 발생, 가해행위와 손해 사이의 인과관계가 있어야 한다.",
    "교통사고처리특례법 제4조 제1항 단서 각 호의 사유에 해당하는 경우에는 피해자의 명시한 의사에 반하여 공소를 제기할 수 있다.",
    "원고의 청구를 기각한다. 소송비용은 원고의 부담으로 한다. 원고가 주장하는 교통사고와 상해 사이의 인과관계를 인정하기 어렵다.",
    "위자료 산정에 있어서는 피해자의 나이, 직업, 재산상태, 생활환경, 정신적 고통의 정도 등 여러 사정을 종합적으로 고려하여야 한다.",
    "임대차보증금 반환 청구 사건에서 임대인은 임차인에게 보증금 전액을 반환할 의무가 있으나, 연체 차임 등을 공제할 수 있다.",
    "근로기준법 제23조 제1항은 사용자는 근로자에게 정당한 이유 없이 해고, 휴직, 정직, 전직, 감봉 그 밖의 징벌을 하지 못한다고 규정하고 있다.",
    "형법 제307조 제1항의 명예훼손죄가 성립하려면 사실을 적시하여 사람의 명예를 훼손하여야 하고, 적시된 사실이 허위인 경우에는 제2항에 의하여 가중처벌된다.",
    "후유장해 등급 판정에 있어서는 맥브라이드 장해평가 방법에 의하고, 노동능력상실률의 평가는 사실인정의 문제이다.",
    "과실상계에 있어서 피해자의 과실은 사회통념이나 신의성실의 원칙에 따라 공동생활에 있어 요구되는 약한 의미의 부주의를 포함한다.",
    "자동차종합보험에서 보험회사는 피보험자가 사고로 타인에게 손해를 가한 경우 피해자에게 직접 보험금을 지급할 의무가 있다.",
    "채무불이행으로 인한 손해배상에 있어서 특별손해는 채무자가 그 사정을 알았거나 알 수 있었을 때에 한하여 배상의 책임이 있다.",
    "국가배상법 제2조에 의하면 국가나 지방자치단체는 공무원이 직무를 집행하면서 고의 또는 과실로 법령을 위반하여 타인에게 손해를 입힌 경우 배상책임을 진다.",
]

QUALITY_THRESHOLDS: dict[str, float] = {
    "pearson": 0.99,
    "spearman": 0.99,
    "top3_match": 3,
    "top5_match": 4,
    "max_diff": 0.05,
}

NUM_WARMUP = 2
NUM_ITERATIONS = 10


# ── 헬퍼 함수 ──────────────────────────────────────────


def _load_tokenizer(model_dir: Path) -> Any:
    """토크나이저 로드."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)


def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    """Sigmoid 활성화 함수."""
    return 1 / (1 + np.exp(-x))


def _get_baseline_scores(tokenizer: Any, model_dir: Path) -> np.ndarray:
    """PyTorch FP32 기준 리랭킹 점수 생성 (품질 비교용).

    AutoModelForSequenceClassification으로 PyTorch 추론 후 sigmoid 적용.
    """
    try:
        import torch
        from transformers import AutoModelForSequenceClassification

        model = AutoModelForSequenceClassification.from_pretrained(
            RERANKER_MODEL, cache_dir=str(MODELS_DIR), trust_remote_code=True,
        )
        model.eval()

        queries = [QUERY] * len(DOCUMENTS)
        inputs = tokenizer(
            queries, DOCUMENTS,
            return_tensors="pt", padding=True, truncation=True, max_length=512,
        )
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits.numpy()
        # logits shape: (batch, num_labels) 또는 (batch,)
        if logits.ndim == 2:
            raw_scores = logits[:, 0]
        else:
            raw_scores = logits.flatten()
        scores = _sigmoid(raw_scores)
        return np.array(scores, dtype=np.float64)
    except Exception as e:
        logger.warning("PyTorch baseline 생성 실패: %s", e)
        return np.array([])


def _pearson_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """두 점수 배열 간 피어슨 상관계수."""
    if a.size == 0 or b.size == 0:
        return -1.0
    corr_matrix = np.corrcoef(a, b)
    return float(corr_matrix[0, 1])


def _spearman_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """두 점수 배열 간 스피어만 순위 상관계수."""
    if a.size == 0 or b.size == 0:
        return -1.0
    from scipy.stats import spearmanr

    corr, _ = spearmanr(a, b)
    return float(corr)


def _run_ort_benchmark(
    session: Any,
    tokenizer: Any,
    label: str,
) -> dict[str, Any]:
    """ORT 세션으로 리랭커 벤치마크 실행.

    입력: (query, document) 쌍 15개를 배치 추론.
    출력: logits → sigmoid 점수 + 레이턴시 통계.
    """
    input_names = [inp.name for inp in session.get_inputs()]

    # Warmup (단일 쌍으로)
    for _ in range(NUM_WARMUP):
        warmup_inputs = tokenizer(
            QUERY, DOCUMENTS[0],
            return_tensors="np", padding=True, truncation=True, max_length=512,
        )
        feed: dict[str, np.ndarray] = {
            "input_ids": warmup_inputs["input_ids"],
            "attention_mask": warmup_inputs["attention_mask"],
        }
        if "token_type_ids" in input_names:
            feed["token_type_ids"] = warmup_inputs.get(
                "token_type_ids", np.zeros_like(warmup_inputs["input_ids"]),
            )
        session.run(None, feed)

    # 배치 추론 벤치마크
    queries = [QUERY] * len(DOCUMENTS)
    latencies: list[float] = []
    last_scores: np.ndarray | None = None

    for _ in range(NUM_ITERATIONS):
        inputs = tokenizer(
            queries, DOCUMENTS,
            return_tensors="np", padding=True, truncation=True, max_length=512,
        )
        feed = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        if "token_type_ids" in input_names:
            feed["token_type_ids"] = inputs.get(
                "token_type_ids", np.zeros_like(inputs["input_ids"]),
            )

        start = time.perf_counter()
        outputs = session.run(None, feed)
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

        if last_scores is None:
            logits = outputs[0]
            if logits.ndim == 2:
                raw = logits[:, 0]
            else:
                raw = logits.flatten()
            last_scores = np.array(_sigmoid(raw), dtype=np.float64)

    median_ms = statistics.median(latencies)
    mean_ms = statistics.mean(latencies)
    std_ms = statistics.stdev(latencies) if len(latencies) > 1 else 0.0

    return {
        "label": label,
        "median_ms": round(median_ms, 1),
        "mean_ms": round(mean_ms, 1),
        "std_ms": round(std_ms, 1),
        "min_ms": round(min(latencies), 1),
        "max_ms": round(max(latencies), 1),
        "scores": last_scores,
    }


# ── 벤치마크 함수 ──────────────────────────────────────


def benchmark_session_config() -> list[dict[str, Any]]:
    """Session Config 최적화 벤치마크."""
    import onnxruntime as ort

    model_file = str(ORT_OPT_DIR / "model_optimized.onnx")
    tokenizer = _load_tokenizer(ORT_OPT_DIR)
    results: list[dict[str, Any]] = []

    # 1a. 현재 설정 (baseline)
    logger.info("\n=== 1a. ORT-opt 현재 설정 (ORT_ENABLE_ALL) ===")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(model_file, opts, providers=["CPUExecutionProvider"])
    results.append(_run_ort_benchmark(session, tokenizer, "ORT-opt (현재 설정)"))
    logger.info("  -> %s: %.1fms", results[-1]["label"], results[-1]["median_ms"])

    # 1b. ORT_DISABLE_ALL (오프라인 최적화 모델용)
    logger.info("\n=== 1b. ORT_DISABLE_ALL ===")
    opts2 = ort.SessionOptions()
    opts2.intra_op_num_threads = 4
    opts2.inter_op_num_threads = 1
    opts2.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts2.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session2 = ort.InferenceSession(model_file, opts2, providers=["CPUExecutionProvider"])
    results.append(_run_ort_benchmark(session2, tokenizer, "ORT-opt (ORT_DISABLE_ALL)"))
    logger.info("  -> %s: %.1fms", results[-1]["label"], results[-1]["median_ms"])

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
    logger.info("  -> %s: %.1fms", results[-1]["label"], results[-1]["median_ms"])

    return results


def benchmark_coreml_ep() -> list[dict[str, Any]]:
    """CoreML ExecutionProvider 벤치마크."""
    import onnxruntime as ort

    if "CoreMLExecutionProvider" not in ort.get_available_providers():
        logger.warning("CoreMLExecutionProvider 미지원, 건너뜀")
        return []

    model_file = str(ORT_OPT_DIR / "model_optimized.onnx")
    tokenizer = _load_tokenizer(ORT_OPT_DIR)
    results: list[dict[str, Any]] = []

    # 2a. CoreML EP (ALL compute units)
    logger.info("\n=== 2a. CoreML EP (ALL) ===")
    try:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        cache_dir = str(ORT_OPT_DIR / ".coreml_cache_rr")
        os.makedirs(cache_dir, exist_ok=True)
        providers: list[Any] = [
            ("CoreMLExecutionProvider", {
                "ModelFormat": "MLProgram",
                "MLComputeUnits": "ALL",
                "ModelCacheDirectory": cache_dir,
            }),
            "CPUExecutionProvider",
        ]
        session = ort.InferenceSession(model_file, opts, providers=providers)
        results.append(_run_ort_benchmark(session, tokenizer, "CoreML EP (ALL)"))
        logger.info("  -> %s: %.1fms", results[-1]["label"], results[-1]["median_ms"])
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
        cache_dir2 = str(ORT_OPT_DIR / ".coreml_cache_rr_ane")
        os.makedirs(cache_dir2, exist_ok=True)
        providers2: list[Any] = [
            ("CoreMLExecutionProvider", {
                "ModelFormat": "MLProgram",
                "MLComputeUnits": "CPUAndNeuralEngine",
                "ModelCacheDirectory": cache_dir2,
            }),
            "CPUExecutionProvider",
        ]
        session2 = ort.InferenceSession(model_file, opts2, providers=providers2)
        results.append(_run_ort_benchmark(session2, tokenizer, "CoreML EP (CPU+ANE)"))
        logger.info("  -> %s: %.1fms", results[-1]["label"], results[-1]["median_ms"])
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
        cache_dir3 = str(ORT_OPT_DIR / ".coreml_cache_rr_cpu")
        os.makedirs(cache_dir3, exist_ok=True)
        providers3: list[Any] = [
            ("CoreMLExecutionProvider", {
                "ModelFormat": "MLProgram",
                "MLComputeUnits": "CPUOnly",
                "ModelCacheDirectory": cache_dir3,
            }),
            "CPUExecutionProvider",
        ]
        session3 = ort.InferenceSession(model_file, opts3, providers=providers3)
        results.append(_run_ort_benchmark(session3, tokenizer, "CoreML EP (CPUOnly)"))
        logger.info("  -> %s: %.1fms", results[-1]["label"], results[-1]["median_ms"])
    except Exception as e:
        logger.error("CoreML EP (CPUOnly) 실패: %s", e)
        results.append({"label": "CoreML EP (CPUOnly)", "error": str(e)})

    return results


def benchmark_qdq_with_config() -> list[dict[str, Any]]:
    """기존 QDQ INT8 + 새 Session Config 조합 벤치마크."""
    import onnxruntime as ort

    if not QDQ_DIR.exists():
        logger.warning("QDQ 모델 없음 (%s), 건너뜀", QDQ_DIR)
        return []

    model_file = str(QDQ_DIR / "model_optimized.onnx")
    tokenizer = _load_tokenizer(QDQ_DIR)
    results: list[dict[str, Any]] = []

    # 3a. QDQ 현재 설정
    logger.info("\n=== 3a. QDQ INT8 (현재 설정) ===")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(model_file, opts, providers=["CPUExecutionProvider"])
    results.append(_run_ort_benchmark(session, tokenizer, "QDQ INT8 (현재 설정)"))
    logger.info("  -> %s: %.1fms", results[-1]["label"], results[-1]["median_ms"])

    # 3b. QDQ + 최적화 Config
    logger.info("\n=== 3b. QDQ INT8 + Session Config 최적화 ===")
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
    logger.info("  -> %s: %.1fms", results[-1]["label"], results[-1]["median_ms"])

    return results


def benchmark_io_binding() -> list[dict[str, Any]]:
    """IO Binding 벤치마크 (CPU EP 오버헤드/이점 측정)."""
    import onnxruntime as ort

    model_file = str(ORT_OPT_DIR / "model_optimized.onnx")
    tokenizer = _load_tokenizer(ORT_OPT_DIR)
    results: list[dict[str, Any]] = []

    # 세션 생성 (기본 설정)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(model_file, opts, providers=["CPUExecutionProvider"])
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]

    # 4a. 기본 session.run (비교 기준)
    logger.info("\n=== 4a. 기본 session.run (IO Binding 비교 기준) ===")
    results.append(_run_ort_benchmark(session, tokenizer, "session.run (기본)"))
    logger.info("  -> %s: %.1fms", results[-1]["label"], results[-1]["median_ms"])

    # 4b. IO Binding
    logger.info("\n=== 4b. IO Binding ===")
    try:
        queries = [QUERY] * len(DOCUMENTS)

        # Warmup
        for _ in range(NUM_WARMUP):
            warmup_inputs = tokenizer(
                QUERY, DOCUMENTS[0],
                return_tensors="np", padding=True, truncation=True, max_length=512,
            )
            io_binding = session.io_binding()
            feed_warmup: dict[str, np.ndarray] = {
                "input_ids": warmup_inputs["input_ids"],
                "attention_mask": warmup_inputs["attention_mask"],
            }
            if "token_type_ids" in input_names:
                feed_warmup["token_type_ids"] = warmup_inputs.get(
                    "token_type_ids", np.zeros_like(warmup_inputs["input_ids"]),
                )
            for name in input_names:
                if name in feed_warmup:
                    io_binding.bind_cpu_input(name, feed_warmup[name])
            for name in output_names:
                io_binding.bind_output(name, "cpu")
            session.run_with_iobinding(io_binding)

        # 벤치마크
        latencies: list[float] = []
        last_scores: np.ndarray | None = None

        for _ in range(NUM_ITERATIONS):
            inputs = tokenizer(
                queries, DOCUMENTS,
                return_tensors="np", padding=True, truncation=True, max_length=512,
            )
            feed: dict[str, np.ndarray] = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            }
            if "token_type_ids" in input_names:
                feed["token_type_ids"] = inputs.get(
                    "token_type_ids", np.zeros_like(inputs["input_ids"]),
                )

            io_binding = session.io_binding()
            for name in input_names:
                if name in feed:
                    io_binding.bind_cpu_input(name, feed[name])
            for name in output_names:
                io_binding.bind_output(name, "cpu")

            start = time.perf_counter()
            session.run_with_iobinding(io_binding)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

            if last_scores is None:
                ort_outputs = io_binding.copy_outputs_to_cpu()
                logits = ort_outputs[0]
                if logits.ndim == 2:
                    raw = logits[:, 0]
                else:
                    raw = logits.flatten()
                last_scores = np.array(_sigmoid(raw), dtype=np.float64)

        median_ms = statistics.median(latencies)
        mean_ms = statistics.mean(latencies)
        std_ms = statistics.stdev(latencies) if len(latencies) > 1 else 0.0

        io_result: dict[str, Any] = {
            "label": "IO Binding (CPU)",
            "median_ms": round(median_ms, 1),
            "mean_ms": round(mean_ms, 1),
            "std_ms": round(std_ms, 1),
            "min_ms": round(min(latencies), 1),
            "max_ms": round(max(latencies), 1),
            "scores": last_scores,
        }
        results.append(io_result)
        logger.info("  -> %s: %.1fms", io_result["label"], io_result["median_ms"])
    except Exception as e:
        logger.error("IO Binding 실패: %s", e)
        results.append({"label": "IO Binding (CPU)", "error": str(e)})

    return results


def run_profiling() -> dict[str, Any]:
    """ORT 프로파일링으로 리랭커 병목 분석."""
    import onnxruntime as ort

    model_file = str(ORT_OPT_DIR / "model_optimized.onnx")
    tokenizer = _load_tokenizer(ORT_OPT_DIR)

    logger.info("\n=== 프로파일링 실행 ===")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.enable_profiling = True
    opts.profile_file_prefix = str(PROJECT_ROOT / "ort_reranker_profile")

    session = ort.InferenceSession(model_file, opts, providers=["CPUExecutionProvider"])

    queries = [QUERY] * len(DOCUMENTS)
    inputs = tokenizer(
        queries, DOCUMENTS,
        return_tensors="np", padding=True, truncation=True, max_length=512,
    )
    input_names = [inp.name for inp in session.get_inputs()]
    feed: dict[str, np.ndarray] = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
    }
    if "token_type_ids" in input_names:
        feed["token_type_ids"] = inputs.get(
            "token_type_ids", np.zeros_like(inputs["input_ids"]),
        )

    session.run(None, feed)
    profile_file = session.end_profiling()
    logger.info("  -> 프로파일 저장: %s", profile_file)

    # 프로파일 분석
    with open(profile_file, encoding="utf-8") as f:
        profile_data = json.load(f)

    # 연산자별 시간 집계
    op_times: dict[str, float] = {}
    for event in profile_data:
        if isinstance(event, dict) and event.get("cat") == "Node":
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
        "top_ops": [
            {
                "op": op,
                "time_ms": round(t / 1000, 1),
                "pct": round(t / total_time * 100, 1),
            }
            for op, t in sorted_ops
        ],
        "total_time_ms": round(total_time / 1000, 1),
    }


# ── 메인 ──────────────────────────────────────────────


def main() -> None:
    """리랭커 신규 최적화 벤치마크 실행."""
    logger.info("=" * 60)
    logger.info("리랭커 ONNX 신규 최적화 벤치마크")
    logger.info("  모델: %s", RERANKER_MODEL)
    logger.info("  쿼리: %s", QUERY)
    logger.info("  문서 수: %d건", len(DOCUMENTS))
    logger.info("=" * 60)

    all_results: list[dict[str, Any]] = []

    # 0. 프로파일링 (병목 분석)
    logger.info("\n" + "-" * 60)
    logger.info("Phase 0: ORT 프로파일링")
    logger.info("-" * 60)
    profile_result = run_profiling()

    # 1. Session Config 최적화
    logger.info("\n" + "-" * 60)
    logger.info("Phase 1: Session Config 최적화")
    logger.info("-" * 60)
    config_results = benchmark_session_config()
    all_results.extend(config_results)

    # 2. CoreML EP
    logger.info("\n" + "-" * 60)
    logger.info("Phase 2: CoreML ExecutionProvider")
    logger.info("-" * 60)
    coreml_results = benchmark_coreml_ep()
    all_results.extend(coreml_results)

    # 3. QDQ INT8 + Config 조합
    logger.info("\n" + "-" * 60)
    logger.info("Phase 3: QDQ INT8 + Session Config")
    logger.info("-" * 60)
    qdq_results = benchmark_qdq_with_config()
    all_results.extend(qdq_results)

    # 4. IO Binding
    logger.info("\n" + "-" * 60)
    logger.info("Phase 4: IO Binding")
    logger.info("-" * 60)
    io_results = benchmark_io_binding()
    all_results.extend(io_results)

    # PyTorch baseline 점수 + 시간 측정
    logger.info("\n" + "-" * 60)
    logger.info("PyTorch FP32 baseline 측정")
    logger.info("-" * 60)
    tokenizer_for_baseline = _load_tokenizer(ORT_OPT_DIR)
    pytorch_start = time.perf_counter()
    baseline_scores = _get_baseline_scores(tokenizer_for_baseline, ORT_OPT_DIR)
    pytorch_elapsed_ms = (time.perf_counter() - pytorch_start) * 1000

    if baseline_scores.size > 0:
        logger.info("  PyTorch FP32 추론: %.1fms (모델 로드 포함)", pytorch_elapsed_ms)
    else:
        logger.warning("  PyTorch baseline 생성 실패, 품질 비교 건너뜀")
        # 기존 벤치마크 참고값 사용
        pytorch_elapsed_ms = 700.0

    # 품질 비교 (Pearson, Spearman)
    logger.info("\n" + "-" * 60)
    logger.info("품질 비교 (vs PyTorch FP32)")
    logger.info("-" * 60)

    for result in all_results:
        scores = result.pop("scores", None)
        if scores is not None and baseline_scores.size > 0:
            pearson = _pearson_correlation(baseline_scores, scores)
            spearman = _spearman_correlation(baseline_scores, scores)
            result["pearson"] = round(pearson, 6)
            result["spearman"] = round(spearman, 6)
            logger.info(
                "  %-35s Pearson=%.4f  Spearman=%.4f",
                result["label"], pearson, spearman,
            )
        elif "error" not in result:
            result["pearson"] = None
            result["spearman"] = None

    # 종합 결과 테이블
    logger.info("\n" + "=" * 60)
    logger.info("종합 결과")
    logger.info("=" * 60)
    logger.info(
        "%-38s %10s %8s %8s %9s",
        "Variant", "Median(ms)", "Speedup", "Pearson", "Spearman",
    )
    logger.info("-" * 75)

    for result in all_results:
        if "error" in result:
            logger.info("%-38s %s", result["label"], f"ERROR: {str(result['error'])[:40]}")
            continue
        median = result["median_ms"]
        speedup = pytorch_elapsed_ms / median if median > 0 else 0
        pearson_str = f"{result.get('pearson', 'N/A')}" if result.get("pearson") is not None else "N/A"
        spearman_str = f"{result.get('spearman', 'N/A')}" if result.get("spearman") is not None else "N/A"
        logger.info(
            "%-38s %10.1f %7.2fx %8s %9s",
            result["label"], median, speedup, pearson_str, spearman_str,
        )

    # JSON 저장
    save_data: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": RERANKER_MODEL,
        "query": QUERY,
        "num_documents": len(DOCUMENTS),
        "num_warmup": NUM_WARMUP,
        "num_iterations": NUM_ITERATIONS,
        "pytorch_baseline_ms": round(pytorch_elapsed_ms, 1),
        "quality_thresholds": QUALITY_THRESHOLDS,
        "profiling": profile_result,
        "results": [
            {k: v for k, v in r.items() if k != "scores"}
            for r in all_results
        ],
    }
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    logger.info("\n결과 저장: %s", RESULTS_FILE)


if __name__ == "__main__":
    main()
