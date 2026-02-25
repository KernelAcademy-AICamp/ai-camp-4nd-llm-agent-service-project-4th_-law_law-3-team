"""ONNX 고급 최적화 벤치마크

이전 벤치마크에서 ONNX 그래프 최적화(O2/O3) 단독으로는 효과가 없었고, O3+INT8만 1.3x 개선.
핵심 원인: optimum의 2GB+ 모델 외부 데이터 처리 버그, Transformer MatMul 지배, 런타임 이중 최적화.

4가지 추가 최적화를 임베딩(KURE-v1) + 리랭커(bge-reranker) 양쪽 모두 실험:
  1. O3+INT8 baseline 재확인
  2. intra_op_num_threads 튜닝
  3. torch.compile (PyTorch 2.x 네이티브 최적화)
  4. onnxruntime.transformers.optimizer 직접 사용 (optimum 버그 우회)

TODO: torch.compile (Phase 3)은 Windows에서 Triton 미지원으로 스킵됨.
  WSL2 Linux 환경에서 재실험 필요 (Triton은 Linux + CUDA 전용).
  WSL2 실험 시: `uv run python scripts/benchmark_advanced_optimization.py --skip-ort-opt --skip-threads`

사용법:
  cd backend && uv run python scripts/benchmark_advanced_optimization.py
  cd backend && uv run python scripts/benchmark_advanced_optimization.py --skip-compile
  cd backend && uv run python scripts/benchmark_advanced_optimization.py --skip-ort-opt
  cd backend && uv run python scripts/benchmark_advanced_optimization.py --skip-threads
  cd backend && uv run python scripts/benchmark_advanced_optimization.py --skip-quality
  cd backend && uv run python scripts/benchmark_advanced_optimization.py --embedding-only
  cd backend && uv run python scripts/benchmark_advanced_optimization.py --reranker-only
"""

import argparse
import gc
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings

warnings.filterwarnings(
    "ignore",
    message=".*incorrect regex pattern.*",
    category=UserWarning,
)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

# ============================================================
# 상수
# ============================================================

SEPARATOR = "=" * 60
WARMUP_RUNS = 2
REPEAT_RUNS = 5

# --- 임베딩 모델 ---
EMB_MODEL_NAME = "nlpai-lab/KURE-v1"
CACHE_DIR = str(PROJECT_ROOT / "data" / "models")
EMB_ONNX_DIR = PROJECT_ROOT / "data" / "models" / "kure-v1-onnx"
EMB_ONNX_O3_INT8_DIR = PROJECT_ROOT / "data" / "models" / "kure-v1-onnx-o3-int8"
EMB_ORT_OPT_DIR = PROJECT_ROOT / "data" / "models" / "kure-v1-ort-optimized"
EMB_ORT_OPT_INT8_DIR = PROJECT_ROOT / "data" / "models" / "kure-v1-ort-opt-int8"

# KURE-v1: XLM-RoBERTa Large
EMB_NUM_HEADS = 16
EMB_HIDDEN_SIZE = 1024

# --- 리랭커 모델 ---
RR_MODEL_NAME = "dragonkue/bge-reranker-v2-m3-ko"
RR_ONNX_DIR = PROJECT_ROOT / "data" / "models" / "reranker-onnx"
RR_ONNX_O3_INT8_DIR = PROJECT_ROOT / "data" / "models" / "reranker-onnx-o3-int8"
RR_ORT_OPT_DIR = PROJECT_ROOT / "data" / "models" / "reranker-ort-optimized"
RR_ORT_OPT_INT8_DIR = PROJECT_ROOT / "data" / "models" / "reranker-ort-opt-int8"

# bge-reranker-v2-m3-ko: XLM-RoBERTa Large (config.json에서 확인)
RR_NUM_HEADS = 16
RR_HIDDEN_SIZE = 1024

DEFAULT_REPORT_PATH = (
    PROJECT_ROOT.parent / "docs" / "04-report" / "features"
    / "advanced-optimization-benchmark.md"
)

# --- 테스트 데이터 (임베딩) ---
EMB_TEST_QUERIES = [
    "교통사고 손해배상 판례",
    "임대차 보증금 반환 청구",
    "근로기준법 해고 부당해고",
    "이혼 재산분할 위자료",
    "명예훼손 형사 고소",
    "상속 포기 절차와 기한",
    "민법 제750조 불법행위 요건",
    "형사소송법 증거능력 배제",
    "행정소송 처분 취소 요건",
    "특허권 침해 금지 청구",
]

EMB_TEST_DOCUMENTS = [
    "피고는 원고에게 금 50,000,000원 및 이에 대한 지연손해금을 지급하라.",
    "임대차보증금 반환 청구 사건에서 임대인은 보증금 전액을 반환할 의무가 있다.",
    "근로기준법 제23조 제1항은 정당한 이유 없이 해고를 하지 못한다고 규정한다.",
    "민법 제750조에 의한 불법행위 손해배상 책임이 성립하려면 위법성이 있어야 한다.",
    "형법 제307조 제1항의 명예훼손죄가 성립하려면 사실을 적시하여야 한다.",
    "이혼 시 재산분할은 혼인 중 쌍방의 협력으로 이룩한 재산을 대상으로 한다.",
    "상속의 포기는 상속개시 있음을 안 날로부터 3월 내에 가정법원에 신고하여야 한다.",
    "형사소송법 제308조의2에 의하면 위법수집증거 배제법칙이 적용된다.",
    "행정소송법 제19조에 의하면 취소소송은 처분일로부터 90일 이내에 제기하여야 한다.",
    "특허법 제126조에 의한 특허권 침해금지청구에서 보호범위가 핵심 쟁점이다.",
    "자동차손해배상 보장법 제3조는 운행자의 배상책임을 규정한다.",
    "후유장해 등급 판정에서는 맥브라이드 장해평가 방법에 의한다.",
    "과실상계에서 피해자의 과실은 약한 의미의 부주의를 포함한다.",
    "국가배상법 제2조는 공무원의 직무상 불법행위에 대한 배상책임을 규정한다.",
    "채무불이행 특별손해는 채무자의 예견가능성이 있을 때 배상책임이 있다.",
    "민사소송법 제202조에 의한 자유심증주의 원칙이 적용된다.",
    "교통사고처리특례법 제4조 단서는 피해자 의사와 관계없이 공소 제기가 가능하다.",
    "자동차종합보험에서 보험회사는 피해자에게 직접 보험금을 지급할 의무가 있다.",
    "위자료 산정은 나이, 직업, 재산상태 등을 종합적으로 고려한다.",
    "불법행위 손해배상 청구권의 소멸시효는 안 날로부터 3년이다.",
]

# --- 테스트 데이터 (리랭커) ---
RR_QUERY = "교통사고 손해배상 판례"
RR_DOCUMENTS = [
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


# ============================================================
# 공통 유틸리티
# ============================================================


def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    """Sigmoid 활성화 함수."""
    return 1 / (1 + np.exp(-x))


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """두 벡터 배열의 평균 pairwise cosine similarity."""
    sims = []
    for i in range(len(a)):
        dot = np.dot(a[i], b[i])
        norm_a = np.linalg.norm(a[i])
        norm_b = np.linalg.norm(b[i])
        if norm_a > 0 and norm_b > 0:
            sims.append(dot / (norm_a * norm_b))
    return float(np.mean(sims)) if sims else 0.0


def _measure_median(
    func: Any,
    warmup: int = WARMUP_RUNS,
    repeat: int = REPEAT_RUNS,
) -> tuple[Any, float]:
    """함수를 여러 번 실행하여 median 시간(ms)과 마지막 결과를 반환."""
    for _ in range(warmup):
        func()

    times: list[float] = []
    result = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = func()
        times.append((time.perf_counter() - t0) * 1000)

    return result, float(np.median(times))


def _copy_tokenizer_files(src_dir: Path, dst_dir: Path) -> None:
    """토크나이저 + config 파일을 복사."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(src_dir), trust_remote_code=True)
    tokenizer.save_pretrained(str(dst_dir))

    config_src = src_dir / "config.json"
    if config_src.exists():
        shutil.copy2(str(config_src), str(dst_dir / "config.json"))


# ============================================================
# 임베딩 함수
# ============================================================


def _encode_pytorch(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """PyTorch FP32 임베딩."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        EMB_MODEL_NAME,
        cache_folder=CACHE_DIR,
        trust_remote_code=True,
        local_files_only=True,
    )
    embeddings = model.encode(
        texts, batch_size=batch_size,
        show_progress_bar=False, normalize_embeddings=True,
    )
    result = np.array(embeddings)
    del model
    gc.collect()
    return result


def _encode_onnx_raw(
    texts: list[str],
    model_dir: Path,
    file_name: str,
    num_threads: int = 0,
) -> np.ndarray:
    """ORT 직접 세션으로 임베딩 생성 (mean pooling + L2 norm)."""
    import onnxruntime as ort
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    model_path = str(model_dir / file_name)

    session_options = ort.SessionOptions()
    if num_threads > 0:
        session_options.intra_op_num_threads = num_threads
        session_options.inter_op_num_threads = 1
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        model_path, sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    input_names = [inp.name for inp in session.get_inputs()]
    inputs = tokenizer(
        texts, return_tensors="np", padding=True,
        truncation=True, max_length=512,
    )
    feed: dict[str, Any] = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
    }
    if "token_type_ids" in input_names:
        feed["token_type_ids"] = inputs.get(
            "token_type_ids", np.zeros_like(inputs["input_ids"]),
        )

    outputs = session.run(None, feed)

    # CLS pooling + L2 norm (KURE-v1은 pooling_mode_cls_token=True)
    emb = outputs[0][:, 0, :]  # CLS 토큰 (index 0)
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.clip(norm, a_min=1e-9, a_max=None)

    del session, tokenizer
    gc.collect()
    return emb


def _encode_compiled(
    texts: list[str],
    mode: str = "default",
    batch_size: int = 32,
) -> np.ndarray:
    """torch.compile로 최적화된 임베딩."""
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        EMB_MODEL_NAME,
        cache_folder=CACHE_DIR,
        trust_remote_code=True,
        local_files_only=True,
    )
    model[0].auto_model = torch.compile(  # type: ignore[assignment]
        model[0].auto_model, mode=mode,  # type: ignore[arg-type]
    )
    embeddings = model.encode(
        texts, batch_size=batch_size,
        show_progress_bar=False, normalize_embeddings=True,
    )
    result = np.array(embeddings)
    del model
    gc.collect()
    return result


# ============================================================
# 리랭커 함수
# ============================================================


def _rerank_pytorch(query: str, documents: list[str]) -> list[float]:
    """PyTorch FP32 리랭킹."""
    import torch
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(RR_MODEL_NAME, activation_fn=torch.nn.Sigmoid())
    pairs = [(query, doc) for doc in documents]
    result = model.predict(pairs)
    scores = [float(s) for s in result]
    del model
    gc.collect()
    return scores


def _rerank_onnx_raw(
    query: str,
    documents: list[str],
    model_dir: Path,
    file_name: str,
    num_threads: int = 0,
) -> list[float]:
    """ORT 직접 세션으로 리랭킹 (배치)."""
    import onnxruntime as ort
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model_path = str(model_dir / file_name)

    session_options = ort.SessionOptions()
    if num_threads > 0:
        session_options.intra_op_num_threads = num_threads
        session_options.inter_op_num_threads = 1
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        model_path, sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    queries = [query] * len(documents)
    inputs = tokenizer(
        queries, documents, return_tensors="np",
        padding=True, truncation=True, max_length=512,
    )

    input_names = [inp.name for inp in session.get_inputs()]
    feed: dict[str, Any] = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
    }
    if "token_type_ids" in input_names:
        feed["token_type_ids"] = inputs.get(
            "token_type_ids", np.zeros_like(inputs["input_ids"]),
        )

    outputs = session.run(None, feed)
    logits = outputs[0][:, 0] if outputs[0].ndim == 2 else outputs[0].flatten()
    scores = _sigmoid(logits).tolist()

    del session, tokenizer
    gc.collect()
    return scores


def _rerank_compiled(
    query: str,
    documents: list[str],
    mode: str = "default",
) -> list[float]:
    """torch.compile로 최적화된 리랭킹."""
    import torch
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(RR_MODEL_NAME, activation_fn=torch.nn.Sigmoid())
    model.model = torch.compile(model.model, mode=mode)  # type: ignore[assignment]
    pairs = [(query, doc) for doc in documents]
    result = model.predict(pairs)
    scores = [float(s) for s in result]
    del model
    gc.collect()
    return scores


# ============================================================
# Phase 0: 환경 준비
# ============================================================


def run_phase_0() -> dict[str, int]:
    """환경 정보 출력."""
    print(f"\n{SEPARATOR}")
    print("  Phase 0: 환경 준비")
    print(SEPARATOR)

    physical_cores = os.cpu_count() // 2 if os.cpu_count() else 1
    logical_cores = os.cpu_count() or 1

    try:
        import psutil
        physical_cores = psutil.cpu_count(logical=False) or physical_cores
    except ImportError:
        pass

    print(f"  OS:              {platform.system()} {platform.release()}")
    print(f"  CPU:             {platform.processor()}")
    print(f"  물리 코어:       {physical_cores}")
    print(f"  논리 코어:       {logical_cores}")

    import onnxruntime as ort
    print(f"  onnxruntime:     {ort.__version__}")

    import torch
    print(f"  PyTorch:         {torch.__version__}")
    print(f"  torch.compile:   {'사용 가능' if hasattr(torch, 'compile') else '미지원'}")

    # 모델 디렉토리 확인
    print("\n  모델 디렉토리 존재 확인:")
    for label, d in [
        ("임베딩 ONNX FP32", EMB_ONNX_DIR),
        ("임베딩 ONNX O3+INT8", EMB_ONNX_O3_INT8_DIR),
        ("리랭커 ONNX FP32", RR_ONNX_DIR),
        ("리랭커 ONNX O3+INT8", RR_ONNX_O3_INT8_DIR),
    ]:
        status = "✅" if d.exists() else "❌"
        print(f"    {status} {label}: {d.name}")

    return {"physical_cores": physical_cores, "logical_cores": logical_cores}


# ============================================================
# Phase 1: Baseline 재측정
# ============================================================


def run_phase_1(
    *,
    run_embedding: bool = True,
    run_reranker: bool = True,
) -> dict[str, dict[str, Any]]:
    """Phase 1: PyTorch FP32 + ONNX O3+INT8 baseline."""
    print(f"\n{SEPARATOR}")
    print("  Phase 1: Baseline 재측정 (PyTorch FP32 vs ONNX O3+INT8)")
    print(SEPARATOR)

    results: dict[str, dict[str, Any]] = {}
    texts = EMB_TEST_QUERIES + EMB_TEST_DOCUMENTS

    if run_embedding:
        print("\n  === 임베딩 ===")

        # PyTorch FP32
        print("  [PyTorch FP32] 측정 중...")
        emb_pt, pt_ms = _measure_median(lambda: _encode_pytorch(texts))
        print(f"    {len(texts)}건 인코딩: {pt_ms:.1f}ms")
        results["emb_pytorch"] = {"time_ms": pt_ms, "embeddings": emb_pt}

        # ONNX O3+INT8
        if EMB_ONNX_O3_INT8_DIR.exists():
            print("  [ONNX O3+INT8] 측정 중...")
            emb_o3int8, o3int8_ms = _measure_median(
                lambda: _encode_onnx_raw(texts, EMB_ONNX_O3_INT8_DIR, "model_quantized.onnx"),
            )
            speedup = pt_ms / o3int8_ms if o3int8_ms > 0 else 0
            print(f"    {len(texts)}건 인코딩: {o3int8_ms:.1f}ms ({speedup:.2f}x)")
            results["emb_o3_int8"] = {
                "time_ms": o3int8_ms, "speedup": speedup, "embeddings": emb_o3int8,
            }
        else:
            print("  [건너뜀] ONNX O3+INT8 디렉토리 없음")

    if run_reranker:
        print("\n  === 리랭커 ===")

        # PyTorch FP32
        print("  [PyTorch FP32] 측정 중...")
        rr_pt, rr_pt_ms = _measure_median(
            lambda: _rerank_pytorch(RR_QUERY, RR_DOCUMENTS),
        )
        print(f"    {len(RR_DOCUMENTS)}건 리랭킹: {rr_pt_ms:.1f}ms")
        results["rr_pytorch"] = {"time_ms": rr_pt_ms, "scores": rr_pt}

        # ONNX O3+INT8
        if RR_ONNX_O3_INT8_DIR.exists():
            print("  [ONNX O3+INT8] 측정 중...")
            rr_o3int8, rr_o3int8_ms = _measure_median(
                lambda: _rerank_onnx_raw(
                    RR_QUERY, RR_DOCUMENTS, RR_ONNX_O3_INT8_DIR, "model_quantized.onnx",
                ),
            )
            speedup = rr_pt_ms / rr_o3int8_ms if rr_o3int8_ms > 0 else 0
            print(f"    {len(RR_DOCUMENTS)}건 리랭킹: {rr_o3int8_ms:.1f}ms ({speedup:.2f}x)")
            results["rr_o3_int8"] = {
                "time_ms": rr_o3int8_ms, "speedup": speedup, "scores": rr_o3int8,
            }
        else:
            print("  [건너뜀] 리랭커 ONNX O3+INT8 디렉토리 없음")

    return results


# ============================================================
# Phase 2: intra_op_num_threads 튜닝
# ============================================================


def run_phase_2(
    hw_info: dict[str, int],
    *,
    run_embedding: bool = True,
    run_reranker: bool = True,
) -> dict[str, dict[str, Any]]:
    """Phase 2: intra_op_num_threads 튜닝."""
    print(f"\n{SEPARATOR}")
    print("  Phase 2: intra_op_num_threads 튜닝")
    print(SEPARATOR)

    physical = hw_info["physical_cores"]
    logical = hw_info["logical_cores"]

    # 중복 제거 + 정렬
    thread_configs = sorted(set([1, 2, 4, physical, logical]))
    print(f"  스레드 설정: {thread_configs}")

    results: dict[str, dict[str, Any]] = {}
    texts = EMB_TEST_QUERIES + EMB_TEST_DOCUMENTS

    if run_embedding and EMB_ONNX_O3_INT8_DIR.exists():
        print("\n  === 임베딩 (ONNX O3+INT8) ===")
        emb_thread_results: dict[int, float] = {}

        for n_threads in thread_configs:
            _, ms = _measure_median(
                lambda nt=n_threads: _encode_onnx_raw(
                    texts, EMB_ONNX_O3_INT8_DIR, "model_quantized.onnx",
                    num_threads=nt,
                ),
            )
            emb_thread_results[n_threads] = ms
            print(f"    threads={n_threads:<3d}: {ms:.1f}ms")

        best_threads = min(emb_thread_results, key=emb_thread_results.get)  # type: ignore[arg-type]
        print(f"  → 최적: threads={best_threads} ({emb_thread_results[best_threads]:.1f}ms)")
        results["emb_threads"] = {
            "results": emb_thread_results,
            "best_threads": best_threads,
            "best_ms": emb_thread_results[best_threads],
        }

    if run_reranker and RR_ONNX_O3_INT8_DIR.exists():
        print("\n  === 리랭커 (ONNX O3+INT8) ===")
        rr_thread_results: dict[int, float] = {}

        for n_threads in thread_configs:
            _, ms = _measure_median(
                lambda nt=n_threads: _rerank_onnx_raw(
                    RR_QUERY, RR_DOCUMENTS, RR_ONNX_O3_INT8_DIR,
                    "model_quantized.onnx", num_threads=nt,
                ),
            )
            rr_thread_results[n_threads] = ms
            print(f"    threads={n_threads:<3d}: {ms:.1f}ms")

        best_threads = min(rr_thread_results, key=rr_thread_results.get)  # type: ignore[arg-type]
        print(f"  → 최적: threads={best_threads} ({rr_thread_results[best_threads]:.1f}ms)")
        results["rr_threads"] = {
            "results": rr_thread_results,
            "best_threads": best_threads,
            "best_ms": rr_thread_results[best_threads],
        }

    return results


# ============================================================
# Phase 3: torch.compile 모드 비교
# ============================================================


def run_phase_3(
    *,
    run_embedding: bool = True,
    run_reranker: bool = True,
) -> dict[str, dict[str, Any]]:
    """Phase 3: torch.compile 모드 비교."""
    print(f"\n{SEPARATOR}")
    print("  Phase 3: torch.compile 모드 비교")
    print(SEPARATOR)

    import torch

    if not hasattr(torch, "compile"):
        print("  [건너뜀] torch.compile 미지원 (PyTorch 2.0+ 필요)")
        return {}

    compile_modes = ["default", "reduce-overhead", "max-autotune"]
    results: dict[str, dict[str, Any]] = {}
    texts = EMB_TEST_QUERIES + EMB_TEST_DOCUMENTS

    if run_embedding:
        print("\n  === 임베딩 ===")
        emb_compile_results: dict[str, float] = {}

        for mode in compile_modes:
            print(f"  [{mode}] 컴파일 + 측정 중 (시간 소요)...")
            try:
                # 워밍업 3회로 컴파일 안정화
                _, ms = _measure_median(
                    lambda m=mode: _encode_compiled(texts, mode=m),
                    warmup=3,
                )
                emb_compile_results[mode] = ms
                print(f"    {mode}: {ms:.1f}ms")
            except Exception as e:
                print(f"    {mode}: 실패 ({e!s:.80s})")

        if emb_compile_results:
            best = min(emb_compile_results, key=emb_compile_results.get)  # type: ignore[arg-type]
            print(f"  → 최적: {best} ({emb_compile_results[best]:.1f}ms)")
            results["emb_compile"] = {
                "results": emb_compile_results,
                "best_mode": best,
                "best_ms": emb_compile_results[best],
            }

    if run_reranker:
        print("\n  === 리랭커 ===")
        rr_compile_results: dict[str, float] = {}

        for mode in compile_modes:
            print(f"  [{mode}] 컴파일 + 측정 중 (시간 소요)...")
            try:
                _, ms = _measure_median(
                    lambda m=mode: _rerank_compiled(RR_QUERY, RR_DOCUMENTS, mode=m),
                    warmup=3,
                )
                rr_compile_results[mode] = ms
                print(f"    {mode}: {ms:.1f}ms")
            except Exception as e:
                print(f"    {mode}: 실패 ({e!s:.80s})")

        if rr_compile_results:
            best = min(rr_compile_results, key=rr_compile_results.get)  # type: ignore[arg-type]
            print(f"  → 최적: {best} ({rr_compile_results[best]:.1f}ms)")
            results["rr_compile"] = {
                "results": rr_compile_results,
                "best_mode": best,
                "best_ms": rr_compile_results[best],
            }

    return results


# ============================================================
# Phase 4: onnxruntime.transformers.optimizer 직접 사용
# ============================================================


def _optimize_with_ort(
    src_onnx_dir: Path,
    src_file: str,
    dst_dir: Path,
    num_heads: int,
    hidden_size: int,
    label: str,
) -> bool:
    """ORT transformer optimizer로 모델 최적화."""
    dst_file = dst_dir / "model_optimized.onnx"
    if dst_file.exists():
        print(f"  [{label}] 이미 존재: {dst_dir.name}")
        return True

    print(f"  [{label}] ORT transformer optimizer 실행 중...")
    try:
        from onnxruntime.transformers.optimizer import optimize_model

        dst_dir.mkdir(parents=True, exist_ok=True)

        optimized = optimize_model(
            str(src_onnx_dir / src_file),
            model_type="bert",
            num_heads=num_heads,
            hidden_size=hidden_size,
        )

        optimized.save_model_to_file(
            str(dst_file),
            use_external_data_format=True,
        )

        # 외부 데이터 파일 검증
        data_file = dst_dir / "model_optimized.onnx.data"
        if data_file.exists():
            size_mb = data_file.stat().st_size / (1024 * 1024)
            print(f"    그래프: {dst_file.stat().st_size // 1024}KB, 가중치: {size_mb:.0f}MB")
        else:
            model_size = dst_file.stat().st_size / (1024 * 1024)
            print(f"    모델 단일 파일: {model_size:.0f}MB")

        _copy_tokenizer_files(src_onnx_dir, dst_dir)
        print(f"    완료: {dst_dir.name}")
        return True
    except Exception as e:
        print(f"    실패: {e!s:.120s}")
        return False


def _quantize_ort_optimized(
    src_dir: Path,
    dst_dir: Path,
    label: str,
) -> bool:
    """ORT 최적화 모델에 INT8 양자화 적용."""
    dst_file = dst_dir / "model_quantized.onnx"
    if dst_file.exists():
        print(f"  [{label}] 이미 존재: {dst_dir.name}")
        return True

    print(f"  [{label}] INT8 양자화 중...")
    try:
        import onnx
        from onnxruntime.quantization import QuantType, quantize_dynamic

        src_model = str(src_dir / "model_optimized.onnx")
        dst_dir.mkdir(parents=True, exist_ok=True)

        quantize_dynamic(
            model_input=src_model,
            model_output=str(dst_file),
            per_channel=True,
            weight_type=QuantType.QInt8,
            extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
        )

        _copy_tokenizer_files(src_dir, dst_dir)
        size_mb = dst_file.stat().st_size / (1024 * 1024)
        print(f"    완료: {size_mb:.0f}MB")
        return True
    except Exception as e:
        print(f"    실패: {e!s:.120s}")
        return False


def run_phase_4(
    *,
    run_embedding: bool = True,
    run_reranker: bool = True,
) -> dict[str, dict[str, Any]]:
    """Phase 4: ORT transformer optimizer 직접 최적화 + 벤치마크."""
    print(f"\n{SEPARATOR}")
    print("  Phase 4: onnxruntime.transformers.optimizer 직접 사용")
    print(SEPARATOR)

    results: dict[str, dict[str, Any]] = {}
    texts = EMB_TEST_QUERIES + EMB_TEST_DOCUMENTS

    if run_embedding and EMB_ONNX_DIR.exists():
        print("\n  === 임베딩 모델 최적화 ===")

        # ORT optimizer 실행
        opt_ok = _optimize_with_ort(
            EMB_ONNX_DIR, "model.onnx", EMB_ORT_OPT_DIR,
            EMB_NUM_HEADS, EMB_HIDDEN_SIZE, "임베딩 ORT opt",
        )
        if opt_ok:
            # ORT opt 벤치마크
            print("  [임베딩 ORT opt] 속도 측정 중...")
            _, ms = _measure_median(
                lambda: _encode_onnx_raw(texts, EMB_ORT_OPT_DIR, "model_optimized.onnx"),
            )
            print(f"    {len(texts)}건 인코딩: {ms:.1f}ms")
            results["emb_ort_opt"] = {"time_ms": ms}

            # ORT opt + INT8
            int8_ok = _quantize_ort_optimized(
                EMB_ORT_OPT_DIR, EMB_ORT_OPT_INT8_DIR, "임베딩 ORT opt+INT8",
            )
            if int8_ok:
                print("  [임베딩 ORT opt+INT8] 속도 측정 중...")
                _, ms = _measure_median(
                    lambda: _encode_onnx_raw(
                        texts, EMB_ORT_OPT_INT8_DIR, "model_quantized.onnx",
                    ),
                )
                print(f"    {len(texts)}건 인코딩: {ms:.1f}ms")
                results["emb_ort_opt_int8"] = {"time_ms": ms}

    if run_reranker and RR_ONNX_DIR.exists():
        print("\n  === 리랭커 모델 최적화 ===")

        opt_ok = _optimize_with_ort(
            RR_ONNX_DIR, "model.onnx", RR_ORT_OPT_DIR,
            RR_NUM_HEADS, RR_HIDDEN_SIZE, "리랭커 ORT opt",
        )
        if opt_ok:
            print("  [리랭커 ORT opt] 속도 측정 중...")
            _, ms = _measure_median(
                lambda: _rerank_onnx_raw(
                    RR_QUERY, RR_DOCUMENTS, RR_ORT_OPT_DIR, "model_optimized.onnx",
                ),
            )
            print(f"    {len(RR_DOCUMENTS)}건 리랭킹: {ms:.1f}ms")
            results["rr_ort_opt"] = {"time_ms": ms}

            int8_ok = _quantize_ort_optimized(
                RR_ORT_OPT_DIR, RR_ORT_OPT_INT8_DIR, "리랭커 ORT opt+INT8",
            )
            if int8_ok:
                print("  [리랭커 ORT opt+INT8] 속도 측정 중...")
                _, ms = _measure_median(
                    lambda: _rerank_onnx_raw(
                        RR_QUERY, RR_DOCUMENTS, RR_ORT_OPT_INT8_DIR, "model_quantized.onnx",
                    ),
                )
                print(f"    {len(RR_DOCUMENTS)}건 리랭킹: {ms:.1f}ms")
                results["rr_ort_opt_int8"] = {"time_ms": ms}

    return results


# ============================================================
# Phase 5: 종합 비교 + 품질 검증 + 보고서
# ============================================================


def _run_quality_check(
    baseline_results: dict[str, dict[str, Any]],
    phase2_results: dict[str, dict[str, Any]],
    phase3_results: dict[str, dict[str, Any]],
    phase4_results: dict[str, dict[str, Any]],
    *,
    run_embedding: bool = True,
    run_reranker: bool = True,
) -> dict[str, dict[str, float]]:
    """품질 검증 (임베딩: cosine, 리랭커: Pearson)."""
    from scipy.stats import pearsonr

    print(f"\n{SEPARATOR}")
    print("  품질 검증")
    print(SEPARATOR)

    quality: dict[str, dict[str, float]] = {}
    texts = EMB_TEST_QUERIES + EMB_TEST_DOCUMENTS

    if run_embedding:
        emb_pt = baseline_results.get("emb_pytorch", {}).get("embeddings")
        if emb_pt is not None:
            print("\n  === 임베딩 품질 (PyTorch FP32 vs 각 방법, cosine) ===")
            emb_methods: list[tuple[str, Any]] = []

            # O3+INT8
            if "emb_o3_int8" in baseline_results:
                emb_methods.append(("ONNX O3+INT8", baseline_results["emb_o3_int8"].get("embeddings")))

            # O3+INT8 + 최적 threads
            if "emb_threads" in phase2_results:
                best_t = phase2_results["emb_threads"]["best_threads"]
                emb = _encode_onnx_raw(
                    texts, EMB_ONNX_O3_INT8_DIR, "model_quantized.onnx",
                    num_threads=best_t,
                )
                emb_methods.append((f"O3+INT8 threads={best_t}", emb))

            # torch.compile (best mode)
            if "emb_compile" in phase3_results:
                best_mode = phase3_results["emb_compile"]["best_mode"]
                emb = _encode_compiled(texts, mode=best_mode)
                emb_methods.append((f"torch.compile ({best_mode})", emb))

            # ORT optimizer
            if EMB_ORT_OPT_DIR.exists():
                emb = _encode_onnx_raw(texts, EMB_ORT_OPT_DIR, "model_optimized.onnx")
                emb_methods.append(("ORT optimizer", emb))

            # ORT optimizer + INT8
            if EMB_ORT_OPT_INT8_DIR.exists():
                emb = _encode_onnx_raw(texts, EMB_ORT_OPT_INT8_DIR, "model_quantized.onnx")
                emb_methods.append(("ORT opt+INT8", emb))

            for label, test_emb in emb_methods:
                if test_emb is not None:
                    cos = _cosine_similarity(emb_pt, test_emb)
                    passed = "PASS" if cos >= 0.985 else "FAIL"
                    print(f"    {label:<30s}: {cos:.6f}  [{passed}]")
                    quality[f"emb_{label}"] = {"cosine": cos}

    if run_reranker:
        rr_pt = baseline_results.get("rr_pytorch", {}).get("scores")
        if rr_pt is not None:
            print("\n  === 리랭커 품질 (PyTorch FP32 vs 각 방법, Pearson) ===")
            rr_methods: list[tuple[str, Any]] = []

            # O3+INT8
            if "rr_o3_int8" in baseline_results:
                rr_methods.append(("ONNX O3+INT8", baseline_results["rr_o3_int8"].get("scores")))

            # O3+INT8 + 최적 threads
            if "rr_threads" in phase2_results:
                best_t = phase2_results["rr_threads"]["best_threads"]
                scores = _rerank_onnx_raw(
                    RR_QUERY, RR_DOCUMENTS, RR_ONNX_O3_INT8_DIR,
                    "model_quantized.onnx", num_threads=best_t,
                )
                rr_methods.append((f"O3+INT8 threads={best_t}", scores))

            # torch.compile (best mode)
            if "rr_compile" in phase3_results:
                best_mode = phase3_results["rr_compile"]["best_mode"]
                scores = _rerank_compiled(RR_QUERY, RR_DOCUMENTS, mode=best_mode)
                rr_methods.append((f"torch.compile ({best_mode})", scores))

            # ORT optimizer
            if RR_ORT_OPT_DIR.exists():
                scores = _rerank_onnx_raw(
                    RR_QUERY, RR_DOCUMENTS, RR_ORT_OPT_DIR, "model_optimized.onnx",
                )
                rr_methods.append(("ORT optimizer", scores))

            # ORT optimizer + INT8
            if RR_ORT_OPT_INT8_DIR.exists():
                scores = _rerank_onnx_raw(
                    RR_QUERY, RR_DOCUMENTS, RR_ORT_OPT_INT8_DIR, "model_quantized.onnx",
                )
                rr_methods.append(("ORT opt+INT8", scores))

            for label, test_scores in rr_methods:
                if test_scores is not None:
                    corr, _ = pearsonr(rr_pt, test_scores)
                    passed = "PASS" if corr >= 0.99 else "FAIL"
                    print(f"    {label:<30s}: {corr:.6f}  [{passed}]")
                    quality[f"rr_{label}"] = {"pearson": float(corr)}

    return quality


def _print_summary_table(
    baseline: dict[str, dict[str, Any]],
    phase2: dict[str, dict[str, Any]],
    phase3: dict[str, dict[str, Any]],
    phase4: dict[str, dict[str, Any]],
    quality: dict[str, dict[str, float]],
    *,
    run_embedding: bool = True,
    run_reranker: bool = True,
) -> str:
    """종합 비교 테이블 출력 + MD 보고서 문자열 반환."""
    lines: list[str] = []

    def p(text: str = "") -> None:
        print(text)
        lines.append(text)

    p(f"\n{SEPARATOR}")
    p("  종합 비교 결과")
    p(SEPARATOR)

    if run_embedding:
        emb_pt_ms = baseline.get("emb_pytorch", {}).get("time_ms", 0)

        p("\n  === 임베딩 비교 테이블 ===")
        p(f"  {'방법':<35s} | {'시간 (ms)':>10s} | {'상대 속도':>10s} | {'품질':>10s}")
        p(f"  {'-' * 35} | {'-' * 10} | {'-' * 10} | {'-' * 10}")
        p(f"  {'PyTorch FP32 (baseline)':<35s} | {emb_pt_ms:>8.1f}ms | {'1.0x':>10s} | {'1.000000':>10s}")

        emb_rows: list[tuple[str, float, str]] = []

        # O3+INT8
        if "emb_o3_int8" in baseline:
            ms = baseline["emb_o3_int8"]["time_ms"]
            cos_str = f"{quality.get('emb_ONNX O3+INT8', {}).get('cosine', 0):.6f}"
            emb_rows.append(("ONNX O3+INT8 (기존 optimum)", ms, cos_str))

        # O3+INT8 + threads
        if "emb_threads" in phase2:
            best_t = phase2["emb_threads"]["best_threads"]
            ms = phase2["emb_threads"]["best_ms"]
            cos_str = f"{quality.get(f'emb_O3+INT8 threads={best_t}', {}).get('cosine', 0):.6f}"
            emb_rows.append((f"O3+INT8 + threads={best_t}", ms, cos_str))

        # torch.compile
        if "emb_compile" in phase3:
            best_mode = phase3["emb_compile"]["best_mode"]
            ms = phase3["emb_compile"]["best_ms"]
            cos_str = f"{quality.get(f'emb_torch.compile ({best_mode})', {}).get('cosine', 0):.6f}"
            emb_rows.append((f"torch.compile ({best_mode})", ms, cos_str))

        # ORT optimizer
        if "emb_ort_opt" in phase4:
            ms = phase4["emb_ort_opt"]["time_ms"]
            cos_str = f"{quality.get('emb_ORT optimizer', {}).get('cosine', 0):.6f}"
            emb_rows.append(("ORT optimizer (직접)", ms, cos_str))

        # ORT opt + INT8
        if "emb_ort_opt_int8" in phase4:
            ms = phase4["emb_ort_opt_int8"]["time_ms"]
            cos_str = f"{quality.get('emb_ORT opt+INT8', {}).get('cosine', 0):.6f}"
            emb_rows.append(("ORT optimizer + INT8", ms, cos_str))

        for label, ms, cos_str in emb_rows:
            speedup = emb_pt_ms / ms if ms > 0 else 0
            p(f"  {label:<35s} | {ms:>8.1f}ms | {speedup:>8.2f}x | {cos_str:>10s}")

    if run_reranker:
        rr_pt_ms = baseline.get("rr_pytorch", {}).get("time_ms", 0)

        p("\n  === 리랭커 비교 테이블 ===")
        p(f"  {'방법':<35s} | {'시간 (ms)':>10s} | {'상대 속도':>10s} | {'품질':>10s}")
        p(f"  {'-' * 35} | {'-' * 10} | {'-' * 10} | {'-' * 10}")
        p(f"  {'PyTorch FP32 (baseline)':<35s} | {rr_pt_ms:>8.1f}ms | {'1.0x':>10s} | {'1.000000':>10s}")

        rr_rows: list[tuple[str, float, str]] = []

        if "rr_o3_int8" in baseline:
            ms = baseline["rr_o3_int8"]["time_ms"]
            pr_str = f"{quality.get('rr_ONNX O3+INT8', {}).get('pearson', 0):.6f}"
            rr_rows.append(("ONNX O3+INT8 (기존)", ms, pr_str))

        if "rr_threads" in phase2:
            best_t = phase2["rr_threads"]["best_threads"]
            ms = phase2["rr_threads"]["best_ms"]
            pr_str = f"{quality.get(f'rr_O3+INT8 threads={best_t}', {}).get('pearson', 0):.6f}"
            rr_rows.append((f"O3+INT8 + threads={best_t}", ms, pr_str))

        if "rr_compile" in phase3:
            best_mode = phase3["rr_compile"]["best_mode"]
            ms = phase3["rr_compile"]["best_ms"]
            pr_str = f"{quality.get(f'rr_torch.compile ({best_mode})', {}).get('pearson', 0):.6f}"
            rr_rows.append((f"torch.compile ({best_mode})", ms, pr_str))

        if "rr_ort_opt" in phase4:
            ms = phase4["rr_ort_opt"]["time_ms"]
            pr_str = f"{quality.get('rr_ORT optimizer', {}).get('pearson', 0):.6f}"
            rr_rows.append(("ORT optimizer (직접)", ms, pr_str))

        if "rr_ort_opt_int8" in phase4:
            ms = phase4["rr_ort_opt_int8"]["time_ms"]
            pr_str = f"{quality.get('rr_ORT opt+INT8', {}).get('pearson', 0):.6f}"
            rr_rows.append(("ORT optimizer + INT8", ms, pr_str))

        for label, ms, pr_str in rr_rows:
            speedup = rr_pt_ms / ms if ms > 0 else 0
            p(f"  {label:<35s} | {ms:>8.1f}ms | {speedup:>8.2f}x | {pr_str:>10s}")

    return "\n".join(lines)


def _generate_report(
    summary_text: str,
    hw_info: dict[str, int],
    baseline: dict[str, dict[str, Any]],
    phase2: dict[str, dict[str, Any]],
    phase3: dict[str, dict[str, Any]],
    phase4: dict[str, dict[str, Any]],
    quality: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    """MD 보고서 생성."""
    from datetime import datetime

    output_path.parent.mkdir(parents=True, exist_ok=True)

    md_lines: list[str] = [
        "# ONNX 고급 최적화 벤치마크 보고서",
        "",
        f"생성일: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 배경",
        "",
        "이전 벤치마크에서 ONNX 그래프 최적화(O2/O3) 단독으로는 효과가 없었고,",
        "O3+INT8만 1.3x 개선이 확인됨. 본 벤치마크는 4가지 추가 최적화를 실험.",
        "",
        "## 환경",
        "",
        f"- OS: {platform.system()} {platform.release()}",
        f"- 물리 코어: {hw_info['physical_cores']}, 논리 코어: {hw_info['logical_cores']}",
        f"- 임베딩 모델: {EMB_MODEL_NAME}",
        f"- 리랭커 모델: {RR_MODEL_NAME}",
        "",
        "## 실험 결과",
        "",
        "### Phase 1: Baseline (PyTorch FP32 vs ONNX O3+INT8)",
        "",
    ]

    for key, label in [("emb_pytorch", "임베딩 PyTorch"), ("emb_o3_int8", "임베딩 O3+INT8"),
                        ("rr_pytorch", "리랭커 PyTorch"), ("rr_o3_int8", "리랭커 O3+INT8")]:
        if key in baseline:
            ms = baseline[key]["time_ms"]
            md_lines.append(f"- {label}: {ms:.1f}ms")

    md_lines.extend(["", "### Phase 2: intra_op_num_threads 튜닝", ""])
    for prefix, label in [("emb_threads", "임베딩"), ("rr_threads", "리랭커")]:
        if prefix in phase2:
            best = phase2[prefix]["best_threads"]
            best_ms = phase2[prefix]["best_ms"]
            md_lines.append(f"- {label} 최적: threads={best} ({best_ms:.1f}ms)")

    md_lines.extend(["", "### Phase 3: torch.compile", ""])
    for prefix, label in [("emb_compile", "임베딩"), ("rr_compile", "리랭커")]:
        if prefix in phase3:
            best = phase3[prefix]["best_mode"]
            best_ms = phase3[prefix]["best_ms"]
            md_lines.append(f"- {label} 최적: {best} ({best_ms:.1f}ms)")

    md_lines.extend(["", "### Phase 4: ORT transformer optimizer", ""])
    for key, label in [("emb_ort_opt", "임베딩 ORT opt"), ("emb_ort_opt_int8", "임베딩 ORT opt+INT8"),
                        ("rr_ort_opt", "리랭커 ORT opt"), ("rr_ort_opt_int8", "리랭커 ORT opt+INT8")]:
        if key in phase4:
            ms = phase4[key]["time_ms"]
            md_lines.append(f"- {label}: {ms:.1f}ms")

    md_lines.extend(["", "### 품질 검증", ""])
    for key, metrics in quality.items():
        metric_name = "cosine" if "cosine" in metrics else "pearson"
        val = metrics[metric_name]
        threshold = 0.985 if metric_name == "cosine" else 0.99
        passed = "PASS" if val >= threshold else "FAIL"
        md_lines.append(f"- {key}: {metric_name}={val:.6f} [{passed}]")

    md_lines.extend(["", "## 종합 비교", "", "```"])
    md_lines.append(summary_text)
    md_lines.extend(["```", ""])

    # 권장사항
    md_lines.extend(["## 권장사항", ""])

    emb_pt = baseline.get("emb_pytorch", {}).get("time_ms", 0)
    rr_pt = baseline.get("rr_pytorch", {}).get("time_ms", 0)

    # 임베딩 최적 방법 찾기
    emb_candidates: list[tuple[str, float]] = []
    if "emb_o3_int8" in baseline:
        emb_candidates.append(("ONNX O3+INT8", baseline["emb_o3_int8"]["time_ms"]))
    if "emb_threads" in phase2:
        emb_candidates.append((f"O3+INT8 threads={phase2['emb_threads']['best_threads']}", phase2["emb_threads"]["best_ms"]))
    if "emb_compile" in phase3:
        emb_candidates.append((f"torch.compile ({phase3['emb_compile']['best_mode']})", phase3["emb_compile"]["best_ms"]))
    if "emb_ort_opt" in phase4:
        emb_candidates.append(("ORT optimizer", phase4["emb_ort_opt"]["time_ms"]))
    if "emb_ort_opt_int8" in phase4:
        emb_candidates.append(("ORT opt+INT8", phase4["emb_ort_opt_int8"]["time_ms"]))

    if emb_candidates:
        best_label, best_ms = min(emb_candidates, key=lambda x: x[1])
        speedup = emb_pt / best_ms if best_ms > 0 else 0
        md_lines.append(f"- **임베딩**: {best_label} ({speedup:.2f}x, {best_ms:.1f}ms)")

    rr_candidates: list[tuple[str, float]] = []
    if "rr_o3_int8" in baseline:
        rr_candidates.append(("ONNX O3+INT8", baseline["rr_o3_int8"]["time_ms"]))
    if "rr_threads" in phase2:
        rr_candidates.append((f"O3+INT8 threads={phase2['rr_threads']['best_threads']}", phase2["rr_threads"]["best_ms"]))
    if "rr_compile" in phase3:
        rr_candidates.append((f"torch.compile ({phase3['rr_compile']['best_mode']})", phase3["rr_compile"]["best_ms"]))
    if "rr_ort_opt" in phase4:
        rr_candidates.append(("ORT optimizer", phase4["rr_ort_opt"]["time_ms"]))
    if "rr_ort_opt_int8" in phase4:
        rr_candidates.append(("ORT opt+INT8", phase4["rr_ort_opt_int8"]["time_ms"]))

    if rr_candidates:
        best_label, best_ms = min(rr_candidates, key=lambda x: x[1])
        speedup = rr_pt / best_ms if best_ms > 0 else 0
        md_lines.append(f"- **리랭커**: {best_label} ({speedup:.2f}x, {best_ms:.1f}ms)")

    md_lines.append("")

    output_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\n  MD 보고서 생성: {output_path}")


# ============================================================
# CLI + main
# ============================================================


def parse_args() -> argparse.Namespace:
    """CLI 인자 파싱."""
    parser = argparse.ArgumentParser(
        description="ONNX 고급 최적화 벤치마크 (임베딩 + 리랭커)",
    )
    parser.add_argument("--skip-compile", action="store_true", help="torch.compile 스킵")
    parser.add_argument("--skip-ort-opt", action="store_true", help="ORT transformer optimizer 스킵")
    parser.add_argument("--skip-threads", action="store_true", help="스레드 튜닝 스킵")
    parser.add_argument("--skip-quality", action="store_true", help="품질 검증 스킵")
    parser.add_argument("--embedding-only", action="store_true", help="임베딩만 실행")
    parser.add_argument("--reranker-only", action="store_true", help="리랭커만 실행")
    parser.add_argument("--no-report", action="store_true", help="MD 보고서 생성 스킵")
    parser.add_argument("--report", type=str, default=None, help="보고서 저장 경로")
    return parser.parse_args()


def main() -> None:
    """벤치마크 메인."""
    args = parse_args()

    run_emb = not args.reranker_only
    run_rr = not args.embedding_only

    print(f"\n{SEPARATOR}")
    print("  ONNX 고급 최적화 벤치마크")
    print(f"  임베딩: {EMB_MODEL_NAME}")
    print(f"  리랭커: {RR_MODEL_NAME}")
    target = "임베딩 + 리랭커"
    if args.embedding_only:
        target = "임베딩만"
    elif args.reranker_only:
        target = "리랭커만"
    print(f"  대상: {target}")
    print(SEPARATOR)

    # Phase 0
    hw_info = run_phase_0()

    # Phase 1
    baseline = run_phase_1(run_embedding=run_emb, run_reranker=run_rr)

    # Phase 2
    phase2: dict[str, dict[str, Any]] = {}
    if not args.skip_threads:
        phase2 = run_phase_2(hw_info, run_embedding=run_emb, run_reranker=run_rr)
    else:
        print("\n  Phase 2: 스킵 (--skip-threads)")

    # Phase 3
    phase3: dict[str, dict[str, Any]] = {}
    if not args.skip_compile:
        phase3 = run_phase_3(run_embedding=run_emb, run_reranker=run_rr)
    else:
        print("\n  Phase 3: 스킵 (--skip-compile)")

    # Phase 4
    phase4: dict[str, dict[str, Any]] = {}
    if not args.skip_ort_opt:
        phase4 = run_phase_4(run_embedding=run_emb, run_reranker=run_rr)
    else:
        print("\n  Phase 4: 스킵 (--skip-ort-opt)")

    # Phase 5: 품질 + 종합
    quality: dict[str, dict[str, float]] = {}
    if not args.skip_quality:
        quality = _run_quality_check(
            baseline, phase2, phase3, phase4,
            run_embedding=run_emb, run_reranker=run_rr,
        )
    else:
        print("\n  품질 검증: 스킵 (--skip-quality)")

    summary_text = _print_summary_table(
        baseline, phase2, phase3, phase4, quality,
        run_embedding=run_emb, run_reranker=run_rr,
    )

    # 보고서
    if not args.no_report:
        report_path = Path(args.report) if args.report else DEFAULT_REPORT_PATH
        _generate_report(
            summary_text, hw_info, baseline, phase2, phase3, phase4, quality,
            report_path,
        )
    else:
        print("\n  MD 보고서: 스킵 (--no-report)")

    print(f"\n{SEPARATOR}")
    print("  벤치마크 완료")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
