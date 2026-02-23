"""ARM/크로스플랫폼 ONNX 최적화 벤치마크

Mac M1/M3, ARM 서버, x86 등 다양한 플랫폼에서
ONNX 최적화 성능을 측정하는 통합 벤치마크 스크립트.

Phase 0: 환경 정보 수집 (platform, CPU cores, SIMD caps, ORT version, ARM/x86)
Phase 1: PyTorch FP32 baseline (CPU + MPS if Mac)
Phase 2: ONNX 원본 FP32 (ORT_ENABLE_ALL)
Phase 3: ORT-최적화 FP32 (Attention Fusion) — fusion 효과 측정
Phase 4: ORT-최적화 FP16 — fusion + FP16 효과 측정
Phase 5: 스레드 최적화 (P코어만 vs 전체코어 비교)
Phase 6: ORT 프로파일링 (노드별 실행시간 top-10)
Phase 7: 종합 보고서 (JSON + 마크다운 테이블)
Phase 8: BF16 Fastmath 벤치마크 (Graviton3 전용, Linux ARM만)
Phase 9: Static Shape 벤치마크 (임베딩 전용, dynamic vs static128)
Phase 10: 복합 최적화 (BF16 + Static128, Graviton3 전용)

각 Phase에서 임베딩 + 리랭커 양쪽 측정:
  - 단일 쿼리 레이턴시 (10회, 워밍업 3회)
  - 품질: cosine similarity vs PyTorch, Pearson correlation (리랭커)
  - 메모리 사용량 (RSS via psutil)

사용법:
  cd backend && uv run python scripts/benchmark_arm_optimization.py
  cd backend && uv run python scripts/benchmark_arm_optimization.py --embedding-only
  cd backend && uv run python scripts/benchmark_arm_optimization.py --reranker-only
  cd backend && uv run python scripts/benchmark_arm_optimization.py --skip-profiling
  cd backend && uv run python scripts/benchmark_arm_optimization.py --skip-bf16-test
  cd backend && uv run python scripts/benchmark_arm_optimization.py --skip-static-test
  cd backend && uv run python scripts/benchmark_arm_optimization.py --output results.json
"""

import argparse
import gc
import json
import os
import platform
import struct
import subprocess
import sys
import time
from datetime import datetime
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

SEPARATOR = "=" * 70
WARMUP_RUNS = 3
REPEAT_RUNS = 10

# --- 임베딩 모델 ---
EMB_MODEL_NAME = "nlpai-lab/KURE-v1"
CACHE_DIR = str(PROJECT_ROOT / "data" / "models")
EMB_ONNX_DIR = PROJECT_ROOT / "data" / "models" / "kure-v1-onnx"
EMB_ORT_OPT_DIR = PROJECT_ROOT / "data" / "models" / "kure-v1-ort-opt"
EMB_ORT_OPT_FP16_DIR = PROJECT_ROOT / "data" / "models" / "kure-v1-ort-opt-fp16"
EMB_ORT_OPT_STATIC128_DIR = PROJECT_ROOT / "data" / "models" / "kure-v1-ort-opt-static128"

# KURE-v1: XLM-RoBERTa Large
EMB_NUM_HEADS = 16
EMB_HIDDEN_SIZE = 1024

# --- 리랭커 모델 ---
RR_MODEL_NAME = "dragonkue/bge-reranker-v2-m3-ko"
RR_ONNX_DIR = PROJECT_ROOT / "data" / "models" / "reranker-onnx"
RR_ORT_OPT_DIR = PROJECT_ROOT / "data" / "models" / "reranker-ort-opt"
RR_ORT_OPT_FP16_DIR = PROJECT_ROOT / "data" / "models" / "reranker-ort-opt-fp16"

# bge-reranker-v2-m3-ko: XLM-RoBERTa Large
RR_NUM_HEADS = 16
RR_HIDDEN_SIZE = 1024

DEFAULT_REPORT_PATH = (
    PROJECT_ROOT.parent / "docs" / "04-report" / "features"
    / "arm-onnx-optimization-benchmark.md"
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

# --- 테스트 데이터 (리랭커) ---
RR_QUERY = "교통사고 손해배상 판례"
RR_DOCUMENTS = [
    "피고는 원고에게 금 50,000,000원 및 이에 대한 지연손해금을 지급하라.",
    "자동차손해배상 보장법 제3조에 의하면 운행자는 배상 책임을 진다.",
    "임대차보증금 반환 청구 사건에서 임대인은 보증금 전액을 반환할 의무가 있다.",
    "근로기준법 제23조 제1항은 정당한 이유 없이 해고를 하지 못한다.",
    "형법 제307조 제1항의 명예훼손죄가 성립하려면 사실을 적시하여야 한다.",
    "교통사고처리특례법 제4조 단서는 피해자 의사와 관계없이 공소 제기 가능.",
    "후유장해 등급 판정에서는 맥브라이드 장해평가 방법에 의한다.",
    "불법행위 손해배상 청구권의 소멸시효는 안 날로부터 3년이다.",
]


# ============================================================
# 공통 유틸리티
# ============================================================


def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    """Sigmoid 활성화 함수."""
    return 1 / (1 + np.exp(-x))


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """두 벡터 배열의 평균 pairwise cosine similarity."""
    sims: list[float] = []
    for i in range(min(len(a), len(b))):
        dot = np.dot(a[i], b[i])
        norm_a = np.linalg.norm(a[i])
        norm_b = np.linalg.norm(b[i])
        if norm_a > 0 and norm_b > 0:
            sims.append(float(dot / (norm_a * norm_b)))
    return float(np.mean(sims)) if sims else 0.0


def _measure_rss_mb() -> float:
    """현재 프로세스의 RSS 메모리(MB)를 반환."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def _measure_latency(
    func: Any,
    warmup: int = WARMUP_RUNS,
    repeat: int = REPEAT_RUNS,
) -> tuple[Any, float, float, float]:
    """함수를 여러 번 실행하여 결과, median, mean, std (ms)를 반환."""
    for _ in range(warmup):
        func()

    times: list[float] = []
    result = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = func()
        times.append((time.perf_counter() - t0) * 1000)

    return result, float(np.median(times)), float(np.mean(times)), float(np.std(times))


def _find_onnx_file(model_dir: Path) -> str | None:
    """ONNX 모델 파일명을 자동 탐색. 없으면 None 반환."""
    candidates = [
        "model_optimized.onnx",
        "model.onnx",
        "model_quantized.onnx",
    ]
    for name in candidates:
        if (model_dir / name).exists():
            return name
    # fallback: 디렉토리 내 .onnx 파일 탐색
    onnx_files = list(model_dir.glob("*.onnx"))
    if onnx_files:
        return onnx_files[0].name
    return None


# ============================================================
# Phase 0: 환경 정보 수집
# ============================================================


def collect_environment_info() -> dict[str, Any]:
    """플랫폼, CPU, GPU, ORT, PyTorch 등 환경 정보를 수집."""
    info: dict[str, Any] = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "is_arm": platform.machine() in ("arm64", "aarch64"),
        "is_mac": platform.system() == "Darwin",
        "is_linux": platform.system() == "Linux",
        "cpu_count": os.cpu_count(),
        "pointer_size": struct.calcsize("P") * 8,
    }

    # ORT 버전 및 프로바이더
    try:
        import onnxruntime as ort
        info["ort_version"] = ort.__version__
        info["ort_providers"] = ort.get_available_providers()
    except ImportError:
        info["ort_version"] = "not installed"
        info["ort_providers"] = []

    # PyTorch
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["mps_available"] = (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            info["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch_version"] = "not installed"
        info["mps_available"] = False
        info["cuda_available"] = False

    # Mac P-core 감지
    if info["is_mac"]:
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.perflevel0.logicalcpu"],
                capture_output=True, text=True, timeout=5,
            )
            info["p_cores"] = int(result.stdout.strip())
        except Exception:
            info["p_cores"] = 4  # M1/M3 기본값

    # 메모리
    try:
        import psutil
        info["total_ram_gb"] = round(
            psutil.virtual_memory().total / (1024**3), 1,
        )
    except ImportError:
        info["total_ram_gb"] = "unknown"

    # 모델 디렉토리 존재 확인
    model_dirs = {
        "emb_onnx": EMB_ONNX_DIR,
        "emb_ort_opt": EMB_ORT_OPT_DIR,
        "emb_ort_opt_fp16": EMB_ORT_OPT_FP16_DIR,
        "rr_onnx": RR_ONNX_DIR,
        "rr_ort_opt": RR_ORT_OPT_DIR,
        "rr_ort_opt_fp16": RR_ORT_OPT_FP16_DIR,
    }
    info["available_models"] = {
        key: d.exists() for key, d in model_dirs.items()
    }

    return info


def print_environment_info(info: dict[str, Any]) -> None:
    """환경 정보를 콘솔에 출력."""
    print(f"\n{SEPARATOR}")
    print("  Phase 0: 환경 정보 수집")
    print(SEPARATOR)

    arch_label = "ARM" if info["is_arm"] else "x86_64"
    print(f"  아키텍처:        {info['machine']} ({arch_label})")
    print(f"  플랫폼:          {info['platform']} {info['platform_release']}")
    print(f"  프로세서:        {info['processor'] or 'N/A'}")
    print(f"  Python:          {info['python']}")
    print(f"  CPU 코어:        {info['cpu_count']}")
    if info["is_mac"] and "p_cores" in info:
        print(f"  P-core (성능):   {info['p_cores']}")
    print(f"  포인터 크기:     {info['pointer_size']}bit")
    print(f"  총 RAM:          {info['total_ram_gb']}GB")

    print(f"\n  onnxruntime:     {info.get('ort_version', 'N/A')}")
    if info.get("ort_providers"):
        print(f"  ORT 프로바이더:  {', '.join(info['ort_providers'])}")
    print(f"  PyTorch:         {info.get('torch_version', 'N/A')}")
    print(f"  MPS (Apple GPU): {'사용 가능' if info.get('mps_available') else '미사용'}")
    print(f"  CUDA (GPU):      {'사용 가능' if info.get('cuda_available') else '미사용'}")
    if info.get("cuda_available") and info.get("cuda_device"):
        print(f"  CUDA 디바이스:   {info['cuda_device']}")

    print("\n  모델 디렉토리 확인:")
    model_labels = {
        "emb_onnx": "임베딩 ONNX FP32",
        "emb_ort_opt": "임베딩 ORT-최적화 FP32",
        "emb_ort_opt_fp16": "임베딩 ORT-최적화 FP16",
        "rr_onnx": "리랭커 ONNX FP32",
        "rr_ort_opt": "리랭커 ORT-최적화 FP32",
        "rr_ort_opt_fp16": "리랭커 ORT-최적화 FP16",
    }
    available = info.get("available_models", {})
    for key, label in model_labels.items():
        status = "OK" if available.get(key) else "없음"
        print(f"    [{status:>3s}] {label}")


# ============================================================
# 임베딩 함수
# ============================================================


def _encode_pytorch(
    texts: list[str],
    device: str = "cpu",
) -> np.ndarray:
    """PyTorch FP32 임베딩 (CPU 또는 MPS)."""
    from sentence_transformers import SentenceTransformer

    kwargs: dict[str, Any] = {
        "cache_folder": CACHE_DIR,
        "trust_remote_code": True,
        "local_files_only": True,
    }
    if device != "cpu":
        kwargs["device"] = device

    model = SentenceTransformer(EMB_MODEL_NAME, **kwargs)
    embeddings = model.encode(
        texts, batch_size=32,
        show_progress_bar=False, normalize_embeddings=True,
    )
    result = np.array(embeddings)

    del model
    if device == "mps":
        try:
            import torch
            torch.mps.empty_cache()
        except Exception:
            pass
    gc.collect()
    return result


def _encode_onnx_raw(
    texts: list[str],
    model_dir: Path,
    file_name: str,
    num_threads: int = 0,
    *,
    enable_bf16_fastmath: bool = False,
    static_max_length: int = 0,
) -> np.ndarray:
    """ORT 직접 세션으로 임베딩 생성 (CLS pooling + L2 norm).

    Args:
        enable_bf16_fastmath: Graviton3 BF16 MMLA fastmath 활성화
        static_max_length: >0이면 고정 길이 패딩 (예: 128)
    """
    import onnxruntime as ort
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), trust_remote_code=True,
    )
    model_path = str(model_dir / file_name)

    session_options = ort.SessionOptions()
    if num_threads > 0:
        session_options.intra_op_num_threads = num_threads
        session_options.inter_op_num_threads = 1
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if enable_bf16_fastmath:
        session_options.add_session_config_entry(
            "mlas.enable_gemm_fastmath_arm64_bfloat16", "1",
        )
    session = ort.InferenceSession(
        model_path, sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    input_names = [inp.name for inp in session.get_inputs()]
    use_static = static_max_length > 0
    inputs = tokenizer(
        texts, return_tensors="np",
        padding="max_length" if use_static else True,
        truncation=True,
        max_length=static_max_length if use_static else 512,
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
    emb = outputs[0][:, 0, :]
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.clip(norm, a_min=1e-9, a_max=None)

    del session, tokenizer
    gc.collect()
    return emb


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
    *,
    enable_bf16_fastmath: bool = False,
) -> list[float]:
    """ORT 직접 세션으로 리랭킹 (배치).

    Args:
        enable_bf16_fastmath: Graviton3 BF16 MMLA fastmath 활성화
    """
    import onnxruntime as ort
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model_path = str(model_dir / file_name)

    session_options = ort.SessionOptions()
    if num_threads > 0:
        session_options.intra_op_num_threads = num_threads
        session_options.inter_op_num_threads = 1
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if enable_bf16_fastmath:
        session_options.add_session_config_entry(
            "mlas.enable_gemm_fastmath_arm64_bfloat16", "1",
        )
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


# ============================================================
# Phase 1: PyTorch FP32 Baseline
# ============================================================


def run_phase_1(
    env_info: dict[str, Any],
    *,
    run_embedding: bool = True,
    run_reranker: bool = True,
) -> dict[str, dict[str, Any]]:
    """Phase 1: PyTorch FP32 baseline (CPU + MPS if Mac)."""
    print(f"\n{SEPARATOR}")
    print("  Phase 1: PyTorch FP32 Baseline")
    print(SEPARATOR)

    results: dict[str, dict[str, Any]] = {}
    texts = EMB_TEST_QUERIES

    if run_embedding:
        print("\n  === 임베딩 ===")

        # CPU baseline
        print("  [PyTorch FP32 / CPU] 측정 중...")
        rss_before = _measure_rss_mb()
        emb_cpu, median_ms, mean_ms, std_ms = _measure_latency(
            lambda: _encode_pytorch(texts, device="cpu"),
        )
        rss_after = _measure_rss_mb()
        print(f"    {len(texts)}건: median={median_ms:.1f}ms, "
              f"mean={mean_ms:.1f}ms, std={std_ms:.1f}ms")
        print(f"    RSS: {rss_after:.0f}MB (delta: {rss_after - rss_before:+.0f}MB)")
        results["emb_pytorch_cpu"] = {
            "median_ms": median_ms, "mean_ms": mean_ms, "std_ms": std_ms,
            "rss_mb": rss_after, "embeddings": emb_cpu,
        }

        # MPS (Mac GPU)
        if env_info.get("mps_available"):
            print("  [PyTorch FP32 / MPS] 측정 중...")
            rss_before = _measure_rss_mb()
            emb_mps, median_ms, mean_ms, std_ms = _measure_latency(
                lambda: _encode_pytorch(texts, device="mps"),
            )
            rss_after = _measure_rss_mb()
            cos = _cosine_similarity(emb_cpu, emb_mps)
            print(f"    {len(texts)}건: median={median_ms:.1f}ms, "
                  f"mean={mean_ms:.1f}ms, std={std_ms:.1f}ms")
            print(f"    RSS: {rss_after:.0f}MB, cosine vs CPU: {cos:.6f}")
            results["emb_pytorch_mps"] = {
                "median_ms": median_ms, "mean_ms": mean_ms, "std_ms": std_ms,
                "rss_mb": rss_after, "cosine_vs_cpu": cos,
                "embeddings": emb_mps,
            }
        else:
            print("  [건너뜀] MPS 미사용 환경")

    if run_reranker:
        print("\n  === 리랭커 ===")

        print("  [PyTorch FP32 / CPU] 측정 중...")
        rss_before = _measure_rss_mb()
        rr_cpu, median_ms, mean_ms, std_ms = _measure_latency(
            lambda: _rerank_pytorch(RR_QUERY, RR_DOCUMENTS),
        )
        rss_after = _measure_rss_mb()
        print(f"    {len(RR_DOCUMENTS)}건: median={median_ms:.1f}ms, "
              f"mean={mean_ms:.1f}ms, std={std_ms:.1f}ms")
        print(f"    RSS: {rss_after:.0f}MB")
        results["rr_pytorch_cpu"] = {
            "median_ms": median_ms, "mean_ms": mean_ms, "std_ms": std_ms,
            "rss_mb": rss_after, "scores": rr_cpu,
        }

    return results


# ============================================================
# Phase 2: ONNX 원본 FP32
# ============================================================


def run_phase_2(
    baseline: dict[str, dict[str, Any]],
    *,
    run_embedding: bool = True,
    run_reranker: bool = True,
) -> dict[str, dict[str, Any]]:
    """Phase 2: ONNX 원본 FP32 (ORT_ENABLE_ALL 기본 런타임 최적화)."""
    print(f"\n{SEPARATOR}")
    print("  Phase 2: ONNX 원본 FP32 (ORT_ENABLE_ALL)")
    print(SEPARATOR)

    results: dict[str, dict[str, Any]] = {}
    texts = EMB_TEST_QUERIES

    if run_embedding:
        if EMB_ONNX_DIR.exists():
            onnx_file = _find_onnx_file(EMB_ONNX_DIR)
            if onnx_file:
                print(f"\n  === 임베딩 ({onnx_file}) ===")
                _measure_rss_mb()  # RSS 기준점
                emb, median_ms, mean_ms, std_ms = _measure_latency(
                    lambda: _encode_onnx_raw(texts, EMB_ONNX_DIR, onnx_file),
                )
                rss_after = _measure_rss_mb()

                # PyTorch baseline 대비 품질
                emb_pt = baseline.get("emb_pytorch_cpu", {}).get("embeddings")
                cos = _cosine_similarity(emb_pt, emb) if emb_pt is not None else 0.0
                pt_ms = baseline.get("emb_pytorch_cpu", {}).get("median_ms", 1.0)
                speedup = pt_ms / median_ms if median_ms > 0 else 0.0

                print(f"    {len(texts)}건: median={median_ms:.1f}ms "
                      f"({speedup:.2f}x vs PyTorch)")
                print(f"    cosine vs PyTorch: {cos:.6f}")
                print(f"    RSS: {rss_after:.0f}MB")
                results["emb_onnx_fp32"] = {
                    "median_ms": median_ms, "mean_ms": mean_ms, "std_ms": std_ms,
                    "speedup": speedup, "cosine": cos,
                    "rss_mb": rss_after, "embeddings": emb,
                }
            else:
                print("  [건너뜀] 임베딩 ONNX 파일 없음")
        else:
            print("  [건너뜀] 임베딩 ONNX 디렉토리 없음")

    if run_reranker:
        if RR_ONNX_DIR.exists():
            onnx_file = _find_onnx_file(RR_ONNX_DIR)
            if onnx_file:
                print(f"\n  === 리랭커 ({onnx_file}) ===")
                _measure_rss_mb()  # RSS 기준점
                scores, median_ms, mean_ms, std_ms = _measure_latency(
                    lambda: _rerank_onnx_raw(
                        RR_QUERY, RR_DOCUMENTS, RR_ONNX_DIR, onnx_file,
                    ),
                )
                rss_after = _measure_rss_mb()

                rr_pt = baseline.get("rr_pytorch_cpu", {}).get("scores")
                pearson = _compute_pearson(rr_pt, scores) if rr_pt else 0.0
                pt_ms = baseline.get("rr_pytorch_cpu", {}).get("median_ms", 1.0)
                speedup = pt_ms / median_ms if median_ms > 0 else 0.0

                print(f"    {len(RR_DOCUMENTS)}건: median={median_ms:.1f}ms "
                      f"({speedup:.2f}x vs PyTorch)")
                print(f"    Pearson vs PyTorch: {pearson:.6f}")
                print(f"    RSS: {rss_after:.0f}MB")
                results["rr_onnx_fp32"] = {
                    "median_ms": median_ms, "mean_ms": mean_ms, "std_ms": std_ms,
                    "speedup": speedup, "pearson": pearson,
                    "rss_mb": rss_after, "scores": scores,
                }
            else:
                print("  [건너뜀] 리랭커 ONNX 파일 없음")
        else:
            print("  [건너뜀] 리랭커 ONNX 디렉토리 없음")

    return results


# ============================================================
# Phase 3: ORT-최적화 FP32 (Attention Fusion)
# ============================================================


def _optimize_with_ort(
    src_onnx_dir: Path,
    src_file: str,
    dst_dir: Path,
    num_heads: int,
    hidden_size: int,
    label: str,
) -> bool:
    """ORT transformer optimizer로 Attention Fusion 최적화 수행."""
    dst_file = dst_dir / "model_optimized.onnx"
    if dst_file.exists():
        print(f"    [{label}] 이미 최적화됨: {dst_dir.name}")
        return True

    print(f"    [{label}] ORT transformer optimizer 실행 중...")
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

        # 외부 데이터 파일 확인
        data_file = dst_dir / "model_optimized.onnx.data"
        if data_file.exists():
            size_mb = data_file.stat().st_size / (1024 * 1024)
            print(f"      그래프: {dst_file.stat().st_size // 1024}KB, "
                  f"가중치: {size_mb:.0f}MB")
        else:
            model_size = dst_file.stat().st_size / (1024 * 1024)
            print(f"      모델: {model_size:.0f}MB (단일 파일)")

        # 토크나이저 복사
        _copy_tokenizer_files(src_onnx_dir, dst_dir)
        print(f"      완료: {dst_dir.name}")
        return True
    except Exception as e:
        print(f"      실패: {e!s:.120s}")
        return False


def _copy_tokenizer_files(src_dir: Path, dst_dir: Path) -> None:
    """토크나이저 + config 파일을 복사."""
    import shutil

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(src_dir), trust_remote_code=True,
    )
    tokenizer.save_pretrained(str(dst_dir))

    config_src = src_dir / "config.json"
    if config_src.exists():
        shutil.copy2(str(config_src), str(dst_dir / "config.json"))


def run_phase_3(
    baseline: dict[str, dict[str, Any]],
    *,
    run_embedding: bool = True,
    run_reranker: bool = True,
) -> dict[str, dict[str, Any]]:
    """Phase 3: ORT-최적화 FP32 (Attention Fusion) 벤치마크."""
    print(f"\n{SEPARATOR}")
    print("  Phase 3: ORT-최적화 FP32 (Attention Fusion)")
    print(SEPARATOR)

    results: dict[str, dict[str, Any]] = {}
    texts = EMB_TEST_QUERIES

    if run_embedding:
        if EMB_ONNX_DIR.exists():
            src_file = _find_onnx_file(EMB_ONNX_DIR)
            if src_file:
                print("\n  === 임베딩 ===")
                opt_ok = _optimize_with_ort(
                    EMB_ONNX_DIR, src_file, EMB_ORT_OPT_DIR,
                    EMB_NUM_HEADS, EMB_HIDDEN_SIZE, "임베딩",
                )
                if opt_ok:
                    opt_file = _find_onnx_file(EMB_ORT_OPT_DIR)
                    if opt_file:
                        print("    속도 측정 중...")
                        _measure_rss_mb()  # RSS 기준점
                        emb, median_ms, mean_ms, std_ms = _measure_latency(
                            lambda: _encode_onnx_raw(
                                texts, EMB_ORT_OPT_DIR, opt_file,
                            ),
                        )
                        rss_after = _measure_rss_mb()

                        emb_pt = baseline.get(
                            "emb_pytorch_cpu", {},
                        ).get("embeddings")
                        cos = (
                            _cosine_similarity(emb_pt, emb)
                            if emb_pt is not None else 0.0
                        )
                        pt_ms = baseline.get(
                            "emb_pytorch_cpu", {},
                        ).get("median_ms", 1.0)
                        speedup = pt_ms / median_ms if median_ms > 0 else 0.0

                        print(f"    {len(texts)}건: median={median_ms:.1f}ms "
                              f"({speedup:.2f}x vs PyTorch)")
                        print(f"    cosine vs PyTorch: {cos:.6f}")
                        print(f"    RSS: {rss_after:.0f}MB")
                        results["emb_ort_opt_fp32"] = {
                            "median_ms": median_ms, "mean_ms": mean_ms,
                            "std_ms": std_ms, "speedup": speedup,
                            "cosine": cos, "rss_mb": rss_after,
                            "embeddings": emb,
                        }
        else:
            print("  [건너뜀] 임베딩 ONNX 원본 디렉토리 없음")

    if run_reranker:
        if RR_ONNX_DIR.exists():
            src_file = _find_onnx_file(RR_ONNX_DIR)
            if src_file:
                print("\n  === 리랭커 ===")
                opt_ok = _optimize_with_ort(
                    RR_ONNX_DIR, src_file, RR_ORT_OPT_DIR,
                    RR_NUM_HEADS, RR_HIDDEN_SIZE, "리랭커",
                )
                if opt_ok:
                    opt_file = _find_onnx_file(RR_ORT_OPT_DIR)
                    if opt_file:
                        print("    속도 측정 중...")
                        _measure_rss_mb()  # RSS 기준점
                        scores, median_ms, mean_ms, std_ms = _measure_latency(
                            lambda: _rerank_onnx_raw(
                                RR_QUERY, RR_DOCUMENTS,
                                RR_ORT_OPT_DIR, opt_file,
                            ),
                        )
                        rss_after = _measure_rss_mb()

                        rr_pt = baseline.get(
                            "rr_pytorch_cpu", {},
                        ).get("scores")
                        pearson = (
                            _compute_pearson(rr_pt, scores) if rr_pt else 0.0
                        )
                        pt_ms = baseline.get(
                            "rr_pytorch_cpu", {},
                        ).get("median_ms", 1.0)
                        speedup = pt_ms / median_ms if median_ms > 0 else 0.0

                        print(f"    {len(RR_DOCUMENTS)}건: "
                              f"median={median_ms:.1f}ms "
                              f"({speedup:.2f}x vs PyTorch)")
                        print(f"    Pearson vs PyTorch: {pearson:.6f}")
                        print(f"    RSS: {rss_after:.0f}MB")
                        results["rr_ort_opt_fp32"] = {
                            "median_ms": median_ms, "mean_ms": mean_ms,
                            "std_ms": std_ms, "speedup": speedup,
                            "pearson": pearson, "rss_mb": rss_after,
                            "scores": scores,
                        }
        else:
            print("  [건너뜀] 리랭커 ONNX 원본 디렉토리 없음")

    return results


# ============================================================
# Phase 4: ORT-최적화 FP16 (Fusion + FP16)
# ============================================================


def _convert_to_fp16(
    src_dir: Path,
    src_file: str,
    dst_dir: Path,
    num_heads: int,
    hidden_size: int,
    label: str,
) -> bool:
    """ORT optimizer + float16 변환 수행."""
    dst_file = dst_dir / "model_optimized.onnx"
    if dst_file.exists():
        print(f"    [{label}] 이미 FP16 변환됨: {dst_dir.name}")
        return True

    print(f"    [{label}] FP16 변환 중...")
    try:
        from onnxruntime.transformers.optimizer import optimize_model

        dst_dir.mkdir(parents=True, exist_ok=True)

        optimized = optimize_model(
            str(src_dir / src_file),
            model_type="bert",
            num_heads=num_heads,
            hidden_size=hidden_size,
        )
        optimized.convert_float_to_float16(
            use_symbolic_shape_infer=True,
            keep_io_types=True,
        )
        optimized.save_model_to_file(
            str(dst_file),
            use_external_data_format=True,
        )

        data_file = dst_dir / "model_optimized.onnx.data"
        if data_file.exists():
            size_mb = data_file.stat().st_size / (1024 * 1024)
            print(f"      그래프: {dst_file.stat().st_size // 1024}KB, "
                  f"가중치: {size_mb:.0f}MB")
        else:
            model_size = dst_file.stat().st_size / (1024 * 1024)
            print(f"      모델: {model_size:.0f}MB (단일 파일)")

        _copy_tokenizer_files(src_dir, dst_dir)
        print(f"      완료: {dst_dir.name}")
        return True
    except Exception as e:
        print(f"      실패: {e!s:.120s}")
        return False


def run_phase_4(
    baseline: dict[str, dict[str, Any]],
    *,
    run_embedding: bool = True,
    run_reranker: bool = True,
) -> dict[str, dict[str, Any]]:
    """Phase 4: ORT-최적화 FP16 벤치마크."""
    print(f"\n{SEPARATOR}")
    print("  Phase 4: ORT-최적화 FP16 (Fusion + Float16)")
    print(SEPARATOR)

    results: dict[str, dict[str, Any]] = {}
    texts = EMB_TEST_QUERIES

    if run_embedding:
        if EMB_ONNX_DIR.exists():
            src_file = _find_onnx_file(EMB_ONNX_DIR)
            if src_file:
                print("\n  === 임베딩 ===")
                fp16_ok = _convert_to_fp16(
                    EMB_ONNX_DIR, src_file, EMB_ORT_OPT_FP16_DIR,
                    EMB_NUM_HEADS, EMB_HIDDEN_SIZE, "임베딩 FP16",
                )
                if fp16_ok:
                    opt_file = _find_onnx_file(EMB_ORT_OPT_FP16_DIR)
                    if opt_file:
                        print("    속도 측정 중...")
                        _measure_rss_mb()  # RSS 기준점
                        emb, median_ms, mean_ms, std_ms = _measure_latency(
                            lambda: _encode_onnx_raw(
                                texts, EMB_ORT_OPT_FP16_DIR, opt_file,
                            ),
                        )
                        rss_after = _measure_rss_mb()

                        emb_pt = baseline.get(
                            "emb_pytorch_cpu", {},
                        ).get("embeddings")
                        cos = (
                            _cosine_similarity(emb_pt, emb)
                            if emb_pt is not None else 0.0
                        )
                        pt_ms = baseline.get(
                            "emb_pytorch_cpu", {},
                        ).get("median_ms", 1.0)
                        speedup = pt_ms / median_ms if median_ms > 0 else 0.0

                        print(f"    {len(texts)}건: median={median_ms:.1f}ms "
                              f"({speedup:.2f}x vs PyTorch)")
                        print(f"    cosine vs PyTorch: {cos:.6f}")
                        print(f"    RSS: {rss_after:.0f}MB")
                        results["emb_ort_opt_fp16"] = {
                            "median_ms": median_ms, "mean_ms": mean_ms,
                            "std_ms": std_ms, "speedup": speedup,
                            "cosine": cos, "rss_mb": rss_after,
                            "embeddings": emb,
                        }
        else:
            print("  [건너뜀] 임베딩 ONNX 원본 디렉토리 없음")

    if run_reranker:
        if RR_ONNX_DIR.exists():
            src_file = _find_onnx_file(RR_ONNX_DIR)
            if src_file:
                print("\n  === 리랭커 ===")
                fp16_ok = _convert_to_fp16(
                    RR_ONNX_DIR, src_file, RR_ORT_OPT_FP16_DIR,
                    RR_NUM_HEADS, RR_HIDDEN_SIZE, "리랭커 FP16",
                )
                if fp16_ok:
                    opt_file = _find_onnx_file(RR_ORT_OPT_FP16_DIR)
                    if opt_file:
                        print("    속도 측정 중...")
                        _measure_rss_mb()  # RSS 기준점
                        scores, median_ms, mean_ms, std_ms = _measure_latency(
                            lambda: _rerank_onnx_raw(
                                RR_QUERY, RR_DOCUMENTS,
                                RR_ORT_OPT_FP16_DIR, opt_file,
                            ),
                        )
                        rss_after = _measure_rss_mb()

                        rr_pt = baseline.get(
                            "rr_pytorch_cpu", {},
                        ).get("scores")
                        pearson = (
                            _compute_pearson(rr_pt, scores) if rr_pt else 0.0
                        )
                        pt_ms = baseline.get(
                            "rr_pytorch_cpu", {},
                        ).get("median_ms", 1.0)
                        speedup = pt_ms / median_ms if median_ms > 0 else 0.0

                        print(f"    {len(RR_DOCUMENTS)}건: "
                              f"median={median_ms:.1f}ms "
                              f"({speedup:.2f}x vs PyTorch)")
                        print(f"    Pearson vs PyTorch: {pearson:.6f}")
                        print(f"    RSS: {rss_after:.0f}MB")
                        results["rr_ort_opt_fp16"] = {
                            "median_ms": median_ms, "mean_ms": mean_ms,
                            "std_ms": std_ms, "speedup": speedup,
                            "pearson": pearson, "rss_mb": rss_after,
                            "scores": scores,
                        }
        else:
            print("  [건너뜀] 리랭커 ONNX 원본 디렉토리 없음")

    return results


# ============================================================
# Phase 5: 스레드 최적화 (P코어 vs 전체코어)
# ============================================================


def _build_thread_configs(env_info: dict[str, Any]) -> list[tuple[str, int]]:
    """플랫폼에 맞는 스레드 설정 목록 생성."""
    thread_configs: list[tuple[str, int]] = []

    if env_info.get("is_mac"):
        p_cores = env_info.get("p_cores", 4)
        thread_configs = [
            ("P코어만", p_cores),
            ("전체코어", os.cpu_count() or 8),
            ("P코어+1", p_cores + 1),
            ("2코어", 2),
        ]
    else:
        cpu_count = os.cpu_count() or 4
        thread_configs = [
            ("전체코어", cpu_count),
            ("절반", max(cpu_count // 2, 1)),
            ("4코어", 4),
            ("2코어", 2),
        ]

    # 중복 제거 (값 기준)
    seen: set[int] = set()
    unique: list[tuple[str, int]] = []
    for label, count in thread_configs:
        if count not in seen:
            seen.add(count)
            unique.append((label, count))
    return unique


def run_phase_5(
    env_info: dict[str, Any],
    baseline: dict[str, dict[str, Any]],
    *,
    run_embedding: bool = True,
    run_reranker: bool = True,
) -> dict[str, dict[str, Any]]:
    """Phase 5: 스레드 최적화 (다양한 스레드 수 비교)."""
    print(f"\n{SEPARATOR}")
    print("  Phase 5: 스레드 최적화")
    print(SEPARATOR)

    results: dict[str, dict[str, Any]] = {}
    texts = EMB_TEST_QUERIES
    thread_configs = _build_thread_configs(env_info)

    print(f"  스레드 설정: {[(label, n) for label, n in thread_configs]}")

    # 측정할 ONNX 디렉토리/파일 결정 (최적화된 모델 우선)
    emb_dir: Path | None = None
    emb_file: str | None = None
    for d in [EMB_ORT_OPT_DIR, EMB_ORT_OPT_FP16_DIR, EMB_ONNX_DIR]:
        if d.exists():
            f = _find_onnx_file(d)
            if f:
                emb_dir = d
                emb_file = f
                break

    rr_dir: Path | None = None
    rr_file: str | None = None
    for d in [RR_ORT_OPT_DIR, RR_ORT_OPT_FP16_DIR, RR_ONNX_DIR]:
        if d.exists():
            f = _find_onnx_file(d)
            if f:
                rr_dir = d
                rr_file = f
                break

    if run_embedding and emb_dir and emb_file:
        print(f"\n  === 임베딩 ({emb_dir.name}/{emb_file}) ===")
        emb_thread_results: list[dict[str, Any]] = []

        for label, n_threads in thread_configs:
            _, median_ms, mean_ms, std_ms = _measure_latency(
                lambda nt=n_threads: _encode_onnx_raw(
                    texts, emb_dir, emb_file,  # type: ignore[arg-type]
                    num_threads=nt,
                ),
            )
            rss = _measure_rss_mb()
            emb_thread_results.append({
                "label": label, "threads": n_threads,
                "median_ms": median_ms, "mean_ms": mean_ms,
                "std_ms": std_ms, "rss_mb": rss,
            })
            print(f"    {label} (threads={n_threads}): "
                  f"median={median_ms:.1f}ms, mean={mean_ms:.1f}ms")

        # 최적 스레드 선택
        best = min(emb_thread_results, key=lambda x: x["median_ms"])
        print(f"  -> 최적: {best['label']} (threads={best['threads']}, "
              f"{best['median_ms']:.1f}ms)")
        results["emb_threads"] = {
            "model_dir": emb_dir.name,
            "all_results": emb_thread_results,
            "best_label": best["label"],
            "best_threads": best["threads"],
            "best_median_ms": best["median_ms"],
        }
    elif run_embedding:
        print("  [건너뜀] 임베딩 ONNX 디렉토리 없음")

    if run_reranker and rr_dir and rr_file:
        print(f"\n  === 리랭커 ({rr_dir.name}/{rr_file}) ===")
        rr_thread_results: list[dict[str, Any]] = []

        for label, n_threads in thread_configs:
            _, median_ms, mean_ms, std_ms = _measure_latency(
                lambda nt=n_threads: _rerank_onnx_raw(
                    RR_QUERY, RR_DOCUMENTS,
                    rr_dir, rr_file,  # type: ignore[arg-type]
                    num_threads=nt,
                ),
            )
            rss = _measure_rss_mb()
            rr_thread_results.append({
                "label": label, "threads": n_threads,
                "median_ms": median_ms, "mean_ms": mean_ms,
                "std_ms": std_ms, "rss_mb": rss,
            })
            print(f"    {label} (threads={n_threads}): "
                  f"median={median_ms:.1f}ms, mean={mean_ms:.1f}ms")

        best = min(rr_thread_results, key=lambda x: x["median_ms"])
        print(f"  -> 최적: {best['label']} (threads={best['threads']}, "
              f"{best['median_ms']:.1f}ms)")
        results["rr_threads"] = {
            "model_dir": rr_dir.name,
            "all_results": rr_thread_results,
            "best_label": best["label"],
            "best_threads": best["threads"],
            "best_median_ms": best["median_ms"],
        }
    elif run_reranker:
        print("  [건너뜀] 리랭커 ONNX 디렉토리 없음")

    return results


# ============================================================
# Phase 6: ORT 프로파일링 (노드별 실행시간 top-10)
# ============================================================


def _run_ort_profiling(
    model_dir: Path,
    file_name: str,
    feed_fn: Any,
    label: str,
    top_n: int = 10,
) -> list[dict[str, Any]]:
    """ORT 프로파일링을 실행하고 상위 노드를 반환."""
    import onnxruntime as ort

    print(f"\n  [{label}] 프로파일링 중...")

    session_options = ort.SessionOptions()
    session_options.enable_profiling = True

    session = ort.InferenceSession(
        str(model_dir / file_name),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    feed = feed_fn(session)

    # 워밍업
    for _ in range(3):
        session.run(None, feed)

    # 프로파일링 실행
    session.run(None, feed)
    profile_file = session.end_profiling()

    del session
    gc.collect()

    # 프로파일 JSON 파싱
    try:
        profile_path = Path(profile_file)
        with profile_path.open("r", encoding="utf-8") as f:
            profile_data = json.load(f)

        # 노드 실행시간 추출
        node_times: list[dict[str, Any]] = []
        events = profile_data if isinstance(profile_data, list) else []

        for event in events:
            if not isinstance(event, dict):
                continue
            cat = event.get("cat", "")
            dur = event.get("dur", 0)
            name = event.get("name", "")
            args = event.get("args", {})

            if cat in ("Node", "kernel") and dur > 0:
                op_name = args.get("op_name", name) if isinstance(args, dict) else name
                node_times.append({
                    "name": name,
                    "op_name": op_name,
                    "duration_us": dur,
                })

        # 상위 N개 정렬
        node_times.sort(key=lambda x: x["duration_us"], reverse=True)
        top_nodes = node_times[:top_n]

        total_us = sum(n["duration_us"] for n in node_times) if node_times else 1
        print(f"    총 노드 수: {len(node_times)}, "
              f"총 실행시간: {total_us / 1000:.1f}ms")
        print(f"    상위 {top_n}개 노드:")

        for i, node in enumerate(top_nodes):
            pct = (node["duration_us"] / total_us) * 100
            dur_ms = node["duration_us"] / 1000
            print(f"      {i+1:2d}. {node['op_name']:<30s} "
                  f"{dur_ms:>8.2f}ms ({pct:>5.1f}%)")

        # 프로파일 파일 정리
        try:
            profile_path.unlink()
        except OSError:
            pass

        return top_nodes

    except Exception as e:
        print(f"    프로파일 파싱 실패: {e!s:.80s}")
        # 프로파일 파일 정리 시도
        try:
            Path(profile_file).unlink()
        except OSError:
            pass
        return []


def run_phase_6(
    *,
    run_embedding: bool = True,
    run_reranker: bool = True,
) -> dict[str, Any]:
    """Phase 6: ORT 프로파일링."""
    print(f"\n{SEPARATOR}")
    print("  Phase 6: ORT 프로파일링 (노드별 실행시간)")
    print(SEPARATOR)

    try:
        import onnxruntime as ort  # noqa: F401
    except ImportError:
        print("  [건너뜀] onnxruntime 미설치")
        return {}

    from transformers import AutoTokenizer

    results: dict[str, Any] = {}
    texts = EMB_TEST_QUERIES

    # 프로파일 대상 디렉토리 결정 (최적화 모델 우선)
    emb_prof_dir: Path | None = None
    emb_prof_file: str | None = None
    for d in [EMB_ORT_OPT_DIR, EMB_ONNX_DIR]:
        if d.exists():
            f = _find_onnx_file(d)
            if f:
                emb_prof_dir = d
                emb_prof_file = f
                break

    rr_prof_dir: Path | None = None
    rr_prof_file: str | None = None
    for d in [RR_ORT_OPT_DIR, RR_ONNX_DIR]:
        if d.exists():
            f = _find_onnx_file(d)
            if f:
                rr_prof_dir = d
                rr_prof_file = f
                break

    if run_embedding and emb_prof_dir and emb_prof_file:
        def _emb_feed_fn(session: Any) -> dict[str, Any]:
            tokenizer = AutoTokenizer.from_pretrained(
                str(emb_prof_dir), trust_remote_code=True,
            )
            inputs = tokenizer(
                texts, return_tensors="np", padding=True,
                truncation=True, max_length=512,
            )
            input_names = [inp.name for inp in session.get_inputs()]
            feed: dict[str, Any] = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            }
            if "token_type_ids" in input_names:
                feed["token_type_ids"] = inputs.get(
                    "token_type_ids",
                    np.zeros_like(inputs["input_ids"]),
                )
            return feed

        top_nodes = _run_ort_profiling(
            emb_prof_dir, emb_prof_file, _emb_feed_fn,
            f"임베딩 ({emb_prof_dir.name})",
        )
        results["emb_profiling"] = {
            "model_dir": emb_prof_dir.name,
            "top_nodes": top_nodes,
        }
    elif run_embedding:
        print("  [건너뜀] 임베딩 프로파일링 대상 없음")

    if run_reranker and rr_prof_dir and rr_prof_file:
        def _rr_feed_fn(session: Any) -> dict[str, Any]:
            tokenizer = AutoTokenizer.from_pretrained(str(rr_prof_dir))
            queries = [RR_QUERY] * len(RR_DOCUMENTS)
            inputs = tokenizer(
                queries, RR_DOCUMENTS, return_tensors="np",
                padding=True, truncation=True, max_length=512,
            )
            input_names = [inp.name for inp in session.get_inputs()]
            feed: dict[str, Any] = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            }
            if "token_type_ids" in input_names:
                feed["token_type_ids"] = inputs.get(
                    "token_type_ids",
                    np.zeros_like(inputs["input_ids"]),
                )
            return feed

        top_nodes = _run_ort_profiling(
            rr_prof_dir, rr_prof_file, _rr_feed_fn,
            f"리랭커 ({rr_prof_dir.name})",
        )
        results["rr_profiling"] = {
            "model_dir": rr_prof_dir.name,
            "top_nodes": top_nodes,
        }
    elif run_reranker:
        print("  [건너뜀] 리랭커 프로파일링 대상 없음")

    return results


# ============================================================
# Pearson 상관계수 유틸리티
# ============================================================


def _compute_pearson(
    scores_a: list[float] | None,
    scores_b: list[float] | None,
) -> float:
    """두 스코어 리스트 간 Pearson 상관계수를 계산."""
    if scores_a is None or scores_b is None:
        return 0.0
    if len(scores_a) != len(scores_b) or len(scores_a) < 2:
        return 0.0
    try:
        from scipy.stats import pearsonr
        corr, _ = pearsonr(scores_a, scores_b)
        return float(corr)
    except ImportError:
        # scipy 없을 때 직접 계산
        a = np.array(scores_a)
        b = np.array(scores_b)
        a_mean = a - a.mean()
        b_mean = b - b.mean()
        numerator = float(np.sum(a_mean * b_mean))
        denominator = float(
            np.sqrt(np.sum(a_mean**2)) * np.sqrt(np.sum(b_mean**2)),
        )
        if denominator < 1e-12:
            return 0.0
        return numerator / denominator


# ============================================================
# Phase 8: BF16 Fastmath 벤치마크 (Graviton3 전용)
# ============================================================


def _detect_arm_bf16_support() -> bool:
    """ARM 서버에서 BF16 MMLA 명령어 지원 여부를 감지한다."""
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8")
        for line in cpuinfo.splitlines():
            if line.startswith("Features"):
                features = line.split(":", 1)[1].strip().split()
                return "bf16" in features
    except (FileNotFoundError, PermissionError, OSError):
        pass
    return False


def run_phase_8(
    baseline: dict[str, dict[str, Any]],
    env_info: dict[str, Any],
    *,
    run_embedding: bool = True,
    run_reranker: bool = True,
) -> dict[str, dict[str, Any]]:
    """Phase 8: BF16 Fastmath 벤치마크 (Graviton3 전용).

    FP32 모델 파일을 유지하면서 MLAS SBGEMM 커널이
    런타임에 BF16 MMLA 경로를 선택하여 GEMM을 가속한다.
    Mac ARM은 BF16 MMLA 미지원이므로 스킵.
    """
    print(f"\n{SEPARATOR}")
    print("  Phase 8: BF16 Fastmath (Graviton3)")
    print(SEPARATOR)

    results: dict[str, dict[str, Any]] = {}
    is_arm = env_info.get("is_arm", False)
    is_mac = env_info.get("is_mac", False)

    if not is_arm or is_mac:
        print("  [스킵] BF16 Fastmath는 Linux ARM (Graviton3+) 전용입니다.")
        print(f"    현재 환경: {'Mac ARM' if is_mac else env_info.get('machine', 'unknown')}")
        results["bf16_skipped"] = {"reason": "not_linux_arm"}
        return results

    bf16_supported = _detect_arm_bf16_support()
    if not bf16_supported:
        print("  [스킵] /proc/cpuinfo에 bf16 feature flag가 없습니다.")
        results["bf16_skipped"] = {"reason": "no_bf16_feature"}
        return results

    print("  BF16 MMLA 지원 감지됨")
    texts = EMB_TEST_QUERIES

    if run_embedding and EMB_ORT_OPT_DIR.exists():
        opt_file = _find_onnx_file(EMB_ORT_OPT_DIR)
        if opt_file:
            print("\n  === 임베딩: BF16 OFF vs ON ===")

            # BF16 OFF (기준)
            print("    [BF16 OFF] 측정 중...")
            emb_off, med_off, mean_off, std_off = _measure_latency(
                lambda: _encode_onnx_raw(texts, EMB_ORT_OPT_DIR, opt_file),
            )

            # BF16 ON
            print("    [BF16 ON] 측정 중...")
            emb_on, med_on, mean_on, std_on = _measure_latency(
                lambda: _encode_onnx_raw(
                    texts, EMB_ORT_OPT_DIR, opt_file,
                    enable_bf16_fastmath=True,
                ),
            )

            cos = _cosine_similarity(emb_off, emb_on)
            max_diff = float(np.max(np.abs(emb_off - emb_on)))
            speedup_vs_off = med_off / med_on if med_on > 0 else 0.0

            pt_ms = baseline.get("emb_pytorch_cpu", {}).get("median_ms", 1.0)
            speedup_vs_pt = pt_ms / med_on if med_on > 0 else 0.0

            print(f"    BF16 OFF: {med_off:.1f}ms")
            print(f"    BF16 ON:  {med_on:.1f}ms ({speedup_vs_off:.2f}x)")
            print(f"    cosine(OFF vs ON): {cos:.6f}")
            print(f"    max_abs_diff: {max_diff:.8f}")

            results["emb_bf16"] = {
                "median_ms_off": med_off, "median_ms_on": med_on,
                "median_ms": med_on,
                "speedup_vs_off": speedup_vs_off,
                "speedup": speedup_vs_pt,
                "cosine_off_vs_on": cos,
                "max_abs_diff": max_diff,
                "mean_ms": mean_on, "std_ms": std_on,
            }

    if run_reranker and RR_ORT_OPT_DIR.exists():
        opt_file = _find_onnx_file(RR_ORT_OPT_DIR)
        if opt_file:
            print("\n  === 리랭커: BF16 OFF vs ON ===")

            print("    [BF16 OFF] 측정 중...")
            scores_off, med_off, _, _ = _measure_latency(
                lambda: _rerank_onnx_raw(
                    RR_QUERY, RR_DOCUMENTS, RR_ORT_OPT_DIR, opt_file,
                ),
            )

            print("    [BF16 ON] 측정 중...")
            scores_on, med_on, mean_on, std_on = _measure_latency(
                lambda: _rerank_onnx_raw(
                    RR_QUERY, RR_DOCUMENTS, RR_ORT_OPT_DIR, opt_file,
                    enable_bf16_fastmath=True,
                ),
            )

            pearson = _compute_pearson(scores_off, scores_on)
            max_diff = max(
                abs(a - b) for a, b in zip(scores_off, scores_on)
            )
            speedup_vs_off = med_off / med_on if med_on > 0 else 0.0

            pt_ms = baseline.get("rr_pytorch_cpu", {}).get("median_ms", 1.0)
            speedup_vs_pt = pt_ms / med_on if med_on > 0 else 0.0

            print(f"    BF16 OFF: {med_off:.1f}ms")
            print(f"    BF16 ON:  {med_on:.1f}ms ({speedup_vs_off:.2f}x)")
            print(f"    Pearson(OFF vs ON): {pearson:.6f}")
            print(f"    max_abs_diff: {max_diff:.8f}")

            results["rr_bf16"] = {
                "median_ms_off": med_off, "median_ms_on": med_on,
                "median_ms": med_on,
                "speedup_vs_off": speedup_vs_off,
                "speedup": speedup_vs_pt,
                "pearson_off_vs_on": pearson,
                "max_abs_diff": max_diff,
                "mean_ms": mean_on, "std_ms": std_on,
            }

    return results


# ============================================================
# Phase 9: Static Shape 벤치마크 (임베딩 전용)
# ============================================================


def run_phase_9(
    baseline: dict[str, dict[str, Any]],
    *,
    run_embedding: bool = True,
) -> dict[str, dict[str, Any]]:
    """Phase 9: Static Shape (batch=1, seq=128) 벤치마크.

    동적 shape(가변 패딩)과 고정 shape(128 고정 패딩)의
    레이턴시 차이를 측정한다. 임베딩 전용 (리랭커는 batch 가변).
    """
    print(f"\n{SEPARATOR}")
    print("  Phase 9: Static Shape (임베딩 전용)")
    print(SEPARATOR)

    results: dict[str, dict[str, Any]] = {}

    if not run_embedding:
        print("  [스킵] 임베딩 미실행 모드")
        return results

    if not EMB_ORT_OPT_STATIC128_DIR.exists():
        print(f"  [스킵] static128 디렉토리 없음: {EMB_ORT_OPT_STATIC128_DIR.name}")
        print("    빌드: uv run python scripts/build_optimized_onnx.py")
        return results

    static_file = _find_onnx_file(EMB_ORT_OPT_STATIC128_DIR)
    if not static_file:
        print("  [스킵] static128 ONNX 모델 파일 없음")
        return results

    dynamic_file = _find_onnx_file(EMB_ORT_OPT_DIR) if EMB_ORT_OPT_DIR.exists() else None
    if not dynamic_file:
        print("  [스킵] dynamic ORT-opt 디렉토리 없음")
        return results

    texts = EMB_TEST_QUERIES

    # 다양한 입력 길이 테스트
    test_cases = [
        ("짧은 쿼리 (2-5어절)", texts[:3]),
        ("중간 쿼리 (5-10어절)", texts),
        ("긴 쿼리 (반복 패딩)", [t * 3 for t in texts[:3]]),
    ]

    print("\n  === Dynamic vs Static128 ===")

    # Dynamic baseline
    print("    [Dynamic] 측정 중...")
    emb_dyn, med_dyn, mean_dyn, std_dyn = _measure_latency(
        lambda: _encode_onnx_raw(texts, EMB_ORT_OPT_DIR, dynamic_file),
    )

    # Static128
    print("    [Static128] 측정 중...")
    emb_static, med_static, mean_static, std_static = _measure_latency(
        lambda: _encode_onnx_raw(
            texts, EMB_ORT_OPT_STATIC128_DIR, static_file,
            static_max_length=128,
        ),
    )

    cos = _cosine_similarity(emb_dyn, emb_static)
    max_diff = float(np.max(np.abs(emb_dyn - emb_static)))
    speedup_vs_dyn = med_dyn / med_static if med_static > 0 else 0.0

    pt_ms = baseline.get("emb_pytorch_cpu", {}).get("median_ms", 1.0)
    speedup_vs_pt = pt_ms / med_static if med_static > 0 else 0.0

    print(f"    Dynamic:   {med_dyn:.1f}ms")
    print(f"    Static128: {med_static:.1f}ms ({speedup_vs_dyn:.2f}x vs Dynamic)")
    print(f"    cosine(Dyn vs Static): {cos:.6f}")
    print(f"    max_abs_diff: {max_diff:.8f}")

    results["emb_static128"] = {
        "median_ms_dynamic": med_dyn,
        "median_ms_static": med_static,
        "median_ms": med_static,
        "speedup_vs_dynamic": speedup_vs_dyn,
        "speedup": speedup_vs_pt,
        "cosine_dyn_vs_static": cos,
        "max_abs_diff": max_diff,
        "mean_ms": mean_static, "std_ms": std_static,
    }

    # 입력 길이별 상세 측정
    print("\n    --- 입력 길이별 레이턴시 ---")
    length_results: list[dict[str, Any]] = []
    for label, test_texts in test_cases:
        _, med_d, _, _ = _measure_latency(
            lambda _t=test_texts: _encode_onnx_raw(
                _t, EMB_ORT_OPT_DIR, dynamic_file,
            ),
        )
        _, med_s, _, _ = _measure_latency(
            lambda _t=test_texts: _encode_onnx_raw(
                _t, EMB_ORT_OPT_STATIC128_DIR, static_file,
                static_max_length=128,
            ),
        )
        ratio = med_d / med_s if med_s > 0 else 0.0
        print(f"    {label}: Dynamic={med_d:.1f}ms, Static={med_s:.1f}ms ({ratio:.2f}x)")
        length_results.append({
            "label": label, "dynamic_ms": med_d,
            "static_ms": med_s, "ratio": ratio,
        })

    results["emb_static128"]["length_tests"] = length_results

    return results


# ============================================================
# Phase 10: 복합 최적화 (BF16 + Static Shape, Graviton3 전용)
# ============================================================


def run_phase_10(
    baseline: dict[str, dict[str, Any]],
    env_info: dict[str, Any],
    *,
    run_embedding: bool = True,
) -> dict[str, dict[str, Any]]:
    """Phase 10: BF16 + Static Shape 복합 최적화 (Graviton3 전용).

    BF16 Fastmath와 Static Shape을 동시에 적용하여
    최대 성능을 측정한다. 임베딩 전용.
    """
    print(f"\n{SEPARATOR}")
    print("  Phase 10: 복합 최적화 (BF16 + Static128)")
    print(SEPARATOR)

    results: dict[str, dict[str, Any]] = {}
    is_arm = env_info.get("is_arm", False)
    is_mac = env_info.get("is_mac", False)

    if not is_arm or is_mac:
        print("  [스킵] Graviton3 전용 (Linux ARM만 지원)")
        return results

    if not _detect_arm_bf16_support():
        print("  [스킵] BF16 미지원 플랫폼")
        return results

    if not run_embedding:
        print("  [스킵] 임베딩 미실행 모드")
        return results

    if not EMB_ORT_OPT_STATIC128_DIR.exists():
        print(f"  [스킵] static128 디렉토리 없음: {EMB_ORT_OPT_STATIC128_DIR.name}")
        return results

    static_file = _find_onnx_file(EMB_ORT_OPT_STATIC128_DIR)
    dynamic_file = _find_onnx_file(EMB_ORT_OPT_DIR) if EMB_ORT_OPT_DIR.exists() else None
    if not static_file or not dynamic_file:
        print("  [스킵] 모델 파일 없음")
        return results

    texts = EMB_TEST_QUERIES

    print("\n  === 임베딩 복합 최적화 비교 ===")

    # (1) Dynamic FP32 (기준)
    print("    [Dynamic FP32] 측정 중...")
    emb_base, med_base, _, _ = _measure_latency(
        lambda: _encode_onnx_raw(texts, EMB_ORT_OPT_DIR, dynamic_file),
    )

    # (2) Dynamic + BF16
    print("    [Dynamic + BF16] 측정 중...")
    emb_bf16, med_bf16, _, _ = _measure_latency(
        lambda: _encode_onnx_raw(
            texts, EMB_ORT_OPT_DIR, dynamic_file,
            enable_bf16_fastmath=True,
        ),
    )

    # (3) Static128 FP32
    print("    [Static128 FP32] 측정 중...")
    emb_st, med_st, _, _ = _measure_latency(
        lambda: _encode_onnx_raw(
            texts, EMB_ORT_OPT_STATIC128_DIR, static_file,
            static_max_length=128,
        ),
    )

    # (4) Static128 + BF16 (최대 최적화)
    print("    [Static128 + BF16] 측정 중...")
    emb_combo, med_combo, mean_combo, std_combo = _measure_latency(
        lambda: _encode_onnx_raw(
            texts, EMB_ORT_OPT_STATIC128_DIR, static_file,
            enable_bf16_fastmath=True,
            static_max_length=128,
        ),
    )

    cos = _cosine_similarity(emb_base, emb_combo)
    max_diff = float(np.max(np.abs(emb_base - emb_combo)))
    pt_ms = baseline.get("emb_pytorch_cpu", {}).get("median_ms", 1.0)

    print("\n    결과 비교:")
    print(f"    {'Variant':<25s} | {'Latency':>10s} | {'vs Base':>8s} | {'vs PyTorch':>10s}")
    print(f"    {'-'*25} | {'-'*10} | {'-'*8} | {'-'*10}")
    for label, med in [
        ("Dynamic FP32", med_base),
        ("Dynamic + BF16", med_bf16),
        ("Static128 FP32", med_st),
        ("Static128 + BF16", med_combo),
    ]:
        vs_base = med_base / med if med > 0 else 0
        vs_pt = pt_ms / med if med > 0 else 0
        print(f"    {label:<25s} | {med:>8.1f}ms | {vs_base:>6.2f}x | {vs_pt:>8.2f}x")

    print(f"\n    cosine(Base vs Combo): {cos:.6f}")
    print(f"    max_abs_diff: {max_diff:.8f}")

    results["emb_combined"] = {
        "median_ms_base": med_base,
        "median_ms_bf16": med_bf16,
        "median_ms_static": med_st,
        "median_ms_combined": med_combo,
        "median_ms": med_combo,
        "speedup_vs_base": med_base / med_combo if med_combo > 0 else 0,
        "speedup": pt_ms / med_combo if med_combo > 0 else 0,
        "cosine_base_vs_combo": cos,
        "max_abs_diff": max_diff,
        "mean_ms": mean_combo, "std_ms": std_combo,
    }

    return results


# ============================================================
# Phase 7: 종합 보고서
# ============================================================


def _collect_summary_rows(
    baseline: dict[str, dict[str, Any]],
    phase2: dict[str, dict[str, Any]],
    phase3: dict[str, dict[str, Any]],
    phase4: dict[str, dict[str, Any]],
    phase5: dict[str, dict[str, Any]],
    *,
    model_type: str,
    phase8: dict[str, dict[str, Any]] | None = None,
    phase9: dict[str, dict[str, Any]] | None = None,
    phase10: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """지정 모델 타입(emb/rr)의 모든 결과를 요약 행으로 수집."""
    rows: list[dict[str, Any]] = []

    prefix = "emb" if model_type == "emb" else "rr"
    quality_key = "cosine" if model_type == "emb" else "pearson"

    # PyTorch CPU baseline
    pt_key = f"{prefix}_pytorch_cpu"
    if pt_key in baseline:
        data = baseline[pt_key]
        rows.append({
            "name": "PyTorch FP32 (CPU)",
            "median_ms": data["median_ms"],
            "speedup": 1.0,
            quality_key: 1.0,
            "rss_mb": data.get("rss_mb", 0),
        })

    # PyTorch MPS
    mps_key = f"{prefix}_pytorch_mps"
    if mps_key in baseline:
        data = baseline[mps_key]
        pt_ms = baseline.get(pt_key, {}).get("median_ms", 1.0)
        rows.append({
            "name": "PyTorch FP32 (MPS)",
            "median_ms": data["median_ms"],
            "speedup": pt_ms / data["median_ms"] if data["median_ms"] > 0 else 0,
            quality_key: data.get("cosine_vs_cpu", 1.0),
            "rss_mb": data.get("rss_mb", 0),
        })

    # ONNX 원본 FP32
    onnx_key = f"{prefix}_onnx_fp32"
    if onnx_key in phase2:
        data = phase2[onnx_key]
        rows.append({
            "name": "ONNX FP32 (원본)",
            "median_ms": data["median_ms"],
            "speedup": data.get("speedup", 0),
            quality_key: data.get(quality_key, 0),
            "rss_mb": data.get("rss_mb", 0),
        })

    # ORT-최적화 FP32
    opt_key = f"{prefix}_ort_opt_fp32"
    if opt_key in phase3:
        data = phase3[opt_key]
        rows.append({
            "name": "ORT-최적화 FP32",
            "median_ms": data["median_ms"],
            "speedup": data.get("speedup", 0),
            quality_key: data.get(quality_key, 0),
            "rss_mb": data.get("rss_mb", 0),
        })

    # ORT-최적화 FP16
    fp16_key = f"{prefix}_ort_opt_fp16"
    if fp16_key in phase4:
        data = phase4[fp16_key]
        rows.append({
            "name": "ORT-최적화 FP16",
            "median_ms": data["median_ms"],
            "speedup": data.get("speedup", 0),
            quality_key: data.get(quality_key, 0),
            "rss_mb": data.get("rss_mb", 0),
        })

    # 스레드 최적화 결과
    thread_key = f"{prefix}_threads"
    if thread_key in phase5:
        thread_data = phase5[thread_key]
        best_label = thread_data.get("best_label", "")
        best_threads = thread_data.get("best_threads", 0)
        best_ms = thread_data.get("best_median_ms", 0)
        pt_ms = baseline.get(pt_key, {}).get("median_ms", 1.0)
        rows.append({
            "name": f"스레드최적 ({best_label}, t={best_threads})",
            "median_ms": best_ms,
            "speedup": pt_ms / best_ms if best_ms > 0 else 0,
            quality_key: 0,  # 별도 품질 측정 미실시
            "rss_mb": 0,
        })

    # BF16 Fastmath (Phase 8)
    if phase8:
        bf16_key = f"{prefix}_bf16"
        if bf16_key in phase8:
            data = phase8[bf16_key]
            qual = data.get("cosine_off_vs_on", 0) if prefix == "emb" else data.get("pearson_off_vs_on", 0)
            rows.append({
                "name": "ORT-opt FP32 + BF16",
                "median_ms": data["median_ms"],
                "speedup": data.get("speedup", 0),
                quality_key: qual,
                "rss_mb": 0,
            })

    # Static Shape (Phase 9, 임베딩만)
    if phase9 and prefix == "emb":
        if "emb_static128" in phase9:
            data = phase9["emb_static128"]
            rows.append({
                "name": "ORT-opt Static128",
                "median_ms": data["median_ms"],
                "speedup": data.get("speedup", 0),
                quality_key: data.get("cosine_dyn_vs_static", 0),
                "rss_mb": 0,
            })

    # 복합 최적화 (Phase 10, 임베딩만)
    if phase10 and prefix == "emb":
        if "emb_combined" in phase10:
            data = phase10["emb_combined"]
            rows.append({
                "name": "Static128 + BF16 (복합)",
                "median_ms": data["median_ms"],
                "speedup": data.get("speedup", 0),
                quality_key: data.get("cosine_base_vs_combo", 0),
                "rss_mb": 0,
            })

    return rows


def generate_report(
    env_info: dict[str, Any],
    baseline: dict[str, dict[str, Any]],
    phase2: dict[str, dict[str, Any]],
    phase3: dict[str, dict[str, Any]],
    phase4: dict[str, dict[str, Any]],
    phase5: dict[str, dict[str, Any]],
    phase6: dict[str, Any],
    *,
    phase8: dict[str, dict[str, Any]] | None = None,
    phase9: dict[str, dict[str, Any]] | None = None,
    phase10: dict[str, dict[str, Any]] | None = None,
    run_embedding: bool = True,
    run_reranker: bool = True,
    output_json: str | None = None,
    output_md: Path | None = None,
) -> None:
    """Phase 7: JSON 결과 저장 + 마크다운 테이블 출력."""
    print(f"\n{SEPARATOR}")
    print("  Phase 7: 종합 보고서")
    print(SEPARATOR)

    # --- JSON 결과 수집 ---
    json_results: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "environment": {
            k: v for k, v in env_info.items()
            if k != "available_models"
        },
        "available_models": env_info.get("available_models", {}),
    }

    # 결과에서 numpy array / list 제거 (JSON 직렬화 불가)
    def _sanitize(data: dict[str, Any]) -> dict[str, Any]:
        sanitized: dict[str, Any] = {}
        for k, v in data.items():
            if isinstance(v, dict):
                sanitized[k] = _sanitize(v)
            elif isinstance(v, np.ndarray):
                continue  # 배열 제외
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                sanitized[k] = v  # dict 리스트는 유지
            elif k in ("embeddings",):
                continue  # 대용량 배열 제외
            else:
                sanitized[k] = v
        return sanitized

    json_results["phase1_baseline"] = _sanitize(baseline)
    json_results["phase2_onnx_fp32"] = _sanitize(phase2)
    json_results["phase3_ort_opt_fp32"] = _sanitize(phase3)
    json_results["phase4_ort_opt_fp16"] = _sanitize(phase4)
    json_results["phase5_threads"] = _sanitize(phase5)
    if phase8:
        json_results["phase8_bf16"] = _sanitize(phase8)
    if phase9:
        json_results["phase9_static"] = _sanitize(phase9)
    if phase10:
        json_results["phase10_combined"] = _sanitize(phase10)
    json_results["phase6_profiling"] = phase6

    # JSON 저장
    if output_json:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n  JSON 결과 저장: {output_path}")

    # --- 콘솔 마크다운 테이블 ---
    md_lines: list[str] = []

    def _print_table(
        title: str,
        rows: list[dict[str, Any]],
        quality_label: str,
    ) -> None:
        print(f"\n  {title}")
        header = (
            f"  {'Variant':<35s} | {'Latency (ms)':>12s} | "
            f"{'Speedup':>8s} | {quality_label:>8s} | {'Memory (MB)':>11s}"
        )
        separator_line = (
            f"  {'-' * 35} | {'-' * 12} | "
            f"{'-' * 8} | {'-' * 8} | {'-' * 11}"
        )
        print(header)
        print(separator_line)
        md_lines.append(f"\n### {title}\n")
        md_lines.append(
            f"| Variant | Latency (ms) | Speedup | "
            f"{quality_label} | Memory (MB) |",
        )
        md_lines.append(
            "|---------|-------------|---------|--------|-------------|",
        )

        for row in rows:
            name = row["name"]
            lat = row["median_ms"]
            spd = row["speedup"]
            qual = row.get(quality_label.lower(), 0)
            mem = row.get("rss_mb", 0)

            qual_str = f"{qual:.4f}" if qual > 0 else "N/A"
            mem_str = f"{mem:.0f}" if mem > 0 else "N/A"

            line = (
                f"  {name:<35s} | {lat:>10.1f}ms | "
                f"{spd:>6.2f}x | {qual_str:>8s} | {mem_str:>11s}"
            )
            print(line)
            md_lines.append(
                f"| {name} | {lat:.1f} | {spd:.2f}x | "
                f"{qual_str} | {mem_str} |",
            )

    if run_embedding:
        emb_rows = _collect_summary_rows(
            baseline, phase2, phase3, phase4, phase5,
            model_type="emb",
            phase8=phase8, phase9=phase9, phase10=phase10,
        )
        if emb_rows:
            _print_table("임베딩 비교", emb_rows, "Cosine")

    if run_reranker:
        rr_rows = _collect_summary_rows(
            baseline, phase2, phase3, phase4, phase5,
            model_type="rr",
            phase8=phase8,
        )
        if rr_rows:
            _print_table("리랭커 비교", rr_rows, "Pearson")

    # --- 마크다운 보고서 파일 ---
    if output_md:
        _write_md_report(
            output_md, env_info, md_lines,
            baseline, phase2, phase3, phase4, phase5, phase6,
            phase8=phase8, phase9=phase9, phase10=phase10,
            run_embedding=run_embedding,
            run_reranker=run_reranker,
        )


def _write_md_report(
    output_path: Path,
    env_info: dict[str, Any],
    table_lines: list[str],
    baseline: dict[str, dict[str, Any]],
    phase2: dict[str, dict[str, Any]],
    phase3: dict[str, dict[str, Any]],
    phase4: dict[str, dict[str, Any]],
    phase5: dict[str, dict[str, Any]],
    phase6: dict[str, Any],
    *,
    phase8: dict[str, dict[str, Any]] | None = None,
    phase9: dict[str, dict[str, Any]] | None = None,
    phase10: dict[str, dict[str, Any]] | None = None,
    run_embedding: bool = True,
    run_reranker: bool = True,
) -> None:
    """마크다운 보고서 파일을 생성."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arch_label = "ARM" if env_info.get("is_arm") else "x86_64"
    lines: list[str] = [
        "# ARM/크로스플랫폼 ONNX 최적화 벤치마크 보고서",
        "",
        f"생성일: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 환경",
        "",
        f"- 아키텍처: {env_info['machine']} ({arch_label})",
        f"- 플랫폼: {env_info['platform']} {env_info.get('platform_release', '')}",
        f"- CPU 코어: {env_info['cpu_count']}",
        f"- 총 RAM: {env_info.get('total_ram_gb', 'N/A')}GB",
        f"- onnxruntime: {env_info.get('ort_version', 'N/A')}",
        f"- PyTorch: {env_info.get('torch_version', 'N/A')}",
        f"- MPS: {'사용 가능' if env_info.get('mps_available') else '미사용'}",
        f"- CUDA: {'사용 가능' if env_info.get('cuda_available') else '미사용'}",
        "",
        "## 모델",
        "",
        f"- 임베딩: `{EMB_MODEL_NAME}` (XLM-RoBERTa Large, 1024차원)",
        f"- 리랭커: `{RR_MODEL_NAME}` (XLM-RoBERTa Large)",
        "",
        "## 벤치마크 결과",
    ]

    # 테이블 삽입
    lines.extend(table_lines)

    # 스레드 최적화 상세
    if phase5:
        lines.extend(["", "### 스레드 최적화 상세", ""])
        for key in ["emb_threads", "rr_threads"]:
            if key in phase5:
                model_label = "임베딩" if key.startswith("emb") else "리랭커"
                data = phase5[key]
                lines.append(f"**{model_label}** ({data.get('model_dir', '')}):")
                lines.append("")
                lines.append("| 설정 | 스레드 수 | Latency (ms) |")
                lines.append("|------|----------|-------------|")
                for r in data.get("all_results", []):
                    lines.append(
                        f"| {r['label']} | {r['threads']} | "
                        f"{r['median_ms']:.1f} |",
                    )
                lines.append(
                    f"\n최적: **{data['best_label']}** "
                    f"(threads={data['best_threads']}, "
                    f"{data['best_median_ms']:.1f}ms)",
                )
                lines.append("")

    # 프로파일링 결과
    if phase6:
        lines.extend(["", "### ORT 프로파일링 상위 노드", ""])
        for key in ["emb_profiling", "rr_profiling"]:
            if key in phase6:
                model_label = "임베딩" if key.startswith("emb") else "리랭커"
                data = phase6[key]
                lines.append(
                    f"**{model_label}** ({data.get('model_dir', '')}):",
                )
                lines.append("")
                lines.append("| # | 연산 | 시간 (ms) |")
                lines.append("|---|------|----------|")
                for i, node in enumerate(data.get("top_nodes", [])[:10]):
                    dur_ms = node["duration_us"] / 1000
                    lines.append(
                        f"| {i+1} | {node['op_name']} | {dur_ms:.2f} |",
                    )
                lines.append("")

    # 권장사항
    lines.extend(["", "## 권장사항", ""])

    for model_type, model_label in [("emb", "임베딩"), ("rr", "리랭커")]:
        if (model_type == "emb" and not run_embedding) or \
           (model_type == "rr" and not run_reranker):
            continue

        all_rows = _collect_summary_rows(
            baseline, phase2, phase3, phase4, phase5,
            model_type=model_type,
            phase8=phase8, phase9=phase9, phase10=phase10,
        )
        # baseline 제외한 최적 방법 찾기
        candidates = [r for r in all_rows if r["name"] != "PyTorch FP32 (CPU)"]
        if candidates:
            best = min(candidates, key=lambda x: x["median_ms"])
            lines.append(
                f"- **{model_label}**: {best['name']} "
                f"({best['speedup']:.2f}x, {best['median_ms']:.1f}ms)",
            )

    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  MD 보고서 생성: {output_path}")


# ============================================================
# CLI + main
# ============================================================


def parse_args() -> argparse.Namespace:
    """CLI 인자 파싱."""
    parser = argparse.ArgumentParser(
        description="ARM/크로스플랫폼 ONNX 최적화 벤치마크 "
                    "(임베딩 + 리랭커)",
    )
    parser.add_argument(
        "--embedding-only", action="store_true",
        help="임베딩만 실행",
    )
    parser.add_argument(
        "--reranker-only", action="store_true",
        help="리랭커만 실행",
    )
    parser.add_argument(
        "--skip-profiling", action="store_true",
        help="ORT 프로파일링 (Phase 6) 스킵",
    )
    parser.add_argument(
        "--skip-fp16", action="store_true",
        help="FP16 변환 (Phase 4) 스킵",
    )
    parser.add_argument(
        "--skip-threads", action="store_true",
        help="스레드 최적화 (Phase 5) 스킵",
    )
    parser.add_argument(
        "--skip-bf16-test", action="store_true",
        help="BF16 Fastmath 벤치마크 (Phase 8) 스킵",
    )
    parser.add_argument(
        "--skip-static-test", action="store_true",
        help="Static Shape 벤치마크 (Phase 9) 스킵",
    )
    parser.add_argument(
        "--skip-combined-test", action="store_true",
        help="복합 최적화 벤치마크 (Phase 10) 스킵",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="JSON 결과 저장 경로 (예: results.json)",
    )
    parser.add_argument(
        "--no-report", action="store_true",
        help="MD 보고서 생성 스킵",
    )
    parser.add_argument(
        "--report", type=str, default=None,
        help="MD 보고서 저장 경로 (기본: docs/04-report/features/)",
    )
    return parser.parse_args()


def main() -> None:
    """벤치마크 메인 실행."""
    args = parse_args()

    run_emb = not args.reranker_only
    run_rr = not args.embedding_only

    print(f"\n{SEPARATOR}")
    print("  ARM/크로스플랫폼 ONNX 최적화 벤치마크")
    print(f"  임베딩: {EMB_MODEL_NAME}")
    print(f"  리랭커: {RR_MODEL_NAME}")
    target = "임베딩 + 리랭커"
    if args.embedding_only:
        target = "임베딩만"
    elif args.reranker_only:
        target = "리랭커만"
    print(f"  대상: {target}")
    print(SEPARATOR)

    # Phase 0: 환경 정보 수집
    env_info = collect_environment_info()
    print_environment_info(env_info)

    # 필수 의존성 확인
    if env_info.get("ort_version") == "not installed":
        print("\n  [오류] onnxruntime이 설치되지 않았습니다.")
        print("  설치: uv add onnxruntime")
        sys.exit(1)

    if env_info.get("torch_version") == "not installed":
        print("\n  [오류] PyTorch가 설치되지 않았습니다.")
        print("  설치: uv pip install torch")
        sys.exit(1)

    # Phase 1: PyTorch FP32 baseline
    baseline = run_phase_1(
        env_info, run_embedding=run_emb, run_reranker=run_rr,
    )

    # Phase 2: ONNX 원본 FP32
    phase2 = run_phase_2(
        baseline, run_embedding=run_emb, run_reranker=run_rr,
    )

    # Phase 3: ORT-최적화 FP32 (Attention Fusion)
    phase3 = run_phase_3(
        baseline, run_embedding=run_emb, run_reranker=run_rr,
    )

    # Phase 4: ORT-최적화 FP16
    phase4: dict[str, dict[str, Any]] = {}
    if not args.skip_fp16:
        phase4 = run_phase_4(
            baseline, run_embedding=run_emb, run_reranker=run_rr,
        )
    else:
        print(f"\n{SEPARATOR}")
        print("  Phase 4: 스킵 (--skip-fp16)")
        print(SEPARATOR)

    # Phase 5: 스레드 최적화
    phase5: dict[str, dict[str, Any]] = {}
    if not args.skip_threads:
        phase5 = run_phase_5(
            env_info, baseline,
            run_embedding=run_emb, run_reranker=run_rr,
        )
    else:
        print(f"\n{SEPARATOR}")
        print("  Phase 5: 스킵 (--skip-threads)")
        print(SEPARATOR)

    # Phase 6: ORT 프로파일링
    phase6: dict[str, Any] = {}
    if not args.skip_profiling:
        phase6 = run_phase_6(
            run_embedding=run_emb, run_reranker=run_rr,
        )
    else:
        print(f"\n{SEPARATOR}")
        print("  Phase 6: 스킵 (--skip-profiling)")
        print(SEPARATOR)

    # Phase 8: BF16 Fastmath (Graviton3)
    phase8: dict[str, dict[str, Any]] = {}
    if not args.skip_bf16_test:
        phase8 = run_phase_8(
            baseline, env_info,
            run_embedding=run_emb, run_reranker=run_rr,
        )
    else:
        print(f"\n{SEPARATOR}")
        print("  Phase 8: 스킵 (--skip-bf16-test)")
        print(SEPARATOR)

    # Phase 9: Static Shape (임베딩 전용)
    phase9: dict[str, dict[str, Any]] = {}
    if not args.skip_static_test:
        phase9 = run_phase_9(
            baseline, run_embedding=run_emb,
        )
    else:
        print(f"\n{SEPARATOR}")
        print("  Phase 9: 스킵 (--skip-static-test)")
        print(SEPARATOR)

    # Phase 10: 복합 최적화 (BF16 + Static, Graviton3)
    phase10: dict[str, dict[str, Any]] = {}
    if not args.skip_combined_test:
        phase10 = run_phase_10(
            baseline, env_info, run_embedding=run_emb,
        )
    else:
        print(f"\n{SEPARATOR}")
        print("  Phase 10: 스킵 (--skip-combined-test)")
        print(SEPARATOR)

    # Phase 7: 종합 보고서
    report_path: Path | None = None
    if not args.no_report:
        report_path = Path(args.report) if args.report else DEFAULT_REPORT_PATH

    generate_report(
        env_info, baseline, phase2, phase3, phase4, phase5, phase6,
        phase8=phase8 or None, phase9=phase9 or None, phase10=phase10 or None,
        run_embedding=run_emb, run_reranker=run_rr,
        output_json=args.output,
        output_md=report_path,
    )

    print(f"\n{SEPARATOR}")
    print("  벤치마크 완료")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
