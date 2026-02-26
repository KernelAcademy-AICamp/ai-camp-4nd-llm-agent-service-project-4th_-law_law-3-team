"""
ONNX 그래프 최적화 미개선 원인 진단 실험

실험 1: ORT_DISABLE_ALL로 이중 최적화 가설 검증
  - 런타임 최적화를 끈 상태에서 원본 vs O2 vs O3 속도 비교
  - 가설: 기본값(ORT_ENABLE_ALL)이 자동 적용되어 오프라인 최적화 효과가 상쇄됨

실험 2: ONNX FP32 품질 이상(cosine 0.76) 원인 규명
  - PyTorch vs ONNX FP32 vs ONNX O2의 임베딩 벡터 직접 비교
  - pooling 전략 차이 확인, 가중치 정합성 확인

사용법:
  cd backend && uv run python scripts/benchmark_onnx_diagnosis.py
  cd backend && uv run python scripts/benchmark_onnx_diagnosis.py --exp1-only
  cd backend && uv run python scripts/benchmark_onnx_diagnosis.py --exp2-only
"""

import argparse
import gc
import os
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

# --- 경로 설정 ---
MODEL_NAME = "nlpai-lab/KURE-v1"
CACHE_DIR = str(PROJECT_ROOT / "data" / "models")
ONNX_DIR = PROJECT_ROOT / "data" / "models" / "kure-v1-onnx"
ONNX_O2_DIR = PROJECT_ROOT / "data" / "models" / "kure-v1-onnx-o2"
ONNX_O3_DIR = PROJECT_ROOT / "data" / "models" / "kure-v1-onnx-o3"

SEPARATOR = "=" * 60
WARMUP_RUNS = 2
REPEAT_RUNS = 5

# --- 테스트 데이터 ---
TEST_QUERIES = [
    "교통사고 손해배상 판례",
    "임대차 보증금 반환 청구",
    "근로기준법 해고 부당해고",
    "이혼 재산분할 위자료",
    "명예훼손 형사 고소",
]

TEST_DOCUMENTS = [
    "피고는 원고에게 금 50,000,000원 및 이에 대한 지연손해금을 지급하라.",
    "임대차보증금 반환 청구 사건에서 임대인은 보증금 전액을 반환할 의무가 있다.",
    "근로기준법 제23조 제1항은 정당한 이유 없이 해고를 하지 못한다고 규정한다.",
    "민법 제750조에 의한 불법행위 손해배상 책임이 성립하려면 위법성이 있어야 한다.",
    "형법 제307조 제1항의 명예훼손죄가 성립하려면 사실을 적시하여야 한다.",
]


# ============================================================
# 공통 유틸리티
# ============================================================


def _encode_pytorch(texts: list[str]) -> np.ndarray:
    """PyTorch SentenceTransformer로 임베딩 생성."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        MODEL_NAME,
        cache_folder=CACHE_DIR,
        trust_remote_code=True,
    )
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    result = np.array(embeddings)
    del model
    gc.collect()
    return result


def _load_onnx_session_raw(
    model_path: str,
    disable_runtime_opt: bool = False,
) -> Any:
    """onnxruntime InferenceSession 직접 생성 (session_options 제어).

    Args:
        model_path: ONNX 모델 파일 경로
        disable_runtime_opt: True면 ORT_DISABLE_ALL 적용
    """
    import onnxruntime as ort

    session_options = ort.SessionOptions()
    if disable_runtime_opt:
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        )
    else:
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

    session = ort.InferenceSession(
        model_path,
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    return session


def _encode_with_raw_session(
    texts: list[str],
    model_dir: Path,
    file_name: str,
    disable_runtime_opt: bool = False,
) -> tuple[np.ndarray, float]:
    """onnxruntime 직접 세션으로 임베딩 생성 + 시간 측정.

    Returns:
        (embeddings, elapsed_ms)
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), trust_remote_code=True
    )
    model_path = str(model_dir / file_name)
    session = _load_onnx_session_raw(model_path, disable_runtime_opt)

    # 모델 입력 이름 확인
    input_names = [inp.name for inp in session.get_inputs()]

    # warm-up
    for _ in range(WARMUP_RUNS):
        inputs = tokenizer(
            texts[:1],
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=512,
        )
        feed: dict[str, Any] = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        if "token_type_ids" in input_names:
            feed["token_type_ids"] = inputs.get(
                "token_type_ids",
                np.zeros_like(inputs["input_ids"]),
            )
        session.run(None, feed)

    # 반복 측정
    times: list[float] = []
    all_embeddings: list[np.ndarray] = []

    for _ in range(REPEAT_RUNS):
        start = time.perf_counter()
        inputs = tokenizer(
            texts,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=512,
        )
        feed = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        if "token_type_ids" in input_names:
            feed["token_type_ids"] = inputs.get(
                "token_type_ids",
                np.zeros_like(inputs["input_ids"]),
            )

        outputs = session.run(None, feed)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

        # mean pooling + L2 norm
        token_embeddings = outputs[0]  # (batch, seq_len, hidden_dim)
        attention_mask = inputs["attention_mask"]
        mask_expanded = np.broadcast_to(
            np.expand_dims(attention_mask, -1), token_embeddings.shape
        )
        sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
        sum_mask = np.clip(
            np.sum(mask_expanded, axis=1), a_min=1e-9, a_max=None
        )
        emb = sum_embeddings / sum_mask
        norm = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.clip(norm, a_min=1e-9, a_max=None)
        all_embeddings.append(emb)

    median_time = float(np.median(times))
    final_embeddings = all_embeddings[-1]

    del session, tokenizer
    gc.collect()
    return final_embeddings, median_time


def _encode_onnx_optimum(
    texts: list[str],
    model_dir: Path,
    file_name: str,
) -> np.ndarray:
    """optimum ORTModelForFeatureExtraction으로 임베딩 생성 (기존 벤치마크 방식)."""
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), trust_remote_code=True
    )
    ort_model = ORTModelForFeatureExtraction.from_pretrained(
        str(model_dir),
        file_name=file_name,
        trust_remote_code=True,
        provider="CPUExecutionProvider",
    )

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    outputs = ort_model(**inputs)
    token_embeddings = outputs.last_hidden_state.detach().numpy()
    attention_mask = inputs["attention_mask"].numpy()
    mask_expanded = np.broadcast_to(
        np.expand_dims(attention_mask, -1), token_embeddings.shape
    )
    sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
    sum_mask = np.clip(
        np.sum(mask_expanded, axis=1), a_min=1e-9, a_max=None
    )
    emb = sum_embeddings / sum_mask
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.clip(norm, a_min=1e-9, a_max=None)

    del tokenizer, ort_model
    gc.collect()
    return emb


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """두 벡터(배열)의 평균 pairwise cosine similarity."""
    sims = []
    for i in range(len(a)):
        dot = np.dot(a[i], b[i])
        norm_a = np.linalg.norm(a[i])
        norm_b = np.linalg.norm(b[i])
        if norm_a > 0 and norm_b > 0:
            sims.append(dot / (norm_a * norm_b))
    return float(np.mean(sims)) if sims else 0.0


# ============================================================
# 실험 1: 이중 최적화 가설 검증
# ============================================================


def run_experiment_1() -> dict[str, Any]:
    """ORT_DISABLE_ALL vs ORT_ENABLE_ALL 속도 비교.

    가설: 런타임 최적화(ORT_ENABLE_ALL)가 기본 적용되므로,
    오프라인 O2/O3 최적화 모델과 원본 모델의 속도가 동일하다.
    ORT_DISABLE_ALL로 런타임 최적화를 끄면 차이가 나타나야 한다.
    """
    print(f"\n{SEPARATOR}")
    print("실험 1: 이중 최적화 가설 검증 (ORT_DISABLE_ALL vs ORT_ENABLE_ALL)")
    print(SEPARATOR)

    results: dict[str, Any] = {}

    # 테스트 대상 모델 설정
    models: list[tuple[str, Path, str]] = []
    if ONNX_DIR.exists():
        models.append(("원본 (FP32)", ONNX_DIR, "model.onnx"))
    if ONNX_O2_DIR.exists():
        models.append(("O2 최적화", ONNX_O2_DIR, "model_optimized.onnx"))
    if ONNX_O3_DIR.exists():
        models.append(("O3 최적화", ONNX_O3_DIR, "model_optimized.onnx"))

    if not models:
        print("  [건너뜀] ONNX 모델 디렉토리가 없습니다.")
        return results

    texts = TEST_QUERIES + TEST_DOCUMENTS

    # 1A: ORT_ENABLE_ALL (기본값) — 런타임 최적화 ON
    print("\n[1A] 런타임 최적화 ON (ORT_ENABLE_ALL) — 기본값")
    print("-" * 50)

    enable_all_times: dict[str, float] = {}
    for label, model_dir, file_name in models:
        _, elapsed = _encode_with_raw_session(
            texts, model_dir, file_name, disable_runtime_opt=False
        )
        enable_all_times[label] = elapsed
        print(f"  {label:20s}: {elapsed:8.1f} ms")

    # 1B: ORT_DISABLE_ALL — 런타임 최적화 OFF
    print("\n[1B] 런타임 최적화 OFF (ORT_DISABLE_ALL)")
    print("-" * 50)

    disable_all_times: dict[str, float] = {}
    for label, model_dir, file_name in models:
        _, elapsed = _encode_with_raw_session(
            texts, model_dir, file_name, disable_runtime_opt=True
        )
        disable_all_times[label] = elapsed
        print(f"  {label:20s}: {elapsed:8.1f} ms")

    # 분석
    print(f"\n{'─' * 60}")
    print("분석: 런타임 최적화 ON vs OFF 비교")
    print(f"{'─' * 60}")
    print(f"  {'모델':20s} | {'ON (ms)':>10s} | {'OFF (ms)':>10s} | {'차이':>10s}")
    print(f"  {'─' * 20} | {'─' * 10} | {'─' * 10} | {'─' * 10}")

    for label in enable_all_times:
        on_time = enable_all_times[label]
        off_time = disable_all_times.get(label, 0)
        diff = off_time - on_time
        diff_pct = (diff / on_time * 100) if on_time > 0 else 0
        print(
            f"  {label:20s} | {on_time:10.1f} | {off_time:10.1f} | "
            f"{diff:+8.1f} ({diff_pct:+.1f}%)"
        )

    # 가설 검증 결론
    if len(enable_all_times) >= 2:
        labels = list(enable_all_times.keys())
        on_spread = max(enable_all_times.values()) - min(enable_all_times.values())
        off_spread = max(disable_all_times.values()) - min(disable_all_times.values())

        print(f"\n  ON 모드 속도 편차: {on_spread:.1f} ms")
        print(f"  OFF 모드 속도 편차: {off_spread:.1f} ms")

        if on_spread < off_spread * 0.5:
            print("\n  ✅ 가설 확인: ON 모드에서는 모델 간 차이가 거의 없음")
            print("     → 런타임이 동일한 최적화를 적용하여 오프라인 최적화 효과 상쇄")
        else:
            print("\n  ❌ 가설 기각: ON 모드에서도 모델 간 유의미한 차이 존재")

    results = {
        "enable_all": enable_all_times,
        "disable_all": disable_all_times,
    }
    return results


# ============================================================
# 실험 2: ONNX FP32 품질 이상 원인 규명
# ============================================================


def run_experiment_2() -> dict[str, Any]:
    """ONNX FP32 cosine 0.76 원인 분석.

    비교 대상:
    1. PyTorch FP32 (baseline)
    2. ONNX FP32 (optimum fallback — 기존 벤치마크 방식)
    3. ONNX FP32 (onnxruntime 직접 세션 — mean pooling 동일 적용)
    4. ONNX O2 (optimum fallback)

    가설:
    A) 외부 데이터 파일 로딩 불일치 → onnxruntime 직접 세션에서도 동일 현상이면 확인
    B) pooling 전략 차이 → 동일 pooling 적용 후 차이 해소되면 확인
    """
    print(f"\n{SEPARATOR}")
    print("실험 2: ONNX FP32 품질 이상 (cosine 0.76) 원인 규명")
    print(SEPARATOR)

    results: dict[str, Any] = {}
    texts = TEST_QUERIES

    # 1. PyTorch 기준 임베딩
    print("\n[2A] PyTorch FP32 임베딩 생성 (baseline)...")
    pytorch_emb = _encode_pytorch(texts)
    print(f"  Shape: {pytorch_emb.shape}, Norm 평균: {np.linalg.norm(pytorch_emb, axis=1).mean():.4f}")

    # 2. ONNX FP32 — optimum fallback (기존 벤치마크 방식)
    if ONNX_DIR.exists():
        print("\n[2B] ONNX FP32 — optimum fallback (기존 벤치마크 방식)...")
        onnx_optimum_emb = _encode_onnx_optimum(texts, ONNX_DIR, "model.onnx")
        cos_optimum = _cosine_similarity(pytorch_emb, onnx_optimum_emb)
        print(f"  Shape: {onnx_optimum_emb.shape}")
        print(f"  vs PyTorch cosine: {cos_optimum:.4f}")

        # 3. ONNX FP32 — onnxruntime 직접 세션 (동일 mean pooling)
        print("\n[2C] ONNX FP32 — onnxruntime 직접 세션 (mean pooling 통일)...")
        onnx_raw_emb, _ = _encode_with_raw_session(
            texts, ONNX_DIR, "model.onnx", disable_runtime_opt=False
        )
        cos_raw = _cosine_similarity(pytorch_emb, onnx_raw_emb)
        print(f"  Shape: {onnx_raw_emb.shape}")
        print(f"  vs PyTorch cosine: {cos_raw:.4f}")

        # optimum vs raw 비교
        cos_optimum_vs_raw = _cosine_similarity(onnx_optimum_emb, onnx_raw_emb)
        print(f"\n  optimum vs raw 직접 비교: {cos_optimum_vs_raw:.4f}")

        results["onnx_fp32_optimum_cosine"] = cos_optimum
        results["onnx_fp32_raw_cosine"] = cos_raw
        results["optimum_vs_raw_cosine"] = cos_optimum_vs_raw
    else:
        print("  [건너뜀] ONNX FP32 모델 디렉토리가 없습니다.")

    # 4. ONNX O2 — optimum fallback (품질 정상 확인)
    if ONNX_O2_DIR.exists():
        print("\n[2D] ONNX O2 — optimum fallback...")
        onnx_o2_emb = _encode_onnx_optimum(texts, ONNX_O2_DIR, "model_optimized.onnx")
        cos_o2 = _cosine_similarity(pytorch_emb, onnx_o2_emb)
        print(f"  vs PyTorch cosine: {cos_o2:.4f}")
        results["onnx_o2_cosine"] = cos_o2

        # O2 raw 세션
        print("\n[2E] ONNX O2 — onnxruntime 직접 세션...")
        onnx_o2_raw_emb, _ = _encode_with_raw_session(
            texts, ONNX_O2_DIR, "model_optimized.onnx", disable_runtime_opt=False
        )
        cos_o2_raw = _cosine_similarity(pytorch_emb, onnx_o2_raw_emb)
        print(f"  vs PyTorch cosine: {cos_o2_raw:.4f}")
        results["onnx_o2_raw_cosine"] = cos_o2_raw
    else:
        print("  [건너뜀] ONNX O2 모델 디렉토리가 없습니다.")

    # 분석 요약
    print(f"\n{'─' * 60}")
    print("분석 요약: ONNX FP32 품질 이상 원인")
    print(f"{'─' * 60}")

    if "onnx_fp32_optimum_cosine" in results and "onnx_fp32_raw_cosine" in results:
        optimum_cos = results["onnx_fp32_optimum_cosine"]
        raw_cos = results["onnx_fp32_raw_cosine"]

        print(f"\n  ONNX FP32 (optimum fallback) vs PyTorch: {optimum_cos:.4f}")
        print(f"  ONNX FP32 (raw session)       vs PyTorch: {raw_cos:.4f}")

        if abs(optimum_cos - raw_cos) < 0.01:
            print("\n  → optimum과 raw 결과가 동일 → pooling 전략 차이는 원인 아님")
            print("  → 가중치 로딩 자체에 문제 (외부 데이터 파일 매핑 오류 가능)")
        elif raw_cos > optimum_cos + 0.1:
            print("\n  → raw 세션이 더 정확 → optimum fallback의 pooling 전략 차이가 원인")
        else:
            print("\n  → 두 방식 모두 품질 저하 → ONNX 변환 자체에서 가중치 손실 발생")

        if "onnx_o2_cosine" in results:
            o2_cos = results["onnx_o2_cosine"]
            print(f"\n  ONNX O2 vs PyTorch: {o2_cos:.4f}")
            if o2_cos > 0.99 and optimum_cos < 0.9:
                print("  → O2는 정상, FP32만 이상 → ONNX 변환 + 최적화가 가중치를 재정렬하여 해소")

    # 5. 추가 진단: 외부 데이터 파일 존재 확인
    print(f"\n{'─' * 60}")
    print("외부 데이터 파일 점검")
    print(f"{'─' * 60}")

    for label, model_dir in [("FP32", ONNX_DIR), ("O2", ONNX_O2_DIR), ("O3", ONNX_O3_DIR)]:
        if not model_dir.exists():
            continue
        onnx_files = list(model_dir.glob("*.onnx"))
        data_files = list(model_dir.glob("*.onnx_data"))
        onnx_sizes = [(f.name, f.stat().st_size / 1024 / 1024) for f in onnx_files]
        data_sizes = [(f.name, f.stat().st_size / 1024 / 1024) for f in data_files]

        print(f"\n  [{label}] {model_dir.name}/")
        for name, size in onnx_sizes:
            print(f"    {name}: {size:.1f} MB")
        for name, size in data_sizes:
            print(f"    {name}: {size:.1f} MB")
        if not data_files:
            print("    (외부 데이터 파일 없음 — 모델이 단일 파일)")

    return results


# ============================================================
# 메인
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ONNX 그래프 최적화 미개선 원인 진단"
    )
    parser.add_argument(
        "--exp1-only",
        action="store_true",
        help="실험 1(이중 최적화 가설)만 실행",
    )
    parser.add_argument(
        "--exp2-only",
        action="store_true",
        help="실험 2(FP32 품질 이상)만 실행",
    )
    args = parser.parse_args()

    print(SEPARATOR)
    print("ONNX 그래프 최적화 미개선 원인 진단")
    print(SEPARATOR)

    # 모델 디렉토리 확인
    print("\n모델 디렉토리 존재 확인:")
    for label, d in [
        ("ONNX FP32", ONNX_DIR),
        ("ONNX O2", ONNX_O2_DIR),
        ("ONNX O3", ONNX_O3_DIR),
    ]:
        status = "✅" if d.exists() else "❌"
        print(f"  {status} {label}: {d}")

    run_exp1 = not args.exp2_only
    run_exp2 = not args.exp1_only

    all_results: dict[str, Any] = {}

    if run_exp1:
        all_results["experiment_1"] = run_experiment_1()

    if run_exp2:
        all_results["experiment_2"] = run_experiment_2()

    # 최종 요약
    print(f"\n{SEPARATOR}")
    print("최종 결론")
    print(SEPARATOR)

    if "experiment_1" in all_results:
        exp1 = all_results["experiment_1"]
        if exp1.get("enable_all") and exp1.get("disable_all"):
            on_vals = list(exp1["enable_all"].values())
            off_vals = list(exp1["disable_all"].values())
            on_spread = max(on_vals) - min(on_vals)
            off_spread = max(off_vals) - min(off_vals)
            print(f"\n  [실험 1] 이중 최적화 가설")
            print(f"    ON 모드 모델 간 편차: {on_spread:.1f} ms")
            print(f"    OFF 모드 모델 간 편차: {off_spread:.1f} ms")
            if on_spread < off_spread * 0.5:
                print("    → ✅ 가설 확인: 런타임 최적화가 오프라인 최적화를 상쇄")
            else:
                print("    → ❌ 가설 기각 또는 추가 검증 필요")

    if "experiment_2" in all_results:
        exp2 = all_results["experiment_2"]
        if "onnx_fp32_optimum_cosine" in exp2:
            print(f"\n  [실험 2] ONNX FP32 품질 이상")
            print(f"    optimum fallback cosine: {exp2['onnx_fp32_optimum_cosine']:.4f}")
            print(f"    raw session cosine:      {exp2.get('onnx_fp32_raw_cosine', 'N/A')}")
            if exp2.get("onnx_o2_cosine"):
                print(f"    O2 cosine:               {exp2['onnx_o2_cosine']:.4f}")

    print(f"\n{SEPARATOR}")
    print("진단 완료")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
