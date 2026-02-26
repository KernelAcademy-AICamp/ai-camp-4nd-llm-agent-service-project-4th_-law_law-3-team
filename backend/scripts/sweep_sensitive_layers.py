"""QDQ INT8 민감 레이어 확장 실험.

목표: cosine >= 0.999 를 달성하는 최소 FP32 레이어 조합을 찾는다.

1단계: 24개 레이어 개별 sensitivity sweep (어떤 레이어가 가장 민감한지)
2단계: 민감 순으로 FP32 레이어를 4→6→8→10→...개 늘리며 cosine + latency 측정
3단계: cosine >= 0.999 달성하는 최소 조합 보고

Usage:
    uv run --no-sync python scripts/sweep_sensitive_layers.py
"""

from __future__ import annotations

import gc
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# 경로 설정
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

DATA_MODELS_DIR = BACKEND_DIR / "data" / "models"
EMB_ONNX_EXPORT_DIR = DATA_MODELS_DIR / "kure-v1-onnx"
EMB_ORT_OPT_DIR = DATA_MODELS_DIR / "kure-v1-ort-opt"

NUM_LAYERS = 24
COSINE_TARGET = 0.999

# 법률 도메인 테스트 쿼리 (10건)
TEST_TEXTS = [
    "교통사고 손해배상 판례",
    "임대차 보증금 반환 청구",
    "근로기준법 해고 부당해고",
    "이혼 재산분할 위자료",
    "명예훼손 형사 고소",
    "상속 유류분 반환",
    "부동산 매매계약 해제",
    "의료사고 과실 책임",
    "개인정보 보호법 위반",
    "특허권 침해 손해배상",
]

ITERATIONS = 10


def _encode_pytorch(texts: list[str]) -> np.ndarray:
    """PyTorch FP32 참조 임베딩."""
    from transformers import AutoModel, AutoTokenizer
    import torch

    model_name = "nlpai-lab/KURE-v1"
    cache_dir = str(DATA_MODELS_DIR)

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
    model.eval()

    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # Mean pooling
    attention_mask = inputs["attention_mask"].unsqueeze(-1).float()
    embeddings = (outputs.last_hidden_state * attention_mask).sum(1) / attention_mask.sum(1)

    # L2 normalize
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    result = embeddings.numpy()

    del model, tokenizer
    gc.collect()
    return result


def _encode_onnx(texts: list[str], model_dir: Path, model_file: str = "model_optimized.onnx") -> np.ndarray:
    """ONNX 모델로 임베딩 생성."""
    import onnxruntime as ort
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="np")

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4
    sess_opts.inter_op_num_threads = 1
    sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess = ort.InferenceSession(
        str(model_dir / model_file),
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"],
    )

    ort_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
    }
    if "token_type_ids" in {inp.name for inp in sess.get_inputs()}:
        ort_inputs["token_type_ids"] = inputs.get(
            "token_type_ids", np.zeros_like(inputs["input_ids"])
        ).astype(np.int64)

    outputs = sess.run(None, ort_inputs)
    last_hidden = outputs[0]

    # Mean pooling
    mask = inputs["attention_mask"].astype(np.float32)[..., np.newaxis]
    embeddings = (last_hidden * mask).sum(axis=1) / mask.sum(axis=1)

    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-12)

    del sess
    return embeddings


def _cosine_pairwise(a: np.ndarray, b: np.ndarray) -> float:
    """두 임베딩 행렬의 pairwise cosine similarity 평균."""
    sims = []
    for i in range(a.shape[0]):
        dot = np.dot(a[i], b[i])
        norm_a = np.linalg.norm(a[i])
        norm_b = np.linalg.norm(b[i])
        if norm_a > 0 and norm_b > 0:
            sims.append(float(dot / (norm_a * norm_b)))
    return float(np.mean(sims)) if sims else 0.0


def _measure_latency(model_dir: Path, model_file: str = "model_optimized.onnx") -> float:
    """Median latency (ms) 측정. ITERATIONS회 반복."""
    import onnxruntime as ort
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4
    sess_opts.inter_op_num_threads = 1
    sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess = ort.InferenceSession(
        str(model_dir / model_file),
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"],
    )

    inputs = tokenizer(TEST_TEXTS, padding=True, truncation=True, max_length=512, return_tensors="np")
    ort_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
    }
    if "token_type_ids" in {inp.name for inp in sess.get_inputs()}:
        ort_inputs["token_type_ids"] = inputs.get(
            "token_type_ids", np.zeros_like(inputs["input_ids"])
        ).astype(np.int64)

    # Warmup
    sess.run(None, ort_inputs)

    latencies = []
    for _ in range(ITERATIONS):
        t0 = time.perf_counter()
        sess.run(None, ort_inputs)
        latencies.append((time.perf_counter() - t0) * 1000)

    del sess
    return float(np.median(latencies))


def _get_layer_node_names(model_path: Path, layer_indices: list[int]) -> list[str]:
    """특정 레이어의 ONNX 노드 이름을 추출."""
    import onnx

    model = onnx.load(str(model_path), load_external_data=False)
    excluded = []
    for node in model.graph.node:
        for idx in layer_indices:
            pattern = f"layer.{idx}."
            alt_pattern = f"layer.{idx}/"
            if pattern in node.name or alt_pattern in node.name:
                excluded.append(node.name)
                break
            for out in node.output:
                if pattern in out or alt_pattern in out:
                    excluded.append(node.name)
                    break
    del model
    return excluded


def _copy_tokenizer(src: Path, dst: Path) -> None:
    """토크나이저 파일 복사."""
    import shutil
    for name in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                  "sentencepiece.bpe.model", "config.json", "vocab.txt"]:
        src_file = src / name
        if src_file.exists():
            shutil.copy2(src_file, dst / name)


def _build_qdq_model(
    fp32_path: Path,
    sensitive_layers: list[int],
    output_dir: Path,
    tokenizer_dir: Path,
) -> bool:
    """주어진 민감 레이어로 QDQ INT8 모델을 빌드."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model_optimized.onnx"

    excluded_nodes = _get_layer_node_names(fp32_path, sensitive_layers)

    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
        nodes_to_exclude=excluded_nodes,
        per_channel=True,
    )

    _copy_tokenizer(tokenizer_dir, output_dir)
    return output_path.exists()


def step1_sensitivity_sweep(fp32_path: Path, pt_ref: np.ndarray) -> list[tuple[int, float]]:
    """1단계: 각 레이어를 개별 FP32로 유지하며 cosine 측정."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    print("\n" + "=" * 60)
    print("1단계: 레이어별 Sensitivity Sweep")
    print("=" * 60)
    print(f"  방법: 레이어 i만 FP32 유지, 나머지 23개 INT8")
    print(f"  → cosine이 높을수록 해당 레이어가 민감 (FP32로 유지해야 함)\n")

    layer_cosines: list[tuple[int, float]] = []

    for layer_idx in range(NUM_LAYERS):
        excluded_nodes = _get_layer_node_names(fp32_path, [layer_idx])

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "model_optimized.onnx"

            quantize_dynamic(
                model_input=str(fp32_path),
                model_output=str(tmp_path),
                weight_type=QuantType.QInt8,
                nodes_to_exclude=excluded_nodes,
                per_channel=True,
            )

            _copy_tokenizer(EMB_ONNX_EXPORT_DIR, Path(tmp_dir))

            onnx_out = _encode_onnx(TEST_TEXTS, Path(tmp_dir))
            cosine = _cosine_pairwise(pt_ref, onnx_out)

        delta = 1.0 - cosine
        layer_cosines.append((layer_idx, cosine))
        print(f"  Layer {layer_idx:2d}: cosine={cosine:.6f}  delta={delta:.6f}")

    # cosine 내림차순 정렬 (가장 민감한 레이어 = FP32로 유지 시 cosine 가장 높음)
    layer_cosines.sort(key=lambda x: x[1], reverse=True)

    print(f"\n  민감도 순위 (FP32 유지 시 cosine 향상이 큰 순):")
    print(f"  {'순위':>4s}  {'Layer':>5s}  {'Cosine':>10s}  {'Delta':>10s}")
    print(f"  {'-' * 35}")
    for rank, (idx, cos) in enumerate(layer_cosines, 1):
        print(f"  {rank:4d}  {idx:5d}  {cos:10.6f}  {1.0 - cos:10.6f}")

    return layer_cosines


def step2_progressive_expansion(
    fp32_path: Path,
    pt_ref: np.ndarray,
    ranked_layers: list[tuple[int, float]],
) -> list[dict]:
    """2단계: 민감 순으로 FP32 레이어를 점진적으로 늘려가며 cosine + latency 측정."""
    print("\n" + "=" * 60)
    print("2단계: FP32 레이어 점진 확장")
    print("=" * 60)
    print(f"  목표: cosine >= {COSINE_TARGET}")
    print(f"  방법: 민감 순으로 FP32 레이어를 4 → 6 → 8 → ... 개 추가\n")

    # 민감도 순 레이어 인덱스
    ranked_indices = [idx for idx, _ in ranked_layers]

    # 테스트할 FP32 레이어 수: 4, 6, 8, 10, 12, 14, 16, 18, 20
    test_counts = list(range(4, 21, 2))

    results = []
    target_achieved = False

    for n_fp32 in test_counts:
        sensitive = sorted(ranked_indices[:n_fp32])

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            success = _build_qdq_model(fp32_path, sensitive, tmp_path, EMB_ONNX_EXPORT_DIR)

            if not success:
                print(f"  FP32={n_fp32:2d}개 {sensitive}: 빌드 실패")
                continue

            onnx_out = _encode_onnx(TEST_TEXTS, tmp_path)
            cosine = _cosine_pairwise(pt_ref, onnx_out)
            latency = _measure_latency(tmp_path)

        marker = " ★ TARGET" if cosine >= COSINE_TARGET else ""
        print(f"  FP32={n_fp32:2d}개: cosine={cosine:.6f}  latency={latency:.1f}ms  layers={sensitive}{marker}")

        result = {
            "n_fp32_layers": n_fp32,
            "sensitive_layers": sensitive,
            "cosine": cosine,
            "latency_ms": latency,
            "meets_target": cosine >= COSINE_TARGET,
        }
        results.append(result)

        if cosine >= COSINE_TARGET and not target_achieved:
            target_achieved = True
            print(f"\n  ✓ cosine >= {COSINE_TARGET} 달성! (FP32 {n_fp32}개)")
            # 목표 달성 후 2개 더 테스트하고 종료 (추세 확인)
            remaining = [c for c in test_counts if c > n_fp32][:2]
            test_counts_after = remaining
            for n_extra in test_counts_after:
                sensitive_extra = sorted(ranked_indices[:n_extra])
                with tempfile.TemporaryDirectory() as tmp_dir2:
                    tmp_path2 = Path(tmp_dir2)
                    success2 = _build_qdq_model(fp32_path, sensitive_extra, tmp_path2, EMB_ONNX_EXPORT_DIR)
                    if success2:
                        onnx_out2 = _encode_onnx(TEST_TEXTS, tmp_path2)
                        cosine2 = _cosine_pairwise(pt_ref, onnx_out2)
                        latency2 = _measure_latency(tmp_path2)
                        print(f"  FP32={n_extra:2d}개: cosine={cosine2:.6f}  latency={latency2:.1f}ms  layers={sensitive_extra}")
                        results.append({
                            "n_fp32_layers": n_extra,
                            "sensitive_layers": sensitive_extra,
                            "cosine": cosine2,
                            "latency_ms": latency2,
                            "meets_target": cosine2 >= COSINE_TARGET,
                        })
            break

    return results


def step3_also_test_ort_opt(pt_ref: np.ndarray) -> dict:
    """ORT-opt FP32 (무손실) 의 cosine + latency 측정."""
    print("\n" + "=" * 60)
    print("참고: ORT-opt FP32 (무손실) 기준선")
    print("=" * 60)

    onnx_out = _encode_onnx(TEST_TEXTS, EMB_ORT_OPT_DIR)
    cosine = _cosine_pairwise(pt_ref, onnx_out)
    latency = _measure_latency(EMB_ORT_OPT_DIR)

    print(f"  ORT-opt FP32: cosine={cosine:.6f}  latency={latency:.1f}ms")
    return {"label": "ORT-opt FP32", "cosine": cosine, "latency_ms": latency}


def main() -> None:
    print("=" * 60)
    print("QDQ INT8 민감 레이어 확장 실험")
    print(f"목표: cosine >= {COSINE_TARGET}")
    print("=" * 60)

    # raw ONNX 모델 경로 확인
    fp32_path = EMB_ONNX_EXPORT_DIR / "model.onnx"
    if not fp32_path.exists():
        print(f"ERROR: raw ONNX 모델 없음: {fp32_path}")
        print("  build_optimized_onnx.py의 Step 1(ONNX export)을 먼저 실행하세요.")
        sys.exit(1)

    print(f"\n소스 모델: {fp32_path}")
    print(f"테스트 쿼리: {len(TEST_TEXTS)}건")
    print(f"반복 횟수: {ITERATIONS}회 (latency)")

    # PyTorch 참조 계산 (1회만)
    print("\nPyTorch FP32 참조 임베딩 계산 중...")
    pt_ref = _encode_pytorch(TEST_TEXTS)
    print(f"  shape: {pt_ref.shape}")

    # 1단계: Sensitivity sweep
    ranked_layers = step1_sensitivity_sweep(fp32_path, pt_ref)

    # 2단계: 점진적 확장
    expansion_results = step2_progressive_expansion(fp32_path, pt_ref, ranked_layers)

    # 3단계: ORT-opt FP32 기준선
    ort_opt_result = step3_also_test_ort_opt(pt_ref)

    # 결과 저장
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cosine_target": COSINE_TARGET,
        "sensitivity_ranking": [
            {"layer": idx, "cosine_when_fp32": cos}
            for idx, cos in ranked_layers
        ],
        "expansion_results": expansion_results,
        "ort_opt_fp32": ort_opt_result,
    }

    output_path = BACKEND_DIR / "sweep_sensitive_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {output_path}")

    # 최종 요약
    print("\n" + "=" * 60)
    print("최종 요약")
    print("=" * 60)

    target_results = [r for r in expansion_results if r["meets_target"]]
    if target_results:
        best = min(target_results, key=lambda r: r["n_fp32_layers"])
        print(f"\n  ★ cosine >= {COSINE_TARGET} 최소 조합:")
        print(f"    FP32 레이어: {best['n_fp32_layers']}개 / {NUM_LAYERS}개")
        print(f"    레이어 목록: {best['sensitive_layers']}")
        print(f"    Cosine: {best['cosine']:.6f}")
        print(f"    Latency: {best['latency_ms']:.1f}ms")
        print(f"\n  비교:")
        print(f"    현재 QDQ INT8 (4개 FP32): 212ms, cosine 0.989")
        print(f"    ORT-opt FP32 (무손실):     {ort_opt_result['latency_ms']:.1f}ms, cosine {ort_opt_result['cosine']:.6f}")
        print(f"    최적 QDQ ({best['n_fp32_layers']}개 FP32):     {best['latency_ms']:.1f}ms, cosine {best['cosine']:.6f}")
    else:
        print(f"\n  cosine >= {COSINE_TARGET}를 달성하지 못했습니다.")
        print(f"  FP32 레이어를 더 늘리거나, ORT-opt FP32 (무손실)를 사용하세요.")


if __name__ == "__main__":
    main()
