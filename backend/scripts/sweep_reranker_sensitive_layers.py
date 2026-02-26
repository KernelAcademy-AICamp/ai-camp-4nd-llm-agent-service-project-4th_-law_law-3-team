"""리랭커 QDQ INT8 민감 레이어 확장 실험.

목표: Pearson >= 0.999 를 달성하는 최소 FP32 레이어 조합을 찾는다.

1단계: 24개 레이어 개별 sensitivity sweep (어떤 레이어가 가장 민감한지)
2단계: 민감 순으로 FP32 레이어를 4→6→8→10→...개 늘리며 Pearson + latency 측정
3단계: Pearson >= 0.999 달성하는 최소 조합 보고

Usage:
    uv run --no-sync python scripts/sweep_reranker_sensitive_layers.py
"""

from __future__ import annotations

import gc
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 경로 설정
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

DATA_MODELS_DIR = BACKEND_DIR / "data" / "models"
RR_ONNX_EXPORT_DIR = DATA_MODELS_DIR / "reranker-onnx-export-tmp"
RR_ORT_OPT_DIR = DATA_MODELS_DIR / "reranker-ort-opt"
RERANKER_MODEL = "dragonkue/bge-reranker-v2-m3-ko"

NUM_LAYERS = 24
PEARSON_TARGET = 0.999
ITERATIONS = 10

# 법률 도메인 테스트 데이터 (query-document 쌍)
QUERY = "교통사고 손해배상 판례"
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


def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    """Sigmoid 활성화 함수."""
    return 1 / (1 + np.exp(-x))


def _predict_pytorch(query: str, documents: list[str]) -> np.ndarray:
    """PyTorch FP32 참조 리랭킹 점수 (sigmoid 적용)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    cache_dir = str(DATA_MODELS_DIR)

    tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL, cache_dir=cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        RERANKER_MODEL, cache_dir=cache_dir,
    )
    model.eval()

    inputs = tokenizer(
        [query] * len(documents),
        documents,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    # CrossEncoder: logits shape = (N, 1) 또는 (N,)
    if logits.ndim == 2:
        raw_scores = logits[:, 0].numpy()
    else:
        raw_scores = logits.flatten().numpy()

    # sigmoid 적용하여 0~1 범위 점수로 변환
    scores: np.ndarray = _sigmoid(raw_scores)

    del model, tokenizer
    gc.collect()
    return scores


def _predict_onnx(
    query: str,
    documents: list[str],
    model_dir: Path,
    model_file: str = "model_optimized.onnx",
) -> np.ndarray:
    """ONNX 모델로 리랭킹 점수 생성 (sigmoid 적용)."""
    import onnxruntime as ort
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    inputs = tokenizer(
        [query] * len(documents),
        documents,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="np",
    )

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

    ort_inputs: dict[str, np.ndarray] = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
    }
    input_names = {inp.name for inp in sess.get_inputs()}
    if "token_type_ids" in input_names:
        ort_inputs["token_type_ids"] = inputs.get(
            "token_type_ids", np.zeros_like(inputs["input_ids"]),
        ).astype(np.int64)

    outputs = sess.run(None, ort_inputs)

    # CrossEncoder: outputs[0] shape = (N, 1) 또는 (N,)
    logits = outputs[0]
    if logits.ndim == 2:
        raw_scores = logits[:, 0]
    else:
        raw_scores = logits.flatten()

    scores: np.ndarray = _sigmoid(raw_scores)

    del sess
    return scores


def _pearson_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """피어슨 상관계수 계산."""
    return float(np.corrcoef(a, b)[0, 1])


def _spearman_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """스피어만 순위 상관계수 계산 (참고용)."""
    from scipy.stats import spearmanr

    corr, _ = spearmanr(a, b)
    return float(corr)


def _measure_latency(
    model_dir: Path,
    model_file: str = "model_optimized.onnx",
) -> float:
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

    inputs = tokenizer(
        [QUERY] * len(DOCUMENTS),
        DOCUMENTS,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="np",
    )
    ort_inputs: dict[str, np.ndarray] = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
    }
    input_names = {inp.name for inp in sess.get_inputs()}
    if "token_type_ids" in input_names:
        ort_inputs["token_type_ids"] = inputs.get(
            "token_type_ids", np.zeros_like(inputs["input_ids"]),
        ).astype(np.int64)

    # Warmup
    sess.run(None, ort_inputs)

    latencies: list[float] = []
    for _ in range(ITERATIONS):
        t0 = time.perf_counter()
        sess.run(None, ort_inputs)
        latencies.append((time.perf_counter() - t0) * 1000)

    del sess
    return float(np.median(latencies))


def _get_layer_node_names(model_path: Path, layer_indices: list[int]) -> list[str]:
    """특정 레이어의 ONNX 노드 이름을 추출.

    Raw ONNX export 노드 패턴 (슬래시 구분):
    - /0/auto_model/encoder/layer.{idx}/attention/...
    ORT-optimized 노드 패턴 (점 구분):
    - roberta.encoder.layer.{idx}.attention...
    """
    import onnx

    model = onnx.load(str(model_path), load_external_data=False)

    layer_patterns: list[str] = []
    for idx in layer_indices:
        # raw ONNX export: 슬래시 구분자
        layer_patterns.append(f"layer.{idx}/")
        # ORT-optimized: 점 구분자
        layer_patterns.append(f"layer.{idx}.")
        # 언더스코어 구분자 (일부 변환기)
        layer_patterns.append(f"layer_{idx}_")
        layer_patterns.append(f"layer_{idx}/")

    excluded: list[str] = []
    for node in model.graph.node:
        node_name = node.name or ""
        for pattern in layer_patterns:
            if pattern in node_name:
                excluded.append(node_name)
                break

    del model
    return excluded


def _copy_tokenizer(src: Path, dst: Path) -> None:
    """토크나이저 파일 복사."""
    import shutil

    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "sentencepiece.bpe.model",
        "config.json",
        "vocab.txt",
    ]:
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


def _resolve_tokenizer_dir() -> Path:
    """토크나이저 소스 디렉토리를 결정한다."""
    if RR_ONNX_EXPORT_DIR.exists():
        return RR_ONNX_EXPORT_DIR
    alt = DATA_MODELS_DIR / "reranker-onnx"
    if alt.exists():
        return alt
    # 최후 수단: ORT-opt 디렉토리
    return RR_ORT_OPT_DIR


def step1_sensitivity_sweep(
    fp32_path: Path,
    pt_ref: np.ndarray,
) -> list[tuple[int, float]]:
    """1단계: 각 레이어를 개별 FP32로 유지하며 Pearson 측정."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    tokenizer_dir = _resolve_tokenizer_dir()

    print("\n" + "=" * 60)
    print("1단계: 레이어별 Sensitivity Sweep")
    print("=" * 60)
    print("  방법: 레이어 i만 FP32 유지, 나머지 23개 INT8")
    print("  → Pearson이 높을수록 해당 레이어가 민감 (FP32로 유지해야 함)\n")

    layer_pearsons: list[tuple[int, float]] = []

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

            _copy_tokenizer(tokenizer_dir, Path(tmp_dir))

            onnx_scores = _predict_onnx(QUERY, DOCUMENTS, Path(tmp_dir))
            pearson = _pearson_correlation(pt_ref, onnx_scores)

        delta = 1.0 - pearson
        layer_pearsons.append((layer_idx, pearson))
        print(f"  Layer {layer_idx:2d}: pearson={pearson:.6f}  delta={delta:.6f}")

    # Pearson 내림차순 정렬 (가장 민감한 레이어 = FP32로 유지 시 Pearson 가장 높음)
    layer_pearsons.sort(key=lambda x: x[1], reverse=True)

    print("\n  민감도 순위 (FP32 유지 시 Pearson 향상이 큰 순):")
    print(f"  {'순위':>4s}  {'Layer':>5s}  {'Pearson':>10s}  {'Delta':>10s}")
    print(f"  {'-' * 35}")
    for rank, (idx, pear) in enumerate(layer_pearsons, 1):
        print(f"  {rank:4d}  {idx:5d}  {pear:10.6f}  {1.0 - pear:10.6f}")

    return layer_pearsons


def step2_progressive_expansion(
    fp32_path: Path,
    pt_ref: np.ndarray,
    ranked_layers: list[tuple[int, float]],
) -> list[dict]:
    """2단계: 민감 순으로 FP32 레이어를 점진적으로 늘려가며 Pearson + latency 측정."""
    tokenizer_dir = _resolve_tokenizer_dir()

    print("\n" + "=" * 60)
    print("2단계: FP32 레이어 점진 확장")
    print("=" * 60)
    print(f"  목표: Pearson >= {PEARSON_TARGET}")
    print("  방법: 민감 순으로 FP32 레이어를 4 → 6 → 8 → ... 개 추가\n")

    # 민감도 순 레이어 인덱스
    ranked_indices = [idx for idx, _ in ranked_layers]

    # 테스트할 FP32 레이어 수: 4, 6, 8, 10, 12, 14, 16, 18, 20
    test_counts = list(range(4, 21, 2))

    results: list[dict] = []
    target_achieved = False

    for n_fp32 in test_counts:
        sensitive = sorted(ranked_indices[:n_fp32])

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            success = _build_qdq_model(fp32_path, sensitive, tmp_path, tokenizer_dir)

            if not success:
                print(f"  FP32={n_fp32:2d}개 {sensitive}: 빌드 실패")
                continue

            onnx_scores = _predict_onnx(QUERY, DOCUMENTS, tmp_path)
            pearson = _pearson_correlation(pt_ref, onnx_scores)
            spearman = _spearman_correlation(pt_ref, onnx_scores)
            latency = _measure_latency(tmp_path)

        meets = pearson >= PEARSON_TARGET
        marker = " ★ TARGET" if meets else ""
        print(
            f"  FP32={n_fp32:2d}개: pearson={pearson:.6f}  "
            f"spearman={spearman:.6f}  latency={latency:.1f}ms  "
            f"layers={sensitive}{marker}"
        )

        result: dict = {
            "n_fp32_layers": n_fp32,
            "sensitive_layers": sensitive,
            "pearson": pearson,
            "spearman": spearman,
            "latency_ms": latency,
            "meets_target": meets,
        }
        results.append(result)

        if meets and not target_achieved:
            target_achieved = True
            print(f"\n  ✓ Pearson >= {PEARSON_TARGET} 달성! (FP32 {n_fp32}개)")
            # 목표 달성 후 2개 더 테스트하고 종료 (추세 확인)
            remaining = [c for c in test_counts if c > n_fp32][:2]
            for n_extra in remaining:
                sensitive_extra = sorted(ranked_indices[:n_extra])
                with tempfile.TemporaryDirectory() as tmp_dir2:
                    tmp_path2 = Path(tmp_dir2)
                    success2 = _build_qdq_model(
                        fp32_path, sensitive_extra, tmp_path2, tokenizer_dir,
                    )
                    if success2:
                        onnx_scores2 = _predict_onnx(QUERY, DOCUMENTS, tmp_path2)
                        pearson2 = _pearson_correlation(pt_ref, onnx_scores2)
                        spearman2 = _spearman_correlation(pt_ref, onnx_scores2)
                        latency2 = _measure_latency(tmp_path2)
                        print(
                            f"  FP32={n_extra:2d}개: pearson={pearson2:.6f}  "
                            f"spearman={spearman2:.6f}  latency={latency2:.1f}ms  "
                            f"layers={sensitive_extra}"
                        )
                        results.append({
                            "n_fp32_layers": n_extra,
                            "sensitive_layers": sensitive_extra,
                            "pearson": pearson2,
                            "spearman": spearman2,
                            "latency_ms": latency2,
                            "meets_target": pearson2 >= PEARSON_TARGET,
                        })
            break

    return results


def step3_also_test_ort_opt(pt_ref: np.ndarray) -> dict:
    """ORT-opt FP32 (무손실) 의 Pearson + latency 측정."""
    print("\n" + "=" * 60)
    print("참고: ORT-opt FP32 (무손실) 기준선")
    print("=" * 60)

    if not RR_ORT_OPT_DIR.exists():
        print(f"  [스킵] ORT-opt 디렉토리 없음: {RR_ORT_OPT_DIR}")
        return {"label": "ORT-opt FP32", "pearson": 0.0, "spearman": 0.0, "latency_ms": 0.0}

    onnx_scores = _predict_onnx(QUERY, DOCUMENTS, RR_ORT_OPT_DIR)
    pearson = _pearson_correlation(pt_ref, onnx_scores)
    spearman = _spearman_correlation(pt_ref, onnx_scores)
    latency = _measure_latency(RR_ORT_OPT_DIR)

    print(f"  ORT-opt FP32: pearson={pearson:.6f}  spearman={spearman:.6f}  latency={latency:.1f}ms")
    return {
        "label": "ORT-opt FP32",
        "pearson": pearson,
        "spearman": spearman,
        "latency_ms": latency,
    }


def main() -> None:
    """리랭커 QDQ INT8 민감 레이어 확장 실험 메인."""
    print("=" * 60)
    print("리랭커 QDQ INT8 민감 레이어 확장 실험")
    print(f"목표: Pearson >= {PEARSON_TARGET}")
    print(f"모델: {RERANKER_MODEL}")
    print("=" * 60)

    # raw ONNX 모델 경로 확인
    fp32_path = RR_ONNX_EXPORT_DIR / "model.onnx"
    if not fp32_path.exists():
        # fallback: reranker-onnx 디렉토리
        alt = DATA_MODELS_DIR / "reranker-onnx" / "model.onnx"
        if alt.exists():
            fp32_path = alt
        else:
            print(f"ERROR: raw ONNX 모델 없음: {fp32_path}")
            print(f"  대체 경로도 없음: {alt}")
            print("  build_optimized_onnx.py의 리랭커 ONNX export를 먼저 실행하세요.")
            sys.exit(1)

    print(f"\n소스 모델: {fp32_path}")
    print(f"테스트 데이터: 쿼리 1건 x 문서 {len(DOCUMENTS)}건")
    print(f"반복 횟수: {ITERATIONS}회 (latency)")

    # PyTorch 참조 계산 (1회만)
    print("\nPyTorch FP32 참조 리랭킹 점수 계산 중...")
    pt_ref = _predict_pytorch(QUERY, DOCUMENTS)
    print(f"  shape: {pt_ref.shape}")
    print(f"  점수 범위: [{pt_ref.min():.4f}, {pt_ref.max():.4f}]")

    # 상위 5개 문서 출력
    ranked = np.argsort(pt_ref)[::-1]
    print("\n  PyTorch 참조 Top-5:")
    for rank, idx in enumerate(ranked[:5]):
        doc_preview = DOCUMENTS[idx][:50]
        print(f"    {rank + 1}. [{idx:>2d}] {pt_ref[idx]:.4f}  {doc_preview}...")

    # 1단계: Sensitivity sweep
    ranked_layers = step1_sensitivity_sweep(fp32_path, pt_ref)

    # 2단계: 점진적 확장
    expansion_results = step2_progressive_expansion(fp32_path, pt_ref, ranked_layers)

    # 3단계: ORT-opt FP32 기준선
    ort_opt_result = step3_also_test_ort_opt(pt_ref)

    # 결과 저장
    output: dict = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pearson_target": PEARSON_TARGET,
        "model": RERANKER_MODEL,
        "test_data": {
            "query": QUERY,
            "num_documents": len(DOCUMENTS),
        },
        "sensitivity_ranking": [
            {"layer": idx, "pearson_when_fp32": pear}
            for idx, pear in ranked_layers
        ],
        "expansion_results": expansion_results,
        "ort_opt_fp32": ort_opt_result,
    }

    output_path = BACKEND_DIR / "sweep_reranker_sensitive_results.json"
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
        print(f"\n  ★ Pearson >= {PEARSON_TARGET} 최소 조합:")
        print(f"    FP32 레이어: {best['n_fp32_layers']}개 / {NUM_LAYERS}개")
        print(f"    레이어 목록: {best['sensitive_layers']}")
        print(f"    Pearson:  {best['pearson']:.6f}")
        print(f"    Spearman: {best['spearman']:.6f}")
        print(f"    Latency:  {best['latency_ms']:.1f}ms")
        print("\n  비교:")
        if ort_opt_result["latency_ms"] > 0:
            print(
                f"    ORT-opt FP32 (무손실):     "
                f"{ort_opt_result['latency_ms']:.1f}ms, "
                f"pearson {ort_opt_result['pearson']:.6f}"
            )
        print(
            f"    최적 QDQ ({best['n_fp32_layers']}개 FP32):     "
            f"{best['latency_ms']:.1f}ms, "
            f"pearson {best['pearson']:.6f}"
        )
    else:
        print(f"\n  Pearson >= {PEARSON_TARGET}를 달성하지 못했습니다.")
        print("  FP32 레이어를 더 늘리거나, ORT-opt FP32 (무손실)를 사용하세요.")


if __name__ == "__main__":
    main()
