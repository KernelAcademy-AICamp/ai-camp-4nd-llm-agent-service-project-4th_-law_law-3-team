"""
리랭커 모델 양자화 벤치마크

PyTorch FP32 vs ONNX FP32 vs ONNX INT8 비교
- 속도 (쿼리-문서 쌍 15건 리랭킹 시간)
- 품질 (점수 상관계수, 순위 일치율)

사용법: cd backend && uv run python scripts/benchmark_reranker_quantize.py
"""

import os
import sys
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

RERANKER_MODEL = "dragonkue/bge-reranker-v2-m3-ko"
ONNX_DIR = PROJECT_ROOT / "data" / "models" / "reranker-onnx"
ONNX_INT8_DIR = PROJECT_ROOT / "data" / "models" / "reranker-onnx-int8"

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

SEPARATOR = "=" * 60


def benchmark_pytorch_fp32() -> tuple[list[float], float]:
    """PyTorch FP32 리랭킹."""
    print(f"\n{SEPARATOR}")
    print("  [1] PyTorch FP32 (baseline)")
    print(SEPARATOR)

    import torch
    from sentence_transformers import CrossEncoder

    t0 = time.monotonic()
    model = CrossEncoder(RERANKER_MODEL, activation_fn=torch.nn.Sigmoid())
    load_time = time.monotonic() - t0
    print(f"  모델 로드: {load_time * 1000:.0f}ms")

    # warm-up
    model.predict([("테스트 쿼리", "테스트 문서")])

    pairs = [(QUERY, doc) for doc in DOCUMENTS]

    # 5회 반복 측정
    all_times: list[float] = []
    scores: list[float] = []
    for trial in range(5):
        t0 = time.monotonic()
        result = model.predict(pairs)
        elapsed = time.monotonic() - t0
        all_times.append(elapsed)
        if trial == 0:
            scores = [float(s) for s in result]

    avg_time = np.mean(all_times)
    print(f"  15건 리랭킹: {avg_time * 1000:.1f}ms (5회 평균)")
    print(f"  건당 평균: {avg_time / len(pairs) * 1000:.1f}ms")

    # 상위 5개 결과
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    print(f"\n  Top-5 (score):")
    for rank, (idx, score) in enumerate(ranked[:5]):
        doc_preview = DOCUMENTS[idx][:50]
        print(f"    {rank + 1}. [{idx:>2d}] {score:.4f}  {doc_preview}...")

    return scores, avg_time


def export_onnx() -> bool:
    """리랭커 ONNX 변환."""
    if ONNX_DIR.exists() and (ONNX_DIR / "model.onnx").exists():
        print(f"  이미 존재: {ONNX_DIR}")
        return True

    print("  ONNX 변환 중...")
    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification

        ort_model = ORTModelForSequenceClassification.from_pretrained(
            RERANKER_MODEL, export=True, trust_remote_code=True,
        )
        ort_model.save_pretrained(str(ONNX_DIR))

        # 토크나이저도 저장
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
        tokenizer.save_pretrained(str(ONNX_DIR))
        print(f"  변환 완료: {ONNX_DIR}")
        return True
    except Exception as e:
        print(f"  변환 실패: {e}")
        return False


def quantize_int8() -> bool:
    """ONNX INT8 양자화."""
    if ONNX_INT8_DIR.exists() and (ONNX_INT8_DIR / "model_quantized.onnx").exists():
        print(f"  이미 존재: {ONNX_INT8_DIR}")
        return True

    print("  INT8 양자화 중...")
    try:
        from optimum.onnxruntime import ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig

        quantizer = ORTQuantizer.from_pretrained(str(ONNX_DIR))
        qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=True)
        ONNX_INT8_DIR.mkdir(parents=True, exist_ok=True)
        quantizer.quantize(save_dir=str(ONNX_INT8_DIR), quantization_config=qconfig)

        # 토크나이저 복사
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(ONNX_DIR))
        tokenizer.save_pretrained(str(ONNX_INT8_DIR))
        print(f"  양자화 완료: {ONNX_INT8_DIR}")
        return True
    except Exception as e:
        print(f"  양자화 실패: {e}")
        return False


def benchmark_onnx(
    model_dir: Path,
    label: str,
    file_name: str = "model.onnx",
) -> tuple[list[float], float]:
    """ONNX 리랭커 벤치마크."""
    print(f"\n{SEPARATOR}")
    print(f"  {label}")
    print(SEPARATOR)

    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer

    t0 = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        str(model_dir), file_name=file_name,
    )
    load_time = time.monotonic() - t0
    print(f"  모델 로드: {load_time * 1000:.0f}ms")

    # warm-up
    inputs = tokenizer(
        "테스트 쿼리", "테스트 문서",
        return_tensors="pt", padding=True, truncation=True,
    )
    _ = ort_model(**inputs)

    # sigmoid 함수
    def sigmoid(x: float) -> float:
        return 1 / (1 + np.exp(-x))

    # 5회 반복 측정
    all_times: list[float] = []
    scores: list[float] = []
    for trial in range(5):
        trial_scores: list[float] = []
        t0 = time.monotonic()
        for doc in DOCUMENTS:
            inputs = tokenizer(
                QUERY, doc,
                return_tensors="pt", padding=True, truncation=True,
                max_length=512,
            )
            outputs = ort_model(**inputs)
            logit = outputs.logits[0][0].item()
            trial_scores.append(sigmoid(logit))
        elapsed = time.monotonic() - t0
        all_times.append(elapsed)
        if trial == 0:
            scores = trial_scores

    avg_time = np.mean(all_times)
    print(f"  15건 리랭킹: {avg_time * 1000:.1f}ms (5회 평균)")
    print(f"  건당 평균: {avg_time / len(DOCUMENTS) * 1000:.1f}ms")

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    print(f"\n  Top-5 (score):")
    for rank, (idx, score) in enumerate(ranked[:5]):
        doc_preview = DOCUMENTS[idx][:50]
        print(f"    {rank + 1}. [{idx:>2d}] {score:.4f}  {doc_preview}...")

    return scores, avg_time


def benchmark_onnx_batched(
    model_dir: Path,
    label: str,
    file_name: str = "model.onnx",
) -> tuple[list[float], float]:
    """ONNX 리랭커 배치 벤치마크."""
    print(f"\n{SEPARATOR}")
    print(f"  {label} (배치)")
    print(SEPARATOR)

    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer

    t0 = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        str(model_dir), file_name=file_name,
    )
    load_time = time.monotonic() - t0
    print(f"  모델 로드: {load_time * 1000:.0f}ms")

    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    # warm-up
    inputs = tokenizer(
        ["테스트"] * 2, ["문서1", "문서2"],
        return_tensors="pt", padding=True, truncation=True,
    )
    _ = ort_model(**inputs)

    queries = [QUERY] * len(DOCUMENTS)

    all_times: list[float] = []
    scores: list[float] = []
    for trial in range(5):
        t0 = time.monotonic()
        inputs = tokenizer(
            queries, DOCUMENTS,
            return_tensors="pt", padding=True, truncation=True,
            max_length=512,
        )
        outputs = ort_model(**inputs)
        logits = outputs.logits[:, 0].detach().numpy()
        trial_scores = sigmoid(logits).tolist()
        elapsed = time.monotonic() - t0
        all_times.append(elapsed)
        if trial == 0:
            scores = trial_scores

    avg_time = np.mean(all_times)
    print(f"  15건 리랭킹 (배치): {avg_time * 1000:.1f}ms (5회 평균)")
    print(f"  건당 평균: {avg_time / len(DOCUMENTS) * 1000:.1f}ms")

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    print(f"\n  Top-5 (score):")
    for rank, (idx, score) in enumerate(ranked[:5]):
        doc_preview = DOCUMENTS[idx][:50]
        print(f"    {rank + 1}. [{idx:>2d}] {score:.4f}  {doc_preview}...")

    return scores, avg_time


def compare_quality(
    baseline_scores: list[float],
    test_scores: list[float],
    label: str,
) -> None:
    """품질 비교."""
    if not test_scores:
        return

    # 피어슨 상관계수
    corr = np.corrcoef(baseline_scores, test_scores)[0, 1]

    # 스피어만 순위 상관계수
    from scipy.stats import spearmanr

    spearman_corr, _ = spearmanr(baseline_scores, test_scores)

    # Top-k 순위 일치
    baseline_rank = np.argsort(baseline_scores)[::-1]
    test_rank = np.argsort(test_scores)[::-1]

    top3_match = len(set(baseline_rank[:3]) & set(test_rank[:3]))
    top5_match = len(set(baseline_rank[:5]) & set(test_rank[:5]))

    # 점수 차이
    diffs = np.array(baseline_scores) - np.array(test_scores)
    max_diff = np.max(np.abs(diffs))
    avg_diff = np.mean(np.abs(diffs))

    print(f"\n  품질 비교 ({label}):")
    print(f"    피어슨 상관계수:   {corr:.6f}")
    print(f"    스피어만 순위상관: {spearman_corr:.6f}")
    print(f"    Top-3 순위 일치:  {top3_match}/3")
    print(f"    Top-5 순위 일치:  {top5_match}/5")
    print(f"    점수 차이 (평균):  {avg_diff:.4f}")
    print(f"    점수 차이 (최대):  {max_diff:.4f}")


def main() -> None:
    """메인."""
    print(f"\n{SEPARATOR}")
    print("  리랭커 양자화 벤치마크")
    print(f"  모델: {RERANKER_MODEL} (568M params)")
    print(f"  쿼리: {QUERY}")
    print(f"  문서 수: {len(DOCUMENTS)}건")
    print(SEPARATOR)

    # 1. PyTorch FP32
    fp32_scores, fp32_time = benchmark_pytorch_fp32()

    # 2. ONNX 변환
    print(f"\n{SEPARATOR}")
    print("  ONNX 변환")
    print(SEPARATOR)
    onnx_ok = export_onnx()

    onnx_scores: list[float] = []
    onnx_time = 0.0
    onnx_batch_scores: list[float] = []
    onnx_batch_time = 0.0
    int8_scores: list[float] = []
    int8_time = 0.0
    int8_batch_scores: list[float] = []
    int8_batch_time = 0.0

    if onnx_ok:
        # 3. ONNX FP32 (개별)
        onnx_scores, onnx_time = benchmark_onnx(ONNX_DIR, "[2] ONNX FP32")
        compare_quality(fp32_scores, onnx_scores, "ONNX FP32 vs PyTorch")

        # 4. ONNX FP32 (배치)
        onnx_batch_scores, onnx_batch_time = benchmark_onnx_batched(
            ONNX_DIR, "[2b] ONNX FP32"
        )

        # 5. INT8 양자화
        print(f"\n{SEPARATOR}")
        print("  INT8 양자화")
        print(SEPARATOR)
        int8_ok = quantize_int8()

        if int8_ok:
            # 6. ONNX INT8 (개별)
            int8_scores, int8_time = benchmark_onnx(
                ONNX_INT8_DIR, "[3] ONNX INT8",
                file_name="model_quantized.onnx",
            )
            compare_quality(fp32_scores, int8_scores, "ONNX INT8 vs PyTorch")

            # 7. ONNX INT8 (배치)
            int8_batch_scores, int8_batch_time = benchmark_onnx_batched(
                ONNX_INT8_DIR, "[3b] ONNX INT8",
                file_name="model_quantized.onnx",
            )
            compare_quality(fp32_scores, int8_batch_scores, "ONNX INT8 배치 vs PyTorch")

    # 최종 비교
    print(f"\n{SEPARATOR}")
    print("  최종 비교 (15건 리랭킹)")
    print(SEPARATOR)

    results = [
        ("PyTorch FP32 (현재)", fp32_time, fp32_scores),
        ("ONNX FP32 (개별)", onnx_time, onnx_scores),
        ("ONNX FP32 (배치)", onnx_batch_time, onnx_batch_scores),
        ("ONNX INT8 (개별)", int8_time, int8_scores),
        ("ONNX INT8 (배치)", int8_batch_time, int8_batch_scores),
    ]

    print(f"\n  {'방식':<25s} {'시간':>10s} {'배수':>8s} {'순위상관':>10s}")
    print(f"  {'-' * 25} {'-' * 10} {'-' * 8} {'-' * 10}")

    for label, t, scores in results:
        if t == 0:
            continue
        ms = f"{t * 1000:.0f}ms"
        speedup = f"{fp32_time / t:.1f}x" if t > 0 else "-"
        if scores and fp32_scores:
            from scipy.stats import spearmanr
            sp, _ = spearmanr(fp32_scores, scores)
            sp_str = f"{sp:.4f}"
        else:
            sp_str = "-"
        print(f"  {label:<25s} {ms:>10s} {speedup:>8s} {sp_str:>10s}")

    # 파이프라인 영향
    if int8_batch_time > 0:
        pipeline_total = 24400  # ms
        rerank_baseline = fp32_time * 1000
        rerank_int8 = int8_batch_time * 1000
        new_total = pipeline_total - rerank_baseline + rerank_int8
        saved = pipeline_total - new_total
        print(f"\n  파이프라인 전체 영향:")
        print(f"    현재:  {pipeline_total:.0f}ms (리랭킹 {rerank_baseline:.0f}ms)")
        print(f"    INT8:  {new_total:.0f}ms (리랭킹 {rerank_int8:.0f}ms)")
        print(f"    절약:  {saved:.0f}ms ({saved / pipeline_total * 100:.1f}%)")


if __name__ == "__main__":
    main()
