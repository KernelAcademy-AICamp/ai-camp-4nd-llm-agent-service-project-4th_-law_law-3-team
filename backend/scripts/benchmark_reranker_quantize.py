"""리랭커 모델 양자화 + 그래프 최적화 종합 벤치마크

비교 대상:
  PyTorch FP32 vs ONNX FP32 vs ONNX INT8
  vs ONNX O2 vs ONNX O3 vs ONNX O3+INT8 vs ONNX FP16

- Phase 0: 환경 준비 (ONNX 변환, INT8 양자화, O2/O3 그래프 최적화, FP16 변환)
- Phase 1: 속도 벤치마크 (개별 + 배치)
- Phase 2: 품질 비교 (Pearson, Spearman, Top-k, 점수 차이)
- Phase 3: 모델 크기 비교
- Phase 4: 종합 요약 + 파이프라인 영향
- Phase 5: MD 보고서 생성

사용법:
  cd backend && uv run python scripts/benchmark_reranker_quantize.py
  cd backend && uv run python scripts/benchmark_reranker_quantize.py --report
  cd backend && uv run python scripts/benchmark_reranker_quantize.py --skip-export --report
  cd backend && uv run python scripts/benchmark_reranker_quantize.py --skip-graph-opt
  cd backend && uv run python scripts/benchmark_reranker_quantize.py --skip-fp16
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

# ============================================================
# 상수
# ============================================================

RERANKER_MODEL = "dragonkue/bge-reranker-v2-m3-ko"
ONNX_DIR = PROJECT_ROOT / "data" / "models" / "reranker-onnx"
ONNX_INT8_DIR = PROJECT_ROOT / "data" / "models" / "reranker-onnx-int8"
ONNX_O2_DIR = PROJECT_ROOT / "data" / "models" / "reranker-onnx-o2"
ONNX_O3_DIR = PROJECT_ROOT / "data" / "models" / "reranker-onnx-o3"
ONNX_O3_INT8_DIR = PROJECT_ROOT / "data" / "models" / "reranker-onnx-o3-int8"
ONNX_FP16_DIR = PROJECT_ROOT / "data" / "models" / "reranker-onnx-fp16"

DEFAULT_REPORT_PATH = (
    PROJECT_ROOT.parent / "docs" / "04-report" / "features"
    / "reranker-onnx-benchmark.md"
)

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
PIPELINE_TOTAL_MS = 24400  # 현재 전체 파이프라인 시간 (ms)

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


# ============================================================
# 유틸리티
# ============================================================


def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    """Sigmoid 활성화 함수."""
    return 1 / (1 + np.exp(-x))


def _copy_tokenizer(src_dir: Path, dst_dir: Path) -> None:
    """토크나이저 + config.json 복사 헬퍼."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(src_dir))
    tokenizer.save_pretrained(str(dst_dir))

    config_src = src_dir / "config.json"
    if config_src.exists():
        shutil.copy2(str(config_src), str(dst_dir / "config.json"))


def _dir_size_mb(path: Path) -> float:
    """디렉토리 크기 (MB)."""
    if not path.exists():
        return 0.0
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def _pass_fail(
    value: float, threshold: float, *, higher_is_better: bool = True
) -> str:
    """PASS/FAIL 판정."""
    if higher_is_better:
        return "PASS" if value >= threshold else "FAIL"
    return "PASS" if value <= threshold else "FAIL"


def _get_model_configs() -> list[tuple[str, Path, str]]:
    """벤치마크 대상 ONNX 모델 (key, dir, file_name) 목록."""
    configs: list[tuple[str, Path, str]] = []
    if ONNX_DIR.exists():
        configs.append(("onnx_fp32", ONNX_DIR, "model.onnx"))
    if ONNX_INT8_DIR.exists():
        configs.append(("onnx_int8", ONNX_INT8_DIR, "model_quantized.onnx"))
    if ONNX_O2_DIR.exists():
        configs.append(("onnx_o2", ONNX_O2_DIR, "model_optimized.onnx"))
    if ONNX_O3_DIR.exists():
        configs.append(("onnx_o3", ONNX_O3_DIR, "model_optimized.onnx"))
    if ONNX_O3_INT8_DIR.exists():
        configs.append(("onnx_o3_int8", ONNX_O3_INT8_DIR, "model_quantized.onnx"))
    if ONNX_FP16_DIR.exists():
        configs.append(("onnx_fp16", ONNX_FP16_DIR, "model.onnx"))
    return configs


# ============================================================
# Phase 0: 모델 변환
# ============================================================


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
    print("\n  Top-5 (score):")
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
        _copy_tokenizer(ONNX_DIR, ONNX_INT8_DIR)
        print(f"  양자화 완료: {ONNX_INT8_DIR}")
        return True
    except Exception as e:
        print(f"  양자화 실패: {e}")
        return False


def optimize_graph(level: str = "O3") -> bool:
    """ONNX 그래프 최적화 (O2 또는 O3).

    Note: ORTOptimizer가 원본 외부 데이터 파일(model.onnx_data)을
    삭제하는 문제가 있어 임시 복사본으로 최적화 후 원본을 복원한다.
    최적화 후 출력 크기 < 1MB면 가중치 미포함으로 판단하여 FAIL 처리.
    """
    from optimum.onnxruntime import ORTOptimizer
    from optimum.onnxruntime.configuration import AutoOptimizationConfig

    target_dir = ONNX_O2_DIR if level == "O2" else ONNX_O3_DIR

    if target_dir.exists() and (target_dir / "model_optimized.onnx").exists():
        print(f"  {level} 최적화 모델 이미 존재: {target_dir}")
        return True

    print(f"  {level} 그래프 최적화 중...")

    onnx_data = ONNX_DIR / "model.onnx_data"
    tmp_copy = ONNX_DIR / "model.onnx_data.backup"
    has_external = onnx_data.exists()

    try:
        if has_external:
            shutil.copy2(str(onnx_data), str(tmp_copy))

        optimizer = ORTOptimizer.from_pretrained(str(ONNX_DIR))
        config = (
            AutoOptimizationConfig.O2()
            if level == "O2"
            else AutoOptimizationConfig.O3()
        )

        target_dir.mkdir(parents=True, exist_ok=True)
        optimizer.optimize(
            save_dir=str(target_dir), optimization_config=config
        )

        # 최적화 후 원본 외부 데이터 복원
        if has_external and not onnx_data.exists():
            shutil.move(str(tmp_copy), str(onnx_data))
            print("  원본 model.onnx_data 복원 완료")
        elif tmp_copy.exists():
            tmp_copy.unlink()

        # 출력 크기 검증 (가중치 미포함 방지)
        opt_file = target_dir / "model_optimized.onnx"
        opt_data = target_dir / "model_optimized.onnx.data"
        if opt_file.exists() and opt_file.stat().st_size < 1_000_000:
            if opt_data.exists() and opt_data.stat().st_size > 1_000_000:
                # 외부 데이터 파일에 가중치 저장됨 (정상)
                print(f"  {level} 최적화: 그래프={opt_file.stat().st_size // 1024}KB, "
                      f"가중치={opt_data.stat().st_size // (1024 * 1024)}MB (외부)")
            else:
                size_bytes = opt_file.stat().st_size
                print(f"  [경고] {level} 최적화 모델 크기 너무 작음 ({size_bytes} bytes)")
                print("  가중치가 포함되지 않았을 수 있습니다.")
                return False

        _copy_tokenizer(ONNX_DIR, target_dir)

        print(f"  {level} 그래프 최적화 완료: {target_dir}")
        return True
    except Exception as e:
        # 실패 시에도 백업 복원
        if has_external and tmp_copy.exists() and not onnx_data.exists():
            shutil.move(str(tmp_copy), str(onnx_data))
        print(f"  {level} 그래프 최적화 실패: {e}")
        return False


def quantize_optimized_int8() -> bool:
    """O3 최적화 모델에 INT8 양자화 적용."""
    if ONNX_O3_INT8_DIR.exists() and (
        ONNX_O3_INT8_DIR / "model_quantized.onnx"
    ).exists():
        print(f"  O3+INT8 모델 이미 존재: {ONNX_O3_INT8_DIR}")
        return True

    if not ONNX_O3_DIR.exists():
        print("  O3 최적화 모델이 없습니다. optimize_graph('O3')를 먼저 실행하세요.")
        return False

    print("  O3+INT8 양자화 중...")
    try:
        import onnx
        from onnxruntime.quantization import QuantType, quantize_dynamic

        src_model = str(ONNX_O3_DIR / "model_optimized.onnx")
        ONNX_O3_INT8_DIR.mkdir(parents=True, exist_ok=True)
        dst_model = str(ONNX_O3_INT8_DIR / "model_quantized.onnx")

        quantize_dynamic(
            model_input=src_model,
            model_output=dst_model,
            per_channel=True,
            weight_type=QuantType.QInt8,
            extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
        )

        _copy_tokenizer(ONNX_O3_DIR, ONNX_O3_INT8_DIR)

        print(f"  O3+INT8 양자화 완료: {ONNX_O3_INT8_DIR}")
        return True
    except Exception as e:
        print(f"  O3+INT8 양자화 실패: {e}")
        return False


def convert_fp16() -> bool:
    """ONNX FP32 → FP16 변환."""
    fp16_model_file = ONNX_FP16_DIR / "model.onnx"
    if fp16_model_file.exists():
        print(f"  FP16 모델 이미 존재: {ONNX_FP16_DIR}")
        return True

    if not (ONNX_DIR / "model.onnx").exists():
        print("  ONNX FP32 모델이 없습니다. export_onnx()를 먼저 실행하세요.")
        return False

    print("  FP16 변환 중...")
    try:
        import onnx
        from onnxconverter_common import float16

        src_path = str(ONNX_DIR / "model.onnx")
        # 외부 데이터 파일이 있는 대형 모델을 위해 load_external_data=True 사용
        model = onnx.load(src_path, load_external_data=True)

        # opset 정보 보존 (float16 변환 시 유실 방지)
        original_opset = list(model.opset_import)

        model_fp16 = float16.convert_float_to_float16(
            model, keep_io_types=True,
        )

        # opset 복원
        if not model_fp16.opset_import:
            for opset in original_opset:
                new_opset = model_fp16.opset_import.add()
                new_opset.CopyFrom(opset)

        ONNX_FP16_DIR.mkdir(parents=True, exist_ok=True)
        onnx.save_model(
            model_fp16,
            str(fp16_model_file),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="model.onnx.data",
            size_threshold=1024,
        )

        # 저장 검증
        if fp16_model_file.stat().st_size < 100:
            print("  [경고] FP16 모델 저장 실패 (파일 크기 비정상)")
            return False

        _copy_tokenizer(ONNX_DIR, ONNX_FP16_DIR)

        print(f"  FP16 변환 완료: {ONNX_FP16_DIR}")
        return True
    except Exception as e:
        print(f"  FP16 변환 실패: {e}")
        return False


# ============================================================
# Phase 1: 속도 벤치마크
# ============================================================


def benchmark_onnx(
    model_dir: Path,
    label: str,
    file_name: str = "model.onnx",
) -> tuple[list[float], float]:
    """ONNX 리랭커 벤치마크 (개별 추론)."""
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
            trial_scores.append(float(_sigmoid(logit)))
        elapsed = time.monotonic() - t0
        all_times.append(elapsed)
        if trial == 0:
            scores = trial_scores

    avg_time = np.mean(all_times)
    print(f"  15건 리랭킹: {avg_time * 1000:.1f}ms (5회 평균)")
    print(f"  건당 평균: {avg_time / len(DOCUMENTS) * 1000:.1f}ms")

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    print("\n  Top-5 (score):")
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
        trial_scores = _sigmoid(logits).tolist()  # type: ignore[union-attr]
        elapsed = time.monotonic() - t0
        all_times.append(elapsed)
        if trial == 0:
            scores = trial_scores

    avg_time = np.mean(all_times)
    print(f"  15건 리랭킹 (배치): {avg_time * 1000:.1f}ms (5회 평균)")
    print(f"  건당 평균: {avg_time / len(DOCUMENTS) * 1000:.1f}ms")

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    print("\n  Top-5 (score):")
    for rank, (idx, score) in enumerate(ranked[:5]):
        doc_preview = DOCUMENTS[idx][:50]
        print(f"    {rank + 1}. [{idx:>2d}] {score:.4f}  {doc_preview}...")

    return scores, avg_time


# ============================================================
# Phase 2: 품질 비교
# ============================================================


def compare_quality(
    baseline_scores: list[float],
    test_scores: list[float],
    label: str,
) -> dict[str, float]:
    """품질 비교. 결과를 출력하고 dict로도 반환."""
    if not test_scores:
        return {}

    from scipy.stats import spearmanr

    # 피어슨 상관계수
    corr = float(np.corrcoef(baseline_scores, test_scores)[0, 1])

    # 스피어만 순위 상관계수
    spearman_corr, _ = spearmanr(baseline_scores, test_scores)
    spearman_corr = float(spearman_corr)

    # Top-k 순위 일치
    baseline_rank = np.argsort(baseline_scores)[::-1]
    test_rank = np.argsort(test_scores)[::-1]

    top3_match = len(set(baseline_rank[:3].tolist()) & set(test_rank[:3].tolist()))
    top5_match = len(set(baseline_rank[:5].tolist()) & set(test_rank[:5].tolist()))

    # 점수 차이
    diffs = np.array(baseline_scores) - np.array(test_scores)
    max_diff = float(np.max(np.abs(diffs)))
    avg_diff = float(np.mean(np.abs(diffs)))

    print(f"\n  품질 비교 ({label}):")
    print(f"    피어슨 상관계수:   {corr:.6f}")
    print(f"    스피어만 순위상관: {spearman_corr:.6f}")
    print(f"    Top-3 순위 일치:  {top3_match}/3")
    print(f"    Top-5 순위 일치:  {top5_match}/5")
    print(f"    점수 차이 (평균):  {avg_diff:.4f}")
    print(f"    점수 차이 (최대):  {max_diff:.4f}")

    return {
        "pearson": corr,
        "spearman": spearman_corr,
        "top3_match": float(top3_match),
        "top5_match": float(top5_match),
        "avg_diff": avg_diff,
        "max_diff": max_diff,
    }


# ============================================================
# Phase 3: 모델 크기 비교
# ============================================================


def run_size_benchmark() -> dict[str, float]:
    """모델 디렉토리 크기 비교."""
    print(f"\n{SEPARATOR}")
    print("  Phase 3: 모델 크기 비교")
    print(SEPARATOR)

    results: dict[str, float] = {}

    # ONNX FP32 크기를 기준으로 사용
    fp32_size = _dir_size_mb(ONNX_DIR)
    results["onnx_fp32"] = fp32_size
    print(f"    ONNX FP32:     {fp32_size:>8.0f}MB")

    model_dir_map: dict[str, Path] = {
        "onnx_int8": ONNX_INT8_DIR,
        "onnx_o2": ONNX_O2_DIR,
        "onnx_o3": ONNX_O3_DIR,
        "onnx_o3_int8": ONNX_O3_INT8_DIR,
        "onnx_fp16": ONNX_FP16_DIR,
    }
    for model_key, model_dir in model_dir_map.items():
        size_mb = _dir_size_mb(model_dir)
        if size_mb > 0:
            results[model_key] = size_mb
            saving = (1 - size_mb / fp32_size) * 100 if fp32_size > 0 else 0
            label = MODEL_LABELS.get(model_key, model_key.upper())
            print(f"    {label:<14s}  {size_mb:>8.0f}MB  ({saving:+.0f}%)")

    return results


# ============================================================
# Phase 4: 종합 요약
# ============================================================


def print_summary(
    speed_results: dict[str, dict[str, float]],
    quality_results: dict[str, dict[str, float]],
    size_results: dict[str, float],
    fp32_time: float,
) -> None:
    """종합 요약 테이블 + 파이프라인 영향."""
    print(f"\n{SEPARATOR}")
    print("  Phase 4: 종합 요약")
    print(SEPARATOR)

    # 종합 비교 테이블
    print(f"\n  {'Variant':<20s} {'개별':>8s} {'배수':>6s} "
          f"{'배치':>8s} {'배수':>6s} {'품질':>6s} {'크기':>8s}")
    print(f"  {'-' * 20} {'-' * 8} {'-' * 6} {'-' * 8} {'-' * 6} {'-' * 6} {'-' * 8}")

    pt_row = speed_results.get("pytorch_fp32", {})
    print(f"  {'PyTorch FP32':<20s} {pt_row.get('time_ms', 0):>6.0f}ms {'1.0x':>6s} "
          f"{'  -':>8s} {'  -':>6s} {'base':>6s} {'  -':>8s}")

    for key in VARIANT_ORDER:
        if key == "pytorch_fp32" or key not in speed_results:
            continue
        r = speed_results[key]
        label = MODEL_LABELS.get(key, key)
        ind_ms = f"{r.get('time_ms', 0):.0f}ms"
        ind_x = f"{r.get('speedup', 0):.1f}x"
        batch_ms = f"{r.get('batch_time_ms', 0):.0f}ms"
        batch_x = f"{r.get('batch_speedup', 0):.1f}x"

        # 품질 판정
        q = quality_results.get(key, {})
        if q:
            p_ok = q.get("pearson", 0) >= QUALITY_THRESHOLDS["pearson"]
            s_ok = q.get("spearman", 0) >= QUALITY_THRESHOLDS["spearman"]
            d_ok = q.get("max_diff", 1) < QUALITY_THRESHOLDS["max_diff"]
            verdict = "PASS" if (p_ok and s_ok and d_ok) else "FAIL"
        else:
            verdict = "-"

        size_str = f"{size_results.get(key, 0):.0f}MB" if key in size_results else "-"
        print(f"  {label:<20s} {ind_ms:>8s} {ind_x:>6s} "
              f"{batch_ms:>8s} {batch_x:>6s} {verdict:>6s} {size_str:>8s}")

    # 파이프라인 영향 분석 (배치 기준, 가장 빠른 PASS variant)
    best_key = ""
    best_batch_ms = fp32_time * 1000
    for key in VARIANT_ORDER:
        if key == "pytorch_fp32" or key not in speed_results:
            continue
        q = quality_results.get(key, {})
        p_ok = q.get("pearson", 0) >= QUALITY_THRESHOLDS["pearson"]
        s_ok = q.get("spearman", 0) >= QUALITY_THRESHOLDS["spearman"]
        d_ok = q.get("max_diff", 1) < QUALITY_THRESHOLDS["max_diff"]
        if p_ok and s_ok and d_ok:
            batch_ms = speed_results[key].get("batch_time_ms", best_batch_ms)
            if batch_ms < best_batch_ms:
                best_batch_ms = batch_ms
                best_key = key

    rerank_baseline_ms = fp32_time * 1000
    print("\n  파이프라인 영향:")
    print(f"    전체 파이프라인: {PIPELINE_TOTAL_MS:.0f}ms")
    print(f"    현재 리랭킹:    {rerank_baseline_ms:.0f}ms (PyTorch FP32)")

    if best_key:
        best_label = MODEL_LABELS.get(best_key, best_key)
        new_total = PIPELINE_TOTAL_MS - rerank_baseline_ms + best_batch_ms
        saved = PIPELINE_TOTAL_MS - new_total
        print(f"    최적 리랭킹:    {best_batch_ms:.0f}ms ({best_label} 배치)")
        print(f"    새 파이프라인:  {new_total:.0f}ms (절약 {saved:.0f}ms, {saved / PIPELINE_TOTAL_MS * 100:.1f}%)")


# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    """CLI 인자 파싱."""
    parser = argparse.ArgumentParser(
        description="리랭커 양자화 + 그래프 최적화 종합 벤치마크"
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="ONNX 변환/양자화 스킵 (이미 변환된 경우)",
    )
    parser.add_argument(
        "--skip-graph-opt",
        action="store_true",
        help="그래프 최적화 (O2/O3) 모델 스킵",
    )
    parser.add_argument(
        "--skip-fp16",
        action="store_true",
        help="FP16 변환 스킵",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="MD 보고서 생성",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default=None,
        help="MD 보고서 저장 경로",
    )
    return parser.parse_args()


def main() -> None:
    """벤치마크 메인."""
    args = parse_args()

    print(f"\n{SEPARATOR}")
    print("  리랭커 양자화 + 그래프 최적화 종합 벤치마크")
    print(f"  모델: {RERANKER_MODEL} (568M params)")
    print(f"  쿼리: {QUERY}")
    print(f"  문서 수: {len(DOCUMENTS)}건")
    print(SEPARATOR)

    # ── Phase 0: 모델 변환 ──────────────────────────────
    if not args.skip_export:
        print(f"\n{SEPARATOR}")
        print("  Phase 0: 모델 변환")
        print(SEPARATOR)

        onnx_ok = export_onnx()
        if onnx_ok:
            quantize_int8()
            if not args.skip_graph_opt:
                optimize_graph("O2")
                o3_ok = optimize_graph("O3")
                if o3_ok:
                    quantize_optimized_int8()
            else:
                print("  그래프 최적화: 스킵 (--skip-graph-opt)")
            if not args.skip_fp16:
                convert_fp16()
            else:
                print("  FP16 변환: 스킵 (--skip-fp16)")
    else:
        print("\n  Phase 0: 스킵 (--skip-export)")

    # ── Phase 1: 속도 벤치마크 ──────────────────────────
    print(f"\n{SEPARATOR}")
    print("  Phase 1: 속도 벤치마크")
    print(SEPARATOR)

    fp32_scores, fp32_time = benchmark_pytorch_fp32()

    speed_results: dict[str, dict[str, float]] = {
        "pytorch_fp32": {
            "time_ms": fp32_time * 1000,
            "speedup": 1.0,
            "per_doc_ms": fp32_time / len(DOCUMENTS) * 1000,
        },
    }

    all_scores: dict[str, list[float]] = {"pytorch_fp32": fp32_scores}

    for key, model_dir, file_name in _get_model_configs():
        label = MODEL_LABELS.get(key, key)

        try:
            # 개별 추론
            scores, avg_time = benchmark_onnx(model_dir, f"[{label}]", file_name)
            speedup = fp32_time / avg_time if avg_time > 0 else 0

            # 배치 추론
            batch_scores, batch_time = benchmark_onnx_batched(
                model_dir, f"[{label}]", file_name,
            )
            batch_speedup = fp32_time / batch_time if batch_time > 0 else 0

            speed_results[key] = {
                "time_ms": avg_time * 1000,
                "speedup": speedup,
                "per_doc_ms": avg_time / len(DOCUMENTS) * 1000,
                "batch_time_ms": batch_time * 1000,
                "batch_speedup": batch_speedup,
                "batch_per_doc_ms": batch_time / len(DOCUMENTS) * 1000,
            }
            all_scores[key] = scores
        except Exception as e:
            print(f"\n  [스킵] {label} 벤치마크 실패: {e}")

    # ── Phase 2: 품질 비교 ──────────────────────────────
    print(f"\n{SEPARATOR}")
    print("  Phase 2: 품질 비교")
    print(SEPARATOR)

    quality_results: dict[str, dict[str, float]] = {}
    for key in VARIANT_ORDER:
        if key == "pytorch_fp32" or key not in all_scores:
            continue
        label = MODEL_LABELS.get(key, key)
        quality = compare_quality(fp32_scores, all_scores[key], label)
        if quality:
            quality_results[key] = quality

    # ── Phase 3: 모델 크기 ──────────────────────────────
    size_results = run_size_benchmark()

    # ── Phase 4: 종합 요약 ──────────────────────────────
    print_summary(speed_results, quality_results, size_results, fp32_time)

    # ── Phase 5: 보고서 생성 ────────────────────────────
    if args.report:
        from scripts.benchmark_reranker_report import generate_report

        report_path = Path(args.report_path) if args.report_path else DEFAULT_REPORT_PATH
        generate_report(
            speed_results=speed_results,
            quality_results=quality_results,
            size_results=size_results,
            output_path=report_path,
        )
        print(f"\n  MD 보고서 생성: {report_path}")


if __name__ == "__main__":
    main()
