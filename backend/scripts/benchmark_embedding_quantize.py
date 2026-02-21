"""
임베딩 모델 양자화 + 그래프 최적화 종합 벤치마크

비교 대상:
  PyTorch FP32 vs ONNX FP32 vs ONNX INT8 vs ONNX O2 vs ONNX O3 vs ONNX O3+INT8

- Phase 0: 환경 준비 (ONNX 변환, INT8 양자화, O2/O3 그래프 최적화)
- Phase 1: 속도 (단일 쿼리 + 배치)
- Phase 2: 임베딩 품질 (Pairwise Cosine, Sim Matrix 상관, Separation)
- Phase 3: LanceDB 검색 품질 (Top-K ID 일치, 순위 상관)
- Phase 4: 모델 크기
- Phase 5: 최종 리포트

사용법:
  cd backend && uv run python scripts/benchmark_embedding_quantize.py
  cd backend && uv run python scripts/benchmark_embedding_quantize.py --skip-search
  cd backend && uv run python scripts/benchmark_embedding_quantize.py --skip-quality
  cd backend && uv run python scripts/benchmark_embedding_quantize.py --skip-export
  cd backend && uv run python scripts/benchmark_embedding_quantize.py --skip-graph-opt
"""

import argparse
import gc
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

MODEL_NAME = "nlpai-lab/KURE-v1"
CACHE_DIR = str(PROJECT_ROOT / "data" / "models")
ONNX_DIR = PROJECT_ROOT / "data" / "models" / "kure-v1-onnx"
ONNX_INT8_DIR = PROJECT_ROOT / "data" / "models" / "kure-v1-onnx-int8"
ONNX_O2_DIR = PROJECT_ROOT / "data" / "models" / "kure-v1-onnx-o2"
ONNX_O3_DIR = PROJECT_ROOT / "data" / "models" / "kure-v1-onnx-o3"
ONNX_O3_INT8_DIR = PROJECT_ROOT / "data" / "models" / "kure-v1-onnx-o3-int8"
LANCEDB_DIR = PROJECT_ROOT / "lancedb_data"
LANCEDB_TABLE = "legal_chunks"

SEPARATOR = "=" * 60
WARMUP_RUNS = 2
REPEAT_RUNS = 5

# --- 테스트 데이터 ---

BENCHMARK_QUERIES = [
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

BENCHMARK_DOCUMENTS = [
    "피고는 원고에게 금 50,000,000원 및 이에 대한 지연손해금을 지급하라. 교통사고로 인한 손해배상 청구 사건에서 피해자의 과실 비율을 30%로 인정하고 치료비 및 위자료를 산정함.",
    "자동차손해배상 보장법 제3조에 의하면 자기를 위하여 자동차를 운행하는 자는 그 운행으로 다른 사람을 사망하게 하거나 부상하게 한 경우에는 그 손해를 배상할 책임을 진다.",
    "불법행위로 인한 손해배상 청구권의 소멸시효는 피해자나 그 법정대리인이 그 손해 및 가해자를 안 날로부터 3년, 불법행위를 한 날로부터 10년이다.",
    "민법 제750조에 의한 불법행위 손해배상 책임이 성립하려면 가해행위의 위법성, 가해자의 고의 또는 과실, 손해의 발생, 가해행위와 손해 사이의 인과관계가 있어야 한다.",
    "임대차보증금 반환 청구 사건에서 임대인은 임차인에게 보증금 전액을 반환할 의무가 있으나, 연체 차임 등을 공제할 수 있다.",
    "근로기준법 제23조 제1항은 사용자는 근로자에게 정당한 이유 없이 해고, 휴직, 정직, 전직, 감봉 그 밖의 징벌을 하지 못한다고 규정하고 있다.",
    "형법 제307조 제1항의 명예훼손죄가 성립하려면 사실을 적시하여 사람의 명예를 훼손하여야 하고, 적시된 사실이 허위인 경우에는 제2항에 의하여 가중처벌된다.",
    "이혼 시 재산분할은 혼인 중 당사자 쌍방의 협력으로 이룩한 재산을 대상으로 하며, 분할의 비율은 기여도에 따라 정하되 부양적 요소도 고려한다.",
    "후유장해 등급 판정에 있어서는 맥브라이드 장해평가 방법에 의하고, 노동능력상실률의 평가는 사실인정의 문제이다.",
    "상속의 포기는 상속개시 있음을 안 날로부터 3월 내에 가정법원에 신고하여야 하며, 상속포기의 효력은 상속개시된 때에 소급하여 발생한다.",
    "형사소송법 제308조의2에 의하면 적법한 절차에 따르지 아니하고 수집한 증거는 증거로 할 수 없다. 이는 위법수집증거 배제법칙이다.",
    "행정소송법 제19조에 의하면 취소소송은 처분 등이 있음을 안 날로부터 90일 이내에 제기하여야 하고, 처분 등이 있은 날로부터 1년을 경과하면 제기하지 못한다.",
    "특허법 제126조에 의한 특허권 침해금지청구에 있어서 침해행위의 존재 및 특허발명의 보호범위에 속하는지 여부가 핵심 쟁점이다.",
    "위자료 산정에 있어서는 피해자의 나이, 직업, 재산상태, 생활환경, 정신적 고통의 정도 등 여러 사정을 종합적으로 고려하여야 한다.",
    "과실상계에 있어서 피해자의 과실은 사회통념이나 신의성실의 원칙에 따라 공동생활에 있어 요구되는 약한 의미의 부주의를 포함한다.",
    "국가배상법 제2조에 의하면 국가나 지방자치단체는 공무원이 직무를 집행하면서 고의 또는 과실로 법령을 위반하여 타인에게 손해를 입힌 경우 배상책임을 진다.",
    "채무불이행으로 인한 손해배상에 있어서 특별손해는 채무자가 그 사정을 알았거나 알 수 있었을 때에 한하여 배상의 책임이 있다.",
    "민사소송법 제202조에 의하면 법원은 변론 전체의 취지와 증거조사의 결과를 참작하여 자유로운 심증으로 사회정의와 형평의 이념에 입각하여 논리와 경험의 법칙에 따라 사실주장이 진실한지 아닌지를 판단한다.",
    "교통사고처리특례법 제4조 제1항 단서 각 호의 사유에 해당하는 경우에는 피해자의 명시한 의사에 반하여 공소를 제기할 수 있다.",
    "자동차종합보험에서 보험회사는 피보험자가 사고로 타인에게 손해를 가한 경우 피해자에게 직접 보험금을 지급할 의무가 있다.",
]

# 유사/비유사 쌍 (품질 비교용)
SIMILAR_PAIRS = [
    ("손해배상 청구권", "손해배상 청구"),
    ("민법 제750조 불법행위", "민법상 불법행위 책임"),
    ("임대차 계약 해지", "임대차계약의 해지"),
    ("형사소송법 제309조", "형사소송법 309조"),
    ("상속 포기 신고", "상속포기 가정법원 신고"),
    ("교통사고 과실 비율", "교통사고 과실상계"),
]

DISSIMILAR_PAIRS = [
    ("민법 제750조 불법행위", "형법 제250조 살인죄"),
    ("손해배상 청구권", "회사 설립 절차"),
    ("임대차 계약", "특허권 침해"),
    ("형사소송법", "민사집행법"),
    ("상속 포기", "조세 감면 신청"),
    ("교통사고 판례", "환경 영향 평가"),
]

# --- 품질 기준 ---

MODEL_LABELS = {
    "pytorch_fp32": "PyTorch FP32",
    "onnx_fp32": "ONNX FP32",
    "onnx_int8": "ONNX INT8",
    "onnx_o2": "ONNX O2",
    "onnx_o3": "ONNX O3",
    "onnx_o3_int8": "ONNX O3+INT8",
}

QUALITY_THRESHOLDS = {
    "pairwise_cosine_mean": 0.995,
    "sim_matrix_pearson": 0.998,
    "separation_diff": 0.05,
    "top3_match_rate": 0.90,
    "top5_match_rate": 0.80,
    "top10_match_rate": 0.70,
    "search_spearman": 0.95,
}


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
    return configs


# ============================================================
# Phase 0: 환경 준비 (ONNX 변환 + INT8 양자화 + 그래프 최적화)
# ============================================================


def _find_model_snapshot_dir() -> Optional[Path]:
    """HuggingFace 캐시에서 원본 모델 snapshot 디렉토리를 찾는다."""
    hf_model_dir = Path(CACHE_DIR) / "models--nlpai-lab--KURE-v1" / "snapshots"
    if not hf_model_dir.exists():
        return None
    snapshots = list(hf_model_dir.iterdir())
    if not snapshots:
        return None
    return snapshots[0]


def _copy_sentence_transformers_config(target_dir: Path) -> None:
    """원본 모델에서 sentence-transformers config 파일들을 복사한다.

    복사 대상:
    - modules.json (ST 모듈 파이프라인)
    - config_sentence_transformers.json (ST 버전/설정)
    - sentence_bert_config.json (SBERT 설정)
    - 1_Pooling/ (pooling 전략: CLS vs mean)
    - 2_Normalize/ (L2 정규화)
    """
    import shutil

    src_dir = _find_model_snapshot_dir()
    if src_dir is None:
        print("  [경고] 원본 모델 snapshot 디렉토리를 찾을 수 없음 → ST config 복사 스킵")
        return

    # 단일 파일 복사
    config_files = [
        "modules.json",
        "config_sentence_transformers.json",
        "sentence_bert_config.json",
    ]
    copied = 0
    for fname in config_files:
        src = src_dir / fname
        if src.exists():
            shutil.copy2(str(src), str(target_dir / fname))
            copied += 1

    # 디렉토리 복사 (1_Pooling, 2_Normalize 등)
    for subdir_name in ["1_Pooling", "2_Normalize"]:
        src_sub = src_dir / subdir_name
        if src_sub.exists():
            dst_sub = target_dir / subdir_name
            if dst_sub.exists():
                shutil.rmtree(str(dst_sub))
            shutil.copytree(str(src_sub), str(dst_sub))
            copied += 1

    print(f"  ST config 복사 완료: {copied}개 항목 → {target_dir.name}/")


def _copy_tokenizer_and_st_config(src_dir: Path, dst_dir: Path) -> None:
    """토크나이저 + sentence-transformers config 파일을 src_dir에서 dst_dir로 복사."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(src_dir), trust_remote_code=True)
    tokenizer.save_pretrained(str(dst_dir))
    _copy_sentence_transformers_config(dst_dir)


def export_onnx() -> bool:
    """ONNX 모델 변환."""
    if ONNX_DIR.exists() and (ONNX_DIR / "model.onnx").exists():
        print(f"  ONNX 모델 이미 존재: {ONNX_DIR}")
        return True

    print("  ONNX 변환 중...")
    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction

        ort_model = ORTModelForFeatureExtraction.from_pretrained(
            MODEL_NAME,
            cache_dir=CACHE_DIR,
            export=True,
            trust_remote_code=True,
        )
        ort_model.save_pretrained(str(ONNX_DIR))

        # 토크나이저 + ST config 복사
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, cache_dir=CACHE_DIR, trust_remote_code=True
        )
        tokenizer.save_pretrained(str(ONNX_DIR))
        _copy_sentence_transformers_config(ONNX_DIR)

        print(f"  ONNX 변환 완료: {ONNX_DIR}")
        return True
    except Exception as e:
        print(f"  ONNX 변환 실패: {e}")
        return False


def quantize_int8() -> bool:
    """ONNX INT8 양자화."""
    if ONNX_INT8_DIR.exists() and (ONNX_INT8_DIR / "model_quantized.onnx").exists():
        print(f"  INT8 모델 이미 존재: {ONNX_INT8_DIR}")
        return True

    print("  INT8 양자화 중...")
    try:
        from optimum.onnxruntime import ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig

        quantizer = ORTQuantizer.from_pretrained(str(ONNX_DIR))
        qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=True)

        ONNX_INT8_DIR.mkdir(parents=True, exist_ok=True)
        quantizer.quantize(
            save_dir=str(ONNX_INT8_DIR),
            quantization_config=qconfig,
        )

        _copy_tokenizer_and_st_config(ONNX_DIR, ONNX_INT8_DIR)

        print(f"  INT8 양자화 완료: {ONNX_INT8_DIR}")
        return True
    except Exception as e:
        print(f"  INT8 양자화 실패: {e}")
        return False


def optimize_graph(level: str = "O3") -> bool:
    """ONNX 그래프 최적화 (O2 또는 O3)."""
    from optimum.onnxruntime import ORTOptimizer
    from optimum.onnxruntime.configuration import AutoOptimizationConfig

    target_dir = ONNX_O2_DIR if level == "O2" else ONNX_O3_DIR

    if target_dir.exists() and (target_dir / "model_optimized.onnx").exists():
        print(f"  {level} 최적화 모델 이미 존재: {target_dir}")
        return True

    print(f"  {level} 그래프 최적화 중...")
    try:
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

        _copy_tokenizer_and_st_config(ONNX_DIR, target_dir)

        print(f"  {level} 그래프 최적화 완료: {target_dir}")
        return True
    except Exception as e:
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
        from optimum.onnxruntime import ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig

        quantizer = ORTQuantizer.from_pretrained(
            str(ONNX_O3_DIR), file_name="model_optimized.onnx"
        )
        qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=True)

        ONNX_O3_INT8_DIR.mkdir(parents=True, exist_ok=True)
        quantizer.quantize(
            save_dir=str(ONNX_O3_INT8_DIR),
            quantization_config=qconfig,
        )

        _copy_tokenizer_and_st_config(ONNX_O3_DIR, ONNX_O3_INT8_DIR)

        print(f"  O3+INT8 양자화 완료: {ONNX_O3_INT8_DIR}")
        return True
    except Exception as e:
        print(f"  O3+INT8 양자화 실패: {e}")
        return False


# ============================================================
# 임베딩 함수 (모델별)
# ============================================================


def _encode_pytorch(
    texts: list[str],
    batch_size: int = 32,
) -> np.ndarray:
    """PyTorch FP32 임베딩."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        MODEL_NAME,
        cache_folder=CACHE_DIR,
        trust_remote_code=True,
        local_files_only=True,
    )
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    result = np.array(embeddings)
    del model
    gc.collect()
    return result


def _load_onnx_model(
    model_dir: Path,
    file_name: str = "model.onnx",
) -> tuple[Any, Any]:
    """ONNX 모델 + 토크나이저 로드. (optimum 사용)"""
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), trust_remote_code=True
    )
    ort_model = ORTModelForFeatureExtraction.from_pretrained(
        str(model_dir), file_name=file_name, trust_remote_code=True
    )
    return tokenizer, ort_model


def _encode_onnx(
    texts: list[str],
    model_dir: Path,
    file_name: str = "model.onnx",
    batch_size: int = 32,
) -> np.ndarray:
    """ONNX 임베딩 (optimum + mean pooling + L2 norm)."""
    # sentence-transformers backend="onnx" 시도
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            str(model_dir),
            backend="onnx",
            trust_remote_code=True,
            local_files_only=True,
            model_kwargs={"file_name": file_name},
        )
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        result = np.array(embeddings)
        del model
        gc.collect()
        return result
    except Exception as e:
        print(f"    [참고] sentence-transformers ONNX 백엔드 실패 → optimum fallback ({e})")

    # fallback: optimum 직접 사용
    tokenizer, ort_model = _load_onnx_model(model_dir, file_name)

    all_embeddings: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
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
        all_embeddings.append(emb)

    result = np.vstack(all_embeddings)
    del tokenizer, ort_model
    gc.collect()
    return result


# ============================================================
# Phase 1: 속도 벤치마크
# ============================================================


def _measure_single_query_speed(
    encode_fn: Callable[..., np.ndarray],
    queries: list[str],
    label: str,
    **kwargs: Any,
) -> float:
    """단일 쿼리 속도 측정 (warm-up + 반복)."""
    # warm-up
    for _ in range(WARMUP_RUNS):
        encode_fn(queries[:1], **kwargs)

    times: list[float] = []
    for q in queries:
        query_times: list[float] = []
        for _ in range(REPEAT_RUNS):
            t0 = time.monotonic()
            encode_fn([q], **kwargs)
            query_times.append(time.monotonic() - t0)
        avg = float(np.median(query_times))
        times.append(avg)

    overall_avg = float(np.mean(times)) * 1000
    print(f"    {label:<30s} {overall_avg:>8.1f}ms (쿼리 {len(queries)}개 × {REPEAT_RUNS}회)")
    return overall_avg


def _measure_batch_speed(
    encode_fn: Callable[..., np.ndarray],
    documents: list[str],
    label: str,
    batch_sizes: Optional[list[int]] = None,
    **kwargs: Any,
) -> dict[int, float]:
    """배치 속도 측정."""
    if batch_sizes is None:
        batch_sizes = [32, 64, 128]

    results: dict[int, float] = {}
    for bs in batch_sizes:
        # warm-up
        encode_fn(documents[:2], batch_size=bs, **kwargs)

        times: list[float] = []
        for _ in range(REPEAT_RUNS):
            t0 = time.monotonic()
            encode_fn(documents, batch_size=bs, **kwargs)
            times.append(time.monotonic() - t0)

        avg_ms = float(np.median(times)) * 1000
        results[bs] = avg_ms

    sizes_str = "  ".join(f"batch={bs}: {ms:.0f}ms" for bs, ms in results.items())
    print(f"    {label:<30s} {sizes_str}")
    return results


def run_speed_benchmark() -> dict[str, dict[str, object]]:
    """Phase 1: 속도 벤치마크."""
    print(f"\n{SEPARATOR}")
    print("  Phase 1: 속도 벤치마크")
    print(SEPARATOR)

    results: dict[str, dict[str, object]] = {}

    # --- PyTorch FP32 ---
    print("\n  [PyTorch FP32]")
    pt_single = _measure_single_query_speed(
        _encode_pytorch, BENCHMARK_QUERIES, "단일 쿼리"
    )
    pt_batch = _measure_batch_speed(
        _encode_pytorch, BENCHMARK_DOCUMENTS, "배치 (20문서)"
    )
    results["pytorch_fp32"] = {"single_ms": pt_single, "batch": pt_batch}

    # --- ONNX 모델 (동적 목록) ---
    for model_key, model_dir, file_name in _get_model_configs():
        label = MODEL_LABELS.get(model_key, model_key.upper())
        print(f"\n  [{label}]")
        single = _measure_single_query_speed(
            _encode_onnx,
            BENCHMARK_QUERIES,
            "단일 쿼리",
            model_dir=model_dir,
            file_name=file_name,
        )
        batch = _measure_batch_speed(
            _encode_onnx,
            BENCHMARK_DOCUMENTS,
            "배치 (20문서)",
            model_dir=model_dir,
            file_name=file_name,
        )
        results[model_key] = {"single_ms": single, "batch": batch}

    return results


# ============================================================
# Phase 2: 임베딩 품질 비교
# ============================================================


def _pairwise_cosine(
    emb_a: np.ndarray, emb_b: np.ndarray
) -> np.ndarray:
    """각 행 간 코사인 유사도 (이미 L2 정규화된 벡터 기준)."""
    return np.sum(emb_a * emb_b, axis=1)


def _sim_matrix(emb: np.ndarray) -> np.ndarray:
    """문서 간 유사도 행렬 (N×N)."""
    return emb @ emb.T


def run_quality_benchmark() -> dict[str, dict[str, float]]:
    """Phase 2: 임베딩 품질 비교."""
    from scipy.stats import pearsonr, spearmanr

    print(f"\n{SEPARATOR}")
    print("  Phase 2: 임베딩 품질 비교")
    print(SEPARATOR)

    # 모든 텍스트를 합쳐서 임베딩
    all_texts = (
        BENCHMARK_QUERIES
        + BENCHMARK_DOCUMENTS
        + [t for pair in SIMILAR_PAIRS for t in pair]
        + [t for pair in DISSIMILAR_PAIRS for t in pair]
    )

    # --- PyTorch FP32 (baseline) ---
    print("  PyTorch FP32 임베딩 생성 중...")
    emb_fp32 = _encode_pytorch(all_texts)

    results: dict[str, dict[str, float]] = {}

    for model_key, model_dir, file_name in _get_model_configs():
        label = MODEL_LABELS.get(model_key, model_key.upper())
        print(f"\n  [{label}] 임베딩 생성 중...")
        emb_test = _encode_onnx(all_texts, model_dir, file_name)

        # 1) Pairwise Cosine: 동일 텍스트 FP32 vs ONNX 벡터 유사도
        pairwise = _pairwise_cosine(emb_fp32, emb_test)
        pairwise_mean = float(np.mean(pairwise))
        pairwise_min = float(np.min(pairwise))

        # 2) Sim Matrix 상관: 문서 간 유사도 행렬 Pearson/Spearman
        n_docs = len(BENCHMARK_DOCUMENTS)
        doc_start = len(BENCHMARK_QUERIES)
        doc_end = doc_start + n_docs

        sim_fp32 = _sim_matrix(emb_fp32[doc_start:doc_end])
        sim_test = _sim_matrix(emb_test[doc_start:doc_end])

        # 상삼각 추출 (대각선 제외)
        triu_idx = np.triu_indices(n_docs, k=1)
        sim_fp32_flat = sim_fp32[triu_idx]
        sim_test_flat = sim_test[triu_idx]

        pearson_corr, _ = pearsonr(sim_fp32_flat, sim_test_flat)
        spearman_corr, _ = spearmanr(sim_fp32_flat, sim_test_flat)

        # 3) Separation 분리도 차이
        pair_start = doc_end
        n_sim = len(SIMILAR_PAIRS)
        n_dissim = len(DISSIMILAR_PAIRS)

        sim_pair_start = pair_start
        sim_pair_end = pair_start + n_sim * 2
        dissim_pair_start = sim_pair_end

        # FP32 유사/비유사 쌍 유사도
        fp32_sim_scores: list[float] = []
        for i in range(n_sim):
            idx_a = sim_pair_start + i * 2
            idx_b = sim_pair_start + i * 2 + 1
            score = float(np.dot(emb_fp32[idx_a], emb_fp32[idx_b]))
            fp32_sim_scores.append(score)

        fp32_dissim_scores: list[float] = []
        for i in range(n_dissim):
            idx_a = dissim_pair_start + i * 2
            idx_b = dissim_pair_start + i * 2 + 1
            score = float(np.dot(emb_fp32[idx_a], emb_fp32[idx_b]))
            fp32_dissim_scores.append(score)

        fp32_separation = float(np.mean(fp32_sim_scores)) - float(
            np.mean(fp32_dissim_scores)
        )

        # Test 유사/비유사 쌍 유사도
        test_sim_scores: list[float] = []
        for i in range(n_sim):
            idx_a = sim_pair_start + i * 2
            idx_b = sim_pair_start + i * 2 + 1
            score = float(np.dot(emb_test[idx_a], emb_test[idx_b]))
            test_sim_scores.append(score)

        test_dissim_scores: list[float] = []
        for i in range(n_dissim):
            idx_a = dissim_pair_start + i * 2
            idx_b = dissim_pair_start + i * 2 + 1
            score = float(np.dot(emb_test[idx_a], emb_test[idx_b]))
            test_dissim_scores.append(score)

        test_separation = float(np.mean(test_sim_scores)) - float(
            np.mean(test_dissim_scores)
        )
        separation_diff = abs(fp32_separation - test_separation)

        metrics = {
            "pairwise_cosine_mean": pairwise_mean,
            "pairwise_cosine_min": pairwise_min,
            "sim_matrix_pearson": float(pearson_corr),
            "sim_matrix_spearman": float(spearman_corr),
            "separation_fp32": fp32_separation,
            "separation_test": test_separation,
            "separation_diff": separation_diff,
        }
        results[model_key] = metrics

        print(f"    Pairwise Cosine Mean: {pairwise_mean:.6f} (min: {pairwise_min:.6f})")
        print(f"    Sim Matrix Pearson:   {pearson_corr:.6f}")
        print(f"    Sim Matrix Spearman:  {spearman_corr:.6f}")
        print(f"    Separation (FP32):    {fp32_separation:.4f}")
        print(f"    Separation (test):    {test_separation:.4f}")
        print(f"    Separation Diff:      {separation_diff:.4f}")

        del emb_test
        gc.collect()

    del emb_fp32
    gc.collect()
    return results


# ============================================================
# Phase 3: LanceDB 검색 품질 비교
# ============================================================


def run_search_benchmark() -> dict[str, dict[str, float]]:
    """Phase 3: LanceDB 검색 품질 비교."""
    from scipy.stats import spearmanr

    print(f"\n{SEPARATOR}")
    print("  Phase 3: LanceDB 검색 품질 비교")
    print(SEPARATOR)

    lance_path = LANCEDB_DIR / f"{LANCEDB_TABLE}.lance"
    if not lance_path.exists():
        print(f"  LanceDB 데이터 없음: {lance_path}")
        print("  --skip-search 옵션 또는 데이터 준비 후 재실행")
        return {}

    import lancedb

    db = lancedb.connect(str(LANCEDB_DIR))
    try:
        table = db.open_table(LANCEDB_TABLE)
    except Exception as e:
        print(f"  LanceDB 테이블 열기 실패: {e}")
        return {}

    top_k_values = [3, 5, 10]
    queries = BENCHMARK_QUERIES[:5]  # 검색 비교는 5개 쿼리로 충분

    # --- FP32 쿼리 벡터로 검색 ---
    print("  PyTorch FP32 쿼리 벡터 생성 중...")
    fp32_query_embs = _encode_pytorch(queries)

    fp32_results: dict[int, list[list[str]]] = {k: [] for k in top_k_values}
    fp32_scores: dict[int, list[list[float]]] = {k: [] for k in top_k_values}

    for i, q_emb in enumerate(fp32_query_embs):
        max_k = max(top_k_values)
        search_results = (
            table.search(q_emb.tolist())
            .metric("cosine")
            .limit(max_k)
            .to_pandas()
        )
        for k in top_k_values:
            top_rows = search_results.head(k)
            ids = top_rows["source_id"].tolist() if "source_id" in top_rows.columns else top_rows.index.tolist()
            distances = top_rows["_distance"].tolist()
            fp32_results[k].append([str(x) for x in ids])
            fp32_scores[k].append([float(1 - d) for d in distances])

    results: dict[str, dict[str, float]] = {}

    for model_key, model_dir, file_name in _get_model_configs():
        label = MODEL_LABELS.get(model_key, model_key.upper())
        print(f"\n  [{label}] 쿼리 벡터 생성 중...")
        test_query_embs = _encode_onnx(queries, model_dir, file_name)

        metrics: dict[str, float] = {}

        for k in top_k_values:
            match_rates: list[float] = []
            for i, q_emb in enumerate(test_query_embs):
                search_results = (
                    table.search(q_emb.tolist())
                    .metric("cosine")
                    .limit(k)
                    .to_pandas()
                )
                ids = search_results["source_id"].tolist() if "source_id" in search_results.columns else search_results.index.tolist()
                test_ids = set(str(x) for x in ids)
                fp32_ids = set(fp32_results[k][i])
                match = len(test_ids & fp32_ids) / k
                match_rates.append(match)

            avg_match = float(np.mean(match_rates))
            metrics[f"top{k}_match_rate"] = avg_match
            print(f"    Top-{k} ID 일치율: {avg_match:.1%}")

        # 순위 Spearman (Top-10 유사도 점수)
        k = 10
        all_fp32_flat: list[float] = []
        all_test_flat: list[float] = []
        for i, q_emb in enumerate(test_query_embs):
            search_results = (
                table.search(q_emb.tolist())
                .metric("cosine")
                .limit(k)
                .to_pandas()
            )
            distances = search_results["_distance"].tolist()
            test_sims = [float(1 - d) for d in distances]
            all_fp32_flat.extend(fp32_scores[k][i])
            all_test_flat.extend(test_sims)

        if len(all_fp32_flat) > 2:
            sp_corr, _ = spearmanr(all_fp32_flat, all_test_flat)
            metrics["search_spearman"] = float(sp_corr)
            print(f"    순위 Spearman:     {sp_corr:.4f}")

        results[model_key] = metrics

        del test_query_embs
        gc.collect()

    del fp32_query_embs
    gc.collect()
    return results


# ============================================================
# Phase 4: 모델 크기 비교
# ============================================================


def _dir_size_mb(path: Path) -> float:
    """디렉토리 크기 (MB)."""
    if not path.exists():
        return 0.0
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def run_size_benchmark() -> dict[str, float]:
    """Phase 4: 모델 크기 비교."""
    print(f"\n{SEPARATOR}")
    print("  Phase 4: 모델 크기 비교")
    print(SEPARATOR)

    # PyTorch 모델 크기 추정 (캐시 디렉토리에서)
    pytorch_dir = Path(CACHE_DIR) / "models--nlpai-lab--KURE-v1"
    pt_size = _dir_size_mb(pytorch_dir)

    results: dict[str, float] = {"pytorch_fp32_mb": pt_size}
    print(f"    PyTorch FP32:  {pt_size:>8.0f}MB")

    model_dir_map: dict[str, Path] = {
        "onnx_fp32": ONNX_DIR,
        "onnx_int8": ONNX_INT8_DIR,
        "onnx_o2": ONNX_O2_DIR,
        "onnx_o3": ONNX_O3_DIR,
        "onnx_o3_int8": ONNX_O3_INT8_DIR,
    }
    for model_key, model_dir in model_dir_map.items():
        size_mb = _dir_size_mb(model_dir)
        results[f"{model_key}_mb"] = size_mb
        if size_mb > 0:
            saving = (1 - size_mb / pt_size) * 100 if pt_size > 0 else 0
            label = MODEL_LABELS.get(model_key, model_key.upper())
            print(f"    {label:<14s}  {size_mb:>8.0f}MB  ({saving:+.0f}%)")

    return results


# ============================================================
# Phase 5: 최종 리포트
# ============================================================


def _pass_fail(value: float, threshold: float, higher_is_better: bool = True) -> str:
    """PASS/FAIL 판정."""
    if higher_is_better:
        return "PASS" if value >= threshold else "FAIL"
    return "PASS" if value <= threshold else "FAIL"


def print_final_report(
    speed_results: dict[str, dict[str, object]],
    quality_results: dict[str, dict[str, float]],
    search_results: dict[str, dict[str, float]],
    size_results: dict[str, float],
) -> None:
    """Phase 5: 최종 리포트 출력."""
    print(f"\n{SEPARATOR}")
    print("  임베딩 모델 양자화 종합 벤치마크 결과")
    print(f"  모델: {MODEL_NAME} (1024차원)")
    print(SEPARATOR)

    model_keys = [k for k, _, _ in _get_model_configs()]

    # --- 1. 속도 - 단일 쿼리 ---
    print("\n  [1. 속도 - 단일 쿼리]")
    print(f"  {'방식':<30s} {'평균':>10s} {'상대 속도':>12s}")
    print(f"  {'-' * 30} {'-' * 10} {'-' * 12}")

    pt_single = float(speed_results.get("pytorch_fp32", {}).get("single_ms", 0))  # type: ignore[arg-type]
    print(f"  {'PyTorch FP32 (baseline)':<30s} {pt_single:>8.1f}ms {'1.0x':>12s}")

    for key in model_keys:
        if key in speed_results:
            label = MODEL_LABELS.get(key, key.upper())
            ms = float(speed_results[key].get("single_ms", 0))  # type: ignore[arg-type]
            speedup = pt_single / ms if ms > 0 else 0
            print(f"  {label:<30s} {ms:>8.1f}ms {speedup:>10.1f}x")

    # --- 2. 속도 - 배치 ---
    print("\n  [2. 속도 - 배치 (20문서)]")
    batch_sizes = [32, 64, 128]
    header = f"  {'방식':<30s}" + "".join(f" {'batch=' + str(b):>12s}" for b in batch_sizes)
    print(header)
    print(f"  {'-' * 30}" + " ".join(f"{'-' * 12}" for _ in batch_sizes))

    all_speed_keys = ["pytorch_fp32", *model_keys]
    for key in all_speed_keys:
        if key in speed_results:
            label = MODEL_LABELS.get(key, key.upper())
            batch_data = speed_results[key].get("batch", {})
            cols = ""
            for b in batch_sizes:
                ms = batch_data.get(b, 0)  # type: ignore[attr-defined]
                cols += f" {float(ms):>10.0f}ms"
            print(f"  {label:<30s}{cols}")

    # --- 3. 임베딩 품질 ---
    if quality_results:
        present_quality_keys = [k for k in model_keys if k in quality_results]
        print("\n  [3. 임베딩 품질]")
        print(f"  {'메트릭':<25s}", end="")
        for key in present_quality_keys:
            label = MODEL_LABELS.get(key, key.upper())
            print(f" {label:>14s}", end="")
        print(f" {'기준':>10s} {'판정':>6s}")
        print(f"  {'-' * 25}", end="")
        for _ in present_quality_keys:
            print(f" {'-' * 14}", end="")
        print(f" {'-' * 10} {'-' * 6}")

        quality_metrics = [
            ("Pairwise Cosine Mean", "pairwise_cosine_mean", QUALITY_THRESHOLDS["pairwise_cosine_mean"], True),
            ("Sim Matrix Pearson", "sim_matrix_pearson", QUALITY_THRESHOLDS["sim_matrix_pearson"], True),
            ("Separation Diff", "separation_diff", QUALITY_THRESHOLDS["separation_diff"], False),
        ]

        for metric_label, metric_key, threshold, higher_is_better in quality_metrics:
            print(f"  {metric_label:<25s}", end="")
            worst_verdict = "PASS"
            for mk in present_quality_keys:
                val = quality_results[mk].get(metric_key, 0)
                print(f" {val:>14.4f}", end="")
                if _pass_fail(val, threshold, higher_is_better) == "FAIL":
                    worst_verdict = "FAIL"

            threshold_str = f">={threshold}" if higher_is_better else f"<{threshold}"
            print(f" {threshold_str:>10s} {worst_verdict:>6s}")

    # --- 4. LanceDB 검색 품질 ---
    if search_results:
        present_search_keys = [k for k in model_keys if k in search_results]
        print("\n  [4. LanceDB 검색 품질]")
        print(f"  {'메트릭':<25s}", end="")
        for key in present_search_keys:
            label = MODEL_LABELS.get(key, key.upper())
            print(f" {label:>14s}", end="")
        print(f" {'기준':>10s} {'판정':>6s}")
        print(f"  {'-' * 25}", end="")
        for _ in present_search_keys:
            print(f" {'-' * 14}", end="")
        print(f" {'-' * 10} {'-' * 6}")

        search_metrics = [
            ("Top-3 ID 일치율", "top3_match_rate", QUALITY_THRESHOLDS["top3_match_rate"], True),
            ("Top-5 ID 일치율", "top5_match_rate", QUALITY_THRESHOLDS["top5_match_rate"], True),
            ("Top-10 ID 일치율", "top10_match_rate", QUALITY_THRESHOLDS["top10_match_rate"], True),
            ("순위 Spearman", "search_spearman", QUALITY_THRESHOLDS["search_spearman"], True),
        ]

        for metric_label, metric_key, threshold, higher_is_better in search_metrics:
            print(f"  {metric_label:<25s}", end="")
            worst_verdict = "PASS"
            for mk in present_search_keys:
                val = search_results[mk].get(metric_key, 0)
                if "일치율" in metric_label:
                    print(f" {val:>13.0%}", end="")
                else:
                    print(f" {val:>14.4f}", end="")
                if _pass_fail(val, threshold, higher_is_better) == "FAIL":
                    worst_verdict = "FAIL"

            threshold_str = f">={threshold:.0%}" if "일치율" in metric_label else f">={threshold}"
            print(f" {threshold_str:>10s} {worst_verdict:>6s}")

    # --- 5. 모델 크기 ---
    if size_results:
        print("\n  [5. 모델 크기]")
        print(f"  {'방식':<30s} {'크기':>10s} {'절약':>8s}")
        print(f"  {'-' * 30} {'-' * 10} {'-' * 8}")

        pt_mb = size_results.get("pytorch_fp32_mb", 0)
        print(f"  {'PyTorch FP32':<30s} {pt_mb:>8.0f}MB {'-':>8s}")
        for key in model_keys:
            size_key = f"{key}_mb"
            mb = size_results.get(size_key, 0)
            if mb > 0:
                label = MODEL_LABELS.get(key, key.upper())
                saving = (1 - mb / pt_mb) * 100 if pt_mb > 0 else 0
                print(f"  {label:<30s} {mb:>8.0f}MB {saving:>+7.0f}%")

    # --- 종합 판정 ---
    print("\n  [종합 판정]")
    for key in model_keys:
        label = MODEL_LABELS.get(key, key.upper())
        all_pass = True
        speed_str = ""

        if key in speed_results:
            ms = float(speed_results[key].get("single_ms", 0))  # type: ignore[arg-type]
            speedup = pt_single / ms if ms > 0 else 0
            speed_str = f"속도 {speedup:.1f}x"

        if key in quality_results:
            q = quality_results[key]
            if q.get("pairwise_cosine_mean", 0) < QUALITY_THRESHOLDS["pairwise_cosine_mean"]:
                all_pass = False
            if q.get("sim_matrix_pearson", 0) < QUALITY_THRESHOLDS["sim_matrix_pearson"]:
                all_pass = False
            if q.get("separation_diff", 1) > QUALITY_THRESHOLDS["separation_diff"]:
                all_pass = False
            quality_str = "품질 동일" if key == "onnx_fp32" else "품질 허용 범위"
        else:
            quality_str = "품질 미측정"

        if key in search_results:
            s = search_results[key]
            if s.get("top3_match_rate", 0) < QUALITY_THRESHOLDS["top3_match_rate"]:
                all_pass = False
            if s.get("search_spearman", 0) < QUALITY_THRESHOLDS["search_spearman"]:
                all_pass = False

        verdict = "PASS" if all_pass else "FAIL"
        parts = [p for p in [speed_str, quality_str] if p]
        detail = ", ".join(parts)
        print(f"  {label}: {verdict} ({detail})")

    print(SEPARATOR)


# ============================================================
# main
# ============================================================


def parse_args() -> argparse.Namespace:
    """CLI 인자 파싱."""
    parser = argparse.ArgumentParser(
        description="KURE-v1 임베딩 양자화 종합 벤치마크"
    )
    parser.add_argument(
        "--skip-search",
        action="store_true",
        help="LanceDB 검색 비교 스킵",
    )
    parser.add_argument(
        "--skip-quality",
        action="store_true",
        help="임베딩 품질 비교 스킵 (속도만 측정)",
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
    return parser.parse_args()


def main() -> None:
    """벤치마크 메인."""
    args = parse_args()

    print(f"\n{SEPARATOR}")
    print("  임베딩 모델 양자화 종합 벤치마크")
    print(f"  모델: {MODEL_NAME} (1024차원)")
    print(f"  쿼리: {len(BENCHMARK_QUERIES)}개, 문서: {len(BENCHMARK_DOCUMENTS)}개")
    print(SEPARATOR)

    # --- Phase 0: 환경 준비 ---
    if not args.skip_export:
        print(f"\n{SEPARATOR}")
        print("  Phase 0: 환경 준비 (ONNX 변환 + INT8 양자화 + 그래프 최적화)")
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
    else:
        print("\n  Phase 0: 스킵 (--skip-export)")

    # --- Phase 1: 속도 벤치마크 ---
    speed_results = run_speed_benchmark()

    # --- Phase 2: 임베딩 품질 비교 ---
    quality_results: dict[str, dict[str, float]] = {}
    if not args.skip_quality:
        quality_results = run_quality_benchmark()
    else:
        print("\n  Phase 2: 스킵 (--skip-quality)")

    # --- Phase 3: LanceDB 검색 품질 비교 ---
    search_results: dict[str, dict[str, float]] = {}
    if not args.skip_search:
        search_results = run_search_benchmark()
    else:
        print("\n  Phase 3: 스킵 (--skip-search)")

    # --- Phase 4: 모델 크기 비교 ---
    size_results = run_size_benchmark()

    # --- Phase 5: 최종 리포트 ---
    print_final_report(speed_results, quality_results, search_results, size_results)


if __name__ == "__main__":
    main()
