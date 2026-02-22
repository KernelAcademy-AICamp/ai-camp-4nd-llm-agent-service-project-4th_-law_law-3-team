"""ONNX 최적화 모델 빌드 스크립트

PyTorch 체크포인트에서 최적화된 ONNX 모델을 빌드합니다.
임베딩(KURE-v1)과 리랭커(bge-reranker-v2-m3-ko) 양쪽 모두 지원.

빌드 단계:
  1. optimum으로 ONNX 내보내기 (opset=17)
  2. onnxruntime.transformers.optimizer로 그래프 퓨전 (FP32)
  3. FP16 변환 (keep_io_types=True)
  4. 퓨전 통계 검증 (Attention >= 24)
  5. PyTorch 대비 수치 동등성 검증 (cosine similarity)
  6. model_versions.json 메타데이터 저장
  7. 토크나이저 + config.json 복사

출력 변형:
  - kure-v1-ort-opt        : 임베딩 Fusion FP32
  - kure-v1-ort-opt-fp16   : 임베딩 Fusion + FP16
  - reranker-ort-opt       : 리랭커 Fusion FP32
  - reranker-ort-opt-fp16  : 리랭커 Fusion + FP16

사용법:
  cd backend && uv run python scripts/build_optimized_onnx.py
  cd backend && uv run python scripts/build_optimized_onnx.py --verify
  cd backend && uv run python scripts/build_optimized_onnx.py --embedding-only
  cd backend && uv run python scripts/build_optimized_onnx.py --reranker-only
  cd backend && uv run python scripts/build_optimized_onnx.py --skip-fp16
  cd backend && uv run python scripts/build_optimized_onnx.py --overwrite
"""

import argparse
import gc
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
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
SUBSEPARATOR = "-" * 50

# --- 임베딩 모델 ---
EMB_MODEL_NAME = "nlpai-lab/KURE-v1"
CACHE_DIR = str(PROJECT_ROOT / "data" / "models")
DATA_MODELS_DIR = PROJECT_ROOT / "data" / "models"

# KURE-v1: XLM-RoBERTa Large
EMB_NUM_HEADS = 16
EMB_HIDDEN_SIZE = 1024
EMB_OPSET = 17
EMB_EXPECTED_ATTENTION_FUSIONS = 24

# ONNX 내보내기 임시 디렉토리 (optimum 출력)
EMB_ONNX_EXPORT_DIR = DATA_MODELS_DIR / "kure-v1-onnx-export-tmp"
# 최종 출력 디렉토리
EMB_ORT_OPT_DIR = DATA_MODELS_DIR / "kure-v1-ort-opt"
EMB_ORT_OPT_FP16_DIR = DATA_MODELS_DIR / "kure-v1-ort-opt-fp16"

# --- 리랭커 모델 ---
RR_MODEL_NAME = "dragonkue/bge-reranker-v2-m3-ko"

# bge-reranker-v2-m3-ko: XLM-RoBERTa Large
RR_NUM_HEADS = 16
RR_HIDDEN_SIZE = 1024
RR_OPSET = 17
RR_EXPECTED_ATTENTION_FUSIONS = 24

# ONNX 내보내기 임시 디렉토리
RR_ONNX_EXPORT_DIR = DATA_MODELS_DIR / "reranker-onnx-export-tmp"
# 최종 출력 디렉토리
RR_ORT_OPT_DIR = DATA_MODELS_DIR / "reranker-ort-opt"
RR_ORT_OPT_FP16_DIR = DATA_MODELS_DIR / "reranker-ort-opt-fp16"

# 메타데이터 파일
MODEL_VERSIONS_PATH = DATA_MODELS_DIR / "model_versions.json"

# 검증 임계값
COSINE_THRESHOLD_FP32 = 0.9999
COSINE_THRESHOLD_FP16 = 0.999
MAX_ABS_DIFF_THRESHOLD_FP32 = 1e-4
MAX_ABS_DIFF_THRESHOLD_FP16 = 5e-3

# --- 테스트 데이터 ---
TEST_TEXTS = [
    "교통사고 손해배상 판례",
    "임대차 보증금 반환 청구",
    "근로기준법 해고 부당해고",
    "이혼 재산분할 위자료",
    "명예훼손 형사 고소",
]

RR_QUERY = "교통사고 손해배상 판례"
RR_DOCUMENTS = [
    "피고는 원고에게 금 50,000,000원 및 이에 대한 지연손해금을 지급하라.",
    "자동차손해배상 보장법 제3조에 의하면 운행자는 손해배상 책임을 진다.",
    "임대차보증금 반환 청구 사건에서 보증금 전액을 반환할 의무가 있다.",
    "근로기준법 제23조 제1항은 정당한 이유 없이 해고를 하지 못한다고 규정한다.",
    "형법 제307조 제1항의 명예훼손죄가 성립하려면 사실을 적시하여야 한다.",
]

# 토크나이저 관련 파일 목록 (복사 대상)
TOKENIZER_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
    "vocab.txt",
]


# ============================================================
# 유틸리티 함수
# ============================================================


def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    """Sigmoid 활성화 함수."""
    return 1.0 / (1.0 + np.exp(-x))


def _cosine_similarity_pairwise(a: np.ndarray, b: np.ndarray) -> float:
    """두 벡터 배열의 평균 pairwise cosine similarity."""
    similarities: list[float] = []
    for i in range(len(a)):
        dot = np.dot(a[i], b[i])
        norm_a = np.linalg.norm(a[i])
        norm_b = np.linalg.norm(b[i])
        if norm_a > 0 and norm_b > 0:
            similarities.append(float(dot / (norm_a * norm_b)))
    return float(np.mean(similarities)) if similarities else 0.0


def _copy_tokenizer_and_config(src_dir: Path, dst_dir: Path) -> int:
    """토크나이저 + config 파일을 src에서 dst로 복사. 복사된 파일 수 반환."""
    copied = 0
    for filename in TOKENIZER_FILES:
        src_file = src_dir / filename
        if src_file.exists():
            shutil.copy2(str(src_file), str(dst_dir / filename))
            copied += 1
    return copied


def _format_size(size_bytes: int) -> str:
    """바이트 수를 읽기 좋은 형태로 변환."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024  # type: ignore[assignment]
    return f"{size_bytes:.1f} TB"


def _dir_total_size(directory: Path) -> int:
    """디렉토리 전체 크기(바이트)."""
    total = 0
    if directory.exists():
        for f in directory.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    return total


def _load_model_versions() -> dict[str, Any]:
    """model_versions.json 로드. 없으면 빈 dict 반환."""
    if MODEL_VERSIONS_PATH.exists():
        with open(MODEL_VERSIONS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_model_versions(versions: dict[str, Any]) -> None:
    """model_versions.json 저장."""
    MODEL_VERSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_VERSIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(versions, f, indent=2, ensure_ascii=False)
    print(f"    메타데이터 저장: {MODEL_VERSIONS_PATH.name}")


def _cleanup_dir(directory: Path) -> None:
    """임시 디렉토리 삭제."""
    if directory.exists():
        shutil.rmtree(directory)
        print(f"    임시 디렉토리 삭제: {directory.name}")


# ============================================================
# Step 1: ONNX 내보내기
# ============================================================


def export_embedding_onnx(
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> bool:
    """임베딩 모델을 ONNX로 내보내기 (optimum 사용)."""
    model_file = output_dir / "model.onnx"
    if model_file.exists() and not overwrite:
        print(f"    이미 존재 (건너뜀): {output_dir.name}/model.onnx")
        return True

    print(f"    ONNX 내보내기: {EMB_MODEL_NAME} -> {output_dir.name}/")
    try:
        from optimum.exporters.onnx import main_export

        output_dir.mkdir(parents=True, exist_ok=True)

        main_export(
            model_name_or_path=EMB_MODEL_NAME,
            output=str(output_dir),
            task="feature-extraction",
            opset=EMB_OPSET,
            cache_dir=CACHE_DIR,
            trust_remote_code=True,
        )

        if not model_file.exists():
            print("    실패: model.onnx 파일이 생성되지 않았습니다")
            return False

        size = model_file.stat().st_size
        print(f"    완료: {_format_size(size)}")
        return True

    except Exception as e:
        print(f"    ONNX 내보내기 실패: {e!s:.200s}")
        return False


def export_reranker_onnx(
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> bool:
    """리랭커 모델을 ONNX로 내보내기 (optimum 사용)."""
    model_file = output_dir / "model.onnx"
    if model_file.exists() and not overwrite:
        print(f"    이미 존재 (건너뜀): {output_dir.name}/model.onnx")
        return True

    print(f"    ONNX 내보내기: {RR_MODEL_NAME} -> {output_dir.name}/")
    try:
        from optimum.exporters.onnx import main_export

        output_dir.mkdir(parents=True, exist_ok=True)

        main_export(
            model_name_or_path=RR_MODEL_NAME,
            output=str(output_dir),
            task="text-classification",
            opset=RR_OPSET,
            cache_dir=CACHE_DIR,
            trust_remote_code=True,
        )

        if not model_file.exists():
            print("    실패: model.onnx 파일이 생성되지 않았습니다")
            return False

        size = model_file.stat().st_size
        print(f"    완료: {_format_size(size)}")
        return True

    except Exception as e:
        print(f"    ONNX 내보내기 실패: {e!s:.200s}")
        return False


# ============================================================
# Step 2: ORT Transformer Optimizer (FP32 Fusion)
# ============================================================


def optimize_model_with_ort(
    onnx_path: Path,
    output_dir: Path,
    *,
    num_heads: int,
    hidden_size: int,
    label: str,
    overwrite: bool = False,
) -> dict[str, int] | None:
    """ORT transformer optimizer로 모델을 최적화하고 퓨전 통계를 반환.

    Returns:
        퓨전 통계 dict (Attention, LayerNormalization 등) 또는 실패 시 None.
    """
    output_file = output_dir / "model_optimized.onnx"
    if output_file.exists() and not overwrite:
        print(f"    [{label}] 이미 존재 (건너뜀): {output_dir.name}/")
        # 기존 모델의 퓨전 통계는 반환할 수 없으므로 빈 dict
        return {}

    print(f"    [{label}] ORT transformer optimizer 실행 중...")
    try:
        from onnxruntime.transformers.optimizer import optimize_model

        output_dir.mkdir(parents=True, exist_ok=True)

        optimized = optimize_model(
            str(onnx_path),
            model_type="bert",
            num_heads=num_heads,
            hidden_size=hidden_size,
            use_external_data_format=True,
        )

        # 퓨전 통계 출력
        stats = optimized.get_fused_operator_statistics()
        attention_count = stats.get("Attention", 0)
        layernorm_count = stats.get("LayerNormalization", 0)
        gelu_count = stats.get("Gelu", 0) + stats.get("FastGelu", 0) + stats.get("BiasGelu", 0)

        print("    퓨전 통계:")
        print(f"      Attention:         {attention_count}")
        print(f"      LayerNormalization: {layernorm_count}")
        print(f"      Gelu/FastGelu:     {gelu_count}")

        if attention_count > 0:
            for key, value in stats.items():
                if key not in ("Attention", "LayerNormalization", "Gelu", "FastGelu", "BiasGelu") and value > 0:
                    print(f"      {key}: {value}")

        # 모델 저장 (외부 데이터 포맷)
        optimized.save_model_to_file(
            str(output_file),
            use_external_data_format=True,
        )

        # 파일 크기 확인
        data_file = output_dir / "model_optimized.onnx.data"
        if data_file.exists():
            graph_size = output_file.stat().st_size
            weight_size = data_file.stat().st_size
            print(f"    그래프: {_format_size(graph_size)}, 가중치: {_format_size(weight_size)}")
        else:
            model_size = output_file.stat().st_size
            print(f"    모델: {_format_size(model_size)}")

        print(f"    [{label}] 최적화 완료: {output_dir.name}/")
        return dict(stats)

    except Exception as e:
        print(f"    [{label}] 최적화 실패: {e!s:.200s}")
        return None


# ============================================================
# Step 3: FP16 변환
# ============================================================


def convert_to_fp16(
    fp32_dir: Path,
    fp16_dir: Path,
    *,
    label: str,
    overwrite: bool = False,
) -> bool:
    """FP32 최적화 모델을 FP16으로 변환.

    2GB+ 모델은 OnnxModel 기반 외부 데이터 포맷으로 처리.
    """
    fp32_path = fp32_dir / "model_optimized.onnx"
    fp16_path = fp16_dir / "model_optimized.onnx"

    if fp16_path.exists() and not overwrite:
        print(f"    [{label}] 이미 존재 (건너뜀): {fp16_dir.name}/")
        return True

    if not fp32_path.exists():
        print(f"    [{label}] FP32 모델 없음: {fp32_path}")
        return False

    print(f"    [{label}] FP16 변환 중...")
    try:
        fp16_dir.mkdir(parents=True, exist_ok=True)

        # 모델 크기 확인 (외부 데이터 포함)
        data_file = fp32_dir / "model_optimized.onnx.data"
        has_external_data = data_file.exists()

        if has_external_data:
            # 2GB+ 모델: OnnxModel 기반 변환
            print("    외부 데이터 파일 감지 -> OnnxModel 기반 FP16 변환")
            import onnx
            from onnxruntime.transformers.onnx_model import OnnxModel

            model = onnx.load(str(fp32_path))
            onnx_model = OnnxModel(model)
            onnx_model.convert_float_to_float16(keep_io_types=True)
            onnx_model.save_model_to_file(
                str(fp16_path),
                use_external_data_format=True,
            )
        else:
            # 소형 모델: 직접 변환
            import onnx
            from onnxruntime.transformers.float16 import convert_float_to_float16

            model = onnx.load(str(fp32_path))
            model_fp16 = convert_float_to_float16(model, keep_io_types=True)
            onnx.save(model_fp16, str(fp16_path))

        # 결과 확인
        fp16_data_file = fp16_dir / "model_optimized.onnx.data"
        if fp16_data_file.exists():
            graph_size = fp16_path.stat().st_size
            weight_size = fp16_data_file.stat().st_size
            print(f"    그래프: {_format_size(graph_size)}, 가중치: {_format_size(weight_size)}")
        else:
            model_size = fp16_path.stat().st_size
            print(f"    모델: {_format_size(model_size)}")

        print(f"    [{label}] FP16 변환 완료: {fp16_dir.name}/")
        return True

    except Exception as e:
        print(f"    [{label}] FP16 변환 실패: {e!s:.200s}")
        return False


# ============================================================
# Step 4: 수치 동등성 검증
# ============================================================


def _encode_pytorch(texts: list[str]) -> np.ndarray:
    """PyTorch FP32 임베딩."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        EMB_MODEL_NAME,
        cache_folder=CACHE_DIR,
        trust_remote_code=True,
        local_files_only=True,
    )
    embeddings = model.encode(
        texts, batch_size=32,
        show_progress_bar=False, normalize_embeddings=True,
    )
    result = np.array(embeddings)
    del model
    gc.collect()
    return result


def _encode_onnx(
    texts: list[str],
    model_dir: Path,
    file_name: str,
) -> np.ndarray:
    """ORT 세션으로 임베딩 생성 (CLS pooling + L2 norm)."""
    import onnxruntime as ort
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    model_path = str(model_dir / file_name)

    session = ort.InferenceSession(
        model_path,
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
    emb = outputs[0][:, 0, :]
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.clip(norm, a_min=1e-9, a_max=None)

    del session, tokenizer
    gc.collect()
    return emb


def _rerank_pytorch(query: str, documents: list[str]) -> np.ndarray:
    """PyTorch FP32 리랭킹 로짓."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(RR_MODEL_NAME, cache_dir=CACHE_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(
        RR_MODEL_NAME, cache_dir=CACHE_DIR,
    )
    model.eval()

    pairs = [(query, doc) for doc in documents]
    inputs = tokenizer(
        [p[0] for p in pairs],
        [p[1] for p in pairs],
        return_tensors="pt",
        padding=True, truncation=True, max_length=512,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.numpy()

    result = logits[:, 0] if logits.ndim == 2 else logits.flatten()
    del model, tokenizer
    gc.collect()
    return result


def _rerank_onnx(
    query: str,
    documents: list[str],
    model_dir: Path,
    file_name: str,
) -> np.ndarray:
    """ORT 세션으로 리랭킹 로짓."""
    import onnxruntime as ort
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model_path = str(model_dir / file_name)

    session = ort.InferenceSession(
        model_path,
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

    del session, tokenizer
    gc.collect()
    return logits


def verify_embedding_quality(
    onnx_dir: Path,
    onnx_file: str,
    *,
    label: str,
    cosine_threshold: float = COSINE_THRESHOLD_FP32,
    max_diff_threshold: float = MAX_ABS_DIFF_THRESHOLD_FP32,
) -> dict[str, float]:
    """임베딩 ONNX 모델의 수치 동등성 검증.

    Returns:
        검증 결과 dict (cosine, max_abs_diff, passed).
    """
    print(f"    [{label}] 수치 동등성 검증 중...")

    try:
        pt_emb = _encode_pytorch(TEST_TEXTS)
        onnx_emb = _encode_onnx(TEST_TEXTS, onnx_dir, onnx_file)

        cosine = _cosine_similarity_pairwise(pt_emb, onnx_emb)
        max_diff = float(np.max(np.abs(pt_emb - onnx_emb)))

        is_passed = cosine >= cosine_threshold and max_diff <= max_diff_threshold
        status = "PASS" if is_passed else "FAIL"

        print(f"      Cosine similarity: {cosine:.6f} (임계값: {cosine_threshold})")
        print(f"      Max abs diff:      {max_diff:.2e} (임계값: {max_diff_threshold:.0e})")
        print(f"      결과: [{status}]")

        return {
            "cosine": cosine,
            "max_abs_diff": max_diff,
            "passed": float(is_passed),
        }
    except Exception as e:
        print(f"      검증 실패: {e!s:.200s}")
        return {"cosine": 0.0, "max_abs_diff": float("inf"), "passed": 0.0}


def verify_reranker_quality(
    onnx_dir: Path,
    onnx_file: str,
    *,
    label: str,
    cosine_threshold: float = COSINE_THRESHOLD_FP32,
    max_diff_threshold: float = MAX_ABS_DIFF_THRESHOLD_FP32,
) -> dict[str, float]:
    """리랭커 ONNX 모델의 수치 동등성 검증.

    Returns:
        검증 결과 dict (cosine, max_abs_diff, rank_correlation, passed).
    """
    print(f"    [{label}] 수치 동등성 검증 중...")

    try:
        pt_logits = _rerank_pytorch(RR_QUERY, RR_DOCUMENTS)
        onnx_logits = _rerank_onnx(RR_QUERY, RR_DOCUMENTS, onnx_dir, onnx_file)

        # 로짓 벡터 간 cosine similarity
        dot = np.dot(pt_logits, onnx_logits)
        norm_pt = np.linalg.norm(pt_logits)
        norm_onnx = np.linalg.norm(onnx_logits)
        cosine = float(dot / (norm_pt * norm_onnx)) if norm_pt > 0 and norm_onnx > 0 else 0.0

        max_diff = float(np.max(np.abs(pt_logits - onnx_logits)))

        # 순위 상관 (Spearman)
        from scipy.stats import spearmanr
        rank_corr, _ = spearmanr(pt_logits, onnx_logits)
        rank_corr = float(rank_corr)

        is_passed = cosine >= cosine_threshold and max_diff <= max_diff_threshold
        status = "PASS" if is_passed else "FAIL"

        print(f"      Cosine similarity:  {cosine:.6f} (임계값: {cosine_threshold})")
        print(f"      Max abs diff:       {max_diff:.2e} (임계값: {max_diff_threshold:.0e})")
        print(f"      Spearman rank corr: {rank_corr:.6f}")
        print(f"      결과: [{status}]")

        # 순위 비교 출력
        pt_scores = _sigmoid(pt_logits)
        onnx_scores = _sigmoid(onnx_logits)
        pt_ranking = np.argsort(-pt_scores)
        onnx_ranking = np.argsort(-onnx_scores)
        is_rank_same = np.array_equal(pt_ranking, onnx_ranking)
        print(f"      순위 일치: {'동일' if is_rank_same else '차이 있음'}")

        return {
            "cosine": cosine,
            "max_abs_diff": max_diff,
            "rank_correlation": rank_corr,
            "passed": float(is_passed),
        }
    except Exception as e:
        print(f"      검증 실패: {e!s:.200s}")
        return {"cosine": 0.0, "max_abs_diff": float("inf"), "rank_correlation": 0.0, "passed": 0.0}


# ============================================================
# 빌드 파이프라인
# ============================================================


def build_embedding_models(
    *,
    overwrite: bool = False,
    skip_fp16: bool = False,
) -> dict[str, Any]:
    """임베딩 모델 최적화 빌드 전체 파이프라인.

    Returns:
        빌드 결과 메타데이터 dict.
    """
    print(f"\n{SEPARATOR}")
    print("  임베딩 모델 최적화 빌드")
    print(f"  모델: {EMB_MODEL_NAME}")
    print(f"  num_heads={EMB_NUM_HEADS}, hidden_size={EMB_HIDDEN_SIZE}, opset={EMB_OPSET}")
    print(SEPARATOR)

    results: dict[str, Any] = {}
    build_start = time.perf_counter()

    # Step 1: ONNX 내보내기
    print(f"\n  {SUBSEPARATOR}")
    print("  Step 1: ONNX 내보내기 (optimum)")
    print(f"  {SUBSEPARATOR}")

    export_ok = export_embedding_onnx(EMB_ONNX_EXPORT_DIR, overwrite=overwrite)
    if not export_ok:
        print("  임베딩 ONNX 내보내기 실패. 빌드 중단.")
        return results

    # Step 2: ORT Optimizer (FP32)
    print(f"\n  {SUBSEPARATOR}")
    print("  Step 2: ORT Transformer Optimizer (FP32)")
    print(f"  {SUBSEPARATOR}")

    onnx_src = EMB_ONNX_EXPORT_DIR / "model.onnx"
    fusion_stats = optimize_model_with_ort(
        onnx_src, EMB_ORT_OPT_DIR,
        num_heads=EMB_NUM_HEADS,
        hidden_size=EMB_HIDDEN_SIZE,
        label="임베딩 FP32",
        overwrite=overwrite,
    )

    if fusion_stats is None:
        print("  임베딩 ORT 최적화 실패. 빌드 중단.")
        return results

    # 퓨전 검증
    attention_count = fusion_stats.get("Attention", 0)
    if attention_count > 0 and attention_count < EMB_EXPECTED_ATTENTION_FUSIONS:
        print(f"  경고: Attention 퓨전 {attention_count}개 "
              f"(예상: >= {EMB_EXPECTED_ATTENTION_FUSIONS})")

    # 토크나이저 복사
    copied = _copy_tokenizer_and_config(EMB_ONNX_EXPORT_DIR, EMB_ORT_OPT_DIR)
    print(f"    토크나이저 파일 복사: {copied}개")

    # Step 3: FP16 변환
    if not skip_fp16:
        print(f"\n  {SUBSEPARATOR}")
        print("  Step 3: FP16 변환")
        print(f"  {SUBSEPARATOR}")

        fp16_ok = convert_to_fp16(
            EMB_ORT_OPT_DIR, EMB_ORT_OPT_FP16_DIR,
            label="임베딩 FP16",
            overwrite=overwrite,
        )

        if fp16_ok:
            copied = _copy_tokenizer_and_config(EMB_ONNX_EXPORT_DIR, EMB_ORT_OPT_FP16_DIR)
            print(f"    토크나이저 파일 복사: {copied}개")

    # Step 4: 수치 동등성 검증
    print(f"\n  {SUBSEPARATOR}")
    print("  Step 4: 수치 동등성 검증")
    print(f"  {SUBSEPARATOR}")

    fp32_quality = verify_embedding_quality(
        EMB_ORT_OPT_DIR, "model_optimized.onnx",
        label="FP32",
        cosine_threshold=COSINE_THRESHOLD_FP32,
        max_diff_threshold=MAX_ABS_DIFF_THRESHOLD_FP32,
    )
    results["kure-v1-ort-opt"] = _build_embedding_meta(
        fusion_stats, fp32_quality, "ort-optimizer-fp32",
    )

    if not skip_fp16 and EMB_ORT_OPT_FP16_DIR.exists():
        fp16_quality = verify_embedding_quality(
            EMB_ORT_OPT_FP16_DIR, "model_optimized.onnx",
            label="FP16",
            cosine_threshold=COSINE_THRESHOLD_FP16,
            max_diff_threshold=MAX_ABS_DIFF_THRESHOLD_FP16,
        )
        results["kure-v1-ort-opt-fp16"] = _build_embedding_meta(
            fusion_stats, fp16_quality, "ort-optimizer-fp16",
        )

    # 임시 디렉토리 정리
    print(f"\n  {SUBSEPARATOR}")
    print("  정리")
    print(f"  {SUBSEPARATOR}")
    _cleanup_dir(EMB_ONNX_EXPORT_DIR)

    # 요약
    build_time = time.perf_counter() - build_start
    print(f"\n  임베딩 빌드 완료 ({build_time:.1f}초)")
    print(f"    FP32: {_format_size(_dir_total_size(EMB_ORT_OPT_DIR))}")
    if not skip_fp16 and EMB_ORT_OPT_FP16_DIR.exists():
        print(f"    FP16: {_format_size(_dir_total_size(EMB_ORT_OPT_FP16_DIR))}")

    return results


def build_reranker_models(
    *,
    overwrite: bool = False,
    skip_fp16: bool = False,
) -> dict[str, Any]:
    """리랭커 모델 최적화 빌드 전체 파이프라인.

    Returns:
        빌드 결과 메타데이터 dict.
    """
    print(f"\n{SEPARATOR}")
    print("  리랭커 모델 최적화 빌드")
    print(f"  모델: {RR_MODEL_NAME}")
    print(f"  num_heads={RR_NUM_HEADS}, hidden_size={RR_HIDDEN_SIZE}, opset={RR_OPSET}")
    print(SEPARATOR)

    results: dict[str, Any] = {}
    build_start = time.perf_counter()

    # Step 1: ONNX 내보내기
    print(f"\n  {SUBSEPARATOR}")
    print("  Step 1: ONNX 내보내기 (optimum)")
    print(f"  {SUBSEPARATOR}")

    export_ok = export_reranker_onnx(RR_ONNX_EXPORT_DIR, overwrite=overwrite)
    if not export_ok:
        print("  리랭커 ONNX 내보내기 실패. 빌드 중단.")
        return results

    # Step 2: ORT Optimizer (FP32)
    print(f"\n  {SUBSEPARATOR}")
    print("  Step 2: ORT Transformer Optimizer (FP32)")
    print(f"  {SUBSEPARATOR}")

    onnx_src = RR_ONNX_EXPORT_DIR / "model.onnx"
    fusion_stats = optimize_model_with_ort(
        onnx_src, RR_ORT_OPT_DIR,
        num_heads=RR_NUM_HEADS,
        hidden_size=RR_HIDDEN_SIZE,
        label="리랭커 FP32",
        overwrite=overwrite,
    )

    if fusion_stats is None:
        print("  리랭커 ORT 최적화 실패. 빌드 중단.")
        return results

    # 퓨전 검증
    attention_count = fusion_stats.get("Attention", 0)
    if attention_count > 0 and attention_count < RR_EXPECTED_ATTENTION_FUSIONS:
        print(f"  경고: Attention 퓨전 {attention_count}개 "
              f"(예상: >= {RR_EXPECTED_ATTENTION_FUSIONS})")

    # 토크나이저 복사
    copied = _copy_tokenizer_and_config(RR_ONNX_EXPORT_DIR, RR_ORT_OPT_DIR)
    print(f"    토크나이저 파일 복사: {copied}개")

    # Step 3: FP16 변환
    if not skip_fp16:
        print(f"\n  {SUBSEPARATOR}")
        print("  Step 3: FP16 변환")
        print(f"  {SUBSEPARATOR}")

        fp16_ok = convert_to_fp16(
            RR_ORT_OPT_DIR, RR_ORT_OPT_FP16_DIR,
            label="리랭커 FP16",
            overwrite=overwrite,
        )

        if fp16_ok:
            copied = _copy_tokenizer_and_config(RR_ONNX_EXPORT_DIR, RR_ORT_OPT_FP16_DIR)
            print(f"    토크나이저 파일 복사: {copied}개")

    # Step 4: 수치 동등성 검증
    print(f"\n  {SUBSEPARATOR}")
    print("  Step 4: 수치 동등성 검증")
    print(f"  {SUBSEPARATOR}")

    fp32_quality = verify_reranker_quality(
        RR_ORT_OPT_DIR, "model_optimized.onnx",
        label="FP32",
        cosine_threshold=COSINE_THRESHOLD_FP32,
        max_diff_threshold=MAX_ABS_DIFF_THRESHOLD_FP32,
    )
    results["reranker-ort-opt"] = _build_reranker_meta(
        fusion_stats, fp32_quality, "ort-optimizer-fp32",
    )

    if not skip_fp16 and RR_ORT_OPT_FP16_DIR.exists():
        fp16_quality = verify_reranker_quality(
            RR_ORT_OPT_FP16_DIR, "model_optimized.onnx",
            label="FP16",
            cosine_threshold=COSINE_THRESHOLD_FP16,
            max_diff_threshold=MAX_ABS_DIFF_THRESHOLD_FP16,
        )
        results["reranker-ort-opt-fp16"] = _build_reranker_meta(
            fusion_stats, fp16_quality, "ort-optimizer-fp16",
        )

    # 임시 디렉토리 정리
    print(f"\n  {SUBSEPARATOR}")
    print("  정리")
    print(f"  {SUBSEPARATOR}")
    _cleanup_dir(RR_ONNX_EXPORT_DIR)

    # 요약
    build_time = time.perf_counter() - build_start
    print(f"\n  리랭커 빌드 완료 ({build_time:.1f}초)")
    print(f"    FP32: {_format_size(_dir_total_size(RR_ORT_OPT_DIR))}")
    if not skip_fp16 and RR_ORT_OPT_FP16_DIR.exists():
        print(f"    FP16: {_format_size(_dir_total_size(RR_ORT_OPT_FP16_DIR))}")

    return results


# ============================================================
# 메타데이터 헬퍼
# ============================================================


def _get_ort_version() -> str:
    """onnxruntime 버전 반환."""
    try:
        import onnxruntime as ort
        return ort.__version__
    except ImportError:
        return "unknown"


def _build_embedding_meta(
    fusion_stats: dict[str, int],
    quality: dict[str, float],
    optimization: str,
) -> dict[str, Any]:
    """임베딩 모델 메타데이터 dict 생성."""
    return {
        "base_model": EMB_MODEL_NAME,
        "optimization": optimization,
        "model_type": "bert",
        "num_heads": EMB_NUM_HEADS,
        "hidden_size": EMB_HIDDEN_SIZE,
        "fusion_stats": fusion_stats,
        "cosine_vs_pytorch": quality.get("cosine", 0.0),
        "max_abs_diff": quality.get("max_abs_diff", float("inf")),
        "verification_passed": bool(quality.get("passed", 0)),
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "ort_version": _get_ort_version(),
        "opset": EMB_OPSET,
    }


def _build_reranker_meta(
    fusion_stats: dict[str, int],
    quality: dict[str, float],
    optimization: str,
) -> dict[str, Any]:
    """리랭커 모델 메타데이터 dict 생성."""
    return {
        "base_model": RR_MODEL_NAME,
        "optimization": optimization,
        "model_type": "bert",
        "num_heads": RR_NUM_HEADS,
        "hidden_size": RR_HIDDEN_SIZE,
        "fusion_stats": fusion_stats,
        "cosine_vs_pytorch": quality.get("cosine", 0.0),
        "max_abs_diff": quality.get("max_abs_diff", float("inf")),
        "rank_correlation": quality.get("rank_correlation", 0.0),
        "verification_passed": bool(quality.get("passed", 0)),
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "ort_version": _get_ort_version(),
        "opset": RR_OPSET,
    }


# ============================================================
# 검증 전용 모드
# ============================================================


def verify_existing_builds(
    *,
    run_embedding: bool = True,
    run_reranker: bool = True,
) -> None:
    """기존 빌드의 수치 동등성만 검증."""
    print(f"\n{SEPARATOR}")
    print("  기존 빌드 검증 모드")
    print(SEPARATOR)

    if run_embedding:
        print("\n  === 임베딩 모델 ===")
        for label, model_dir, threshold_cos, threshold_diff in [
            ("FP32", EMB_ORT_OPT_DIR, COSINE_THRESHOLD_FP32, MAX_ABS_DIFF_THRESHOLD_FP32),
            ("FP16", EMB_ORT_OPT_FP16_DIR, COSINE_THRESHOLD_FP16, MAX_ABS_DIFF_THRESHOLD_FP16),
        ]:
            model_file = model_dir / "model_optimized.onnx"
            if model_file.exists():
                verify_embedding_quality(
                    model_dir, "model_optimized.onnx",
                    label=label,
                    cosine_threshold=threshold_cos,
                    max_diff_threshold=threshold_diff,
                )
            else:
                print(f"    [{label}] 모델 없음: {model_dir.name}/")

    if run_reranker:
        print("\n  === 리랭커 모델 ===")
        for label, model_dir, threshold_cos, threshold_diff in [
            ("FP32", RR_ORT_OPT_DIR, COSINE_THRESHOLD_FP32, MAX_ABS_DIFF_THRESHOLD_FP32),
            ("FP16", RR_ORT_OPT_FP16_DIR, COSINE_THRESHOLD_FP16, MAX_ABS_DIFF_THRESHOLD_FP16),
        ]:
            model_file = model_dir / "model_optimized.onnx"
            if model_file.exists():
                verify_reranker_quality(
                    model_dir, "model_optimized.onnx",
                    label=label,
                    cosine_threshold=threshold_cos,
                    max_diff_threshold=threshold_diff,
                )
            else:
                print(f"    [{label}] 모델 없음: {model_dir.name}/")


# ============================================================
# 환경 정보 출력
# ============================================================


def print_environment() -> None:
    """환경 정보 출력."""
    import platform as _platform

    print(f"\n{SEPARATOR}")
    print("  환경 정보")
    print(SEPARATOR)

    print(f"  OS:          {_platform.system()} {_platform.release()}")
    print(f"  Python:      {_platform.python_version()}")

    try:
        import onnxruntime as ort
        print(f"  onnxruntime: {ort.__version__}")
        print(f"  providers:   {ort.get_available_providers()}")
    except ImportError:
        print("  onnxruntime: 미설치")

    try:
        import onnx
        print(f"  onnx:        {onnx.__version__}")
    except ImportError:
        print("  onnx:        미설치")

    try:
        import optimum
        print(f"  optimum:     {optimum.__version__}")
    except ImportError:
        print("  optimum:     미설치")

    try:
        import torch
        print(f"  PyTorch:     {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  CUDA:        {torch.version.cuda}")
            print(f"  GPU:         {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("  PyTorch:     미설치")

    print(f"\n  모델 디렉토리: {DATA_MODELS_DIR}")
    print("  출력 디렉토리 상태:")
    for label, directory in [
        ("임베딩 FP32", EMB_ORT_OPT_DIR),
        ("임베딩 FP16", EMB_ORT_OPT_FP16_DIR),
        ("리랭커 FP32", RR_ORT_OPT_DIR),
        ("리랭커 FP16", RR_ORT_OPT_FP16_DIR),
    ]:
        if directory.exists():
            size = _format_size(_dir_total_size(directory))
            has_model = (directory / "model_optimized.onnx").exists()
            status = f"존재 ({size})" if has_model else "불완전"
        else:
            status = "없음"
        print(f"    {label:<14s}: {directory.name} [{status}]")


# ============================================================
# CLI + main
# ============================================================


def parse_args() -> argparse.Namespace:
    """CLI 인자 파싱."""
    parser = argparse.ArgumentParser(
        description="ONNX 최적화 모델 빌드 (임베딩 + 리랭커)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  uv run python scripts/build_optimized_onnx.py           # 전체 빌드
  uv run python scripts/build_optimized_onnx.py --verify   # 기존 빌드 검증
  uv run python scripts/build_optimized_onnx.py --embedding-only
  uv run python scripts/build_optimized_onnx.py --reranker-only
  uv run python scripts/build_optimized_onnx.py --skip-fp16
  uv run python scripts/build_optimized_onnx.py --overwrite
        """,
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="기존 빌드의 수치 동등성만 검증 (빌드하지 않음)",
    )
    parser.add_argument(
        "--embedding-only", action="store_true",
        help="임베딩 모델만 빌드",
    )
    parser.add_argument(
        "--reranker-only", action="store_true",
        help="리랭커 모델만 빌드",
    )
    parser.add_argument(
        "--skip-fp16", action="store_true",
        help="FP16 변환 스킵",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="기존 모델 덮어쓰기 (기본: 이미 존재하면 건너뜀)",
    )
    return parser.parse_args()


def main() -> None:
    """빌드 메인."""
    args = parse_args()

    run_emb = not args.reranker_only
    run_rr = not args.embedding_only

    print(f"\n{SEPARATOR}")
    print("  ONNX 최적화 모델 빌드")
    print(f"  임베딩: {EMB_MODEL_NAME}")
    print(f"  리랭커: {RR_MODEL_NAME}")
    target = "임베딩 + 리랭커"
    if args.embedding_only:
        target = "임베딩만"
    elif args.reranker_only:
        target = "리랭커만"
    mode = "검증 모드" if args.verify else "빌드 모드"
    print(f"  대상: {target} ({mode})")
    print(SEPARATOR)

    # 환경 정보
    print_environment()

    # 검증 전용 모드
    if args.verify:
        verify_existing_builds(run_embedding=run_emb, run_reranker=run_rr)
        print(f"\n{SEPARATOR}")
        print("  검증 완료")
        print(SEPARATOR)
        return

    # 빌드 모드
    total_start = time.perf_counter()
    all_meta: dict[str, Any] = {}

    if run_emb:
        emb_meta = build_embedding_models(
            overwrite=args.overwrite,
            skip_fp16=args.skip_fp16,
        )
        all_meta.update(emb_meta)

    if run_rr:
        rr_meta = build_reranker_models(
            overwrite=args.overwrite,
            skip_fp16=args.skip_fp16,
        )
        all_meta.update(rr_meta)

    # model_versions.json 업데이트
    if all_meta:
        print(f"\n{SEPARATOR}")
        print("  메타데이터 저장")
        print(SEPARATOR)

        versions = _load_model_versions()
        versions.update(all_meta)
        _save_model_versions(versions)

        # 저장된 내용 요약 출력
        print("\n  model_versions.json 내용:")
        for key, meta in all_meta.items():
            cosine = meta.get("cosine_vs_pytorch", 0)
            passed = meta.get("verification_passed", False)
            status = "PASS" if passed else "FAIL"
            print(f"    {key}: cosine={cosine:.6f} [{status}]")

    # 최종 요약
    total_time = time.perf_counter() - total_start
    print(f"\n{SEPARATOR}")
    print("  빌드 완료 요약")
    print(SEPARATOR)
    print(f"  총 소요 시간: {total_time:.1f}초 ({total_time / 60:.1f}분)")

    if run_emb:
        print("\n  임베딩 출력:")
        for label, directory in [
            ("FP32", EMB_ORT_OPT_DIR),
            ("FP16", EMB_ORT_OPT_FP16_DIR),
        ]:
            if directory.exists():
                size = _format_size(_dir_total_size(directory))
                print(f"    {label}: {directory} ({size})")
            elif label == "FP16" and args.skip_fp16:
                print(f"    {label}: 스킵됨 (--skip-fp16)")
            else:
                print(f"    {label}: 빌드 실패")

    if run_rr:
        print("\n  리랭커 출력:")
        for label, directory in [
            ("FP32", RR_ORT_OPT_DIR),
            ("FP16", RR_ORT_OPT_FP16_DIR),
        ]:
            if directory.exists():
                size = _format_size(_dir_total_size(directory))
                print(f"    {label}: {directory} ({size})")
            elif label == "FP16" and args.skip_fp16:
                print(f"    {label}: 스킵됨 (--skip-fp16)")
            else:
                print(f"    {label}: 빌드 실패")

    if MODEL_VERSIONS_PATH.exists():
        print(f"\n  메타데이터: {MODEL_VERSIONS_PATH}")

    print(f"\n{SEPARATOR}")
    print("  완료")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
