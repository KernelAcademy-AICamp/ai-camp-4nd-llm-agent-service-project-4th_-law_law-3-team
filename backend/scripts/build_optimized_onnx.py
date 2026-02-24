"""ONNX 최적화 모델 빌드 스크립트

PyTorch 체크포인트에서 최적화된 ONNX 모델을 빌드합니다.
임베딩(KURE-v1)과 리랭커(bge-reranker-v2-m3-ko) 양쪽 모두 지원.

빌드 단계:
  1. optimum으로 ONNX 내보내기 (opset=17)
  1b. onnx-simplifier로 그래프 단순화 (constant folding, dead node 제거)
  2. onnxruntime.transformers.optimizer로 그래프 퓨전 (FP32)
  3. FP16 변환 (keep_io_types=True)
  4. 퓨전 통계 검증 (Attention >= 24)
  5. PyTorch 대비 수치 동등성 검증 (cosine similarity)
  6. model_versions.json 메타데이터 저장
  7. 토크나이저 + config.json 복사

출력 변형 (현재 유효):
  - kure-v1-ort-opt            : 임베딩 Fusion FP32 (무손실, cosine 1.0)
  - kure-v1-ort-opt-qdq        : 임베딩 QDQ 선택적 INT8 (16 FP32 레이어, cosine 0.999)
  - reranker-ort-opt           : 리랭커 Fusion FP32
  - reranker-ort-opt-qdq       : 리랭커 QDQ 선택적 INT8

삭제된 변형 (벤치마크 결과 역효과/실패):
  - fp16, static128, static64, static64-qdq (Mac ARM에서 역효과)

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
EMB_ORT_OPT_STATIC128_DIR = DATA_MODELS_DIR / "kure-v1-ort-opt-static128"
EMB_ORT_OPT_STATIC64_DIR = DATA_MODELS_DIR / "kure-v1-ort-opt-static64"
EMB_ORT_OPT_QDQ_DIR = DATA_MODELS_DIR / "kure-v1-ort-opt-qdq"
EMB_ORT_OPT_STATIC64_QDQ_DIR = DATA_MODELS_DIR / "kure-v1-ort-opt-static64-qdq"

# Static shape 설정
EMB_STATIC_BATCH_SIZE = 1
EMB_STATIC_SEQ_LENGTH = 128
EMB_STATIC64_SEQ_LENGTH = 64

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
RR_ORT_OPT_QDQ_DIR = DATA_MODELS_DIR / "reranker-ort-opt-qdq"

# 메타데이터 파일
MODEL_VERSIONS_PATH = DATA_MODELS_DIR / "model_versions.json"

# 검증 임계값
COSINE_THRESHOLD_FP32 = 0.9999
COSINE_THRESHOLD_FP16 = 0.999
COSINE_THRESHOLD_QDQ = 0.985  # INT8 양자화: 실측 임베딩 ~0.988, 리랭커 ~0.9998
MAX_ABS_DIFF_THRESHOLD_FP32 = 1e-4
MAX_ABS_DIFF_THRESHOLD_FP16 = 5e-3
MAX_ABS_DIFF_THRESHOLD_FP16_RR = 1.5e-2  # 리랭커는 분류 logit → 임베딩보다 FP16 diff가 큼
MAX_ABS_DIFF_THRESHOLD_QDQ = 0.03  # 임베딩 INT8: 정규화 벡터, 실측 ~0.021
MAX_ABS_DIFF_THRESHOLD_QDQ_RR = 0.5  # 리랭커 INT8: logit 스케일이 커서 abs diff 큼, 실측 ~0.45

# QDQ 기본 민감 레이어 (XLM-RoBERTa Large 24 layers)
# sweep_sensitive_layers.py 실측 결과: cosine >= 0.999 달성에 16개 FP32 필요.
# INT8 레이어 = [3, 4, 5, 7, 8, 10, 11, 16] (8개만 양자화)
DEFAULT_SENSITIVE_LAYERS = [0, 1, 2, 6, 9, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23]
NUM_TRANSFORMER_LAYERS = 24

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

# --- 캘리브레이션 데이터 (Static Quantization용) ---
CALIBRATION_TEXTS = [
    # 민사
    "교통사고 손해배상 판례",
    "임대차 보증금 반환 청구",
    "매매 계약 해제 위약금",
    "불법행위 손해배상 책임",
    "채무불이행 손해배상",
    "계약 위반 손해배상 청구",
    "소유권 이전 등기 청구",
    "부당이득 반환 청구",
    # 형사
    "사기죄 구성요건 판단",
    "횡령 배임 형사 처벌 기준",
    "특수폭행 상해 형사사건",
    "음주운전 면허취소 처벌",
    "마약류 관리에 관한 법률 위반",
    "명예훼손 형사 고소 요건",
    # 가사
    "이혼 재산분할 위자료 산정",
    "양육권 변경 청구 요건",
    "상속 유류분 반환 청구",
    "친권 행사 제한 심판",
    "혼인무효 확인 소송",
    # 행정
    "건축허가 취소 행정소송",
    "과징금 부과처분 취소",
    "영업정지 처분 취소 청구",
    "개발행위허가 불허가 처분",
    "정보공개 거부처분 취소",
    # 노동
    "부당해고 구제 신청",
    "퇴직금 미지급 청구",
    "산업재해 보상 청구",
    "직장 내 괴롭힘 손해배상",
    "임금체불 진정 신고",
    # 부동산
    "부동산 매매 계약서 작성",
    "전세 보증금 반환 소송",
    "건물 명도 청구 소송",
    "토지 수용 보상금 청구",
    "재건축 조합 분쟁 해결",
    # 지식재산
    "특허 침해 금지 청구",
    "상표권 침해 손해배상",
    "저작권 침해 형사 고소",
    "영업비밀 침해 금지 가처분",
    # 세법/금융
    "양도소득세 과세처분 취소",
    "부가가치세 경정 청구",
    "상속세 과세표준 산정 기준",
    "법인세 부당행위 계산 부인",
    # 회사법
    "주주총회 결의 무효 확인",
    "이사 배임 책임 추궁 소송",
    "합병 무효 소송 요건",
    "신주발행 무효 청구",
    # 헌법
    "기본권 침해 헌법소원 심판",
    "법률 위헌 심판 청구",
    "평등권 침해 여부 판단 기준",
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
            library_name="transformers",
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
# Step 1b: ONNX Graph Surgery (onnx-simplifier)
# ============================================================


def simplify_onnx_model(
    model_dir: Path,
    *,
    label: str,
    overwrite: bool = False,
) -> dict[str, Any] | None:
    """onnx-simplifier로 그래프를 단순화한다.

    constant folding 강화, dead node 제거, reshape 최적화를 수행.
    수학적 동등 변환이므로 품질 손실 없음.

    Returns:
        단순화 전후 통계 dict 또는 실패 시 None.
    """
    model_path = model_dir / "model.onnx"
    if not model_path.exists():
        print(f"    [{label}] 모델 파일 없음: {model_path}")
        return None

    print(f"    [{label}] onnx-simplifier 실행 중...")
    try:
        import onnx
        import onnxsim

        model = onnx.load(str(model_path))

        # 단순화 전 통계
        nodes_before = len(model.graph.node)
        size_before = model_path.stat().st_size

        # simplify 실행
        model_simplified, check = onnxsim.simplify(model)

        if not check:
            print(f"    [{label}] simplify 검증 실패 (원본 유지)")
            return {
                "success": False,
                "nodes_before": nodes_before,
                "nodes_after": nodes_before,
                "nodes_removed": 0,
                "size_before": size_before,
                "size_after": size_before,
            }

        # 단순화 후 통계
        nodes_after = len(model_simplified.graph.node)
        nodes_removed = nodes_before - nodes_after

        # 원본 덮어쓰기 (simplify는 무손실이므로 안전)
        onnx.save(model_simplified, str(model_path))
        size_after = model_path.stat().st_size
        size_diff_pct = (1 - size_after / size_before) * 100 if size_before > 0 else 0

        print(f"    노드: {nodes_before} → {nodes_after} ({nodes_removed}개 제거)")
        print(f"    크기: {_format_size(size_before)} → {_format_size(size_after)} ({size_diff_pct:+.1f}%)")
        print(f"    [{label}] simplify 완료")

        return {
            "success": True,
            "nodes_before": nodes_before,
            "nodes_after": nodes_after,
            "nodes_removed": nodes_removed,
            "size_before": size_before,
            "size_after": size_after,
            "size_reduction_pct": size_diff_pct,
        }

    except ImportError:
        print(f"    [{label}] onnx-simplifier 미설치 (건너뜀)")
        print("    설치: uv add --dev onnx-simplifier")
        return None
    except Exception as e:
        print(f"    [{label}] simplify 실패: {e!s:.200s}")
        return None


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
# Step 3b: Static Shape 변환 (임베딩 전용)
# ============================================================


def convert_to_static_shape(
    fp32_dir: Path,
    static_dir: Path,
    *,
    batch_size: int = EMB_STATIC_BATCH_SIZE,
    seq_length: int = EMB_STATIC_SEQ_LENGTH,
    label: str,
    overwrite: bool = False,
) -> bool:
    """FP32 최적화 모델을 고정 shape로 변환.

    동적 축(batch_size, sequence_length)을 고정하여
    추가 constant folding, reshape 제거를 유도한다.
    """
    fp32_path = fp32_dir / "model_optimized.onnx"
    static_path = static_dir / "model_optimized.onnx"

    if static_path.exists() and not overwrite:
        print(f"    [{label}] 이미 존재 (건너뜀): {static_dir.name}/")
        return True

    if not fp32_path.exists():
        print(f"    [{label}] FP32 모델 없음: {fp32_path}")
        return False

    print(f"    [{label}] Static shape 변환 중 (batch={batch_size}, seq={seq_length})...")
    try:
        import onnx
        from onnxruntime.tools.make_dynamic_shape_fixed import make_input_shape_fixed

        static_dir.mkdir(parents=True, exist_ok=True)

        # 외부 데이터가 있으면 data_path 기준으로 로드
        data_file = fp32_dir / "model_optimized.onnx.data"
        if data_file.exists():
            model = onnx.load(str(fp32_path), load_external_data=False)
            from onnx.external_data_helper import load_external_data_for_model

            load_external_data_for_model(model, str(fp32_dir))
        else:
            model = onnx.load(str(fp32_path))

        # ORT 1.23.2+: 입력별로 고정 shape 적용
        input_shape_map = {
            "input_ids": [batch_size, seq_length],
            "attention_mask": [batch_size, seq_length],
        }
        for input_name, fixed_shape in input_shape_map.items():
            make_input_shape_fixed(model.graph, input_name, fixed_shape)

        # 외부 데이터 모델: 가중치를 외부 파일로 저장
        if data_file.exists():
            onnx.save_model(
                model,
                str(static_path),
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location="model_optimized.onnx.data",
                size_threshold=1024,
            )
        else:
            onnx.save(model, str(static_path))

        # 결과 확인
        static_data_file = static_dir / "model_optimized.onnx.data"
        if static_data_file.exists():
            graph_size = static_path.stat().st_size
            weight_size = static_data_file.stat().st_size
            print(f"    그래프: {_format_size(graph_size)}, 가중치: {_format_size(weight_size)}")
        else:
            model_size = static_path.stat().st_size
            print(f"    모델: {_format_size(model_size)}")

        print(f"    [{label}] Static shape 변환 완료: {static_dir.name}/")
        return True

    except Exception as e:
        print(f"    [{label}] Static shape 변환 실패: {e!s:.200s}")
        return False


# ============================================================
# Step 3c: QDQ 선택적 양자화 (민감 레이어 제외 INT8)
# ============================================================


class LegalTextCalibrationReader:
    """법률 텍스트 calibration 데이터 리더 (quantize_static용).

    onnxruntime.quantization.CalibrationDataReader를 구현하여
    Static Quantization의 activation range 캘리브레이션에 사용한다.
    """

    def __init__(
        self,
        model_dir: Path,
        texts: list[str],
        *,
        max_length: int = 512,
        model_type: str = "embedding",
    ) -> None:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), trust_remote_code=True,
        )
        self.max_length = max_length
        self._idx = 0

        # 모델 입력 이름 파악
        model_path = model_dir / "model.onnx"
        if not model_path.exists():
            model_path = model_dir / "model_optimized.onnx"
        session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"],
        )
        input_names = {inp.name for inp in session.get_inputs()}
        del session

        # 모든 피드를 미리 준비
        self._feeds: list[dict[str, np.ndarray]] = []
        for text in texts:
            if model_type == "reranker":
                inputs = self.tokenizer(
                    [text], [text],
                    return_tensors="np",
                    padding="max_length", truncation=True,
                    max_length=self.max_length,
                )
            else:
                inputs = self.tokenizer(
                    [text],
                    return_tensors="np",
                    padding="max_length", truncation=True,
                    max_length=self.max_length,
                )

            feed: dict[str, np.ndarray] = {}
            for name in input_names:
                if name in inputs:
                    feed[name] = inputs[name]
                elif name == "token_type_ids":
                    feed[name] = np.zeros_like(inputs["input_ids"])
            self._feeds.append(feed)

    def get_next(self) -> dict[str, np.ndarray] | None:
        """다음 캘리브레이션 배치를 반환. 끝이면 None."""
        if self._idx >= len(self._feeds):
            return None
        feed = self._feeds[self._idx]
        self._idx += 1
        return feed

    def rewind(self) -> None:
        """캘리브레이션 리더를 처음으로 되감는다."""
        self._idx = 0


def _get_layer_node_names(
    model_path: Path,
    layer_indices: list[int],
) -> list[str]:
    """ONNX 모델에서 특정 레이어 인덱스에 해당하는 노드 이름을 추출한다.

    Raw ONNX export 노드 패턴 (슬래시 구분):
    - /0/auto_model/encoder/layer.{idx}/attention/...
    ORT-optimized 노드 패턴 (점 구분):
    - roberta.encoder.layer.{idx}.attention...
    """
    import onnx

    # 노드 이름만 필요하므로 가중치 데이터 로드 불필요 (>2GB external data 안전)
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

    excluded_nodes: list[str] = []
    for node in model.graph.node:
        node_name = node.name or ""
        for pattern in layer_patterns:
            if pattern in node_name:
                excluded_nodes.append(node_name)
                break

    return excluded_nodes


def quantize_qdq(
    fp32_dir: Path,
    qdq_dir: Path,
    *,
    sensitive_layers: list[int] | None = None,
    per_channel: bool = True,
    use_static: bool = False,
    calibration_texts: list[str] | None = None,
    model_type: str = "embedding",
    label: str,
    overwrite: bool = False,
) -> dict[str, Any] | None:
    """FP32 ONNX 모델에 QDQ 선택적 양자화를 적용한다.

    민감 레이어(sensitive_layers)의 노드는 양자화에서 제외하여
    FP32를 유지하고, 나머지는 INT8로 양자화한다.

    Args:
        per_channel: 채널별 독립 scale/zero_point 사용 (정밀도 향상).
        use_static: True면 static quantization + entropy calibration 사용.
        calibration_texts: static quantization 캘리브레이션 텍스트.
        model_type: "embedding" 또는 "reranker".

    Returns:
        양자화 결과 통계 dict 또는 실패 시 None.
    """
    # raw ONNX export에서 양자화해야 함 (ORT-fused node는 양자화 불가)
    # 따라서 fp32_dir은 ONNX export 디렉토리를 가리켜야 함
    fp32_path = fp32_dir / "model.onnx"
    if not fp32_path.exists():
        # ORT-optimized 모델 fallback
        fp32_path = fp32_dir / "model_optimized.onnx"

    qdq_path = qdq_dir / "model_optimized.onnx"

    if qdq_path.exists() and not overwrite:
        print(f"    [{label}] 이미 존재 (건너뜀): {qdq_dir.name}/")
        return {}

    if not fp32_path.exists():
        print(f"    [{label}] FP32 모델 없음: {fp32_path}")
        return None

    if sensitive_layers is None:
        sensitive_layers = DEFAULT_SENSITIVE_LAYERS

    quant_mode = "static (entropy)" if use_static else "dynamic"
    print(f"    [{label}] QDQ 선택적 양자화 중... (per_channel={per_channel}, {quant_mode})")
    print(f"    민감 레이어 (FP32 유지): {sensitive_layers}")

    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        qdq_dir.mkdir(parents=True, exist_ok=True)

        # External data 처리: >2GB 모델은 model.onnx + model.onnx_data로 분리됨
        # quantize_dynamic이 안정적으로 로드하도록 단일 파일 통합
        quantize_input = fp32_path
        consolidated_path: Path | None = None

        ext_data = fp32_path.parent / (fp32_path.name + "_data")
        if not ext_data.exists():
            ext_data = fp32_path.parent / (fp32_path.name + ".data")

        if ext_data.exists():
            import onnx as _onnx

            total_size = fp32_path.stat().st_size + ext_data.stat().st_size
            print(f"    External data 감지: {ext_data.name} (총 {_format_size(total_size)})")

            if total_size < 2 * 1024**3:
                # 2GB 미만: 단일 파일로 통합하여 양자화 안정성 확보
                consolidated_path = qdq_dir / "_source_fp32.onnx"
                _model = _onnx.load(str(fp32_path))
                _onnx.save(_model, str(consolidated_path))
                del _model
                quantize_input = consolidated_path
                print(f"    단일 파일 통합: {_format_size(consolidated_path.stat().st_size)}")
            else:
                print("    2GB 이상: 원본 경로 사용 (절대 경로 기반)")

        # 민감 레이어 노드 이름 추출
        excluded_nodes = _get_layer_node_names(quantize_input, sensitive_layers)
        print(f"    제외 노드 수: {len(excluded_nodes)}")

        if use_static:
            # Static quantization + Entropy calibration (Step 3)
            from onnxruntime.quantization import CalibrationMethod
            from onnxruntime.quantization import quantize_static as _quantize_static

            cal_texts = calibration_texts or CALIBRATION_TEXTS
            print(f"    Entropy calibration: {len(cal_texts)}개 텍스트")

            cal_reader = LegalTextCalibrationReader(
                fp32_dir, cal_texts, model_type=model_type,
            )

            _quantize_static(
                model_input=str(quantize_input),
                model_output=str(qdq_path),
                calibration_data_reader=cal_reader,
                calibrate_method=CalibrationMethod.Entropy,
                weight_type=QuantType.QInt8,
                nodes_to_exclude=excluded_nodes,
                per_channel=per_channel,
            )
        else:
            # Dynamic quantization + per-channel (Step 1)
            quantize_dynamic(
                model_input=str(quantize_input),
                model_output=str(qdq_path),
                weight_type=QuantType.QInt8,
                nodes_to_exclude=excluded_nodes,
                per_channel=per_channel,
            )

        # 통합 임시 파일 정리
        if consolidated_path and consolidated_path.exists():
            consolidated_path.unlink()
            print("    통합 임시 파일 정리 완료")

        if not qdq_path.exists():
            print(f"    [{label}] 양자화 실패: 출력 파일 없음")
            return None

        # 크기 비교
        fp32_size = fp32_path.stat().st_size
        # 외부 데이터 파일이 있으면 합산
        data_file = fp32_dir / (fp32_path.name + ".data")
        if data_file.exists():
            fp32_size += data_file.stat().st_size

        qdq_size = qdq_path.stat().st_size
        qdq_data_file = qdq_dir / (qdq_path.name + ".data")
        if qdq_data_file.exists():
            qdq_size += qdq_data_file.stat().st_size

        compression = (1 - qdq_size / fp32_size) * 100 if fp32_size > 0 else 0

        print(f"    FP32: {_format_size(fp32_size)} → QDQ INT8: {_format_size(qdq_size)} ({compression:.1f}% 압축)")
        print(f"    [{label}] QDQ 양자화 완료: {qdq_dir.name}/")

        return {
            "sensitive_layers": sensitive_layers,
            "excluded_nodes_count": len(excluded_nodes),
            "fp32_size": fp32_size,
            "qdq_size": qdq_size,
            "compression_pct": compression,
            "per_channel": per_channel,
            "use_static": use_static,
        }

    except ImportError:
        print(f"    [{label}] onnxruntime.quantization 미설치")
        return None
    except Exception as e:
        print(f"    [{label}] QDQ 양자화 실패: {e!s:.200s}")
        return None


# ============================================================
# Step 3d: Layer-wise Sensitivity Sweep
# ============================================================


def run_sensitivity_sweep(
    fp32_dir: Path,
    *,
    num_layers: int = NUM_TRANSFORMER_LAYERS,
    label: str,
    model_type: str = "embedding",
    top_n: int = 8,
) -> list[int]:
    """각 레이어를 개별 INT8 양자화하며 cosine 변화를 측정한다.

    레이어 i만 INT8, 나머지 FP32 → cosine 측정 (num_layers회 반복).
    cosine 하락이 큰 상위 top_n개를 민감 레이어로 반환한다.

    Q-BERT 논문: BERT 계열은 입출력 외에도 중간 attention-heavy
    레이어(10-15번)가 양자화에 민감할 수 있음.

    Args:
        fp32_dir: raw ONNX export 디렉토리 (model.onnx 포함).
        num_layers: 트랜스포머 레이어 수 (XLM-RoBERTa Large = 24).
        label: 출력 레이블.
        model_type: "embedding" 또는 "reranker".
        top_n: FP32로 유지할 민감 레이어 수.

    Returns:
        민감 레이어 인덱스 리스트 (레이어 순서, top_n개).
    """
    import tempfile

    fp32_path = fp32_dir / "model.onnx"
    if not fp32_path.exists():
        fp32_path = fp32_dir / "model_optimized.onnx"

    if not fp32_path.exists():
        print(f"    [{label}] FP32 모델 없음: {fp32_dir}")
        return []

    print(f"\n    [{label}] Sensitivity sweep 시작 ({num_layers} 레이어, top_n={top_n})")

    from onnxruntime.quantization import QuantType, quantize_dynamic

    # PyTorch 참조를 한 번만 계산 (24회 재사용)
    print("    PyTorch 참조 계산 중...")
    if model_type == "embedding":
        pt_ref = _encode_pytorch(TEST_TEXTS)
    else:
        pt_ref = _rerank_pytorch(RR_QUERY, RR_DOCUMENTS)

    layer_cosines: list[tuple[int, float]] = []
    sweep_start = time.perf_counter()

    for layer_idx in range(num_layers):
        # layer_idx만 FP32 유지, 나머지 모두 INT8 양자화
        # → 출력 ~700MB (대부분 INT8), 디스크 효율적
        # → "이 레이어를 FP32로 유지하면 품질이 얼마나 올라가는가" 직접 측정
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

            # 토크나이저 복사 (추론에 필요)
            _copy_tokenizer_and_config(fp32_dir, Path(tmp_dir))

            # ONNX 추론 → cosine 측정
            if model_type == "embedding":
                onnx_out = _encode_onnx(
                    TEST_TEXTS, Path(tmp_dir), "model_optimized.onnx",
                )
                cosine = _cosine_similarity_pairwise(pt_ref, onnx_out)
            else:
                onnx_out = _rerank_onnx(
                    RR_QUERY, RR_DOCUMENTS,
                    Path(tmp_dir), "model_optimized.onnx",
                )
                dot = np.dot(pt_ref, onnx_out)
                norm_pt = np.linalg.norm(pt_ref)
                norm_onnx = np.linalg.norm(onnx_out)
                cosine = (
                    float(dot / (norm_pt * norm_onnx))
                    if norm_pt > 0 and norm_onnx > 0 else 0.0
                )

            layer_cosines.append((layer_idx, cosine))
            delta = 1.0 - cosine
            print(f"      Layer {layer_idx:2d}: cosine={cosine:.6f}  delta={delta:.6f}")

    sweep_time = time.perf_counter() - sweep_start

    # cosine 내림차순 정렬 (FP32 유지 시 효과가 큰 레이어 = 가장 민감)
    layer_cosines.sort(key=lambda x: x[1], reverse=True)

    # 결과 요약
    print(f"\n    [{label}] Sensitivity sweep 결과 ({sweep_time:.1f}초):")
    print("    (각 행: 해당 레이어만 FP32, 나머지 INT8)")
    print(f"    {'순위':>4s}  {'Layer':>5s}  {'Cosine':>10s}  {'Delta':>10s}")
    print(f"    {'-' * 35}")

    for rank, (idx, cos) in enumerate(layer_cosines, 1):
        delta = 1.0 - cos
        marker = " ← FP32 유지" if rank <= top_n else ""
        print(f"    {rank:4d}  {idx:5d}  {cos:10.6f}  {delta:10.6f}{marker}")

    sensitive = sorted([idx for idx, _ in layer_cosines[:top_n]])
    print(f"\n    민감 레이어 (상위 {top_n}개): {sensitive}")

    # 메모리 정리
    del pt_ref
    gc.collect()

    return sensitive


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
    use_static_quantize: bool = False,
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

    # Step 1b: onnx-simplifier
    print(f"\n  {SUBSEPARATOR}")
    print("  Step 1b: ONNX Graph Surgery (onnx-simplifier)")
    print(f"  {SUBSEPARATOR}")

    simplify_stats = simplify_onnx_model(
        EMB_ONNX_EXPORT_DIR,
        label="임베딩",
        overwrite=overwrite,
    )
    results["_emb_simplify_stats"] = simplify_stats or {}

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

    # Step 3b: Static Shape 변환 (batch=1, seq=128)
    print(f"\n  {SUBSEPARATOR}")
    print("  Step 3b: Static Shape 변환 (batch=1, seq=128)")
    print(f"  {SUBSEPARATOR}")

    static_ok = convert_to_static_shape(
        EMB_ORT_OPT_DIR, EMB_ORT_OPT_STATIC128_DIR,
        label="임베딩 Static128",
        overwrite=overwrite,
    )

    if static_ok:
        copied = _copy_tokenizer_and_config(EMB_ONNX_EXPORT_DIR, EMB_ORT_OPT_STATIC128_DIR)
        print(f"    토크나이저 파일 복사: {copied}개")

    # Step 3b-2: Static Shape 변환 (batch=1, seq=64)
    print(f"\n  {SUBSEPARATOR}")
    print("  Step 3b-2: Static Shape 변환 (batch=1, seq=64)")
    print(f"  {SUBSEPARATOR}")

    static64_ok = convert_to_static_shape(
        EMB_ORT_OPT_DIR, EMB_ORT_OPT_STATIC64_DIR,
        seq_length=EMB_STATIC64_SEQ_LENGTH,
        label="임베딩 Static64",
        overwrite=overwrite,
    )

    if static64_ok:
        copied = _copy_tokenizer_and_config(EMB_ONNX_EXPORT_DIR, EMB_ORT_OPT_STATIC64_DIR)
        print(f"    토크나이저 파일 복사: {copied}개")

    # Step 3c: QDQ 선택적 양자화 (raw ONNX에서 양자화)
    print(f"\n  {SUBSEPARATOR}")
    print("  Step 3c: QDQ 선택적 양자화 (민감 레이어 FP32 유지)")
    print(f"  {SUBSEPARATOR}")

    # 재실행 시 tmp 디렉토리가 이미 rename되었을 수 있으므로 fallback
    emb_qdq_source = EMB_ONNX_EXPORT_DIR
    emb_raw_fallback = DATA_MODELS_DIR / "kure-v1-onnx"
    if not emb_qdq_source.exists() and emb_raw_fallback.exists():
        emb_qdq_source = emb_raw_fallback
        print(f"    QDQ 소스 fallback: {emb_raw_fallback.name}/")

    qdq_stats = quantize_qdq(
        emb_qdq_source, EMB_ORT_OPT_QDQ_DIR,
        label="임베딩 QDQ",
        use_static=use_static_quantize,
        model_type="embedding",
        overwrite=overwrite,
    )
    if qdq_stats is not None and qdq_stats:
        copied = _copy_tokenizer_and_config(emb_qdq_source, EMB_ORT_OPT_QDQ_DIR)
        print(f"    토크나이저 파일 복사: {copied}개")

    # Step 3c-2: Static64 기반 QDQ 양자화
    if static64_ok:
        print(f"\n  {SUBSEPARATOR}")
        print("  Step 3c-2: Static64 + QDQ 양자화 (민감 레이어 FP32 유지)")
        print(f"  {SUBSEPARATOR}")

        qdq_static64_stats = quantize_qdq(
            emb_qdq_source, EMB_ORT_OPT_STATIC64_QDQ_DIR,
            label="임베딩 Static64 QDQ",
            use_static=use_static_quantize,
            model_type="embedding",
            overwrite=overwrite,
        )
        if qdq_static64_stats is not None and qdq_static64_stats:
            # Static shape 변환 적용 (QDQ 모델에 static64 shape 고정)
            static64_qdq_model_path = EMB_ORT_OPT_STATIC64_QDQ_DIR / "model_optimized.onnx"
            if static64_qdq_model_path.exists():
                convert_to_static_shape(
                    EMB_ORT_OPT_STATIC64_QDQ_DIR, EMB_ORT_OPT_STATIC64_QDQ_DIR,
                    seq_length=EMB_STATIC64_SEQ_LENGTH,
                    label="임베딩 Static64 QDQ shape 고정",
                    overwrite=True,
                )
            copied = _copy_tokenizer_and_config(emb_qdq_source, EMB_ORT_OPT_STATIC64_QDQ_DIR)
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

    if EMB_ORT_OPT_STATIC128_DIR.exists():
        static_quality = verify_embedding_quality(
            EMB_ORT_OPT_STATIC128_DIR, "model_optimized.onnx",
            label="Static128",
            cosine_threshold=COSINE_THRESHOLD_FP32,
            max_diff_threshold=MAX_ABS_DIFF_THRESHOLD_FP32,
        )
        results["kure-v1-ort-opt-static128"] = _build_embedding_meta(
            fusion_stats, static_quality, "ort-optimizer-fp32-static128",
        )

    if EMB_ORT_OPT_QDQ_DIR.exists():
        qdq_quality = verify_embedding_quality(
            EMB_ORT_OPT_QDQ_DIR, "model_optimized.onnx",
            label="QDQ",
            cosine_threshold=COSINE_THRESHOLD_QDQ,
            max_diff_threshold=MAX_ABS_DIFF_THRESHOLD_QDQ,
        )
        results["kure-v1-ort-opt-qdq"] = _build_embedding_meta(
            fusion_stats, qdq_quality, "ort-optimizer-qdq-int8",
        )

    if EMB_ORT_OPT_STATIC64_DIR.exists():
        static64_quality = verify_embedding_quality(
            EMB_ORT_OPT_STATIC64_DIR, "model_optimized.onnx",
            label="Static64",
            cosine_threshold=COSINE_THRESHOLD_FP32,
            max_diff_threshold=MAX_ABS_DIFF_THRESHOLD_FP32,
        )
        results["kure-v1-ort-opt-static64"] = _build_embedding_meta(
            fusion_stats, static64_quality, "ort-optimizer-fp32-static64",
        )

    if EMB_ORT_OPT_STATIC64_QDQ_DIR.exists():
        static64_qdq_quality = verify_embedding_quality(
            EMB_ORT_OPT_STATIC64_QDQ_DIR, "model_optimized.onnx",
            label="Static64 QDQ",
            cosine_threshold=COSINE_THRESHOLD_QDQ,
            max_diff_threshold=MAX_ABS_DIFF_THRESHOLD_QDQ,
        )
        results["kure-v1-ort-opt-static64-qdq"] = _build_embedding_meta(
            fusion_stats, static64_qdq_quality, "ort-optimizer-qdq-int8-static64",
        )

    # 임시 디렉토리 → 벤치마크용 raw ONNX 보존
    print(f"\n  {SUBSEPARATOR}")
    print("  정리")
    print(f"  {SUBSEPARATOR}")
    emb_raw_dir = DATA_MODELS_DIR / "kure-v1-onnx"
    if EMB_ONNX_EXPORT_DIR.exists():
        if emb_raw_dir.exists():
            shutil.rmtree(emb_raw_dir)
        EMB_ONNX_EXPORT_DIR.rename(emb_raw_dir)
        print(f"    raw ONNX 보존: {emb_raw_dir.name}/")
    else:
        print("    raw ONNX export 없음 (이미 정리됨)")

    # 요약
    build_time = time.perf_counter() - build_start
    print(f"\n  임베딩 빌드 완료 ({build_time:.1f}초)")
    print(f"    FP32: {_format_size(_dir_total_size(EMB_ORT_OPT_DIR))}")
    if not skip_fp16 and EMB_ORT_OPT_FP16_DIR.exists():
        print(f"    FP16: {_format_size(_dir_total_size(EMB_ORT_OPT_FP16_DIR))}")
    if EMB_ORT_OPT_STATIC128_DIR.exists():
        print(f"    Static128: {_format_size(_dir_total_size(EMB_ORT_OPT_STATIC128_DIR))}")
    if EMB_ORT_OPT_STATIC64_DIR.exists():
        print(f"    Static64:  {_format_size(_dir_total_size(EMB_ORT_OPT_STATIC64_DIR))}")
    if EMB_ORT_OPT_QDQ_DIR.exists():
        print(f"    QDQ INT8:  {_format_size(_dir_total_size(EMB_ORT_OPT_QDQ_DIR))}")
    if EMB_ORT_OPT_STATIC64_QDQ_DIR.exists():
        print(f"    Static64 QDQ: {_format_size(_dir_total_size(EMB_ORT_OPT_STATIC64_QDQ_DIR))}")

    return results


def build_reranker_models(
    *,
    overwrite: bool = False,
    skip_fp16: bool = False,
    use_static_quantize: bool = False,
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

    # Step 1b: onnx-simplifier
    print(f"\n  {SUBSEPARATOR}")
    print("  Step 1b: ONNX Graph Surgery (onnx-simplifier)")
    print(f"  {SUBSEPARATOR}")

    simplify_stats = simplify_onnx_model(
        RR_ONNX_EXPORT_DIR,
        label="리랭커",
        overwrite=overwrite,
    )
    results["_rr_simplify_stats"] = simplify_stats or {}

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

    # Step 3b: QDQ 선택적 양자화 (raw ONNX에서 양자화)
    print(f"\n  {SUBSEPARATOR}")
    print("  Step 3b: QDQ 선택적 양자화 (민감 레이어 FP32 유지)")
    print(f"  {SUBSEPARATOR}")

    # 재실행 시 tmp 디렉토리가 이미 rename되었을 수 있으므로 fallback
    rr_qdq_source = RR_ONNX_EXPORT_DIR
    rr_raw_fallback = DATA_MODELS_DIR / "reranker-onnx"
    if not rr_qdq_source.exists() and rr_raw_fallback.exists():
        rr_qdq_source = rr_raw_fallback
        print(f"    QDQ 소스 fallback: {rr_raw_fallback.name}/")

    qdq_stats = quantize_qdq(
        rr_qdq_source, RR_ORT_OPT_QDQ_DIR,
        label="리랭커 QDQ",
        use_static=use_static_quantize,
        model_type="reranker",
        overwrite=overwrite,
    )
    if qdq_stats is not None and qdq_stats:
        copied = _copy_tokenizer_and_config(rr_qdq_source, RR_ORT_OPT_QDQ_DIR)
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
            max_diff_threshold=MAX_ABS_DIFF_THRESHOLD_FP16_RR,
        )
        results["reranker-ort-opt-fp16"] = _build_reranker_meta(
            fusion_stats, fp16_quality, "ort-optimizer-fp16",
        )

    if RR_ORT_OPT_QDQ_DIR.exists():
        qdq_quality = verify_reranker_quality(
            RR_ORT_OPT_QDQ_DIR, "model_optimized.onnx",
            label="QDQ",
            cosine_threshold=COSINE_THRESHOLD_QDQ,
            max_diff_threshold=MAX_ABS_DIFF_THRESHOLD_QDQ_RR,
        )
        results["reranker-ort-opt-qdq"] = _build_reranker_meta(
            fusion_stats, qdq_quality, "ort-optimizer-qdq-int8",
        )

    # 임시 디렉토리 → 벤치마크용 raw ONNX 보존
    print(f"\n  {SUBSEPARATOR}")
    print("  정리")
    print(f"  {SUBSEPARATOR}")
    rr_raw_dir = DATA_MODELS_DIR / "reranker-onnx"
    if RR_ONNX_EXPORT_DIR.exists():
        if rr_raw_dir.exists():
            shutil.rmtree(rr_raw_dir)
        RR_ONNX_EXPORT_DIR.rename(rr_raw_dir)
        print(f"    raw ONNX 보존: {rr_raw_dir.name}/")
    else:
        print("    raw ONNX export 없음 (이미 정리됨)")

    # 요약
    build_time = time.perf_counter() - build_start
    print(f"\n  리랭커 빌드 완료 ({build_time:.1f}초)")
    print(f"    FP32: {_format_size(_dir_total_size(RR_ORT_OPT_DIR))}")
    if not skip_fp16 and RR_ORT_OPT_FP16_DIR.exists():
        print(f"    FP16: {_format_size(_dir_total_size(RR_ORT_OPT_FP16_DIR))}")
    if RR_ORT_OPT_QDQ_DIR.exists():
        print(f"    QDQ INT8:  {_format_size(_dir_total_size(RR_ORT_OPT_QDQ_DIR))}")

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
            ("Static128", EMB_ORT_OPT_STATIC128_DIR, COSINE_THRESHOLD_FP32, MAX_ABS_DIFF_THRESHOLD_FP32),
            ("QDQ", EMB_ORT_OPT_QDQ_DIR, COSINE_THRESHOLD_QDQ, MAX_ABS_DIFF_THRESHOLD_QDQ),
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
            ("FP16", RR_ORT_OPT_FP16_DIR, COSINE_THRESHOLD_FP16, MAX_ABS_DIFF_THRESHOLD_FP16_RR),
            ("QDQ", RR_ORT_OPT_QDQ_DIR, COSINE_THRESHOLD_QDQ, MAX_ABS_DIFF_THRESHOLD_QDQ_RR),
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
        from importlib.metadata import version as _pkg_version
        print(f"  optimum:     {_pkg_version('optimum')}")
    except Exception:
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
        ("임베딩 S128", EMB_ORT_OPT_STATIC128_DIR),
        ("임베딩 QDQ", EMB_ORT_OPT_QDQ_DIR),
        ("리랭커 FP32", RR_ORT_OPT_DIR),
        ("리랭커 FP16", RR_ORT_OPT_FP16_DIR),
        ("리랭커 QDQ", RR_ORT_OPT_QDQ_DIR),
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
  uv run python scripts/build_optimized_onnx.py                    # 전체 빌드
  uv run python scripts/build_optimized_onnx.py --verify            # 기존 빌드 검증
  uv run python scripts/build_optimized_onnx.py --embedding-only
  uv run python scripts/build_optimized_onnx.py --reranker-only
  uv run python scripts/build_optimized_onnx.py --skip-fp16
  uv run python scripts/build_optimized_onnx.py --overwrite
  uv run python scripts/build_optimized_onnx.py --sensitivity-sweep # 레이어 민감도 분석
  uv run python scripts/build_optimized_onnx.py --static-quantize   # Static 양자화
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
    parser.add_argument(
        "--sensitivity-sweep", action="store_true",
        help="레이어별 민감도 분석 (24개 레이어 순회, cosine 측정)",
    )
    parser.add_argument(
        "--sensitivity-top-n", type=int, default=8,
        help="민감도 상위 N개 레이어를 FP32로 유지 (기본: 8)",
    )
    parser.add_argument(
        "--static-quantize", action="store_true",
        help="Static Quantization + Entropy Calibration 사용",
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

    # Sensitivity Sweep 모드
    if args.sensitivity_sweep:
        print(f"\n{SEPARATOR}")
        print("  레이어별 민감도 분석 (Sensitivity Sweep)")
        print(SEPARATOR)
        # raw ONNX에서 sweep (ORT-optimized fused node는 quantize 불가)
        emb_raw_dir = DATA_MODELS_DIR / "kure-v1-onnx"
        rr_raw_dir = DATA_MODELS_DIR / "reranker-onnx"
        if run_emb:
            if not emb_raw_dir.exists():
                print(f"\n  [ERROR] raw ONNX 모델 없음: {emb_raw_dir}")
                print("  먼저 기본 빌드를 실행하세요.")
            else:
                sensitive = run_sensitivity_sweep(
                    emb_raw_dir,
                    label="임베딩",
                    model_type="embedding",
                    top_n=args.sensitivity_top_n,
                )
                print(f"\n  → 추천 DEFAULT_SENSITIVE_LAYERS = {sensitive}")
        if run_rr:
            if not rr_raw_dir.exists():
                print(f"\n  [ERROR] raw ONNX 모델 없음: {rr_raw_dir}")
                print("  먼저 기본 빌드를 실행하세요.")
            else:
                sensitive = run_sensitivity_sweep(
                    rr_raw_dir,
                    label="리랭커",
                    model_type="reranker",
                    top_n=args.sensitivity_top_n,
                )
                print(f"\n  → 추천 sensitive_layers = {sensitive}")
        print(f"\n{SEPARATOR}")
        print("  민감도 분석 완료")
        print(SEPARATOR)
        return

    # 빌드 모드
    total_start = time.perf_counter()
    all_meta: dict[str, Any] = {}

    if run_emb:
        emb_meta = build_embedding_models(
            overwrite=args.overwrite,
            skip_fp16=args.skip_fp16,
            use_static_quantize=args.static_quantize,
        )
        all_meta.update(emb_meta)

    if run_rr:
        rr_meta = build_reranker_models(
            overwrite=args.overwrite,
            skip_fp16=args.skip_fp16,
            use_static_quantize=args.static_quantize,
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
            ("Static128", EMB_ORT_OPT_STATIC128_DIR),
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
