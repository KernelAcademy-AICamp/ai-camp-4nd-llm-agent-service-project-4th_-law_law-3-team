"""
ONNX 세션 싱글턴 관리

플랫폼별 최적 스레드 설정, 세션 로드, warmup을 담당한다.
Feature Flag(USE_ONNX_EMBEDDING, USE_ONNX_RERANKER)가 활성화된 경우에만 로드.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)

# 모델 저장 기본 디렉토리
_MODELS_DIR = Path(__file__).parent.parent.parent.parent / "data" / "models"

# Variant → 디렉토리 이름 매핑
_EMB_VARIANT_MAP: dict[str, str] = {
    "ort-opt": "kure-v1-ort-opt",
    "ort-opt-qdq": "kure-v1-ort-opt-qdq",
}
_RR_VARIANT_MAP: dict[str, str] = {
    "ort-opt": "reranker-ort-opt",
    "ort-opt-qdq": "reranker-ort-opt-qdq",
}

# ONNX 모델 파일명 후보 (우선순위순)
_MODEL_FILE_CANDIDATES = ["model_optimized.onnx", "model.onnx"]

# Static shape variant 접미사에서 max_length 추출 정규식
_STATIC_LENGTH_PATTERN = re.compile(r"static(\d+)")


def _parse_static_length(variant: str) -> int | None:
    """variant 문자열에서 static shape의 max_length를 추출한다.

    예: "ort-opt-static128" → 128, "ort-opt-static64-qdq" → 64, "ort-opt" → None
    """
    match = _STATIC_LENGTH_PATTERN.search(variant)
    if match:
        return int(match.group(1))
    return None


@dataclass
class PlatformConfig:
    """플랫폼별 ORT 세션 설정."""

    intra_op_threads: int
    inter_op_threads: int = 1
    execution_mode: str = "sequential"  # sequential | parallel
    enable_bf16_fastmath: bool = False  # Graviton3 BF16 MMLA fastmath
    description: str = ""


@dataclass
class OnnxSessionHolder:
    """로드된 ONNX 세션과 토크나이저를 보관."""

    session: Any = None  # ort.InferenceSession
    tokenizer: Any = None  # transformers.AutoTokenizer
    model_dir: Optional[Path] = None
    variant: str = ""
    input_names: list[str] = field(default_factory=list)
    output_names: list[str] = field(default_factory=list)
    use_io_binding: bool = False
    is_loaded: bool = False


# 모듈 레벨 싱글턴
_embedding_holder = OnnxSessionHolder()
_reranker_holder = OnnxSessionHolder()
_init_lock = threading.Lock()


def _detect_platform_config() -> PlatformConfig:
    """현재 플랫폼에 최적화된 ORT 스레드 설정을 감지한다."""
    system = platform.system()
    machine = platform.machine()
    is_mac = system == "Darwin"
    is_arm = machine in ("arm64", "aarch64")
    cpu_count = os.cpu_count() or 4

    # 사용자 지정값이 있으면 우선
    if settings.ONNX_INTRA_OP_THREADS > 0:
        return PlatformConfig(
            intra_op_threads=settings.ONNX_INTRA_OP_THREADS,
            description=f"사용자 지정 ({settings.ONNX_INTRA_OP_THREADS} threads)",
        )

    # Mac M1/M3: P코어(성능코어)만 사용
    if is_mac and is_arm:
        p_cores = _detect_mac_p_cores()
        return PlatformConfig(
            intra_op_threads=p_cores,
            description=f"Mac ARM P코어 ({p_cores})",
        )

    # ARM 서버 (Graviton 등): 전체 물리코어 + BF16 fastmath 감지
    if is_arm:
        bf16_enabled = _detect_arm_bf16_support() and settings.ONNX_ENABLE_BF16_FASTMATH
        bf16_label = " + BF16 fastmath" if bf16_enabled else ""
        return PlatformConfig(
            intra_op_threads=cpu_count,
            enable_bf16_fastmath=bf16_enabled,
            description=f"ARM 서버 ({cpu_count} cores{bf16_label})",
        )

    # x86: 전체 코어
    return PlatformConfig(
        intra_op_threads=cpu_count,
        description=f"x86 ({cpu_count} cores)",
    )


def _detect_arm_bf16_support() -> bool:
    """ARM 서버에서 BF16 MMLA 명령어 지원 여부를 감지한다.

    Graviton3+ 프로세서는 /proc/cpuinfo의 Features에 bf16을 포함한다.
    Mac ARM은 이 함수 호출 전에 분기되므로 Linux ARM만 대상.
    """
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8")
        for line in cpuinfo.splitlines():
            if line.startswith("Features"):
                features = line.split(":", 1)[1].strip().split()
                return "bf16" in features
    except (FileNotFoundError, PermissionError, OSError):
        pass
    return False


def _detect_mac_p_cores() -> int:
    """Mac Silicon의 P코어(성능코어) 수를 감지한다."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.perflevel0.logicalcpu"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except (FileNotFoundError, ValueError, subprocess.TimeoutExpired):
        pass
    # M1/M3 기본값
    return 4


def _find_model_file(model_dir: Path) -> Optional[Path]:
    """디렉토리에서 ONNX 모델 파일을 찾는다."""
    for candidate in _MODEL_FILE_CANDIDATES:
        path = model_dir / candidate
        if path.exists():
            return path
    return None


def _resolve_model_dir(variant: str, variant_map: dict[str, str]) -> Optional[Path]:
    """variant 문자열을 실제 디렉토리 경로로 변환한다."""
    dir_name = variant_map.get(variant)
    if dir_name is None:
        logger.error("알 수 없는 ONNX variant: %s (지원: %s)", variant, list(variant_map.keys()))
        return None
    model_dir = _MODELS_DIR / dir_name
    if not model_dir.exists():
        logger.warning("ONNX 모델 디렉토리 없음: %s", model_dir)
        return None
    return model_dir


def _create_session(
    model_dir: Path,
    platform_config: PlatformConfig,
) -> Any:
    """ORT InferenceSession을 생성한다."""
    import onnxruntime as ort  # type: ignore[import-untyped]

    model_file = _find_model_file(model_dir)
    if model_file is None:
        raise FileNotFoundError(f"ONNX 모델 파일 없음: {model_dir}")

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = platform_config.intra_op_threads
    session_options.inter_op_num_threads = platform_config.inter_op_threads

    if platform_config.execution_mode == "sequential":
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    else:
        session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

    # 그래프 최적화 레벨: 이미 오프라인 최적화된 모델이므로 기본 유지
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Graviton3 BF16 fastmath: FP32 GEMM을 내부 BF16 MMLA로 가속
    if platform_config.enable_bf16_fastmath:
        session_options.add_session_config_entry(
            "mlas.enable_gemm_fastmath_arm64_bfloat16", "1",
        )
        logger.info("BF16 fastmath 활성화 (Graviton3 MMLA)")

    # EP 자동 감지: CUDA 사용 가능하면 우선, 아니면 CPU fallback
    available_providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in available_providers:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(
        str(model_file),
        sess_options=session_options,
        providers=providers,
    )

    active_providers = session.get_providers()
    logger.info(
        "ONNX 세션 생성: %s (providers=%s, threads=%d, mode=%s)",
        model_file.name,
        active_providers,
        platform_config.intra_op_threads,
        platform_config.execution_mode,
    )
    return session


def _load_tokenizer(model_dir: Path) -> Any:
    """토크나이저를 로드한다."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)  # type: ignore[no-untyped-call]


def load_embedding_session() -> bool:
    """임베딩 ONNX 세션을 로드한다. 성공 여부를 반환."""
    global _embedding_holder

    if not settings.USE_ONNX_EMBEDDING:
        return False

    with _init_lock:
        if _embedding_holder.is_loaded:
            return True

        model_dir = _resolve_model_dir(settings.ONNX_EMBEDDING_VARIANT, _EMB_VARIANT_MAP)
        if model_dir is None:
            return False

        try:
            platform_config = _detect_platform_config()
            session = _create_session(model_dir, platform_config)
            tokenizer = _load_tokenizer(model_dir)

            _embedding_holder.session = session
            _embedding_holder.tokenizer = tokenizer
            _embedding_holder.model_dir = model_dir
            _embedding_holder.variant = settings.ONNX_EMBEDDING_VARIANT
            _embedding_holder.input_names = [inp.name for inp in session.get_inputs()]
            _embedding_holder.output_names = [out.name for out in session.get_outputs()]
            _embedding_holder.use_io_binding = settings.ONNX_ENABLE_IO_BINDING
            _embedding_holder.is_loaded = True

            logger.info(
                "ONNX 임베딩 세션 로드 완료: variant=%s, dir=%s, io_binding=%s",
                settings.ONNX_EMBEDDING_VARIANT,
                model_dir,
                _embedding_holder.use_io_binding,
            )
            return True
        except Exception:
            logger.exception("ONNX 임베딩 세션 로드 실패")
            return False


def load_reranker_session() -> bool:
    """리랭커 ONNX 세션을 로드한다. 성공 여부를 반환."""
    global _reranker_holder

    if not settings.USE_ONNX_RERANKER:
        return False

    with _init_lock:
        if _reranker_holder.is_loaded:
            return True

        model_dir = _resolve_model_dir(settings.ONNX_RERANKER_VARIANT, _RR_VARIANT_MAP)
        if model_dir is None:
            return False

        try:
            platform_config = _detect_platform_config()
            session = _create_session(model_dir, platform_config)
            tokenizer = _load_tokenizer(model_dir)

            _reranker_holder.session = session
            _reranker_holder.tokenizer = tokenizer
            _reranker_holder.model_dir = model_dir
            _reranker_holder.variant = settings.ONNX_RERANKER_VARIANT
            _reranker_holder.input_names = [inp.name for inp in session.get_inputs()]
            _reranker_holder.output_names = [out.name for out in session.get_outputs()]
            _reranker_holder.use_io_binding = settings.ONNX_ENABLE_IO_BINDING
            _reranker_holder.is_loaded = True

            logger.info(
                "ONNX 리랭커 세션 로드 완료: variant=%s, dir=%s, io_binding=%s",
                settings.ONNX_RERANKER_VARIANT,
                model_dir,
                _reranker_holder.use_io_binding,
            )
            return True
        except Exception:
            logger.exception("ONNX 리랭커 세션 로드 실패")
            return False


def warmup_embedding() -> None:
    """임베딩 세션 warmup (JIT 컴파일 트리거)."""
    if not _embedding_holder.is_loaded:
        return
    try:
        encode_embedding_onnx("warm-up")
        logger.info("ONNX 임베딩 warmup 완료")
    except Exception:
        logger.exception("ONNX 임베딩 warmup 실패")


def warmup_reranker() -> None:
    """리랭커 세션 warmup (JIT 컴파일 트리거)."""
    if not _reranker_holder.is_loaded:
        return
    try:
        predict_reranker_onnx("warm-up", ["warm-up"])
        logger.info("ONNX 리랭커 warmup 완료")
    except Exception:
        logger.exception("ONNX 리랭커 warmup 실패")


def is_embedding_onnx_loaded() -> bool:
    """임베딩 ONNX 세션이 로드되었는지 확인."""
    return _embedding_holder.is_loaded


def is_reranker_onnx_loaded() -> bool:
    """리랭커 ONNX 세션이 로드되었는지 확인."""
    return _reranker_holder.is_loaded


def encode_embedding_onnx(query: str) -> list[float]:
    """ONNX 세션으로 임베딩 벡터를 생성한다.

    CLS pooling + L2 정규화 적용 (KURE-v1 기본 설정).

    Args:
        query: 검색 쿼리 텍스트

    Returns:
        임베딩 벡터 (float 리스트)

    Raises:
        RuntimeError: ONNX 세션이 로드되지 않은 경우
    """
    holder = _embedding_holder
    if not holder.is_loaded:
        raise RuntimeError("ONNX 임베딩 세션이 로드되지 않았습니다")

    # Static shape variant: 고정 길이 패딩 (예: static64 → 64, static128 → 128)
    static_length = _parse_static_length(holder.variant)
    inputs = holder.tokenizer(
        query,
        return_tensors="np",
        padding="max_length" if static_length else True,
        truncation=True,
        max_length=static_length if static_length else 512,
    )

    feed: dict[str, Any] = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
    }
    if "token_type_ids" in holder.input_names:
        feed["token_type_ids"] = inputs.get(
            "token_type_ids",
            np.zeros_like(inputs["input_ids"]),
        )

    # IO Binding: numpy→ORT 데이터 복사 오버헤드 제거
    if holder.use_io_binding:
        binding = holder.session.io_binding()
        for name, arr in feed.items():
            binding.bind_cpu_input(name, arr)
        for out_name in holder.output_names:
            binding.bind_output(out_name, "cpu")
        holder.session.run_with_iobinding(binding)
        raw_outputs = binding.copy_outputs_to_cpu()
    else:
        raw_outputs = holder.session.run(None, feed)

    # CLS pooling (index 0) + L2 정규화
    emb = raw_outputs[0][:, 0, :]
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.clip(norm, a_min=1e-9, a_max=None)

    result: list[float] = emb[0].tolist()
    return result


def encode_embedding_onnx_batch(
    texts: list[str],
    batch_size: int = 32,
    normalize: bool = True,
) -> list[list[float]]:
    """ONNX 세션으로 텍스트 배치의 임베딩 벡터를 생성한다.

    인제스트 파이프라인용 배치 함수. CLS pooling + L2 정규화 적용.

    Args:
        texts: 텍스트 목록
        batch_size: 서브배치 크기 (OOM 방지)
        normalize: L2 정규화 적용 여부

    Returns:
        임베딩 벡터 목록 (각 벡터는 float 리스트)

    Raises:
        RuntimeError: ONNX 세션이 로드되지 않은 경우
    """
    holder = _embedding_holder
    if not holder.is_loaded:
        raise RuntimeError("ONNX 임베딩 세션이 로드되지 않았습니다")

    static_length = _parse_static_length(holder.variant)
    all_embeddings: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        sub_texts = texts[start : start + batch_size]

        inputs = holder.tokenizer(
            sub_texts,
            return_tensors="np",
            padding="max_length" if static_length else True,
            truncation=True,
            max_length=static_length if static_length else 512,
        )

        feed: dict[str, Any] = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        if "token_type_ids" in holder.input_names:
            feed["token_type_ids"] = inputs.get(
                "token_type_ids",
                np.zeros_like(inputs["input_ids"]),
            )

        if holder.use_io_binding:
            binding = holder.session.io_binding()
            for name, arr in feed.items():
                binding.bind_cpu_input(name, arr)
            for out_name in holder.output_names:
                binding.bind_output(out_name, "cpu")
            holder.session.run_with_iobinding(binding)
            raw_outputs = binding.copy_outputs_to_cpu()
        else:
            raw_outputs = holder.session.run(None, feed)

        # CLS pooling (index 0)
        emb = raw_outputs[0][:, 0, :]

        if normalize:
            norm = np.linalg.norm(emb, axis=1, keepdims=True)
            emb = emb / np.clip(norm, a_min=1e-9, a_max=None)

        all_embeddings.extend(emb.tolist())

    return all_embeddings


def predict_reranker_onnx(query: str, documents: list[str]) -> list[float]:
    """ONNX 세션으로 리랭킹 점수를 계산한다.

    Args:
        query: 검색 쿼리
        documents: 문서 텍스트 목록

    Returns:
        sigmoid 적용된 점수 목록 (0~1)

    Raises:
        RuntimeError: ONNX 세션이 로드되지 않은 경우
    """
    holder = _reranker_holder
    if not holder.is_loaded:
        raise RuntimeError("ONNX 리랭커 세션이 로드되지 않았습니다")

    queries = [query] * len(documents)
    inputs = holder.tokenizer(
        queries,
        documents,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=512,
    )

    feed: dict[str, Any] = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
    }
    if "token_type_ids" in holder.input_names:
        feed["token_type_ids"] = inputs.get(
            "token_type_ids",
            np.zeros_like(inputs["input_ids"]),
        )

    # IO Binding: numpy→ORT 데이터 복사 오버헤드 제거
    if holder.use_io_binding:
        binding = holder.session.io_binding()
        for name, arr in feed.items():
            binding.bind_cpu_input(name, arr)
        for out_name in holder.output_names:
            binding.bind_output(out_name, "cpu")
        holder.session.run_with_iobinding(binding)
        raw_outputs = binding.copy_outputs_to_cpu()
    else:
        raw_outputs = holder.session.run(None, feed)
    logits = raw_outputs[0]

    # Sigmoid 적용
    if logits.ndim == 2:
        logits = logits[:, 0]
    scores: np.ndarray = 1.0 / (1.0 + np.exp(-logits))

    return [float(s) for s in scores]


def get_platform_info() -> dict[str, Any]:
    """현재 플랫폼 정보와 ONNX 세션 상태를 반환한다."""
    config = _detect_platform_config()
    return {
        "platform": platform.system(),
        "machine": platform.machine(),
        "is_arm": platform.machine() in ("arm64", "aarch64"),
        "cpu_count": os.cpu_count(),
        "intra_op_threads": config.intra_op_threads,
        "thread_description": config.description,
        "embedding_loaded": _embedding_holder.is_loaded,
        "embedding_variant": _embedding_holder.variant,
        "reranker_loaded": _reranker_holder.is_loaded,
        "reranker_variant": _reranker_holder.variant,
    }
