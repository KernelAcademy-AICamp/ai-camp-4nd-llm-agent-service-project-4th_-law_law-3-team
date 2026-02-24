"""
ONNX 세션 싱글턴 관리

플랫폼별 최적 스레드 설정, 세션 로드, warmup을 담당한다.
Feature Flag(USE_ONNX_EMBEDDING, USE_ONNX_RERANKER)가 활성화된 경우에만 로드.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
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
    "ort-opt-qdq-6fp32": "reranker-ort-opt-qdq-6fp32",
}

# ONNX 모델 파일명 후보 (우선순위순)
_MODEL_FILE_CANDIDATES = ["model_optimized.onnx", "model.onnx"]


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
    pooling_strategy: str = "cls"  # cls | mean
    is_loaded: bool = False


# 모듈 레벨 싱글턴
_embedding_holder = OnnxSessionHolder()
_reranker_holder = OnnxSessionHolder()
_init_lock = threading.Lock()

# 런타임 비활성화 플래그 (settings 직접 변경 방지)
_onnx_embedding_disabled = False
_onnx_reranker_disabled = False


def disable_onnx_embedding() -> None:
    """ONNX 임베딩을 런타임에 비활성화한다 (PyTorch 폴백)."""
    global _onnx_embedding_disabled
    _onnx_embedding_disabled = True
    logger.warning("ONNX 임베딩 런타임 비활성화 → PyTorch 폴백")


def disable_onnx_reranker() -> None:
    """ONNX 리랭커를 런타임에 비활성화한다 (PyTorch 폴백)."""
    global _onnx_reranker_disabled
    _onnx_reranker_disabled = True
    logger.warning("ONNX 리랭커 런타임 비활성화 → PyTorch 폴백")


def is_onnx_embedding_active() -> bool:
    """ONNX 임베딩이 로드되었고 비활성화되지 않았는지 확인."""
    return _embedding_holder.is_loaded and not _onnx_embedding_disabled


def is_onnx_reranker_active() -> bool:
    """ONNX 리랭커가 로드되었고 비활성화되지 않았는지 확인."""
    return _reranker_holder.is_loaded and not _onnx_reranker_disabled


# 배치 임베딩 동시성 제어 (ORT 세션은 thread-safe하지만 메모리 폭발 방지)
_batch_semaphore = threading.Semaphore(1)

# 추론 타임아웃용 스레드풀
_inference_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="onnx-infer")


def _apply_pooling(
    raw_output: np.ndarray,
    attention_mask: np.ndarray,
    strategy: str = "cls",
) -> np.ndarray:
    """토큰 임베딩에서 문장 임베딩을 추출한다.

    Args:
        raw_output: 모델 출력 (batch, seq_len, hidden_dim)
        attention_mask: 어텐션 마스크 (batch, seq_len)
        strategy: "cls" (CLS 토큰) 또는 "mean" (평균 풀링)

    Returns:
        문장 임베딩 (batch, hidden_dim)
    """
    if strategy == "mean":
        mask = attention_mask[:, :, np.newaxis].astype(np.float32)
        summed = np.sum(raw_output * mask, axis=1)
        counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
        result: np.ndarray = summed / counts
        return result
    # cls (기본값)
    return raw_output[:, 0, :]


def _run_with_timeout(session: Any, feed: dict[str, Any]) -> list[Any]:
    """ORT 추론을 타임아웃 제한으로 실행한다.

    Args:
        session: ort.InferenceSession
        feed: 입력 텐서 딕셔너리

    Returns:
        추론 결과 리스트

    Raises:
        TimeoutError: 추론이 타임아웃 초과 시
        RuntimeError: 추론 실행 실패 시
    """
    timeout = settings.ONNX_INFERENCE_TIMEOUT_SECONDS
    future = _inference_executor.submit(session.run, None, feed)
    try:
        result: list[Any] = future.result(timeout=timeout)
        return result
    except FuturesTimeoutError:
        logger.error("ONNX 추론 타임아웃 (%.1f초 초과)", timeout)
        future.cancel()
        raise TimeoutError(f"ONNX 추론이 {timeout}초 내에 완료되지 않았습니다")


def _detect_cgroup_cpu_limit() -> int | None:
    """컨테이너 cgroup CPU 제한을 감지한다.

    cgroup v2 → v1 순으로 탐색. 제한이 없거나 읽기 실패 시 None 반환.
    """
    # cgroup v2: /sys/fs/cgroup/cpu.max → "quota period" (예: "200000 100000" = 2코어)
    try:
        cpu_max = Path("/sys/fs/cgroup/cpu.max").read_text(encoding="utf-8").strip()
        quota_str, period_str = cpu_max.split()
        if quota_str != "max":
            return max(1, int(quota_str) // int(period_str))
    except (FileNotFoundError, PermissionError, OSError, ValueError):
        pass

    # cgroup v1: /sys/fs/cgroup/cpu/cpu.cfs_quota_us + cpu.cfs_period_us
    try:
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text(encoding="utf-8").strip())
        if quota > 0:
            period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text(encoding="utf-8").strip())
            return max(1, quota // period)
    except (FileNotFoundError, PermissionError, OSError, ValueError):
        pass

    return None


def _detect_platform_config() -> PlatformConfig:
    """현재 플랫폼에 최적화된 ORT 스레드 설정을 감지한다."""
    system = platform.system()
    machine = platform.machine()
    is_mac = system == "Darwin"
    is_arm = machine in ("arm64", "aarch64")
    cpu_count = _detect_cgroup_cpu_limit() or os.cpu_count() or 4

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
            ["/usr/sbin/sysctl", "-n", "hw.perflevel0.logicalcpu"],
            capture_output=True,
            text=True,
            timeout=5,
            env={"PATH": "/usr/sbin:/usr/bin"},
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
        supported = list(variant_map.keys())
        logger.error(
            "알 수 없는 ONNX variant: '%s'\n"
            "  지원 variant: %s\n"
            "  해결: .env 파일에서 ONNX_EMBEDDING_VARIANT 또는 ONNX_RERANKER_VARIANT를 "
            "위 목록 중 하나로 설정하세요.",
            variant,
            supported,
        )
        return None
    model_dir = _MODELS_DIR / dir_name
    if not model_dir.exists():
        logger.error(
            "ONNX 모델 디렉토리 없음: %s\n"
            "  해결: scripts/build_optimized_onnx.py를 실행하여 모델을 빌드하거나, "
            "data/models/%s 디렉토리를 수동 배치하세요.",
            model_dir,
            dir_name,
        )
        return None
    return model_dir


def _verify_model_integrity(model_dir: Path, variant_key: str) -> bool:
    """ONNX 모델 파일의 무결성을 검증한다.

    Args:
        model_dir: 모델 디렉토리 경로
        variant_key: variant 식별자 (로깅용)

    Returns:
        True이면 무결성 통과
    """
    model_file = _find_model_file(model_dir)
    if model_file is None:
        logger.error("ONNX 모델 파일 없음: %s", model_dir)
        return False

    # 최소 파일 크기 검증 (1MB 미만이면 손상 가능성)
    min_size_bytes = 1_000_000
    file_size = model_file.stat().st_size
    if file_size < min_size_bytes:
        logger.error(
            "ONNX 모델 파일이 너무 작음 (손상 가능): %s (%d bytes, 최소 %d bytes)",
            model_file.name,
            file_size,
            min_size_bytes,
        )
        return False

    # model_versions.json의 verification_passed 확인 (있을 경우)
    versions_file = model_dir / "model_versions.json"
    if versions_file.exists():
        import json

        try:
            versions = json.loads(versions_file.read_text(encoding="utf-8"))
            if not versions.get("verification_passed", True):
                logger.error(
                    "ONNX 모델 검증 실패 기록: %s (model_versions.json)",
                    variant_key,
                )
                return False
        except (json.JSONDecodeError, OSError):
            pass

    return True


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

    # 그래프 최적화 레벨: 이미 오프라인 최적화된 모델이므로 런타임 재최적화 불필요
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

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

    return AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=False)  # type: ignore[no-untyped-call]


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

        if not _verify_model_integrity(model_dir, settings.ONNX_EMBEDDING_VARIANT):
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

        if not _verify_model_integrity(model_dir, settings.ONNX_RERANKER_VARIANT):
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

    inputs = holder.tokenizer(
        query,
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
        raw_outputs = _run_with_timeout(holder.session, feed)

    # Pooling + L2 정규화
    emb = _apply_pooling(raw_outputs[0], feed["attention_mask"], holder.pooling_strategy)
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

    all_embeddings: list[list[float]] = []

    with _batch_semaphore:
        for start in range(0, len(texts), batch_size):
            sub_texts = texts[start : start + batch_size]

            inputs = holder.tokenizer(
                sub_texts,
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

            if holder.use_io_binding:
                binding = holder.session.io_binding()
                for name, arr in feed.items():
                    binding.bind_cpu_input(name, arr)
                for out_name in holder.output_names:
                    binding.bind_output(out_name, "cpu")
                holder.session.run_with_iobinding(binding)
                raw_outputs = binding.copy_outputs_to_cpu()
            else:
                raw_outputs = _run_with_timeout(holder.session, feed)

            # Pooling
            emb = _apply_pooling(raw_outputs[0], feed["attention_mask"], holder.pooling_strategy)

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
        raw_outputs = _run_with_timeout(holder.session, feed)
    logits = raw_outputs[0]

    # Sigmoid 적용
    if logits.ndim == 2:
        logits = logits[:, 0]
    scores: np.ndarray = 1.0 / (1.0 + np.exp(-logits))

    return [float(s) for s in scores]


def cleanup_sessions() -> None:
    """ONNX 세션과 스레드풀을 정리한다 (서버 종료 시 호출)."""
    global _embedding_holder, _reranker_holder

    if _embedding_holder.is_loaded:
        _embedding_holder.session = None
        _embedding_holder.tokenizer = None
        _embedding_holder.is_loaded = False
        logger.info("ONNX 임베딩 세션 해제")

    if _reranker_holder.is_loaded:
        _reranker_holder.session = None
        _reranker_holder.tokenizer = None
        _reranker_holder.is_loaded = False
        logger.info("ONNX 리랭커 세션 해제")

    _inference_executor.shutdown(wait=False)
    logger.info("ONNX 추론 스레드풀 종료")


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
