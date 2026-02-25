"""torch.compile 성능 미개선 원인 진단

기존 벤치마크(`benchmark_advanced_optimization.py`)에서 torch.compile이
PyTorch FP32보다 느리게 나온 원인을 분리 측정:

  1. 모델 로드 vs 컴파일 vs 추론 시간 분리
  2. 모델 재생성 오버헤드 (기존 벤치마크 방식) vs 모델 영속 방식 비교
  3. 입력 크기별 스케일링 (torch.compile 효과가 나타나는 임계점)
  4. GPU 커널 프로파일링 (CUDA event timing)

사용법:
  cd backend && uv run --no-sync python scripts/benchmark_compile_diagnosis.py
"""

import gc
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings

warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*", category=UserWarning)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

# ============================================================
# 상수
# ============================================================

SEPARATOR = "=" * 60
EMB_MODEL_NAME = "nlpai-lab/KURE-v1"
CACHE_DIR = str(PROJECT_ROOT / "data" / "models")

TEST_QUERIES = [
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

TEST_DOCUMENTS = [
    "피고는 원고에게 금 50,000,000원 및 이에 대한 지연손해금을 지급하라.",
    "임대차보증금 반환 청구 사건에서 임대인은 보증금 전액을 반환할 의무가 있다.",
    "근로기준법 제23조 제1항은 정당한 이유 없이 해고를 하지 못한다고 규정한다.",
    "민법 제750조에 의한 불법행위 손해배상 책임이 성립하려면 위법성이 있어야 한다.",
    "형법 제307조 제1항의 명예훼손죄가 성립하려면 사실을 적시하여야 한다.",
    "이혼 시 재산분할은 혼인 중 쌍방의 협력으로 이룩한 재산을 대상으로 한다.",
    "상속의 포기는 상속개시 있음을 안 날로부터 3월 내에 가정법원에 신고하여야 한다.",
    "형사소송법 제308조의2에 의하면 위법수집증거 배제법칙이 적용된다.",
    "행정소송법 제19조에 의하면 취소소송은 처분일로부터 90일 이내에 제기하여야 한다.",
    "특허법 제126조에 의한 특허권 침해금지청구에서 보호범위가 핵심 쟁점이다.",
    "자동차손해배상 보장법 제3조는 운행자의 배상책임을 규정한다.",
    "후유장해 등급 판정에서는 맥브라이드 장해평가 방법에 의한다.",
    "과실상계에서 피해자의 과실은 약한 의미의 부주의를 포함한다.",
    "국가배상법 제2조는 공무원의 직무상 불법행위에 대한 배상책임을 규정한다.",
    "채무불이행 특별손해는 채무자의 예견가능성이 있을 때 배상책임이 있다.",
    "민사소송법 제202조에 의한 자유심증주의 원칙이 적용된다.",
    "교통사고처리특례법 제4조 단서는 피해자 의사와 관계없이 공소 제기가 가능하다.",
    "자동차종합보험에서 보험회사는 피해자에게 직접 보험금을 지급할 의무가 있다.",
    "위자료 산정은 나이, 직업, 재산상태 등을 종합적으로 고려한다.",
    "불법행위 손해배상 청구권의 소멸시효는 안 날로부터 3년이다.",
]


def _measure_times(func: Any, repeat: int = 5) -> list[float]:
    """함수를 여러 번 실행하여 각 실행 시간(ms)을 반환."""
    times: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        func()
        times.append((time.perf_counter() - t0) * 1000)
    return times


# ============================================================
# 진단 1: 모델 로드 vs 컴파일 vs 추론 시간 분리
# ============================================================


def diagnose_time_breakdown() -> dict[str, Any]:
    """모델 로드, 컴파일, 추론 시간을 분리 측정."""
    import torch
    from sentence_transformers import SentenceTransformer

    print(f"\n{SEPARATOR}")
    print("  진단 1: 시간 분해 (로드 / 컴파일 / 추론)")
    print(SEPARATOR)

    texts = TEST_QUERIES + TEST_DOCUMENTS
    results: dict[str, Any] = {}

    # --- 모델 로드 시간 ---
    load_times: list[float] = []
    for i in range(3):
        t0 = time.perf_counter()
        model = SentenceTransformer(
            EMB_MODEL_NAME,
            cache_folder=CACHE_DIR,
            trust_remote_code=True,
            local_files_only=True,
        )
        load_times.append((time.perf_counter() - t0) * 1000)
        if i < 2:
            del model
            gc.collect()

    avg_load = float(np.median(load_times))
    print(f"  모델 로드:     {avg_load:.0f}ms (median of {len(load_times)})")
    results["load_ms"] = avg_load

    # --- PyTorch 추론 시간 (모델 영속) ---
    # 워밍업
    model.encode(texts[:5], show_progress_bar=False, normalize_embeddings=True)

    pt_times = _measure_times(
        lambda: model.encode(
            texts, batch_size=32, show_progress_bar=False,
            normalize_embeddings=True,
        ),
        repeat=7,
    )
    pt_median = float(np.median(pt_times))
    print(f"  PyTorch 추론:  {pt_median:.1f}ms (median), "
          f"[{', '.join(f'{t:.0f}' for t in pt_times)}]")
    results["pytorch_inference_ms"] = pt_median
    results["pytorch_inference_all"] = pt_times

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # --- torch.compile 시간 분해 ---
    for mode in ["default", "reduce-overhead", "max-autotune"]:
        print(f"\n  --- torch.compile mode={mode} ---")

        # 모델 로드
        model = SentenceTransformer(
            EMB_MODEL_NAME,
            cache_folder=CACHE_DIR,
            trust_remote_code=True,
            local_files_only=True,
        )

        # 컴파일 (래핑만, 실제 컴파일은 첫 forward에서)
        t0 = time.perf_counter()
        model[0].auto_model = torch.compile(  # type: ignore[assignment]
            model[0].auto_model, mode=mode,  # type: ignore[arg-type]
        )
        wrap_ms = (time.perf_counter() - t0) * 1000
        print(f"    torch.compile 래핑: {wrap_ms:.1f}ms")

        # 첫 번째 forward (JIT 컴파일 트리거)
        t0 = time.perf_counter()
        model.encode(texts[:5], show_progress_bar=False, normalize_embeddings=True)
        first_ms = (time.perf_counter() - t0) * 1000
        print(f"    첫 forward (JIT 컴파일): {first_ms:.0f}ms")

        # 두 번째 forward (shape 변경 시 재컴파일 가능)
        t0 = time.perf_counter()
        model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        second_ms = (time.perf_counter() - t0) * 1000
        print(f"    두 번째 forward (전체 데이터): {second_ms:.0f}ms")

        # 세 번째 이후 (안정 상태)
        compile_times = _measure_times(
            lambda: model.encode(
                texts, batch_size=32, show_progress_bar=False,
                normalize_embeddings=True,
            ),
            repeat=7,
        )
        compile_median = float(np.median(compile_times))
        print(f"    안정 추론: {compile_median:.1f}ms (median), "
              f"[{', '.join(f'{t:.0f}' for t in compile_times)}]")

        speedup = pt_median / compile_median if compile_median > 0 else 0
        print(f"    vs PyTorch: {speedup:.2f}x")

        results[f"compile_{mode}"] = {
            "wrap_ms": wrap_ms,
            "first_forward_ms": first_ms,
            "second_forward_ms": second_ms,
            "stable_inference_ms": compile_median,
            "stable_inference_all": compile_times,
            "speedup": speedup,
        }

        del model
        gc.collect()
        torch.cuda.empty_cache()

    return results


# ============================================================
# 진단 2: 기존 벤치마크 방식 재현 (모델 재생성)
# ============================================================


def diagnose_recreation_overhead() -> dict[str, Any]:
    """기존 벤치마크처럼 매 호출마다 모델 재생성하는 방식의 오버헤드 측정."""
    import torch
    from sentence_transformers import SentenceTransformer

    print(f"\n{SEPARATOR}")
    print("  진단 2: 모델 재생성 오버헤드 (기존 벤치마크 방식)")
    print(SEPARATOR)

    texts = TEST_QUERIES + TEST_DOCUMENTS
    results: dict[str, Any] = {}

    # PyTorch (매 호출 재생성)
    def pytorch_recreate() -> np.ndarray:
        m = SentenceTransformer(
            EMB_MODEL_NAME, cache_folder=CACHE_DIR,
            trust_remote_code=True, local_files_only=True,
        )
        emb = m.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        result = np.array(emb)
        del m
        gc.collect()
        return result

    pt_times = _measure_times(pytorch_recreate, repeat=5)
    pt_median = float(np.median(pt_times))
    print(f"  PyTorch (재생성): {pt_median:.0f}ms (median), "
          f"[{', '.join(f'{t:.0f}' for t in pt_times)}]")
    results["pytorch_recreate_ms"] = pt_median

    # torch.compile (매 호출 재생성)
    def compile_recreate(mode: str = "default") -> np.ndarray:
        m = SentenceTransformer(
            EMB_MODEL_NAME, cache_folder=CACHE_DIR,
            trust_remote_code=True, local_files_only=True,
        )
        m[0].auto_model = torch.compile(m[0].auto_model, mode=mode)  # type: ignore[assignment]
        emb = m.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        result = np.array(emb)
        del m
        gc.collect()
        return result

    for mode in ["default"]:
        times = _measure_times(lambda m=mode: compile_recreate(m), repeat=5)
        median = float(np.median(times))
        speedup = pt_median / median if median > 0 else 0
        print(f"  compile({mode}, 재생성): {median:.0f}ms (median), "
              f"[{', '.join(f'{t:.0f}' for t in times)}]")
        print(f"    vs PyTorch 재생성: {speedup:.2f}x")
        results[f"compile_{mode}_recreate_ms"] = median

    print(f"\n  [분석] 재생성 방식의 문제점:")
    print(f"    - 매 호출마다 모델 로드 + compile 래핑 + JIT 컴파일이 반복됨")
    print(f"    - torch.compile의 '컴파일 1회 → 추론 N회' 설계 전제가 무효화됨")

    return results


# ============================================================
# 진단 3: 입력 크기별 스케일링
# ============================================================


def diagnose_scaling() -> dict[str, Any]:
    """입력 크기를 늘려가며 torch.compile 효과 임계점 탐색."""
    import torch
    from sentence_transformers import SentenceTransformer

    print(f"\n{SEPARATOR}")
    print("  진단 3: 입력 크기별 스케일링")
    print(SEPARATOR)

    base_texts = TEST_QUERIES + TEST_DOCUMENTS  # 30건
    sizes = [10, 30, 60, 120, 240]
    results: dict[str, Any] = {}

    # PyTorch 모델 (영속)
    pt_model = SentenceTransformer(
        EMB_MODEL_NAME, cache_folder=CACHE_DIR,
        trust_remote_code=True, local_files_only=True,
    )
    pt_model.encode(base_texts[:5], show_progress_bar=False, normalize_embeddings=True)

    # Compiled 모델 (영속)
    compiled_model = SentenceTransformer(
        EMB_MODEL_NAME, cache_folder=CACHE_DIR,
        trust_remote_code=True, local_files_only=True,
    )
    compiled_model[0].auto_model = torch.compile(  # type: ignore[assignment]
        compiled_model[0].auto_model, mode="default",  # type: ignore[arg-type]
    )
    # 워밍업 (JIT 컴파일)
    compiled_model.encode(base_texts[:5], show_progress_bar=False, normalize_embeddings=True)
    compiled_model.encode(base_texts, show_progress_bar=False, normalize_embeddings=True)

    print(f"  {'N':>5s} | {'PyTorch (ms)':>12s} | {'Compiled (ms)':>13s} | {'상대 속도':>10s}")
    print(f"  {'-' * 5} | {'-' * 12} | {'-' * 13} | {'-' * 10}")

    for n in sizes:
        # 텍스트 반복으로 원하는 크기 생성
        repeat_count = max(1, n // len(base_texts) + 1)
        texts = (base_texts * repeat_count)[:n]

        pt_times = _measure_times(
            lambda t=texts: pt_model.encode(
                t, batch_size=32, show_progress_bar=False,
                normalize_embeddings=True,
            ),
            repeat=5,
        )
        pt_ms = float(np.median(pt_times))

        c_times = _measure_times(
            lambda t=texts: compiled_model.encode(
                t, batch_size=32, show_progress_bar=False,
                normalize_embeddings=True,
            ),
            repeat=5,
        )
        c_ms = float(np.median(c_times))

        speedup = pt_ms / c_ms if c_ms > 0 else 0
        marker = " ★" if speedup > 1.0 else ""
        print(f"  {n:>5d} | {pt_ms:>10.1f}ms | {c_ms:>11.1f}ms | {speedup:>8.2f}x{marker}")
        results[f"n_{n}"] = {
            "pytorch_ms": pt_ms, "compiled_ms": c_ms, "speedup": speedup,
        }

    del pt_model, compiled_model
    gc.collect()
    torch.cuda.empty_cache()

    return results


# ============================================================
# 진단 4: CUDA 커널 레벨 프로파일링
# ============================================================


def diagnose_cuda_overhead() -> dict[str, Any]:
    """CUDA 이벤트 타이밍으로 GPU 커널 오버헤드 측정."""
    import torch
    from sentence_transformers import SentenceTransformer

    if not torch.cuda.is_available():
        print("\n  진단 4: CUDA 미사용, 스킵")
        return {}

    print(f"\n{SEPARATOR}")
    print("  진단 4: CUDA 커널 오버헤드 분석")
    print(SEPARATOR)

    texts = TEST_QUERIES + TEST_DOCUMENTS
    results: dict[str, Any] = {}

    for label, use_compile in [("PyTorch", False), ("torch.compile", True)]:
        model = SentenceTransformer(
            EMB_MODEL_NAME, cache_folder=CACHE_DIR,
            trust_remote_code=True, local_files_only=True,
        )
        if use_compile:
            model[0].auto_model = torch.compile(  # type: ignore[assignment]
                model[0].auto_model, mode="default",  # type: ignore[arg-type]
            )
            # 워밍업 (JIT 컴파일)
            model.encode(texts[:5], show_progress_bar=False, normalize_embeddings=True)
            model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        else:
            model.encode(texts[:5], show_progress_bar=False, normalize_embeddings=True)

        # CUDA 이벤트 타이밍
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()

        gpu_times: list[float] = []
        wall_times: list[float] = []
        for _ in range(5):
            torch.cuda.synchronize()
            start_event.record()
            t0 = time.perf_counter()
            model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
            wall_ms = (time.perf_counter() - t0) * 1000
            end_event.record()
            torch.cuda.synchronize()
            gpu_ms = start_event.elapsed_time(end_event)
            gpu_times.append(gpu_ms)
            wall_times.append(wall_ms)

        gpu_median = float(np.median(gpu_times))
        wall_median = float(np.median(wall_times))
        overhead = wall_median - gpu_median

        print(f"  [{label}]")
        print(f"    GPU 시간:  {gpu_median:.1f}ms (median)")
        print(f"    Wall 시간: {wall_median:.1f}ms (median)")
        print(f"    Python/CPU 오버헤드: {overhead:.1f}ms ({overhead / wall_median * 100:.1f}%)")

        results[label] = {
            "gpu_ms": gpu_median,
            "wall_ms": wall_median,
            "overhead_ms": overhead,
            "gpu_times": gpu_times,
            "wall_times": wall_times,
        }

        del model
        gc.collect()
        torch.cuda.empty_cache()

    return results


# ============================================================
# 진단 5: dynamo 가드 체크 + 재컴파일 횟수
# ============================================================


def diagnose_dynamo_guards() -> dict[str, Any]:
    """torch._dynamo 가드 체크 및 재컴파일 횟수 측정."""
    import torch
    from sentence_transformers import SentenceTransformer

    print(f"\n{SEPARATOR}")
    print("  진단 5: torch._dynamo 가드/재컴파일 분석")
    print(SEPARATOR)

    texts = TEST_QUERIES + TEST_DOCUMENTS
    results: dict[str, Any] = {}

    model = SentenceTransformer(
        EMB_MODEL_NAME, cache_folder=CACHE_DIR,
        trust_remote_code=True, local_files_only=True,
    )

    # dynamo 카운터 초기화
    torch._dynamo.reset()
    compiled_auto_model = torch.compile(model[0].auto_model, mode="default")
    model[0].auto_model = compiled_auto_model  # type: ignore[assignment]

    # 여러 input shape으로 호출하여 재컴파일 관찰
    shapes_tested = [5, 10, 20, 30]
    for n in shapes_tested:
        t0 = time.perf_counter()
        model.encode(texts[:n], batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        ms = (time.perf_counter() - t0) * 1000
        print(f"    N={n:>3d}: {ms:.0f}ms")

    # 동일 shape 반복
    print("    --- 동일 shape 반복 (N=30) ---")
    for i in range(5):
        t0 = time.perf_counter()
        model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        ms = (time.perf_counter() - t0) * 1000
        print(f"    반복 {i + 1}: {ms:.0f}ms")

    # dynamo 통계 출력
    try:
        explain_output = torch._dynamo.explain(model[0].auto_model)
        print(f"\n    Graph breaks: {explain_output.graph_break_count}")
        print(f"    Graph count: {explain_output.graph_count}")
        results["graph_breaks"] = explain_output.graph_break_count
        results["graph_count"] = explain_output.graph_count
    except Exception as e:
        print(f"    dynamo.explain 실패: {e!s:.80s}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return results


# ============================================================
# 메인
# ============================================================


def main() -> None:
    """진단 실행."""
    import torch

    print(f"\n{SEPARATOR}")
    print("  torch.compile 성능 미개선 원인 진단")
    print(f"  모델: {EMB_MODEL_NAME}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(SEPARATOR)

    all_results: dict[str, Any] = {}

    # 진단 1: 시간 분해
    all_results["time_breakdown"] = diagnose_time_breakdown()

    # 진단 2: 모델 재생성 오버헤드
    all_results["recreation"] = diagnose_recreation_overhead()

    # 진단 3: 입력 크기별 스케일링
    all_results["scaling"] = diagnose_scaling()

    # 진단 4: CUDA 커널 오버헤드
    all_results["cuda"] = diagnose_cuda_overhead()

    # 진단 5: dynamo 가드/재컴파일
    all_results["dynamo"] = diagnose_dynamo_guards()

    # 종합 분석
    print(f"\n{SEPARATOR}")
    print("  종합 분석")
    print(SEPARATOR)

    tb = all_results.get("time_breakdown", {})
    pt_inf = tb.get("pytorch_inference_ms", 0)

    print(f"\n  1. 시간 구성 비교:")
    print(f"     PyTorch 추론 (영속 모델):  {pt_inf:.1f}ms")
    for mode in ["default", "reduce-overhead", "max-autotune"]:
        key = f"compile_{mode}"
        if key in tb:
            d = tb[key]
            print(f"     compile({mode}):")
            print(f"       래핑: {d['wrap_ms']:.1f}ms | "
                  f"첫 forward(JIT): {d['first_forward_ms']:.0f}ms | "
                  f"안정 추론: {d['stable_inference_ms']:.1f}ms "
                  f"({d['speedup']:.2f}x)")

    rec = all_results.get("recreation", {})
    if rec:
        print(f"\n  2. 모델 재생성 영향:")
        print(f"     PyTorch (매번 재생성): {rec.get('pytorch_recreate_ms', 0):.0f}ms")
        print(f"     PyTorch (영속):       {pt_inf:.1f}ms")
        if pt_inf > 0:
            ratio = rec.get("pytorch_recreate_ms", 0) / pt_inf
            print(f"     재생성 오버헤드:      {ratio:.1f}x (추론 대비)")
        print(f"     → 기존 벤치마크는 '로드+추론' vs '로드+컴파일+추론'을 비교한 것")

    cuda = all_results.get("cuda", {})
    if cuda:
        pt_cuda = cuda.get("PyTorch", {})
        c_cuda = cuda.get("torch.compile", {})
        if pt_cuda and c_cuda:
            print(f"\n  3. GPU vs CPU 오버헤드:")
            print(f"     PyTorch    — GPU: {pt_cuda['gpu_ms']:.1f}ms, "
                  f"오버헤드: {pt_cuda['overhead_ms']:.1f}ms")
            print(f"     Compiled   — GPU: {c_cuda['gpu_ms']:.1f}ms, "
                  f"오버헤드: {c_cuda['overhead_ms']:.1f}ms")
            if pt_cuda["gpu_ms"] > 0:
                gpu_speedup = pt_cuda["gpu_ms"] / c_cuda["gpu_ms"]
                print(f"     GPU 커널 속도 비교: {gpu_speedup:.2f}x")

    scaling = all_results.get("scaling", {})
    if scaling:
        print(f"\n  4. 입력 크기별 손익분기점:")
        found_breakeven = False
        for key in sorted(scaling.keys()):
            d = scaling[key]
            n = int(key.split("_")[1])
            marker = "★ 개선" if d["speedup"] > 1.0 else "  미개선"
            print(f"     N={n:>4d}: {d['speedup']:.2f}x  {marker}")
            if d["speedup"] > 1.0 and not found_breakeven:
                found_breakeven = True
        if not found_breakeven:
            print(f"     → 테스트 범위 내에서 torch.compile 개선 없음")

    print(f"\n{SEPARATOR}")
    print("  진단 완료")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
