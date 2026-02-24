"""
임베딩 모델 양자화 벤치마크

FP32 (PyTorch) vs ONNX FP32 vs ONNX INT8 비교
사용법: cd backend && uv run python scripts/benchmark_embedding_quantize.py
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

QUERIES = [
    "교통사고 손해배상 판례",
    "임대차 보증금 반환 청구",
    "근로기준법 해고 부당해고",
    "이혼 재산분할 위자료",
    "명예훼손 형사 고소",
]

MODEL_NAME = "nlpai-lab/KURE-v1"
CACHE_DIR = str(PROJECT_ROOT / "data" / "models")
ONNX_DIR = PROJECT_ROOT / "data" / "models" / "kure-v1-onnx"
ONNX_INT8_DIR = PROJECT_ROOT / "data" / "models" / "kure-v1-onnx-int8"

SEPARATOR = "=" * 60


def benchmark_pytorch_fp32() -> list[float]:
    """PyTorch FP32 (현재 방식) 벤치마크."""
    print(f"\n{SEPARATOR}")
    print("  [1] PyTorch FP32 (baseline)")
    print(SEPARATOR)

    from sentence_transformers import SentenceTransformer

    t0 = time.monotonic()
    model = SentenceTransformer(
        MODEL_NAME,
        cache_folder=CACHE_DIR,
        trust_remote_code=True,
        local_files_only=True,
    )
    load_time = time.monotonic() - t0
    print(f"  모델 로드: {load_time * 1000:.0f}ms")

    # warm-up
    model.encode("테스트", show_progress_bar=False, normalize_embeddings=True)

    times = []
    for q in QUERIES:
        t0 = time.monotonic()
        emb = model.encode(q, show_progress_bar=False, normalize_embeddings=True)
        elapsed = time.monotonic() - t0
        times.append(elapsed)
        dim = len(emb)
        print(f"  {q[:20]:<20s} → {elapsed * 1000:>8.1f}ms  (dim={dim})")

    avg = sum(times) / len(times)
    print(f"  평균: {avg * 1000:.1f}ms")
    return times


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
        print(f"  ONNX 변환 완료: {ONNX_DIR}")
        return True
    except Exception as e:
        print(f"  ONNX 변환 실패: {e}")
        return False


def quantize_int8() -> bool:
    """ONNX INT8 양자화.

    양자화 후 SentenceTransformer 로드를 위해:
    1. model_quantized.onnx → model.onnx 리네임
    2. config/tokenizer 파일을 FP32 ONNX 디렉토리에서 복사
    """
    if ONNX_INT8_DIR.exists() and (ONNX_INT8_DIR / "model.onnx").exists():
        print(f"  INT8 모델 이미 존재: {ONNX_INT8_DIR}")
        return True

    print("  INT8 양자화 중...")
    try:
        import shutil

        from optimum.onnxruntime import ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig

        quantizer = ORTQuantizer.from_pretrained(str(ONNX_DIR))
        qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=True)

        ONNX_INT8_DIR.mkdir(parents=True, exist_ok=True)
        quantizer.quantize(
            save_dir=str(ONNX_INT8_DIR),
            quantization_config=qconfig,
        )

        # model_quantized.onnx → model.onnx 리네임
        # SentenceTransformer(backend="onnx")는 model.onnx를 찾음
        quantized_path = ONNX_INT8_DIR / "model_quantized.onnx"
        target_path = ONNX_INT8_DIR / "model.onnx"
        if quantized_path.exists() and not target_path.exists():
            quantized_path.rename(target_path)
            print("  model_quantized.onnx → model.onnx 리네임")

        # config/tokenizer 파일 복사 (FP32 ONNX → INT8)
        copied = 0
        for f in ONNX_DIR.iterdir():
            if f.is_file() and f.suffix in (".json", ".txt") and f.name != "ort_config.json":
                dst = ONNX_INT8_DIR / f.name
                if not dst.exists():
                    shutil.copy2(f, dst)
                    copied += 1
        if copied:
            print(f"  config/tokenizer 파일 {copied}개 복사 완료")

        print(f"  INT8 양자화 완료: {ONNX_INT8_DIR}")
        return True
    except Exception as e:
        print(f"  INT8 양자화 실패: {e}")
        return False


def benchmark_onnx(model_dir: Path, label: str) -> list[float]:
    """ONNX 모델 벤치마크."""
    print(f"\n{SEPARATOR}")
    print(f"  {label}")
    print(SEPARATOR)

    from sentence_transformers import SentenceTransformer

    # ONNX 백엔드 사용하여 로드
    # sentence-transformers >= 3.x는 backend 파라미터 지원
    try:
        t0 = time.monotonic()
        model = SentenceTransformer(
            str(model_dir),
            backend="onnx",
            trust_remote_code=True,
            local_files_only=True,
        )
        load_time = time.monotonic() - t0
        print(f"  모델 로드: {load_time * 1000:.0f}ms")
    except Exception as e:
        print(f"  ONNX 백엔드 로드 실패: {e}")
        print("  대체 방법으로 시도...")

        try:
            from optimum.onnxruntime import ORTModelForFeatureExtraction
            from transformers import AutoTokenizer

            t0 = time.monotonic()
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_dir), trust_remote_code=True
            )
            ort_model = ORTModelForFeatureExtraction.from_pretrained(
                str(model_dir), trust_remote_code=True
            )
            load_time = time.monotonic() - t0
            print(f"  모델 로드 (optimum): {load_time * 1000:.0f}ms")

            # optimum 직접 사용
            import numpy as np

            times = []
            for q in QUERIES:
                t0 = time.monotonic()
                inputs = tokenizer(q, return_tensors="pt", padding=True, truncation=True)
                outputs = ort_model(**inputs)
                # mean pooling
                token_embeddings = outputs.last_hidden_state.detach().numpy()
                attention_mask = inputs["attention_mask"].numpy()
                mask_expanded = np.broadcast_to(
                    np.expand_dims(attention_mask, -1), token_embeddings.shape
                )
                sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
                sum_mask = np.clip(np.sum(mask_expanded, axis=1), a_min=1e-9, a_max=None)
                emb = sum_embeddings / sum_mask
                # normalize
                norm = np.linalg.norm(emb, axis=1, keepdims=True)
                emb = emb / norm
                elapsed = time.monotonic() - t0
                times.append(elapsed)
                dim = emb.shape[1]
                print(f"  {q[:20]:<20s} → {elapsed * 1000:>8.1f}ms  (dim={dim})")

            avg = sum(times) / len(times)
            print(f"  평균: {avg * 1000:.1f}ms")
            return times
        except Exception as e2:
            print(f"  대체 방법도 실패: {e2}")
            return []

    # warm-up
    model.encode("테스트", show_progress_bar=False, normalize_embeddings=True)

    times = []
    for q in QUERIES:
        t0 = time.monotonic()
        emb = model.encode(q, show_progress_bar=False, normalize_embeddings=True)
        elapsed = time.monotonic() - t0
        times.append(elapsed)
        dim = len(emb)
        print(f"  {q[:20]:<20s} → {elapsed * 1000:>8.1f}ms  (dim={dim})")

    avg = sum(times) / len(times)
    print(f"  평균: {avg * 1000:.1f}ms")
    return times


def print_comparison(
    fp32_times: list[float],
    onnx_times: list[float],
    int8_times: list[float],
) -> None:
    """비교 결과 출력."""
    print(f"\n{SEPARATOR}")
    print("  비교 결과")
    print(SEPARATOR)

    def avg_ms(times: list[float]) -> float:
        return (sum(times) / len(times)) * 1000 if times else 0

    fp32_avg = avg_ms(fp32_times)
    onnx_avg = avg_ms(onnx_times)
    int8_avg = avg_ms(int8_times)

    print(f"\n  {'방식':<25s} {'평균':>10s} {'상대 속도':>12s}")
    print(f"  {'-' * 25} {'-' * 10} {'-' * 12}")

    print(f"  {'PyTorch FP32 (baseline)':<25s} {fp32_avg:>8.1f}ms {'1.0x':>12s}")

    if onnx_times:
        speedup = fp32_avg / onnx_avg if onnx_avg > 0 else 0
        print(f"  {'ONNX FP32':<25s} {onnx_avg:>8.1f}ms {speedup:>10.1f}x")

    if int8_times:
        speedup = fp32_avg / int8_avg if int8_avg > 0 else 0
        print(f"  {'ONNX INT8':<25s} {int8_avg:>8.1f}ms {speedup:>10.1f}x")

    if int8_times and fp32_avg > 0:
        saved = fp32_avg - int8_avg
        pct = saved / fp32_avg * 100
        print(f"\n  INT8 적용 시 쿼리당 {saved:.0f}ms 절약 ({pct:.0f}% 감소)")

    # 파이프라인 전체 영향 추정
    if int8_times:
        pipeline_total = 24400  # 벤치마크 기준 총 파이프라인 시간 (ms)
        embed_baseline = fp32_avg
        embed_int8 = int8_avg
        new_total = pipeline_total - embed_baseline + embed_int8
        print("\n  파이프라인 전체 영향 (추정):")
        print(f"    현재:  {pipeline_total:.0f}ms (임베딩 {embed_baseline:.0f}ms)")
        print(f"    INT8:  {new_total:.0f}ms (임베딩 {embed_int8:.0f}ms)")
        saved_total = pipeline_total - new_total
        print(f"    절약:  {saved_total:.0f}ms ({saved_total / pipeline_total * 100:.1f}%)")


def main() -> None:
    """벤치마크 메인."""
    print(f"\n{SEPARATOR}")
    print("  임베딩 모델 양자화 벤치마크")
    print(f"  모델: {MODEL_NAME}")
    print(f"  쿼리 수: {len(QUERIES)}")
    print(SEPARATOR)

    # 1. PyTorch FP32 baseline
    fp32_times = benchmark_pytorch_fp32()

    # 2. ONNX 변환
    print(f"\n{SEPARATOR}")
    print("  ONNX 모델 변환")
    print(SEPARATOR)
    onnx_ok = export_onnx()

    onnx_times: list[float] = []
    int8_times: list[float] = []

    if onnx_ok:
        # 3. ONNX FP32 벤치마크
        onnx_times = benchmark_onnx(ONNX_DIR, "[2] ONNX FP32")

        # 4. INT8 양자화
        print(f"\n{SEPARATOR}")
        print("  INT8 양자화")
        print(SEPARATOR)
        int8_ok = quantize_int8()

        if int8_ok:
            # 5. ONNX INT8 벤치마크
            int8_times = benchmark_onnx(ONNX_INT8_DIR, "[3] ONNX INT8")

    # 6. 비교
    print_comparison(fp32_times, onnx_times, int8_times)


if __name__ == "__main__":
    main()
