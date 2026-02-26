# 리랭커 ONNX 최적화 벤치마크 보고서

> 생성일: 2026-02-24
> 모델: `dragonkue/bge-reranker-v2-m3-ko` (XLM-RoBERTa Large, 568M params, 24 layers)
> 환경: Mac ARM (Apple Silicon), ORT 1.23.2, CPU EP
> 테스트: 쿼리 1개 × 문서 15건, 10회 반복 (중앙값)

## 1. 요약

| Variant | Latency | vs PyTorch | Pearson | 권장 |
|---------|--------:|-----------:|--------:|:----:|
| PyTorch FP32 (baseline) | 3238ms | 1.00x | — | |
| **ORT-opt FP32** | 1815ms | **1.78x** | 1.0000 | 무손실 |
| **QDQ INT8 (4 FP32)** | 920ms | **3.52x** | 0.9999 | **최적** |

**최종 권장**: `ort-opt-qdq` (4 FP32 레이어 [2,6,7,15]) — 품질(Pearson 0.9999) + 속도(3.52x) 최적 밸런스

## 2. ONNX Export

### 2.1 Export 방식

`optimum` 라이브러리가 `transformers 5.0.0`과 호환되지 않아 (`get_parameter_dtype` import 에러), `torch.onnx.export`로 직접 변환.

| 항목 | 값 |
|------|-----|
| Export 방식 | `torch.onnx.export(dynamo=False)` (레거시 TorchScript tracer) |
| Opset | 14 |
| 동적 축 | batch, seq_len (입출력 모두) |
| 외부 데이터 | 단일 파일 (`model.onnx.data`, 2.12GB) |

> **주의**: PyTorch 2.10의 기본 dynamo 기반 exporter는 shape inference 에러를 유발하므로 `dynamo=False` 필수.

### 2.2 ORT 그래프 최적화 (Fusion)

| Fusion 타입 | 개수 | 비고 |
|------------|-----:|------|
| BiasGelu | 24 | GELU 활성화 (레이어당 1개) |
| SkipLayerNormalization | 48 | 잔차연결+정규화 (레이어당 2개) |
| LayerNormalization | 1 | 최종 출력용 |

레거시 tracer가 dynamo 대비 **SkipLayerNormalization fusion 48개 추가** 달성 (dynamo: 0개).

## 3. ORT 프로파일링 — 병목 분석

| 연산자 | 시간 | 비율 |
|--------|-----:|-----:|
| MatMul | 1778ms | 93.9% |
| BiasGelu | 41ms | 2.2% |
| SkipLayerNormalization | 14ms | 0.7% |
| FusedMatMul | 12ms | 0.6% |
| Gather | 11ms | 0.6% |
| 기타 (Transpose, Add, Where, Softmax, IsNaN) | 36ms | 2.0% |
| **총합** | **1893ms** | **100%** |

MatMul이 **93.9%** 로 절대적 병목. INT8 양자화가 MatMul 가속에 직결되므로 QDQ의 효과가 큰 구조.

## 4. Session Config / EP / IO Binding 벤치마크

### 4.1 결과 요약

| Variant | Median(ms) | vs PyTorch | Pearson | Spearman |
|---------|----------:|-----------:|--------:|---------:|
| **ORT-opt (기본 ORT_ENABLE_ALL)** | **1815** | **1.78x** | 1.0 | 1.0 |
| ORT-opt (ORT_DISABLE_ALL) | 1904 | 1.70x | 1.0 | 1.0 |
| ORT-opt (전체 Config 최적화) | 1961 | 1.65x | 1.0 | 1.0 |
| CoreML EP (ALL) | 2003 | 1.62x | 1.0 | 1.0 |
| CoreML EP (CPU+ANE) | 2094 | 1.55x | 1.0 | 1.0 |
| CoreML EP (CPUOnly) | 2108 | 1.54x | 1.0 | 1.0 |
| session.run (기본) | 2107 | 1.54x | 1.0 | 1.0 |
| IO Binding (CPU) | 2138 | 1.51x | 1.0 | 1.0 |

### 4.2 분석

- **기본 ORT_ENABLE_ALL이 최적** — 추가 Session Config 튜닝(denormal, spinning, GELU approx)은 오히려 역효과
- **CoreML EP 비효과적** — 1121개 노드 중 399개만 CoreML 지원 (35.6%), 나머지 CPU fallback으로 오버헤드 발생
- **IO Binding 비효과적** — CPU EP에서는 메모리 복사 오버헤드가 이점을 상회
- 모든 variant에서 **Pearson=1.0, Spearman=1.0** (품질 무손실)

### 4.3 임베딩 모델과 비교

| 항목 | 임베딩 (KURE-v1) | 리랭커 (bge-reranker) |
|------|:-:|:-:|
| Session Config 효과 | 역효과 | 역효과 |
| CoreML EP | 역효과 | 역효과 (35.6% 노드만 지원) |
| IO Binding | 역효과 | 역효과 |
| 기본 설정 최적 | O | O |

결론: Mac ARM CPU EP에서 **기본 ORT_ENABLE_ALL 설정이 양쪽 모델 모두에서 최적**.

## 5. QDQ INT8 민감 레이어 Sweep

### 5.1 1단계 — 개별 레이어 민감도

각 레이어를 개별적으로 FP32 유지(나머지 23개 INT8)하여 Pearson 측정.

| 순위 | Layer | Pearson | 비고 |
|-----:|------:|--------:|------|
| 1 | **6** | 0.99907 | 가장 민감 |
| 2 | **7** | 0.99904 | |
| 3 | **15** | 0.99889 | |
| 4 | **2** | 0.99885 | |
| 5 | 16 | 0.99871 | |
| 6 | 10 | 0.99870 | |
| ... | ... | ... | |
| 23 | 3 | 0.99710 | |
| 24 | 9 | 0.99692 | 가장 덜 민감 |

전체 레이어 Pearson 범위: 0.9969 ~ 0.9991 (임베딩 대비 분산이 작음).

### 5.2 2단계 — FP32 레이어 점진 확장

민감도 순으로 FP32 레이어를 늘려가며 Pearson >= 0.999 달성 여부 확인.

| FP32 수 | 레이어 목록 | Pearson | Spearman | Latency | 목표 달성 |
|---------:|:-----------|--------:|---------:|--------:|:---------:|
| **4** | **[2, 6, 7, 15]** | **0.99997** | 0.968 | **920ms** | **O** |
| 6 | [2, 6, 7, 10, 15, 16] | 0.99939 | 0.993 | 981ms | O |
| 8 | [2, 5, 6, 7, 10, 12, 15, 16] | 0.99754 | 0.986 | 1109ms | X |

**4개 FP32 레이어**만으로 Pearson >= 0.999 달성 (임베딩은 16개 필요).

### 5.3 기준선 비교

| Variant | Latency | vs PyTorch | Pearson |
|---------|--------:|-----------:|--------:|
| PyTorch FP32 | 3238ms | 1.00x | — |
| ORT-opt FP32 | 2102ms | 1.54x | 1.0000 |
| **QDQ INT8 (4 FP32)** | **920ms** | **3.52x** | **0.9999** |

QDQ INT8이 ORT-opt FP32 대비 **2.29x** 추가 가속.

### 5.4 임베딩 모델과 비교

| 항목 | 임베딩 (KURE-v1) | 리랭커 (bge-reranker) |
|------|:-:|:-:|
| Pearson >= 0.999 필요 FP32 수 | 16 / 24 | **4 / 24** |
| INT8 양자화 비율 | 33% (8개) | **83% (20개)** |
| QDQ vs ORT-opt 가속 | 1.30x | **2.29x** |
| QDQ vs PyTorch 가속 | 8.97x | **3.52x** |

리랭커가 임베딩보다 **INT8 양자화에 훨씬 덜 민감** — 4개 레이어만 FP32로 유지해도 충분.

## 6. 최종 권장 Variant

| Variant | 용도 | Latency | Pearson | 모델 크기 |
|---------|------|--------:|--------:|----------:|
| `ort-opt` | 무손실 필요 시 | 1815ms | 1.0000 | ~2.1GB |
| `ort-opt-qdq` | **프로덕션 권장** | 920ms | 0.9999 | ~544MB |

### 코드 설정

```bash
# .env
USE_ONNX_RERANKER=true
ONNX_RERANKER_VARIANT=ort-opt-qdq   # 또는 ort-opt (무손실)
```

### 민감 레이어 설정

```python
# build_optimized_onnx.py
DEFAULT_SENSITIVE_LAYERS_RR = [2, 6, 7, 15]  # 4개 FP32, 20개 INT8
```

## 7. 빌드 가이드

### 7.1 ONNX Export (torch.onnx.export)

```bash
cd backend
# transformers 5.0.0 + optimum 비호환 → torch.onnx.export 직접 사용
# dynamo=False 필수 (레거시 TorchScript tracer)
uv run --no-sync python scripts/build_optimized_onnx.py --reranker-only
```

### 7.2 벤치마크 재현

```bash
cd backend
# Session Config + CoreML + IO Binding + 프로파일링
uv run --no-sync python scripts/benchmark_reranker_optimizations.py

# 민감 레이어 sweep
uv run --no-sync python scripts/sweep_reranker_sensitive_layers.py
```

## 8. 관련 파일

| 파일 | 역할 |
|------|------|
| `scripts/build_optimized_onnx.py` | ONNX 모델 빌드 (export → optimize → quantize) |
| `scripts/benchmark_reranker_optimizations.py` | Session Config / CoreML / IO Binding 벤치마크 |
| `scripts/sweep_reranker_sensitive_layers.py` | QDQ INT8 민감 레이어 sweep |
| `app/services/rag/onnx_session.py` | ONNX 세션 로딩 (variant 매핑) |
| `benchmark_reranker_opt_results.json` | 벤치마크 결과 JSON |
| `sweep_reranker_sensitive_results.json` | Sweep 결과 JSON |
| **[`reranker-onnx-variant-test-guide.md`](./reranker-onnx-variant-test-guide.md)** | **팀원용 variant 테스트 가이드 (다운로드, .env 설정, 수동 추론)** |
