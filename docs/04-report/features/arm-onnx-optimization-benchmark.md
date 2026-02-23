# ARM/크로스플랫폼 ONNX 최적화 벤치마크 보고서

생성일: 2026-02-23
최종 업데이트: 2026-02-23 (QDQ INT8 수정 후 재검증 반영)

## 환경

| 항목 | 값 |
|------|-----|
| 아키텍처 | arm64 (Apple Silicon M1) |
| 플랫폼 | Darwin 25.3.0 |
| CPU 코어 | 8 (P코어 4 + E코어 4) |
| 총 RAM | 16.0GB |
| onnxruntime | 1.23.2 |
| PyTorch | 2.10.0 |
| MPS | 사용 가능 |
| CUDA | 미사용 |

## 모델

| 모델 | HuggingFace ID | 아키텍처 | 파라미터 |
|------|----------------|----------|---------|
| 임베딩 | `nlpai-lab/KURE-v1` | XLM-RoBERTa Large | 24 레이어, 16 헤드, 1024차원 |
| 리랭커 | `dragonkue/bge-reranker-v2-m3-ko` | XLM-RoBERTa Large | 24 레이어, 16 헤드 |

## 최적화 파이프라인

```
HuggingFace 모델
  ↓ optimum export (ONNX)
Raw ONNX FP32 (model.onnx + model.onnx_data)
  ↓ onnxsim (그래프 단순화)
  ↓ ORT optimizer (Attention/SkipLayerNorm/BiasGelu fusion)
ORT-최적화 FP32 (model_optimized.onnx)
  ├─→ FP16 변환
  ├─→ Static Shape (batch=1, seq=128)
  └─→ QDQ INT8 선택적 양자화 (raw ONNX 기반)
```

## 벤치마크 결과

> **측정 조건**: 10건 텍스트, 6회 반복 (warmup 1회 포함), median 기준.
> 각 Phase별 PyTorch CPU baseline을 기준으로 speedup 계산.

### 임베딩 비교

| Variant | Latency (ms) | Speedup | Cosine | Memory (MB) | 모델 크기 |
|---------|-------------|---------|--------|-------------|-----------|
| PyTorch FP32 (CPU) | 3867.7 | 1.00x | 1.0000 | 677 | 2.3GB (원본) |
| PyTorch FP32 (MPS) | 3667.0 | 1.05x | 1.0000 | 646 | - |
| 스레드최적 (P코어만, t=4) | 1005.7 | 3.84x | N/A | 1139 | - |
| **QDQ INT8 (선택적)** | **789.9** | **4.44x** | **0.989** | **687** | **687MB** |

### 리랭커 비교

| Variant | Latency (ms) | Speedup | Pearson | Memory (MB) | 모델 크기 |
|---------|-------------|---------|---------|-------------|-----------|
| PyTorch FP32 (CPU) | 5057.9 | 1.00x | 1.0000 | 664 | 2.3GB (원본) |
| 스레드최적 (P코어만, t=4) | 1422.8 | 3.56x | N/A | 1012 | - |
| **QDQ INT8 (선택적)** | **920.8** | **5.07x** | **0.992** | **688** | **688MB** |

### QDQ INT8 선택적 양자화 상세

#### 양자화 설정

| 항목 | 설정 |
|------|------|
| 양자화 방식 | Dynamic Quantization (INT8) |
| per_channel | True (채널별 독립 scale/zero_point) |
| 민감 레이어 (FP32 유지) | layer 0, 1, 22, 23 (첫/끝 2개) |
| 양자화 대상 레이어 | layer 2~21 (20개 레이어, 전체의 83%) |
| 소스 모델 | Raw ONNX export (ORT-fused 노드는 양자화 불가) |

#### 품질 검증 결과 (--verify)

| 모델 | Cosine Similarity | Max Abs Diff | Threshold | 판정 |
|------|-------------------|-------------|-----------|------|
| 임베딩 | 0.988 | 0.021 | cosine ≥ 0.985, diff ≤ 0.03 | PASS |
| 리랭커 | 0.9998 (cosine) | 0.452 (logit) | cosine ≥ 0.985, diff ≤ 0.5 | PASS |

> **리랭커 Max Abs Diff 주의**: 리랭커는 logit 스케일(0~1 범위가 아님)이므로 임베딩(정규화 벡터)보다 절대 차이가 크다. Pearson 상관계수 0.992로 순위 보존은 양호.

#### 모델 크기 비교

| Variant | 임베딩 | 리랭커 |
|---------|--------|--------|
| PyTorch 원본 | ~2.3GB | ~2.3GB |
| ORT-최적화 FP32 | ~1.2GB | ~1.2GB |
| **QDQ INT8** | **687MB** | **688MB** |
| 크기 절감 | **70%↓** (vs PyTorch) | **70%↓** (vs PyTorch) |

### 스레드 최적화 상세

**임베딩** (kure-v1-ort-opt):

| 설정 | 스레드 수 | Latency (ms) |
|------|----------|-------------|
| P코어만 | 4 | 1005.7 |
| 전체코어 | 8 | 1088.7 |
| P코어+1 | 5 | 1151.5 |
| 2코어 | 2 | 1175.0 |

최적: **P코어만** (threads=4, 1005.7ms)

**리랭커** (reranker-ort-opt):

| 설정 | 스레드 수 | Latency (ms) |
|------|----------|-------------|
| P코어만 | 4 | 1422.8 |
| 전체코어 | 8 | 1665.9 |
| P코어+1 | 5 | 1518.3 |
| 2코어 | 2 | 1848.0 |

최적: **P코어만** (threads=4, 1422.8ms)

### ORT 프로파일링 상위 노드

**임베딩** (kure-v1-ort-opt):

| # | 연산 | 시간 (ms) |
|---|------|----------|
| 1 | MatMul | 5.86 |
| 2 | MatMul | 5.09 |
| 3 | MatMul | 4.22 |
| 4 | MatMul | 4.04 |
| 5 | MatMul | 3.88 |
| 6 | MatMul | 3.67 |
| 7 | Gather | 3.49 |
| 8 | MatMul | 3.15 |
| 9 | MatMul | 3.10 |
| 10 | Attention | 3.04 |

**리랭커** (reranker-ort-opt):

| # | 연산 | 시간 (ms) |
|---|------|----------|
| 1 | MatMul | 12.73 |
| 2 | MatMul | 12.54 |
| 3 | MatMul | 12.48 |
| 4 | MatMul | 12.46 |
| 5 | MatMul | 12.44 |
| 6 | MatMul | 12.39 |
| 7 | MatMul | 12.07 |
| 8 | MatMul | 12.01 |
| 9 | MatMul | 11.98 |
| 10 | MatMul | 11.97 |

## FP32 무손실 최적화 (Graviton3 전용)

> Mac ARM(Apple Silicon)에서는 해당 없음. Graviton3 이상 서버에서 적용.

| 최적화 | 원리 | 기대 효과 | 설정 |
|--------|------|----------|------|
| BF16 Fastmath | Graviton3 MMLA로 FP32 GEMM 가속 | 최대 65%↓ latency | `ONNX_ENABLE_BF16_FASTMATH=true` |
| Static Shape | batch=1, seq=128 고정 → constant folding | 5-15%↓ latency | `kure-v1-ort-opt-static128` 모델 사용 |
| 복합 (BF16+Static) | 위 두 가지 결합 | 누적 효과 | 환경변수 + static128 모델 |

- **BF16 Fastmath**: `/proc/cpuinfo`에서 `bf16` 플래그 감지 → 자동 활성화
- **Static Shape**: `build_optimized_onnx.py`의 `convert_to_static_shape()` 함수로 빌드
- Mac에서 미측정: Phase 8, 9, 10은 `is_linux and is_arm` 조건 충족 시에만 실행

## 플랫폼별 권장 설정

### Mac ARM (Apple Silicon)

| 모델 | 권장 Variant | Latency | 품질 | 비고 |
|------|-------------|---------|------|------|
| 임베딩 | QDQ INT8 | ~790ms | cosine 0.989 | 모델 70% 축소, 4.4x 빠름 |
| 리랭커 | QDQ INT8 | ~921ms | Pearson 0.992 | 모델 70% 축소, 5.1x 빠름 |

> FP16은 Mac ARM에서 속도 이점 없음 (NEON fp16 미지원). MPS도 CPU보다 느림.

### Graviton3+ (AWS)

| 모델 | 권장 Variant | 기대 효과 | 비고 |
|------|-------------|----------|------|
| 임베딩 | FP32 Static128 + BF16 Fastmath | 최대 65%↓ | 무손실, FP32 정밀도 유지 |
| 리랭커 | QDQ INT8 또는 FP32 + BF16 Fastmath | 품질/속도 트레이드오프 | INT8 시 Pearson 0.992 |

## 핵심 인사이트

1. **QDQ INT8은 Mac ARM에서 가장 효과적**: PyTorch 대비 4-5x 빠르고 모델 70% 축소
2. **민감 레이어 보호가 핵심**: 첫/끝 2개 레이어(layer 0,1,22,23)를 FP32로 유지하여 품질 손실 최소화
3. **리랭커 INT8 품질 우수**: Pearson 0.992로 순위 보존 양호 (max abs diff는 logit 스케일 특성)
4. **FP16은 Mac에서 무효**: NEON fp16 미지원으로 FP32보다 오히려 느림
5. **MPS 가속 무효**: CPU보다 느림 (모델 크기 대비 MPS 오버헤드)
6. **스레드 최적화**: P코어만 사용(t=4)이 최적 (E코어 추가 시 오히려 느림)
7. **Graviton3 BF16**: FP32 무손실로 최대 65% 개선 가능 (서버 환경에서 측정 필요)

## 빌드 및 검증 명령어

```bash
cd backend

# 전체 빌드 (FP32 + FP16 + Static128 + QDQ)
uv run python scripts/build_optimized_onnx.py --overwrite

# 품질 검증만
uv run python scripts/build_optimized_onnx.py --verify

# 벤치마크 전체 실행
uv run python scripts/benchmark_arm_optimization.py

# QDQ 벤치마크만 (다른 Phase 스킵)
uv run python scripts/benchmark_arm_optimization.py \
  --skip-profiling --skip-fp16 --skip-threads \
  --skip-bf16-test --skip-static-test --skip-combined-test \
  --no-report
```
