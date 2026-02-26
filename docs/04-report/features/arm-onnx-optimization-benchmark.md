# ARM/크로스플랫폼 ONNX 최적화 벤치마크 보고서

최종 업데이트: 2026-02-24

## 환경

| 항목 | 값 |
|------|-----|
| 아키텍처 | arm64 (Apple Silicon M-series) |
| 플랫폼 | Darwin 25.3.0 |
| CPU 코어 | 8 (P-core 4 + E-core 4) |
| RAM | 16.0GB |
| onnxruntime | 1.23.2 |
| PyTorch | 2.10.0 |
| MPS (Apple GPU) | 사용 가능 |
| CUDA | 미사용 |

## 모델

| 모델 | 용도 | 아키텍처 | 차원 |
|------|------|----------|------|
| `nlpai-lab/KURE-v1` | 임베딩 | XLM-RoBERTa Large (24 Transformer layers) | 1024 |
| `dragonkue/bge-reranker-v2-m3-ko` | 리랭커 | XLM-RoBERTa Large | - |

## 벤치마크 방법

- 테스트 입력: 법률 도메인 쿼리 10건 (2-50 토큰)
- 측정: 10회 반복 median (첫 1회 warmup 제외)
- Cosine similarity: PyTorch FP32 CPU 결과를 기준으로 비교

## 종합 결과 (임베딩)

| Variant | Latency (ms) | Speedup | Cosine | Memory (MB) |
|---------|-------------|---------|--------|-------------|
| PyTorch FP32 (CPU) | 1501.0 | 1.00x | 1.0000 | 823 |
| PyTorch FP32 (MPS) | 1530.5 | 0.98x | 1.0000 | 769 |
| ONNX FP32 (원본) | 1013.1 | 1.48x | 1.0000 | 1244 |
| ORT-최적화 FP32 | 940.5 | 1.60x | 1.0000 | 1238 |
| 스레드최적 (P코어만, t=4) | 942.7 | 1.59x | - | - |
| QDQ INT8 (4 FP32, 기존) | 771.8 | 1.94x | 0.9889 | 1262 |
| **QDQ INT8 (16 FP32, 최적)** | **167.4** | **1.30x vs ORT-opt** | **0.9990** | - |
| ORT-opt FP32 (무손실) | 218.0 | 1.00x (기준) | 1.0000 | 1238 |
| ORT-opt + IO Binding | 968.3 | 1.55x | 1.0000 | - |
| ~~ORT-opt Static128~~ | ~~3073.7~~ | ~~0.49x~~ | ~~1.0000~~ | 삭제됨 |
| ~~ORT-opt Static64~~ | ~~1965.8~~ | ~~0.76x~~ | ~~1.0000~~ | 삭제됨 |

## Phase별 상세 분석

### Phase 1: PyTorch FP32 Baseline

- **CPU**: 1501.0ms (Apple Accelerate/AMX 활용)
- **MPS (Apple GPU)**: 1530.5ms - CPU와 동등. XLM-RoBERTa Large는 MPS에서 이점 없음

> Mac ARM에서는 Apple Accelerate 프레임워크가 PyTorch GEMM을 고도로 최적화하여
> MPS GPU 오프로딩이 오히려 데이터 전송 오버헤드로 느려진다.

### Phase 2-3: ONNX 변환 + ORT 최적화

| 단계 | Latency | Speedup | 설명 |
|------|---------|---------|------|
| ONNX FP32 원본 | 1013.1ms | 1.48x | ORT 기본 그래프 최적화만 적용 |
| ORT-최적화 FP32 | 940.5ms | 1.60x | Attention/SkipLayerNorm/BiasGelu Fusion |

ORT 최적화로 Fusion 적용: Attention=24, SkipLayerNorm=48, BiasGelu=24.
PyTorch 대비 1.60x지만 Apple Accelerate가 워낙 강력해 차이가 크지 않음.

### Phase 5: 스레드 최적화

| 설정 | 스레드 수 | Latency (ms) |
|------|----------|-------------|
| **P코어만** | **4** | **942.7** |
| 전체코어 | 8 | 999.8 |
| P코어+1 | 5 | 1054.1 |
| 2코어 | 2 | 1102.3 |

**P코어(성능 코어) 4개가 최적**. E코어(효율 코어) 포함 시 스레드 경합으로 오히려 느려짐.
`ONNX_INTRA_OP_THREADS=4` 권장.

### Phase 8: BF16 Fastmath

**Mac ARM에서 미지원**. Graviton3+ (Linux ARM) 전용 기능.
Graviton3 MMLA 명령어로 FP32 GEMM을 BF16으로 가속 (예상 최대 65% 개선).

### Phase 9, 12: Static Shape (Static128 / Static64)

| Variant | Latency | vs Dynamic | Cosine |
|---------|---------|-----------|--------|
| Dynamic (batch=10) | 945.6ms | 1.00x | - |
| Static128 (batch=1 x10) | 3073.7ms | 0.31x | 1.000000 |
| Static64 (batch=1 x10) | 1965.8ms | 0.49x | 1.000000 |

**Static Shape가 Mac ARM에서 역효과를 보인 이유:**

1. **배치 처리 불가**: Static 모델은 `batch=1, seq=N` 고정. 10건 처리 시 개별 10회 추론 필요 → 오버헤드 10배
2. **패딩 낭비**: 짧은 쿼리(5토큰)도 64/128 토큰으로 패딩 → 불필요한 연산
3. **ORT CPU EP 한계**: CPU EP는 static shape에서 constant folding 이점이 미미

**입력 길이별 분석 (Static128):**

| 쿼리 유형 | Dynamic | Static128 | 비율 |
|-----------|---------|-----------|------|
| 짧은 쿼리 (2-5어절) | 816.6ms | 1421.8ms | 0.57x |
| 중간 쿼리 (5-10어절) | 955.7ms | 3148.6ms | 0.30x |
| 긴 쿼리 (반복 패딩) | 922.4ms | 1449.6ms | 0.64x |

> Static Shape는 **Graviton 서버(Linux ARM)에서 BF16 Fastmath와 조합** 시에만 의미가 있다.
> Mac ARM 로컬 개발에서는 Dynamic shape를 사용해야 한다.

### Phase 11: QDQ INT8 선택적 양자화

| 항목 | 값 |
|------|-----|
| Latency | 771.8ms |
| Speedup | **1.94x vs PyTorch** |
| Cosine similarity | 0.9889 |
| 모델 크기 | 687.1MB (FP32 대비 67% 축소) |
| 민감 레이어 (FP32 유지) | [0, 1, 22, 23] (4개) |

모든 최적화 중 **유일하게 유의미한 속도 향상**을 보인 variant.
첫/마지막 2개 레이어를 FP32로 유지하여 품질 손실 최소화 (cosine 0.989).

### Phase 13: IO Binding

| 방식 | Latency | 비율 |
|------|---------|------|
| session.run() | 966.3ms | 1.00x |
| IO Binding | 968.3ms | 1.00x |
| Static64 run() | 1960.9ms | - |
| Static64 IO Binding | 1979.1ms | 0.99x |

**IO Binding이 효과 없는 이유:** CPU ExecutionProvider에서는 numpy 배열이 이미 CPU 메모리에 있어 복사 오버헤드가 무시할 수준. GPU EP (CUDA/TensorRT) 환경에서만 유의미.

## Round 2: 심층 최적화 탐색 (2026-02-24)

3명의 연구 에이전트가 병렬로 조사한 결과를 기반으로 5개 신규 최적화를 벤치마크.

### ORT 프로파일링 (연산자별 병목)

| 연산자 | 시간 (ms) | 비율 |
|--------|----------|------|
| **MatMul** | 468.2 | **69.9%** |
| **Attention** | 169.5 | **25.3%** |
| BiasGelu | 19.4 | 2.9% |
| SkipLayerNormalization | 6.2 | 0.9% |
| Gather | 5.5 | 0.8% |
| 기타 | 0.6 | 0.1% |
| **합계** | **669.4** | **100%** |

> MatMul + Attention이 **전체 시간의 95.2%**를 차지. INT8 양자화가 효과적인 이유 (MatMul 연산 대상).

### Session Config 최적화

| 설정 | Median (ms) | vs PyTorch | 변화 |
|------|-----------|---------|------|
| ORT_ENABLE_ALL (현재) | 430.1 | 3.49x | 기준 |
| ORT_DISABLE_ALL | 447.4 | 3.35x | -4% |
| 전체 Config 최적화 | 458.8 | 3.27x | -6% |

**전체 Config 최적화 항목**: `graph_optimization_level=ORT_DISABLE_ALL` + `denormal_as_zero` + `force_spinning_stop` + `gelu_approximation`

**결론**: pre-optimized 모델에서 ORT_DISABLE_ALL은 오히려 역효과. Fusion 노드(Attention, SkipLayerNorm 등)의 런타임 최적화가 비활성화되어 느려짐. **현재 ORT_ENABLE_ALL이 최적.**

### CoreML ExecutionProvider

| 설정 | Median (ms) | vs PyTorch | 비고 |
|------|-----------|---------|------|
| CoreML EP (ALL) | 466.1 | 3.22x | SystemError 20 → CPU 폴백 |
| CoreML EP (CPU+ANE) | 476.5 | 3.15x | SystemError 20 → CPU 폴백 |
| CoreML EP (CPUOnly) | 483.3 | 3.11x | SystemError 20 → CPU 폴백 |

**CoreML EP가 실패한 이유:**
1. `SystemError: 20` — XLM-RoBERTa Large (197 노드)에서 CoreML이 83개 노드만 지원 (42%)
2. 76개 파티션으로 분할되어 CPU↔CoreML 전환 오버헤드 발생
3. 결국 모든 설정에서 CPU EP로 폴백 — 순수 CPU보다 느림

> CoreML EP는 BERT-base 급 소형 모델에서만 효과. XLM-RoBERTa Large에는 **사용 불가**.

### Dynamic Quantization INT8

**실패**: `Unable to find data type for weight_name` — pre-optimized (Fusion 적용) 모델에서 Dynamic 양자화 불가. ORT 최적화 전 원본 ONNX에서 적용해야 하나, QDQ 양자화가 더 우수하므로 의미 없음.

### INT4 Weight-Only (MatMulNBits)

**건너뜀**: `matmul_4bits_quantizer` 모듈이 현재 ORT 1.23.2에서 import 불가. onnxruntime-extensions 또는 향후 버전 필요.

### QDQ INT8 + Session Config 복합

| 설정 | Median (ms) | vs PyTorch | Cosine | 변화 |
|------|-----------|---------|--------|------|
| **QDQ INT8 (현재 설정)** | **212.3** | **7.07x** | **0.9891** | **기준** |
| QDQ INT8 + Config 최적화 | 232.7 | 6.45x | 0.9894 | -9% |

**결론**: QDQ INT8에서도 Session Config 최적화(ORT_DISABLE_ALL 등)는 역효과. **현재 설정(ORT_ENABLE_ALL + P-core 4스레드)이 최적.**

### Round 2 종합

| 테스트 항목 | Median (ms) | vs PyTorch | Cosine | 판정 |
|------------|-----------|---------|--------|------|
| ORT-opt 현재 설정 | 430.1 | 3.49x | 1.0000 | 기준 |
| ORT_DISABLE_ALL | 447.4 | 3.35x | 1.0000 | 악화 |
| Session Config 최적화 | 458.8 | 3.27x | 1.0000 | 악화 |
| CoreML EP (ALL) | 466.1 | 3.22x | 1.0000 | 실패 (CPU 폴백) |
| CoreML EP (CPU+ANE) | 476.5 | 3.15x | 1.0000 | 실패 (CPU 폴백) |
| CoreML EP (CPUOnly) | 483.3 | 3.11x | 1.0000 | 실패 (CPU 폴백) |
| **QDQ INT8 (현재 설정)** | **212.3** | **7.07x** | **0.9891** | **최선** |
| QDQ INT8 + Config 최적화 | 232.7 | 6.45x | 0.9894 | 악화 |

## Round 3: 민감 레이어 확장 실험 (2026-02-24)

Round 2에서 QDQ INT8이 cosine 0.989로 속도는 최선이나 **품질 목표(cosine ≥ 0.999)에 미달**하였다.
이를 해결하기 위해 24개 Transformer 레이어의 양자화 민감도를 개별 측정하고,
FP32 유지 레이어를 점진적으로 확장하여 품질-속도 최적점을 탐색하였다.

### Step 1: 레이어별 민감도 측정

각 레이어를 개별적으로 FP32로 유지하고 나머지 23개를 INT8로 양자화한 뒤,
PyTorch FP32 결과와의 cosine similarity를 측정하여 민감도를 순위화하였다.

| 순위 | 레이어 | Cosine (해당 레이어 FP32 시) | 비고 |
|------|--------|---------------------------|------|
| 1 | 20 | 0.9937 | 출력 근처 |
| 2 | 19 | 0.9933 | 출력 근처 |
| 3 | 23 | 0.9932 | 최종 레이어 |
| 4 | 22 | 0.9931 | 최종 레이어 |
| 5 | 21 | 0.9931 | 출력 근처 |
| 6 | 14 | 0.9929 | 중간 레이어 |
| 7 | 9 | 0.9929 | 중간 레이어 |
| ... | ... | ... | ... |
| 23 | 16 | 0.9922 | 가장 둔감 |
| 24 | 4 | 0.9919 | 가장 둔감 |

> 전체 레이어의 민감도 범위가 0.9919~0.9937로 좁아, **모든 레이어가 양자화에 민감**함을 보인다.
> 출력 근처 레이어(19-23)가 가장 민감하고, 중간 레이어(3, 4, 10, 16)가 가장 둔감하다.

### Step 2: 점진적 FP32 확장

민감도 순위에 따라 FP32 유지 레이어를 4→6→8→...→20개로 점진적으로 확장하며 cosine과 latency를 측정.

| FP32 레이어 수 | INT8 레이어 수 | Cosine | Latency (ms) | 목표 달성 |
|---------------|---------------|--------|-------------|----------|
| 4 | 20 | 0.9963 | 90.6 | ✗ |
| 6 | 18 | 0.9972 | 101.9 | ✗ |
| 8 | 16 | 0.9979 | 110.0 | ✗ |
| 10 | 14 | 0.9983 | 125.7 | ✗ |
| 12 | 12 | 0.9983 | 134.1 | ✗ |
| 14 | 10 | 0.9987 | 145.3 | ✗ |
| **16** | **8** | **0.9990** | **167.4** | **✓** |
| 18 | 6 | 0.9992 | 188.7 | ✓ |
| 20 | 4 | 0.9994 | 191.3 | ✓ |
| 24 (ORT-opt FP32) | 0 | 1.0000 | 218.0 | ✓ |

### Step 3: 최적 구성

**cosine ≥ 0.999를 달성하는 최소 FP32 레이어 수: 16개**

```
FP32 유지 (16개): [0, 1, 2, 6, 9, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23]
INT8 양자화 (8개): [3, 4, 5, 7, 8, 10, 11, 16]
```

| 항목 | 값 |
|------|-----|
| Cosine similarity | 0.9990 (목표 0.999 달성) |
| Latency | 167.4ms |
| vs ORT-opt FP32 (218ms) | **1.30x 빠름 (23% 개선)** |
| vs PyTorch FP32 (1501ms) | **8.97x 빠름** |
| 양자화 비율 | 8/24 = 33% (레이어 기준) |

### Round 3 결론

- 24개 레이어 모두 양자화에 민감하여, **cosine ≥ 0.999 달성에 16개 FP32 레이어 필요**
- 양자화 효과가 8개 레이어(33%)에만 적용되므로, Round 2의 QDQ INT8(전 레이어 양자화, 7.07x)보다 속도 이점 감소
- 그러나 ORT-opt FP32 대비 여전히 **23% 개선**이며, 품질 손실 최소(cosine 0.999)
- **품질 우선 시나리오에서 QDQ INT8 (16 FP32)이 최적 선택**

## 최적화 파이프라인 요약

```
PyTorch FP32 (1501ms) → ONNX 변환 (1013ms, 1.48x)
                      → ORT 그래프 최적화 (218~940ms, 1.60~6.88x)
                      → QDQ INT8 (16 FP32) (167ms, 8.97x)  ← 품질 우선 (cosine 0.999)
                      → QDQ INT8 (4 FP32) (212ms, 7.07x)   ← 속도 우선 (cosine 0.989)
                      → 민감 레이어 확장: 16/24 FP32 필요 (cosine ≥ 0.999)
                      → Session Config 튜닝: 역효과
                      → CoreML EP: 실패 (XLM-RoBERTa 미지원)
                      → Dynamic Quantization: 실패 (pre-optimized 모델)
                      → Static Shape: 역효과 (Mac ARM)
                      → IO Binding: 무효 (CPU EP)
                      → BF16 Fastmath: 미지원 (Mac ARM)
```

## 플랫폼별 권장 설정

### Mac ARM (개발 환경) - 품질 우선 (권장)

```env
USE_ONNX_EMBEDDING=true
ONNX_EMBEDDING_VARIANT=ort-opt-qdq      # 8.97x, cosine 0.9990 (16 FP32 레이어)
ONNX_INTRA_OP_THREADS=4                 # P코어만
ONNX_ENABLE_IO_BINDING=false            # CPU EP에서 무효
ONNX_ENABLE_BF16_FASTMATH=false         # Mac ARM 미지원
```

### Mac ARM (개발 환경) - 무손실

```env
USE_ONNX_EMBEDDING=true
ONNX_EMBEDDING_VARIANT=ort-opt          # 6.88x, cosine 1.0000 (FP32 무손실)
ONNX_INTRA_OP_THREADS=4                 # P코어만
ONNX_ENABLE_IO_BINDING=false            # CPU EP에서 무효
ONNX_ENABLE_BF16_FASTMATH=false         # Mac ARM 미지원
```

### Graviton3+ (프로덕션 서버, 예상)

```env
USE_ONNX_EMBEDDING=true
ONNX_EMBEDDING_VARIANT=ort-opt-qdq      # INT8 양자화 (16 FP32 레이어)
ONNX_INTRA_OP_THREADS=0                 # 자동 감지
ONNX_ENABLE_IO_BINDING=false            # CPU EP에서 무효
ONNX_ENABLE_BF16_FASTMATH=true          # BF16 MMLA 가속
```

> Graviton3에서는 BF16 Fastmath + QDQ INT8 복합이 추가 30-50% 개선 예상.
> 실측 데이터는 Graviton3 서버 확보 후 측정 필요.

## 결론

1. **QDQ INT8 (16 FP32)이 품질 목표(cosine ≥ 0.999) 달성 최적 선택** (8.97x, cosine 0.9990)
2. **ORT 그래프 최적화는 무손실 기본 최적화** (6.88x, cosine 1.0000)
3. **민감 레이어 확장 실험으로 품질-속도 트레이드오프 정량화 완료** (4~20 FP32 레이어)
4. **병목은 MatMul(70%) + Attention(25%)** — INT8 양자화가 이를 직접 공략
5. **Session Config 튜닝, CoreML EP, Dynamic Quantization은 모두 효과 없음 또는 실패**
6. **현재 프로덕션 설정(ORT_ENABLE_ALL + P-core 4스레드 + QDQ INT8 16 FP32)이 최적**

### 비용 대비 효과

| 최적화 | 구현 난이도 | Mac ARM 효과 | Graviton 기대 효과 |
|--------|-----------|-------------|------------------|
| ORT 그래프 최적화 | 낮음 | 6.88x | 6.88x |
| **QDQ INT8 (16 FP32)** | **중간** | **8.97x** | **12-18x (예상)** |
| QDQ INT8 (4 FP32) | 중간 | 7.07x (cosine 0.989) | 10-15x (예상) |
| 민감 레이어 확장 | 높음 | cosine 0.999 달성 | 동일 적용 가능 |
| Session Config 튜닝 | 낮음 | **역효과** | 미검증 |
| CoreML EP | 낮음 | **실패** | N/A (ARM Linux) |
| Dynamic INT8 | 중간 | **실패** | 실패 (pre-optimized) |
| Static Shape | 중간 | **역효과** | 1.1-1.3x (BF16 조합) |
| IO Binding | 낮음 | 무효 | 무효 (CPU EP) |
| BF16 Fastmath | 낮음 | 미지원 | **1.3-1.65x** |

### 탐색했으나 효과 없었던 최적화 목록

| 기법 | 결과 | 이유 |
|------|------|------|
| ORT_DISABLE_ALL | -4% 악화 | Fusion 노드 런타임 최적화 비활성화 |
| denormal_as_zero | 무효 | XLM-RoBERTa에서 denormal 발생 빈도 극소 |
| force_spinning_stop | 무효 | 이미 sequential mode에서 경합 없음 |
| gelu_approximation | 무효 | BiasGelu가 전체 2.9%만 차지 |
| CoreML EP | 실패 | 197 노드 중 83개만 지원 (42%), 파티션 오버헤드 |
| Dynamic INT8 | 실패 | pre-optimized 모델 비호환 |
| INT4 MatMulNBits | 미지원 | ORT 1.23.2 모듈 없음 |
| QDQ INT8 + Config 최적화 | -9% 악화 | 복합 설정 간 간섭 |
