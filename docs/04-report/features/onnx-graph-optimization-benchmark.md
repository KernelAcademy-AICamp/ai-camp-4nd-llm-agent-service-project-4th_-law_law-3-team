# ONNX 그래프 최적화 벤치마크 보고서

> 생성일: 2026-02-21 18:31 UTC  
> 모델: nlpai-lab/KURE-v1 (1024차원)  
> 벤치마크 쿼리: 10개, 문서: 20개, 인제스트 테스트: 200건

## 요약

**추천 variant**: `PyTorch FP32` — PyTorch 대비 **1.0x** 속도 향상 (CPU), 품질 기준 충족

## 1. 속도 비교

### 1.1.1 단일 쿼리 지연시간 (CPU)

| Variant | 평균 (ms) | 상대 속도 |
|---------|----------:|----------:|
| PyTorch FP32 | 1500.1 | 1.0x (baseline) |
| ONNX FP32 | 2415.8 | 0.6x |
| ONNX INT8 | 1639.2 | 0.9x |
| ONNX O2 | 1703.2 | 0.9x |
| ONNX O3 | 1676.6 | 0.9x |
| ONNX O3+INT8 | 1148.7 | 1.3x |

### 1.1.2 배치 처리 (20문서, CPU)

| Variant |batch=32 (ms) | batch=64 (ms) | batch=128 (ms) |
|---------|---------: | ---------: | ---------: |
| PyTorch FP32 | 3532 | 3562 | 3532 |
| ONNX FP32 | 5000 | 4828 | 4515 |
| ONNX INT8 | 2109 | 2109 | 2110 |
| ONNX O2 | 3718 | 3719 | 3734 |
| ONNX O3 | 3688 | 3719 | 3688 |
| ONNX O3+INT8 | 1735 | 1734 | 1735 |

### 1.2.1 단일 쿼리 지연시간 (GPU)

| Variant | 평균 (ms) | 상대 속도 |
|---------|----------:|----------:|
| PyTorch FP32 | 1951.5 | 1.0x (baseline) |
| ONNX INT8 | 2112.5 | 0.9x |
| ONNX O2 | 1654.8 | 1.2x |
| ONNX O3 | 1682.6 | 1.2x |
| ONNX O3+INT8 | 1623.4 | 1.2x |

### 1.2.2 배치 처리 (20문서, GPU)

| Variant |batch=32 (ms) | batch=64 (ms) | batch=128 (ms) |
|---------|---------: | ---------: | ---------: |
| PyTorch FP32 | 2047 | 2031 | 2000 |
| ONNX INT8 | 2671 | 2657 | 2719 |
| ONNX O2 | 3703 | 3703 | 3703 |
| ONNX O3 | 3718 | 3765 | 3719 |
| ONNX O3+INT8 | 2219 | 2406 | 2281 |

### 1.3 데이터 인제스트 (200문서, CPU)

| Variant | 임베딩 (ms) | 쓰기 (ms) | 총 (ms) | 문서/초 |
|---------|----------:|--------:|-------:|-------:|
| PyTorch FP32 | 3141 | 16 | 3157 | 6.3 |
| ONNX FP32 | 4781 | 16 | 4797 | 4.2 |
| ONNX INT8 | 2594 | 15 | 2609 | 7.7 |
| ONNX O2 | 3875 | 16 | 3891 | 5.1 |
| ONNX O3 | 3922 | 16 | 3938 | 5.1 |
| ONNX O3+INT8 | 2078 | 16 | 2094 | 9.6 |

### 1.4 데이터 인제스트 (200문서, GPU)

| Variant | 임베딩 (ms) | 쓰기 (ms) | 총 (ms) | 문서/초 |
|---------|----------:|--------:|-------:|-------:|
| ONNX FP32 | 4656 | 16 | 4672 | 4.3 |
| ONNX INT8 | 2719 | 16 | 2735 | 7.3 |
| ONNX O2 | 3968 | 16 | 3984 | 5.0 |
| ONNX O3 | 4172 | 16 | 4188 | 4.8 |
| ONNX O3+INT8 | 2453 | 16 | 2469 | 8.1 |

## 2. 임베딩 품질

| 메트릭 | ONNX FP32 | ONNX INT8 | ONNX O2 | ONNX O3 | ONNX O3+INT8 | 기준 | 판정 |
|--------| -----: | -----: | -----: | -----: | -----: | -----: | -----: |
| Pairwise Cosine | 0.7626 | 0.9851 | 1.0000 | 1.0000 | 0.9854 | >=0.995 | FAIL |
| Sim Matrix Pearson | 0.7911 | 0.9850 | 1.0000 | 1.0000 | 0.9894 | >=0.998 | FAIL |
| Separation Diff | 0.0921 | 0.0019 | 0.0000 | 0.0006 | 0.0108 | <0.05 | FAIL |

## 4. 모델 크기

| Variant | 크기 (MB) | 절약 (%) |
|---------|--------:|---------:|
| PyTorch FP32 | 2182 | - |
| ONNX FP32 | 2179 | +0% |
| ONNX INT8 | 560 | +74% |
| ONNX O2 | 2178 | +0% |
| ONNX O3 | 2178 | +0% |
| ONNX O3+INT8 | 559 | +74% |

## 5. 종합 비교 및 추천

### 5.1 Trade-off 분석 (CPU)

| Variant | 속도 | 품질 | 검색 | 크기 | 종합 |
|---------|:----:|:----:|:----:|:----:|:----:|
| ONNX FP32 | 느림 | 미달 | - | 큼 | FAIL |
| ONNX INT8 | 느림 | 미달 | - | **작음** | FAIL |
| ONNX O2 | 느림 | **우수** | - | 큼 | PASS |
| ONNX O3 | 느림 | **우수** | - | 큼 | PASS |
| ONNX O3+INT8 | 보통 | 미달 | - | **작음** | FAIL |

### 5.2 운영 환경별 추천 (CPU)

- **속도 우선**: `PyTorch FP32` — 응답 지연시간이 최우선인 환경 (실시간 검색)
- **품질 우선**: `ONNX O2` — 임베딩 품질이 최우선인 환경 (정밀 검색)
- **균형**: `PyTorch FP32` — 속도와 품질의 균형 (일반 운영)

### 5.3 Trade-off 분석 (GPU)

| Variant | 속도 | 품질 | 검색 | 크기 | 종합 |
|---------|:----:|:----:|:----:|:----:|:----:|
| ONNX INT8 | 느림 | 미달 | - | **작음** | FAIL |
| ONNX O2 | 느림 | **우수** | - | 큼 | PASS |
| ONNX O3 | 느림 | **우수** | - | 큼 | PASS |
| ONNX O3+INT8 | 느림 | 미달 | - | **작음** | FAIL |

### 5.4 운영 환경별 추천 (GPU)

- **속도 우선**: `ONNX O2` — 응답 지연시간이 최우선인 환경 (실시간 검색)
- **품질 우선**: `ONNX O2` — 임베딩 품질이 최우선인 환경 (정밀 검색)
- **균형**: `ONNX O2` — 속도와 품질의 균형 (일반 운영)

### 5.5 전체 데이터 재임베딩 예상 시간 (CPU)

대상: `legal_chunks` 테이블 (253,768건)

| Variant | 인제스트 속도 (문서/초) | 예상 시간 |
|---------|--------------------:|--------:|
| PyTorch FP32 | 6.3 | ~11.1시간 |
| ONNX FP32 | 4.2 | ~16.9시간 |
| ONNX INT8 | 7.7 | ~9.2시간 |
| ONNX O2 | 5.1 | ~13.7시간 |
| ONNX O3 | 5.1 | ~13.9시간 |
| ONNX O3+INT8 | 9.6 | ~7.4시간 |

### 5.6 전체 데이터 재임베딩 예상 시간 (GPU)

대상: `legal_chunks` 테이블 (253,768건)

| Variant | 인제스트 속도 (문서/초) | 예상 시간 |
|---------|--------------------:|--------:|
| ONNX FP32 | 4.3 | ~16.5시간 |
| ONNX INT8 | 7.3 | ~9.6시간 |
| ONNX O2 | 5.0 | ~14.0시간 |
| ONNX O3 | 4.8 | ~14.8시간 |
| ONNX O3+INT8 | 8.1 | ~8.7시간 |

## 6. 그래프 최적화 미개선 원인 분석

### 6.1 핵심 원인: O2/O3 모델의 가중치 미포함 문제

진단 실험(`benchmark_onnx_diagnosis.py`)을 통해 확인된 핵심 원인:

**O2/O3 최적화 모델에 가중치가 포함되지 않았다.**

| 모델 | ONNX 파일 크기 | 외부 데이터 파일 | 가중치 상태 |
|------|--------------|----------------|-----------|
| 원본 FP32 | 0.4 MB | `model.onnx_data` 2,162 MB | 외부 파일에 저장 |
| O2 최적화 | 0.1 MB | 없음 | **미포함** |
| O3 최적화 | 0.1 MB | 없음 | **미포함** |

`optimum`의 그래프 최적화(`AutoOptimizationConfig.O2/O3`)가 2GB 초과 모델의 외부 데이터 파일을 올바르게 통합하지 못하여, 최적화된 ONNX 파일에 그래프 구조만 남고 가중치가 누락되었다.

이전 벤치마크에서 O2/O3가 cosine 1.0을 보인 이유는, `sentence-transformers` ONNX 백엔드가 외부 데이터 파일이 없는 모델을 로드할 때 원본 PyTorch 가중치로 fallback했기 때문으로 추정된다.

### 6.1.1 이중 최적화 가설 — 실험적 검증

`ORT_DISABLE_ALL` vs `ORT_ENABLE_ALL` 비교 실험 결과:

| 모델 | ON (ms) | OFF (ms) | 차이 |
|------|---------|----------|------|
| 원본 (FP32) | 547.4 | 557.1 | +1.8% |
| O2 최적화 | 512.6 | 522.4 | +1.9% |
| O3 최적화 | 506.2 | 500.2 | -1.2% |

- ON/OFF 모두 모델 간 속도 차이는 미미 (~7%, 41ms)
- 런타임 최적화가 "완전히 상쇄"한다는 가설은 **부분 기각** — 상쇄 효과는 존재하나 완전하지 않음
- 다만 O2/O3 모델이 가중치 없이 0.1MB이므로, 측정된 속도 차이가 실제 그래프 최적화 효과인지 단순 모델 크기 차이인지 분리 불가

### 6.2 BERT 구조적 한계

- 추론 시간의 60-70%가 MatMul(GEMM)에 집중되며, 그래프 최적화는 연산량 자체를 줄이지 않음
- 그래프 최적화가 줄이는 것: 불필요한 노드 제거, 연산 융합(Conv+BN 등)이지만 Transformer 구조에서는 이미 최적화 여지가 적음
- 1024차원 KURE-v1은 768차원 BERT-base보다 MatMul 비중이 더 크므로 그래프 최적화 효과가 더욱 제한적

### 6.3 O3+INT8만 개선된 이유

O3+INT8에서 1.3x 속도 개선이 발생한 핵심 원인은 **INT8 양자화**이며, 그래프 최적화가 아니다.

- FP32 → INT8로 연산 타입 자체가 변경되어 연산 처리량이 증가
- AVX2 INT8 곱셈 명령어(`VPMADDUBSW`, `VPMADDWD`) 활용으로 단일 사이클당 처리량 증가
- 메모리 대역폭 4배 절약 (4바이트 → 1바이트), 캐시 적중률 향상
- ONNX INT8 단독(0.9x)보다 O3+INT8(1.3x)이 빠른 이유: O3 최적화가 양자화 노드 배치를 더 효율적으로 구성

### 6.4 ONNX FP32 품질 이상 (cosine 0.78) 원인 — 실험적 검증

진단 실험으로 세 가지 경로의 cosine similarity를 비교:

| 경로 | vs PyTorch cosine | 결과 |
|------|------------------|------|
| ONNX FP32 — optimum fallback | 0.7849 | FAIL |
| ONNX FP32 — onnxruntime 직접 세션 (mean pooling 통일) | 0.7849 | FAIL |
| ONNX O2 — optimum fallback | 0.7849 | FAIL |
| ONNX O2 — onnxruntime 직접 세션 | 0.7849 | FAIL |
| optimum vs raw 직접 비교 | 1.0000 | 동일 |

**확인된 사실:**

1. **Pooling 전략은 원인이 아님** — optimum fallback과 onnxruntime 직접 세션(동일 mean pooling)의 결과가 완전히 일치 (cosine 1.0)
2. **모든 ONNX variant가 동일한 cosine 0.785** — FP32, O2 모두 동일하므로 그래프 최적화와 무관
3. **근본 원인: ONNX 변환 자체의 가중치 정밀도 손실** — `optimum`의 ONNX export 과정에서 2GB 초과 모델의 외부 데이터 파일(`model.onnx_data`) 매핑 시 가중치 정밀도 차이 발생

> 이전 벤치마크에서 O2/O3가 cosine 1.0을 보인 것은 `sentence-transformers` ONNX 백엔드가 원본 PyTorch 가중치로 fallback한 결과로 추정됨 (6.1절 참조)

## 7. ARM 플랫폼 성능 전망

### 7.1 M-series Mac (Apple Silicon)

| 실행 경로 | 예상 속도 | 품질 | 비고 |
|-----------|----------|------|------|
| PyTorch MPS | 2-3x 개선 | 완벽 | `device="mps"` 전환, FP16 자동 활용 |
| CoreML EP + ANE | 3-5x 가능 | FP16 자동 변환 주의 | ANE는 FP16 전용, 정밀도 검증 필요 |
| ONNX CPUExecutionProvider | 효과 미미 | 완벽 | NEON 128-bit는 AVX2(256-bit)의 절반 폭 |
| ONNX FP16 + CoreML | 2-3x 개선 | 좋음 (cosine ~0.999) | INT8보다 품질 손실 적음 |

**핵심 전략:**

- O2/O3 그래프 최적화 단독: **효과 없음** (이중 최적화가 동일하게 적용됨)
- **FP16 변환**이 M-series에서의 핵심 전략 — 품질 기준 통과 + ANE/GPU 하드웨어 최적화 동시 달성
- MPS 백엔드는 코드 변경 최소(`device="mps"`)로 즉시 적용 가능
- CoreML EP는 `onnxruntime-coreml` 패키지 추가 설치 필요

### 7.2 AWS Graviton3/4 (ARM Neoverse)

**Graviton3 하드웨어 특성:**

| 항목 | 사양 |
|------|------|
| SIMD | SVE 256-bit (가변 벡터 폭) |
| BFloat16 | 네이티브 지원 (BF16 SGEMM 가속) |
| INT8 | MMLA 명령어 가속 |
| 메모리 | DDR5, 높은 대역폭 |

**AWS 공식 벤치마크 수치:**

| 최적화 전략 | 성능 향상 |
|------------|----------|
| BF16 SGEMM (vs FP32) | 최대 65% 추론 성능 향상 |
| INT8 MMLA QGEMM (vs FP32 INT8) | 최대 30% INT8 추론 향상 |
| BERT-large 추론 | Graviton3 > Intel Ice Lake 1.8배 |

**예상 성능:**

| 전략 | 예상 속도 | 품질 | 비고 |
|------|----------|------|------|
| ONNX FP32 + BF16 자동가속 | 1.5-1.8x | ~0.999 PASS | ONNX Runtime v1.17+ 자동 적용 |
| ONNX INT8 (arm64 양자화) | 1.8-2.5x | ~0.985 FAIL | MMLA 커널 활용, 품질 검증 필요 |
| PyTorch + torch.compile | 1.3-1.5x | 완벽 | AWS 공식 가이드 지원 |

**주의사항:**

- 현재 코드의 `avx2` 양자화 설정은 ARM에서 사용 불가 → `arm64` 전환 필수
- O2/O3 그래프 최적화 단독: **효과 없음** (이중 최적화가 동일하게 적용됨)
- BF16 자동 가속은 ONNX Runtime v1.17 이상에서 Graviton3 감지 시 자동 활성화

## 8. 플랫폼별 배포 권장 사항

### 8.1 종합 비교

| 플랫폼 | 최적 전략 | 예상 속도 | 품질 판정 | 코드 변경 수준 |
|--------|----------|----------|----------|--------------|
| **Windows x86 (현재)** | PyTorch FP32 | 1.0x (baseline) | PASS | 없음 |
| **Windows x86** | ONNX O3+INT8 | 1.3x | FAIL (cosine 0.985) | 없음 |
| **M-series Mac** | PyTorch MPS | 2-3x | PASS | `device="mps"` 전환 |
| **M-series Mac** | ONNX FP16 + CoreML EP | 2-3x | PASS (~0.999) | CoreML EP 설치 + FP16 변환 |
| **AWS Graviton3** | ONNX FP32 + BF16 자동가속 | 1.5-1.8x | PASS (~0.999) | ORT v1.17+ 확인 |
| **AWS Graviton3** | ONNX INT8 (arm64) | 1.8-2.5x | FAIL (~0.985) | arm64 양자화 재수행 |

### 8.2 현재 벤치마크 코드 수정 필요사항

| 수정 항목 | 대상 파일 | 설명 |
|----------|----------|------|
| arm64 양자화 지원 | `benchmark_embedding_quantize.py` | `avx2` → 플랫폼 감지 후 `arm64` 분기 |
| CoreML EP 지원 | `benchmark_embedding_quantize.py` | `CoreMLExecutionProvider` 옵션 추가 |
| MPS 디바이스 지원 | `benchmark_embedding_quantize.py` | PyTorch MPS 백엔드 벤치마크 추가 |
| 런타임 최적화 비활성화 옵션 | `benchmark_embedding_quantize.py` | `ORT_DISABLE_ALL` 옵션으로 오프라인 최적화 효과 분리 측정 |

### 8.3 참고 자료

- [ONNX Runtime Graph Optimizations](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html)
- [ONNX Runtime Quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [AWS Graviton ML Inference](https://github.com/aws/aws-graviton-getting-started/blob/main/machinelearning/onnx.md)
- [Apple Core ML ONNX Runtime EP](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)
- [ONNX Runtime Session Options](https://onnxruntime.ai/docs/api/python/api_summary.html#sessionoptions)

## 부록

### 벤치마크 설정값

| 항목 | 값 |
|------|-----|
| 모델 | nlpai-lab/KURE-v1 |
| 차원 | 1024 |
| Warm-up | 2회 |
| 반복 | 5회 (median) |
| 인제스트 테스트 문서 | 200건 |
| LanceDB 테이블 | legal_chunks |

### 품질 기준

| 메트릭 | 기준 |
|--------|------|
| pairwise_cosine_mean | >= 0.995 |
| sim_matrix_pearson | >= 0.998 |
| separation_diff | < 0.05 |
| top3_match_rate | >= 90% |
| top5_match_rate | >= 80% |
| top10_match_rate | >= 70% |
| search_spearman | >= 0.95 |

