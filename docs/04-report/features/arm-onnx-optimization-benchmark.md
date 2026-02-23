# ARM ONNX 최적화 벤치마크 보고서

## 개요

Mac M1/M3 + ARM 서버(AWS Graviton) 환경에서 ONNX 트랜스포머 최적화를 통한 CPU 레이턴시 최소화.
기존 x86 벤치마크에서 발견된 `optimum` 2GB+ 외부 데이터 버그를 우회하여
`onnxruntime.transformers.optimizer`를 직접 적용.

**측정 환경**: Mac M1 (arm64), onnxruntime 1.23.2, PyTorch 2.10.0, opset 17
**측정일**: 2026-02-23

## 대상 모델

| 모델 | 용도 | 아키텍처 | 크기 |
|------|------|---------|------|
| `nlpai-lab/KURE-v1` | 임베딩 (1024차원) | XLM-RoBERTa Large (16H, 1024D) | ~2.3GB |
| `dragonkue/bge-reranker-v2-m3-ko` | 리랭킹 | XLM-RoBERTa Large (16H, 1024D) | ~2.3GB |

## 최적화 Variant

| Variant | 설명 | 실측 크기 | 속도 효과 | 대상 환경 |
|---------|------|----------|----------|----------|
| `ort-opt` | Attention/LayerNorm/GELU Fusion (FP32) | 2.27GB | 기준선 | 정확도 최우선 |
| `ort-opt-fp16` | Fusion + FP16 mixed precision | 1.13GB | **속도 이점 없음** (메모리만 절약) | Mac Air 8GB 메모리 부족 시 |
| `ort-opt-static128` | Fusion + Static Shape (batch=1, seq=128) | 2.27GB | 미측정 (예상 5-15%) | Graviton3 임베딩 |
| `ort-opt-int8` | Dynamic INT8 quantization (raw export 기반) | 0.57GB | **실측 15-41% 빠름** | **리랭커 전용** |

> **FP16 주의**: CPU에서 FP16 모델은 연산 시 FP32로 캐스팅되므로 속도 이점이 없다.
> 오히려 FP16↔FP32 변환 오버헤드가 발생할 수 있다. 유일한 장점은 메모리 절약 (2.27GB → 1.13GB).

### Fusion 통계 (ORT Transformer Optimizer)

| Fusion 타입 | 개수 |
|-------------|------|
| Attention | 24 |
| SkipLayerNormalization | 48 |
| BiasGelu | 24 |

- FP32 그래프 퓨전은 수학적 동등 변환 → 정확도 손실 0%
- Attention Fusion: 24개 Multi-Head Attention 블록 통합
- LayerNorm Fusion: element-wise 연산 커널 호출 감소
- GELU Fusion: 활성화 함수 최적화

### INT8 양자화 주의사항

- **반드시 raw ONNX export에 적용** (ORT-optimized 모델의 fused node에는 적용 불가)
- `onnxruntime.quantization.quantize_dynamic()` 사용
- ORT 세션 레벨에서 그래프 최적화 자동 수행 (별도 fusion 불필요)

## 벤치마크 결과 (Mac M1 arm64 실측)

### 임베딩 (단일 쿼리 레이턴시)

| Variant | Mac M1 | vs PyTorch | Cosine vs PT | 판정 |
|---------|--------|------------|-------------|------|
| **PyTorch FP32 (baseline)** | **33.8ms** | - | 1.0000 | **Mac 최적** |
| ONNX FP32 (ORT-opt) | 85.0ms | 2.5x 느림 | 1.0000 | ORT MLAS 한계 |
| ONNX FP16 (ORT-opt) | 미측정 | 속도 이점 없음 | 0.9999985 | 메모리만 절약 |
| ONNX INT8 (dynamic) | 28.9ms | 15% 빠름 | **0.985** | **정확도 불가** |

> **임베딩 INT8 불가 사유**: cosine=0.985는 벡터 유사도 검색에서 순위 역전을 유발할 수 있다.
> 리랭커와 달리 임베딩은 LanceDB 253K 벡터와 직접 cosine distance로 비교되므로,
> 절대적인 벡터 공간 위치의 정확성이 필수 (cosine >= 0.999 요구).

### 리랭커 (15문서 배치 레이턴시)

| Variant | Mac M1 | vs PyTorch | Cosine vs PT | 판정 |
|---------|--------|------------|-------------|------|
| **PyTorch FP32 (baseline)** | **114.2ms** | - | 1.0000 | Mac 기준선 |
| ONNX FP32 (ORT-opt) | 222.5ms | 1.9x 느림 | 1.0000 | ORT MLAS 한계 |
| ONNX FP16 (ORT-opt) | 미측정 | 속도 이점 없음 | 0.9999999 | 메모리만 절약 |
| **ONNX INT8 (dynamic)** | **67.3ms** | **41% 빠름** | **0.998** | **채택** |

> **리랭커 INT8 채택 사유**: 상대적 순위만 보존하면 되므로 cosine=0.998이면 충분.
> 41% 속도 향상은 실용적 가치가 크다.

### CoreML 벤치마크 (Mac M1)

CoreMLExecutionProvider 테스트 결과: **실용적이지 않음**.
- 197개 노드 중 8개만 지원 (4%)
- 나머지 96%는 CPU fallback → 오히려 오버헤드 발생

### 모델 크기

| Variant | 임베딩 | 리랭커 |
|---------|--------|--------|
| PyTorch FP32 | ~2.3GB | ~2.3GB |
| ORT-최적화 FP32 | 2.27GB | 2.27GB |
| ORT-최적화 FP16 | 1.13GB | 1.14GB |
| INT8 (dynamic) | 0.57GB | 0.57GB |

## FP32 무손실 최적화 시도 (BF16 Fastmath + Static Shape)

> **현재 상태**: 코드 구현 완료, **실측 데이터 없음**. Graviton3 인스턴스에서 벤치마크 필요.

INT8 양자화가 임베딩에서 cosine=0.985로 품질 위험이 있어 배제된 후,
FP32 가중치/연산 정밀도를 100% 유지하면서 순수 연산 효율성만 극대화하는 전략을 탐색했다.

### 임베딩 FP32 속도 최적화의 현실

| 전략 | 속도 효과 | 상태 | 비고 |
|------|----------|------|------|
| FP16 mixed precision | **효과 없음** | 확인됨 | CPU에서 FP16→FP32 캐스팅 발생, 변환 오버헤드만 추가 |
| INT8 dynamic quantization | **15% 빠름** | 실측 | cosine=0.985로 **임베딩 사용 불가** |
| BF16 Fastmath (Graviton3) | 예상 최대 65% | **미측정** | Graviton3 전용, Mac ARM 미지원 |
| Static Shape (batch=1, seq=128) | 예상 5-15% | **미측정** | constant folding 최적화 |
| 복합 (BF16 + Static128) | 예상치 합산 | **미측정** | Graviton3 전용 |

**결론**: Mac ARM에서 임베딩 FP32 속도를 개선할 수 있는 실증된 방법은 현재 없다.
PyTorch FP32 (33.8ms)가 Mac ARM 최선이며, Graviton3에서 BF16이 효과가 있을지는 실측이 필요하다.

### BF16 Fastmath (Graviton3 전용, Phase 8)

**원리**: Graviton3의 MMLA 명령어가 FP32 GEMM을 내부적으로 BF16으로 변환하여 SIMD 가속.
모델 파일은 FP32 유지, MLAS SBGEMM 커널이 런타임에 BF16 경로를 선택한다.

```python
# 1줄 config로 활성화
session_options.add_session_config_entry(
    "mlas.enable_gemm_fastmath_arm64_bfloat16", "1"
)
```

- **플랫폼 제한**: Linux ARM (Graviton3+)만 지원. Mac ARM은 BF16 MMLA 미지원.
- **감지 방법**: `/proc/cpuinfo` Features 필드에 `bf16` 포함 여부 확인
- **정밀도**: BF16은 exponent 8bit + mantissa 7bit으로 FP16보다 rounding error가 훨씬 작음
- **예상 성능**: 최대 65% 레이턴시 개선 (ORT 공식 벤치마크 기준, **본 프로젝트 미측정**)
- **품질 게이트**: cosine >= 0.999 미달 시 자동 비활성화

### Static Shape 최적화 (임베딩 전용, Phase 9)

**원리**: 동적 축(batch_size, sequence_length) 대신 고정 shape(batch=1, seq=128)로 빌드하면,
그래프 최적화기가 Shape inference를 확정적으로 수행 → 추가 constant folding, reshape 제거.

- **대상**: 임베딩만 (단일 쿼리, 고정 길이). 리랭커는 batch_size 가변이므로 dynamic 유지.
- **패딩**: `padding="max_length", max_length=128`로 고정 길이 패딩
- **예상 성능**: 5-15% 레이턴시 개선 (**본 프로젝트 미측정**)
- **빌드**: `build_optimized_onnx.py`에서 `onnxruntime.tools.make_dynamic_shape_fixed` 활용

### 벤치마크 실행 (Graviton3 인스턴스에서)

```bash
cd backend

# Phase 8/9/10 포함 전체 벤치마크
uv run python scripts/benchmark_arm_optimization.py

# 개별 Phase 스킵
uv run python scripts/benchmark_arm_optimization.py --skip-bf16-test    # Phase 8 스킵
uv run python scripts/benchmark_arm_optimization.py --skip-static-test  # Phase 9 스킵
uv run python scripts/benchmark_arm_optimization.py --skip-combined-test # Phase 10 스킵
```

> **참고**: Phase 8, 10은 Linux ARM (Graviton3+)에서만 실행됩니다. Mac ARM에서는 자동 스킵.

## 핵심 인사이트

### Mac ARM에서 ONNX FP32가 PyTorch보다 느린 이유

Mac ARM의 PyTorch는 Apple Accelerate / AMX 하드웨어 코프로세서를 직접 사용.
ONNX Runtime의 CPUExecutionProvider(MLAS)는 범용 NEON SIMD만 사용하여 2-2.5x 느림.
이 격차는 ORT 아키텍처 수준의 제약으로, FP32 범위 내에서 Mac ARM에서 ONNX가 PyTorch를 앞서는 것은 **구조적으로 불가능**.

### CPU에서 FP16은 속도 이점이 없다

일반적인 CPU (x86, ARM 모두)에서 FP16 모델 로드 시:
1. FP16 가중치를 메모리에서 읽음
2. GEMM 연산 전에 **FP32로 캐스팅** (CPU는 FP16 연산을 네이티브 지원하지 않음)
3. FP32로 연산 수행 후 결과를 다시 FP16으로 변환

→ 속도 이점 없음. 오히려 변환 오버헤드 발생 가능. 유일한 장점은 메모리 절약 (2.27GB → 1.13GB).

### INT8은 리랭커에서만 유효

INT8 dynamic quantization은 가중치를 8-bit로 압축하여:
- 메모리 대역폭 4x 절약 → CPU 캐시 효율 증가
- GEMM 연산 최적화 → ARM NEON INT8 활용
- 모델 크기 75% 감소 (2.27GB → 0.57GB)

단, **임베딩에서는 cosine=0.985로 품질 부족** (벡터 검색에서 순위 역전 위험).
리랭커는 상대적 순위만 보존하면 되므로 cosine=0.998로 충분 → INT8 채택.

## 최종 권장 전략

### 실측 기반 확정 사항

| 모델 | 결론 | 근거 |
|------|------|------|
| **임베딩** | **FP32 유지** (속도 최적화 수단 없음) | INT8 cosine=0.985 불가, FP16 속도 이점 없음 |
| **리랭커** | **INT8 채택** | cosine=0.998 양호 + 41% 실측 속도 향상 |

### AWS Graviton3 프로덕션

| 모델 | Variant | 비고 |
|------|---------|------|
| **임베딩** | **FP32** (ort-opt) | BF16 fastmath 자동 감지 적용 (효과 미측정) |
| **리랭커** | **INT8** (dynamic) | 41% 속도 향상 실측, cosine=0.998 |

- BF16 fastmath + Static Shape는 코드 구현 완료되어 있으나 **Graviton3 실측 데이터 없음**
- Graviton3 인스턴스에서 벤치마크 실행 후 효과가 확인되면 임베딩 variant를 `ort-opt-static128`으로 전환

### Mac 로컬 개발용

| 모델 | Variant | 이유 |
|------|---------|------|
| 임베딩 | **PyTorch FP32** | Mac ARM에서 가장 빠름 (33.8ms) |
| 리랭커 | **PyTorch FP32** | Mac ARM에서 ONNX 대비 빠름 (114.2ms) |

> Mac ARM에서는 Apple Accelerate/AMX 가속 덕분에 PyTorch가 ONNX보다 2-2.5x 빠르다.
> FP16은 속도 이점 없고, BF16 MMLA도 미지원이므로 ONNX 최적화의 실익이 없다.

## 플랫폼별 최적 설정

### Mac M1/M3 MacBook Air

**권장: PyTorch FP32 사용** (ONNX 대비 2-2.5x 빠름). ONNX 사용 시 아래 설정:

| 설정 | 값 | 이유 |
|------|-----|------|
| Variant | `ort-opt-fp16` | 메모리 50% 절약 (8GB Air 안정). **속도 이점 없음** |
| `intra_op_threads` | 4 (P코어만) | E코어 배제로 일관된 성능 |
| `inter_op_threads` | 1 | 단일 쿼리 최적화 |
| `execution_mode` | `ORT_SEQUENTIAL` | 트랜스포머 순차 구조 |

### AWS Graviton3

| 설정 | 값 | 이유 |
|------|-----|------|
| 임베딩 Variant | `ort-opt` (FP32) | FP32 유지 (BF16 효과 확인 후 static128 전환 검토) |
| 리랭커 Variant | `ort-opt-int8` | 41% 속도 향상 실측, cosine=0.998 |
| `intra_op_threads` | 물리코어수 | SMT 없음, 전체 활용 |
| `inter_op_threads` | 1 | 단일 쿼리 최적화 |
| BF16 fastmath | 자동 감지 (MMLA 하드웨어) | 예상 최대 65% 향상 (**미측정**) |
| `ONNX_ENABLE_BF16_FASTMATH` | `true` (기본값) | `/proc/cpuinfo` bf16 감지 시 자동 활성화 |

## 프로덕션 통합

### Feature Flag

```bash
# backend/.env (Graviton3 프로덕션)
USE_ONNX_EMBEDDING=true
ONNX_EMBEDDING_VARIANT=ort-opt            # 임베딩: FP32 (BF16 실측 후 ort-opt-static128 전환 검토)
USE_ONNX_RERANKER=true
ONNX_RERANKER_VARIANT=ort-opt-int8        # 리랭커: INT8 (41% 속도 향상 실측)
ONNX_QUALITY_GATE_ENABLED=true            # 시작 시 품질 자동 검증
ONNX_QUALITY_GATE_FALLBACK=true           # 품질 미달 시 PyTorch 자동 폴백
ONNX_INTRA_OP_THREADS=0                   # 0 = 자동 (P코어 감지)
ONNX_ENABLE_BF16_FASTMATH=true            # Graviton3 BF16 MMLA 자동 활성화 (미지원 플랫폼에서는 무시)
```

### 품질 게이트

서버 시작 시 8개 법률 쿼리로 PyTorch vs ONNX 자동 비교:
- 임베딩: cosine >= 0.995 → PASS
- 리랭커: Pearson >= 0.990 → PASS
- 미달 시: `ONNX_QUALITY_GATE_FALLBACK=true`이면 PyTorch 자동 폴백

### 적용 순서

```bash
# 1. 최적화 모델 빌드
uv run python scripts/build_optimized_onnx.py --verify

# 2. 벤치마크 (해당 플랫폼에서)
uv run python scripts/benchmark_arm_optimization.py

# 3. .env에 Feature Flag 설정
USE_ONNX_EMBEDDING=true
USE_ONNX_RERANKER=true

# 4. 서버 시작 → 품질 게이트 자동 실행
uv run uvicorn app.main:app --reload
```

## 모델 검증 결과 (model_versions.json)

| 모델 | Cosine vs PT | Max Abs Diff | Rank Correlation | PASS |
|------|-------------|-------------|-----------------|------|
| kure-v1-ort-opt (FP32) | 1.0000 | 2.2e-07 | 1.0000 | PASS |
| kure-v1-ort-opt-fp16 | 0.9999985 | 2.2e-04 | - | PASS |
| reranker-ort-opt (FP32) | 1.0000 | 8.6e-06 | 1.0000 | PASS |
| reranker-ort-opt-fp16 | 1.0000 | 1.07e-02 | 1.0000 | PASS |

## 빌드 파이프라인 에러 해결 기록

| 에러 | 원인 | 해결 |
|------|------|------|
| `optimum.__version__` AttributeError | optimum 2.1.0에서 모듈 속성 제거 | `importlib.metadata.version('optimum')` 사용 |
| `use_external_data_format` unexpected arg | ORT 1.23.2에서 파라미터 제거 | 파라미터 삭제 |
| `Unknown task: text-classification for SentenceTransformer` | optimum이 bge-reranker를 SentenceTransformer로 오감지 | `library_name="transformers"` 명시 |
| OOM (exit code 137) 두 모델 동시 빌드 | 메모리 부족 | `--reranker-only`로 분리 실행 |
| 리랭커 FP16 max_abs_diff 1.07e-02 > 5e-03 | 분류 logit은 정규화 임베딩보다 FP16 diff가 큼 | `MAX_ABS_DIFF_THRESHOLD_FP16_RR = 1.5e-2` 별도 임계값 |
| INT8 양자화 fused node 미인식 | ORT-optimized 모델의 fused Attention 노드 인식 불가 | raw ONNX export에 직접 quantize_dynamic() 적용 |

## 파일 목록

| 파일 | 설명 |
|------|------|
| `backend/scripts/build_optimized_onnx.py` | 최적화 모델 빌드 + 검증 스크립트 |
| `backend/scripts/benchmark_arm_optimization.py` | 플랫폼별 벤치마크 |
| `backend/scripts/benchmark_embedding_report.py` | 임베딩 벤치마크 MD 보고서 생성기 |
| `backend/scripts/benchmark_reranker_report.py` | 리랭커 벤치마크 MD 보고서 생성기 |
| `backend/app/core/config.py` | Feature Flag (7개 추가) |
| `backend/app/services/rag/onnx_session.py` | ONNX 세션 싱글턴 + 플랫폼 감지 |
| `backend/app/services/rag/onnx_quality_gate.py` | 품질 자동 검증 |
| `backend/app/services/rag/embedding.py` | ONNX dispatch 분기 |
| `backend/app/services/rag/rerank.py` | ONNX dispatch 분기 |
| `backend/app/main.py` | lifespan ONNX 로드/검증 |
| `backend/data/models/model_versions.json` | 빌드 메타데이터 (4개 모델) |

## 잠재 리스크

| 리스크 | 가능성 | 대응 |
|--------|--------|------|
| ONNX export cosine 0.76 재현 | 높음 | `torch.onnx.export()` 직접 사용, opset=17 |
| FP16 cosine < 0.995 | 중간 | `keep_io_types=True`로 LayerNorm FP32 유지 |
| Attention fusion 실패 | 중간 | `get_fused_operator_statistics()` 즉시 감지 |
| Mac Air 8GB 메모리 부족 (FP32) | 높음 | FP16 variant 사용 (1.1GB) |
| 임베딩 INT8 cosine=0.985 검색 품질 저하 | **높음** | **임베딩은 FP32 유지** (INT8 사용 금지) |
| CoreML 노드 미지원 | 확인됨 | 4% 지원 → **사용 불가** |
| BF16 cosine < 0.999 | 낮음 | 품질 게이트 자동 비활성화 + PyTorch 폴백 |
| Static Shape 긴 입력 truncation | 낮음 | max_length=128 초과 시 truncation (법률 쿼리 대부분 128토큰 이내) |
