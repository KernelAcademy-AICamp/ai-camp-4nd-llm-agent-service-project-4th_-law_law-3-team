# ONNX Runtime 최신 최적화 기법 리서치

작성일: 2026-02-24
담당: researcher-ort

## 목적

현재 KURE-v1 (XLM-RoBERTa Large, 24 Transformer layers) 임베딩 모델의 ONNX Runtime 추론 최적화에서
**QDQ INT8 = 772ms (1.94x vs PyTorch 1501ms)** 이 최선 결과이다.
이 문서는 아직 시도하지 않은 ORT 최적화 기법을 조사하여 추가 성능 개선 가능성을 탐색한다.

## 이미 시도한 기법 (제외 대상)

| 기법 | 결과 | 비고 |
|------|------|------|
| ORT 그래프 최적화 (Fusion) | 940ms (1.60x) | Attention/SkipLayerNorm/BiasGelu |
| QDQ INT8 선택적 양자화 | 772ms (1.94x) | 첫/마지막 2레이어 FP32 유지 |
| Static Shape (128/64) | 역효과 | Mac ARM에서 batch=1 루프 오버헤드 |
| IO Binding | 무효 | CPU EP에서 복사 오버헤드 없음 |
| BF16 Fastmath | 미지원 | Mac ARM 미지원, Graviton3+ 전용 |
| P코어 스레드 최적화 (4 threads) | 942ms | ORT-opt FP32와 동등 |
| MPS GPU | 무효 | 데이터 전송 오버헤드 |

---

## 1. INT4 Weight-Only 양자화 (MatMulNBits)

### 개요

ORT 1.22+에서 도입된 `MatMulNBits` 연산자를 활용하여 가중치를 4비트로 블록 단위 양자화하는 기법.
활성화는 FP32 유지, 가중치만 INT4로 압축하여 메모리 대역폭 병목을 완화한다.

### 핵심 특징

- **블록 단위 양자화**: 블록 크기(최소 16, 2의 거듭제곱)로 가중치 그룹화
- **지원 알고리즘**: RTN (기본), HQQ, GPTQ
- **ONNX opset 21 필요**: 자동 업그레이드 가능
- **KleidiAI 통합**: ORT 1.22+에서 ARM Neon/SVE2/SME 가속 지원

### 적용 방법

```python
from onnxruntime.quantization import matmul_4bits_quantizer

quantizer = matmul_4bits_quantizer.MatMul4BitsQuantizer(
    model_path="model_optimized.onnx",
    block_size=32,
    is_symmetric=True,  # INT4 대칭 양자화
    accuracy_level=4,   # 최고 정확도
    algorithm="RTN",
)
quantizer.process()
quantizer.model.save("model_int4.onnx")
```

### 기대 효과

- **임베딩 모델 3x 속도 향상** 사례 보고 (nixiesearch 벤치마크)
- 메모리 대역폭 감소 → CPU 캐시 효율 개선
- KleidiAI의 ARM 최적화 커널이 MatMulNBits를 직접 가속

### 주의사항

- Mac ARM (Apple Silicon)에서 MatMulNBits 커널 지원 여부 확인 필요
- KleidiAI의 SME/SVE2 최적화는 Armv9 (Graviton3+, Cortex-X3+) 대상
- Apple M-series는 Neon만 지원 → 가속 폭이 Armv9 대비 제한적
- 임베딩 품질(cosine similarity) 검증 필수

### 적용 가능성: ★★★★☆ (높음)

INT8보다 더 공격적인 양자화로 추가 속도 개선 가능성 있음.
단, Apple Silicon의 MatMulNBits 커널 성능 실측이 필요.

---

## 2. 선택적 양자화 튜닝 (Selective Quantization Tuning)

### 개요

모든 레이어를 동일하게 양자화하는 대신, 레이어별 민감도를 측정하여
정확도 손실이 큰 레이어만 FP32로 유지하고 나머지를 양자화하는 기법.

### 핵심 메커니즘 (TuneQn 프레임워크)

1. **QDQ Error**: 원본과 양자화-역양자화 결과 간 FP32 차이 측정
2. **XModel Error**: 완전 양자화 모델에서 각 레이어가 유발하는 상대 오차 계산
3. 두 메트릭을 50:50 가중 합산하여 민감도 순위 결정
4. 가장 민감한 레이어부터 순차적으로 FP32 면제

### 현재 상태와의 차이

현재 QDQ INT8은 "첫/마지막 2레이어 FP32 유지"라는 **경험적 규칙**을 적용 중.
선택적 튜닝은 **데이터 기반 측정**으로 최적 면제 레이어 조합을 탐색한다.

### 기대 효과

- 정확도 손실 최대 54% 감소 (동일 양자화 비율 대비)
- 또는 동일 정확도에서 더 많은 레이어 양자화 → 추가 속도 개선
- 현재 cosine 0.9889 → 0.995+ 도달 가능성

### 주의사항

- TuneQn은 현재 트랜스포머 모델 미지원 (확장 예정)
- 수동으로 레이어별 프로파일링 + 실험 반복 필요
- 캘리브레이션 데이터셋(법률 도메인 쿼리) 준비 필요

### 적용 가능성: ★★★☆☆ (중간)

원리는 유효하지만 자동화 도구가 트랜스포머를 미지원하므로 수동 구현 필요.

---

## 3. 구조적 프루닝 (Structured Pruning) + ONNX 재학습

### 개요

트랜스포머의 어텐션 헤드, FFN 뉴런, 또는 전체 레이어를 제거하여
모델 크기와 연산량을 직접 줄이는 기법.

### 프루닝 유형

| 유형 | 대상 | CPU 가속 | 정확도 영향 |
|------|------|---------|-----------|
| **어텐션 헤드 프루닝** | 16개 헤드 중 일부 제거 | 중간 | 낮음~중간 |
| **FFN 뉴런 프루닝** | 4096 뉴런 중 일부 제거 | 중간 | 낮음~중간 |
| **레이어 프루닝** | 24개 레이어 중 일부 제거 | **높음** | 중간~높음 |

### 레이어 프루닝 상세

24-layer XLM-RoBERTa에서 중간 레이어(예: 9~16번)를 제거하면:
- **12-layer 모델**: 이론상 ~2x 속도 향상 (772ms → ~400ms 가능)
- **18-layer 모델**: ~1.3x 속도 향상
- fine-tuning 없이도 상위/하위 레이어만으로 임베딩 품질 유지 가능성

### ONNXPruner 활용

```
원본 모델 → ONNX 변환 → ONNXPruner (node association tree)
→ 구조적 프루닝 → ORT fine-tuning (10 epochs) → 양자화
```

- ONNXPruner는 ViT-B/16에서 검증됨 (트랜스포머 호환)
- L1/L2-norm, Hrank 평가 기준 지원
- **트리 레벨 평가**: 단일 노드가 아닌 연결 패턴 전체를 고려

### 기대 효과

- 레이어 프루닝 + INT8 조합 시 **3~4x 속도 향상** 가능
- BERT-Base 사례: 45ms → 23ms (ONNX Runtime 최적화 포함)
- 프루닝 후 모델 크기 50% 이상 감소

### 주의사항

- fine-tuning 데이터셋 필요 (법률 도메인 임베딩 학습 데이터)
- 임베딩 품질 검증 필수 (RAG recall/MRR 영향 평가)
- ONNXPruner의 트랜스포머 연산자 지원 범위 확인 필요
- 학습 인프라(GPU) 필요

### 적용 가능성: ★★★☆☆ (중간)

가장 큰 속도 향상 잠재력이 있지만, fine-tuning 인프라와 데이터가 필요.

---

## 4. 지식 증류 (Knowledge Distillation) → 소형 모델

### 개요

24-layer KURE-v1을 교사(teacher)로, 6~12 layer 소형 모델을 학생(student)으로
증류하여 근사한 임베딩 품질을 유지하면서 속도를 대폭 향상.

### 증류 전략

| 학생 모델 | 레이어 | 예상 속도 | 예상 품질 |
|----------|--------|----------|----------|
| XLM-RoBERTa Base (12L) | 12 | ~3x | 90-95% |
| DistilXLM-RoBERTa (6L) | 6 | ~5x | 85-92% |
| MiniLM (6L, 384d) | 6 | ~8x | 80-90% |

### sentence-transformers 증류

```python
from sentence_transformers import SentenceTransformer, distillation

teacher = SentenceTransformer("nlpai-lab/KURE-v1")
student = SentenceTransformer("xlm-roberta-base")  # 12-layer
# 증류 학습 후 ONNX 변환 + INT8 양자화
```

### 기대 효과

- 증류 모델 + ORT INT8: **5~10x 속도 향상** 가능
- 메모리 사용량 50-75% 감소
- 배포 크기 대폭 축소

### 주의사항

- 법률 도메인 증류용 학습 데이터 필요 (쿼리-문서 쌍)
- 증류 후 RAG 검색 품질 재평가 필수 (Recall@10, MRR)
- 학습 인프라(GPU) 필요
- 증류는 최적화가 아닌 **모델 교체**에 해당

### 적용 가능성: ★★☆☆☆ (낮음-중간)

최대 효과가 가능하지만 학습 데이터/인프라 요구사항이 높고,
기존 KURE-v1의 법률 도메인 성능을 보존하기 어려울 수 있음.

---

## 5. ORT 프로파일링 기반 병목 분석 + 타깃 최적화

### 개요

ORT의 내장 프로파일러로 연산자별 실행 시간을 측정하여
실제 병목 연산자를 식별하고 해당 연산자만 집중 최적화하는 접근.

### 프로파일링 방법

```python
import onnxruntime as ort

options = ort.SessionOptions()
options.enable_profiling = True
options.profile_file_prefix = "kure_v1_profile"

session = ort.InferenceSession("model.onnx", options)
# 추론 실행 ...
profile_file = session.end_profiling()
# → kure_v1_profile_*.json 생성
```

### 분석 도구

```bash
# ORT 내장 트랜스포머 프로파일러
python -m onnxruntime.transformers.profiler --input kure_v1_profile.json
# → 상위 비용 노드, 연산자 유형별 시간 집계 출력

# Chrome에서 시각화
# chrome://tracing → JSON 파일 로드
```

### 기대 효과

- 실제 병목이 MatMul인지, Attention인지, LayerNorm인지 정확히 파악
- 병목 연산자에 맞춤 최적화 (예: MatMul이 병목이면 INT4 양자화 우선)
- 불필요한 최적화 시도 방지

### 적용 가능성: ★★★★★ (매우 높음)

다른 모든 최적화에 앞서 수행해야 할 **필수 단계**.
비용 없이 즉시 실행 가능하며, 이후 최적화 방향을 결정하는 근거가 됨.

---

## 6. Dynamic Quantization (동적 양자화)

### 개요

현재 사용 중인 QDQ (정적 양자화)와 달리, 가중치만 미리 양자화하고
활성화는 추론 시 동적으로 양자화 파라미터를 계산하는 방식.

### 정적 vs 동적 양자화

| 항목 | 정적 (QDQ, 현재) | 동적 |
|------|----------------|------|
| 가중치 | 미리 양자화 | 미리 양자화 |
| 활성화 | 캘리브레이션 기반 고정 | 추론 시 동적 계산 |
| 캘리브레이션 | 필요 | **불필요** |
| 정확도 | 높음 | **더 높음** (입력 적응) |
| 속도 | 빠름 | 약간 느림 (동적 계산 오버헤드) |

### 적용 방법

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="model_optimized.onnx",
    model_output="model_dynamic_int8.onnx",
    weight_type=QuantType.QInt8,
    op_types_to_quantize=["MatMul", "Attention"],
)
```

### 기대 효과

- 캘리브레이션 불필요 → 즉시 적용 가능
- 트랜스포머에서는 동적 양자화가 정적보다 정확도 우수한 경우가 많음
- sentence-transformers ONNX INT8 동적 양자화: **3.08x 속도 향상** 보고

### 주의사항

- 현재 QDQ INT8 (1.94x)과 성능 비교 필요
- 동적 계산 오버헤드로 정적 양자화보다 느릴 수 있음
- ARM에서는 dot-product 명령어로 INT8 연산 가속

### 적용 가능성: ★★★★☆ (높음)

캘리브레이션 없이 즉시 적용 가능하며, 정확도가 더 나을 수 있음.
현재 QDQ와 직접 비교 벤치마크 추천.

---

## 7. ONNX 그래프 수술 (Graph Surgery) — 불필요 연산 제거

### 개요

ONNX 모델 그래프에서 추론에 불필요한 노드(학습 전용 노드, 중복 Cast,
사용되지 않는 출력 등)를 수동으로 제거하여 그래프를 경량화.

### 활용 도구

- **onnx-tool**: ONNX 모델 파싱, 편집, 최적화, 프로파일링
- **ORT Model Editor API** (v1.22+): 프로그래밍 방식으로 ONNX 모델 편집
- **onnx.helper**: ONNX 공식 그래프 조작 유틸리티

### 가능한 최적화

| 최적화 | 설명 | 기대 효과 |
|--------|------|----------|
| 불필요 Cast 제거 | FP32→FP32 등 무의미한 타입 변환 | 미미 |
| 미사용 출력 제거 | pooler_output 등 불필요 분기 제거 | 소폭 |
| Constant Folding 강화 | ORT가 놓친 상수 연산 미리 계산 | 소폭 |
| Shape 추론 최적화 | 불필요한 동적 shape 계산 제거 | 소폭~중간 |

### 적용 가능성: ★★☆☆☆ (낮음)

ORT의 기본 그래프 최적화가 이미 대부분 처리.
수동 그래프 수술의 추가 이점은 제한적.

---

## 8. KleidiAI 네이티브 빌드 (ORT 소스 빌드)

### 개요

ORT 1.22+에서 통합된 KleidiAI 라이브러리를 활성화하기 위해
소스에서 ORT를 빌드하여 ARM MLAS 최적화 커널을 직접 활용.

### KleidiAI가 가속하는 연산

| 연산 | 설명 | ORT 버전 |
|------|------|---------|
| SGEMM | 단정밀도 행렬곱 | 1.22+ |
| IGEMM | 정수 행렬곱 | 1.22+ |
| Dynamic Quantized MatMul | 동적 양자화 행렬곱 | 1.22+ |
| MatMulNBits | N-bit 가중치 행렬곱 | 1.22+ |
| Conv2D (SME2) | SME2 가속 합성곱 | 1.23+ |

### Apple Silicon 제약

| ARM 기능 | Apple M-series | Graviton3+ | Cortex-X3+ |
|----------|---------------|------------|------------|
| Neon | O | O | O |
| SVE2 | X | O | O |
| SME | X | X | O (일부) |

Apple Silicon은 Neon만 지원하므로 KleidiAI의 SVE2/SME 최적화는 사용 불가.
단, Neon 기반 SGEMM/IGEMM 최적화는 기대 가능.

### 빌드 방법

```bash
git clone https://github.com/microsoft/onnxruntime.git
cd onnxruntime
./build.sh --config Release --parallel \
  --cmake_extra_defines onnxruntime_USE_KLEIDIAI=ON
pip install build/Linux/Release/dist/onnxruntime-*.whl
```

### 기대 효과

- pip 배포판 대비 ARM 최적화 커널 활성화
- SGEMM/IGEMM 성능 향상 (정확한 수치는 실측 필요)
- Graviton3+ 서버에서는 **최대 2.6x 향상** 보고

### 주의사항

- 소스 빌드에 시간 소요 (30분~1시간)
- pip 배포판에 KleidiAI가 이미 포함되었을 가능성 (확인 필요)
- Apple Silicon에서의 실효성은 미지수

### 적용 가능성: ★★★☆☆ (중간)

Graviton3+ 서버 배포 시 높은 가치. Apple Silicon에서는 제한적.

---

## 9. CoreML Execution Provider (Apple Silicon 전용)

### 개요

Apple의 CoreML 프레임워크를 ORT의 Execution Provider로 활용하여
Apple Neural Engine (ANE) 및 Apple GPU를 통한 하드웨어 가속을 시도하는 기법.

### 설정 방법

```python
import onnxruntime as ort

providers = [
    ('CoreMLExecutionProvider', {
        "ModelFormat": "MLProgram",        # NeuralNetwork(구형) vs MLProgram(신형)
        "MLComputeUnits": "ALL",           # CPU_ONLY, CPU_AND_GPU, ALL(ANE 포함)
        "RequireStaticInputShapes": True,  # 정적 shape 강제 (성능 향상)
    }),
    'CPUExecutionProvider',  # fallback
]
session = ort.InferenceSession("model.onnx", providers=providers)
```

### 트랜스포머 모델 호환성 문제

| 연산자 | NeuralNetwork | MLProgram | 제약사항 |
|--------|:------------:|:---------:|---------|
| MatMul | △ | △ | 상수 가중치만 (NeuralNetwork), transA==0 (MLProgram) |
| LayerNormalization | X | O | NeuralNetwork 미지원 |
| Attention (fused) | X | X | **전용 연산자 없음** |
| Gelu | X | O | MLProgram만 지원 |
| Softmax | O | O | - |

**핵심 제약**: Fused Attention 연산자를 지원하지 않아 ORT 최적화된 트랜스포머 모델의
상당 부분이 CPU EP로 fallback됨. 이는 EP 간 데이터 전송 오버헤드를 유발.

### Mac ARM 호환: O (Apple Silicon 전용)
### 구현 난이도: 중간
### 예상 성능 향상: 불확실 (Attention fallback으로 인해 MPS와 유사한 역효과 가능)

### 적용 가능성: ★★☆☆☆ (낮음)

Fused Attention 미지원으로 XLM-RoBERTa 같은 트랜스포머에서는 실효성이 낮음.
이미 MPS GPU가 역효과(1530ms)를 보인 것과 유사한 패턴이 예상됨.

---

## 10. XNNPACK Execution Provider

### 개요

Google의 XNNPACK 라이브러리 기반 EP. ARM Neon SIMD를 활용한
부동소수점/양자화 추론에 최적화된 저수준 커널 제공.

### 지원 연산자

| 연산자 | 지원 | 제약사항 |
|--------|:----:|---------|
| MatMul | O | 2D만 지원 |
| Gemm | O | 2D만 지원 |
| Conv | O | 2D만, 상수 가중치 |
| Softmax | O | opset 13 미만 또는 마지막 축만 |
| QLinearConv | O | 양자화 합성곱 |
| Attention | **X** | 미지원 |
| LayerNorm | **X** | 미지원 |

### 스레드 최적화 설정

```python
import onnxruntime as ort

options = ort.SessionOptions()
options.add_session_config_entry("session.allow_spinning", "0")
options.intra_op_num_threads = 1  # ORT 스레드풀은 1로

providers = [
    ('XnnpackExecutionProvider', {
        'intra_op_num_threads': 4,  # XNNPACK 자체 스레드풀 = P코어 수
    }),
    'CPUExecutionProvider',
]
```

### Mac ARM 호환: O (ARM Neon)
### 구현 난이도: 중간 (소스 빌드 필요할 수 있음)
### 예상 성능 향상: 낮음~중간 (MatMul/Gemm 가속은 가능하나 Attention fallback)

### 적용 가능성: ★★☆☆☆ (낮음)

CoreML EP와 동일 문제 — Attention/LayerNorm 미지원으로 트랜스포머에서 효과 제한.
CNN/경량 모델에는 유효하지만 XLM-RoBERTa Large에는 부적합.

---

## 11. ONNX Opset 업그레이드

### 개요

ONNX 모델의 opset 버전을 최신으로 업그레이드하여 새로운 연산자와
최적화된 구현을 활용하는 기법.

### 최신 Opset 주요 변경

| Opset | 주요 변경 | 트랜스포머 관련성 |
|-------|----------|----------------|
| 18 | RMSNormalization, RotaryEmbedding 추가 | 높음 (최신 트랜스포머) |
| 19 | FLOAT8E8M0 지원 추가 | 낮음 (FP8 양자화) |
| 20 | AffineGrid, ImageDecoder 등 | 낮음 (이미지 처리) |
| 21 | INT4/UINT4 데이터 타입 | 높음 (4-bit 양자화 전제) |

### 적용 방법

```python
import onnx
from onnx import version_converter

model = onnx.load("model_opset17.onnx")
converted = version_converter.convert_version(model, 21)
onnx.save(converted, "model_opset21.onnx")
```

### 기대 효과

- Opset 21은 INT4 MatMulNBits의 전제 조건
- RMSNormalization/RotaryEmbedding은 XLM-RoBERTa에 직접 해당하지 않음 (BERT 계열은 LayerNorm + absolute position)
- 기존 연산자의 최적화된 재구현 가능성은 있으나 벤치마크 필요

### Mac ARM 호환: O
### 구현 난이도: 낮음
### 예상 성능 향상: 미미~낮음 (단독으로는 효과 제한, INT4와 조합 시 유효)

### 적용 가능성: ★★★☆☆ (중간)

INT4 양자화(기법 #1)를 적용하려면 opset 21이 필수이므로 전제 조건으로 유효.
단독으로는 성능 향상 기대하기 어려움.

---

## 12. ORT Extensions / Custom Operators

### 개요

ORT에서 기본 지원하지 않는 연산자를 C++/CUDA로 직접 구현하여 등록하거나,
`onnxruntime-extensions` 패키지의 사전 구축된 커스텀 연산자를 활용하는 기법.

### 가능한 활용

| 커스텀 연산자 | 용도 | 효과 |
|-------------|------|------|
| 토크나이저 통합 | 전처리를 모델 그래프 내부로 | 파이프라인 간소화 |
| Fused MHA (커스텀) | 최적화된 Multi-Head Attention | 잠재적 속도 향상 |
| 도메인 특화 후처리 | mean pooling 등 통합 | 미미한 오버헤드 감소 |

### 주의사항

- CPU EP와 CUDA EP에서만 커스텀 연산자 지원
- C++ 구현 + 빌드 필요 → 난이도 높음
- ORT 내장 Fused Attention이 이미 충분히 최적화됨

### Mac ARM 호환: O (CPU EP)
### 구현 난이도: 높음
### 예상 성능 향상: 낮음 (기존 ORT 최적화가 이미 대부분 커버)

### 적용 가능성: ★☆☆☆☆ (매우 낮음)

개발 비용 대비 효과가 미미. ORT의 기존 트랜스포머 최적화가 이미 충분.

---

## 우선순위 정리

| 순위 | 기법 | 난이도 | 기대 효과 | Mac ARM 호환 | 즉시 적용 |
|------|------|--------|----------|:----------:|:--------:|
| **1** | **ORT 프로파일링 병목 분석** | 낮음 | 방향 설정 | O | O |
| **2** | **동적 양자화 (Dynamic INT8)** | 낮음 | QDQ와 비교 | O | O |
| **3** | **INT4 Weight-Only (MatMulNBits)** | 중간 | 높음 | O | O |
| **4** | **선택적 양자화 튜닝** | 중간 | 중간 | O | △ |
| **5** | ONNX Opset 21 업그레이드 | 낮음 | 낮음 (INT4 전제) | O | O |
| **6** | KleidiAI 소스 빌드 | 중간 | 중간 | △ | △ |
| **7** | 구조적 프루닝 | 높음 | 높음 | O | X |
| **8** | 지식 증류 | 높음 | 매우 높음 | O | X |
| **9** | CoreML EP | 중간 | 불확실 | O (전용) | △ |
| **10** | XNNPACK EP | 중간 | 낮음 | O | △ |
| **11** | 그래프 수술 | 낮음 | 낮음 | O | O |
| **12** | ORT Custom Operators | 높음 | 낮음 | O | X |

## 권장 실행 계획

### 즉시 실행 (코드만으로 가능)

1. **프로파일링** → 병목 연산자 식별
2. **동적 양자화** → QDQ INT8 (772ms)과 직접 비교
3. **INT4 Weight-Only** → Opset 21 업그레이드 + MatMulNBits 적용 후 속도/정확도 측정

### 중기 (환경 구축 필요)

4. **선택적 양자화** → 레이어별 민감도 수동 측정
5. **KleidiAI 소스 빌드** → 특히 Graviton3 배포 대상 시

### 장기 (학습 인프라 필요)

6. **구조적 프루닝** → 24L → 12~18L 축소
7. **지식 증류** → 소형 모델 학습

### 비추천 (효과 대비 비용 높음)

8. CoreML EP → Attention fallback 문제
9. XNNPACK EP → 트랜스포머 연산자 미지원
10. ORT Custom Operators → 기존 최적화로 충분

---

## 참고 자료

- [ONNX Runtime Quantization 공식 문서](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [ONNX Runtime Transformer Optimizer](https://onnxruntime.ai/docs/performance/transformers-optimization.html)
- [KleidiAI + ORT 통합 블로그](https://onnxruntime.ai/blogs/arm-microsoft-kleidiai)
- [ORT v1.22 Release Notes](https://github.com/microsoft/onnxruntime/releases/tag/v1.22.0)
- [ORT v1.23 Release Notes](https://github.com/microsoft/onnxruntime/releases/tag/v1.23.0)
- [ONNXPruner 논문](https://arxiv.org/abs/2404.08016)
- [Selective Quantization Tuning (TuneQn)](https://arxiv.org/abs/2507.12196)
- [Sentence-Transformers 효율성 가이드](https://sbert.net/docs/sentence_transformer/usage/efficiency.html)
- [ORT 프로파일링 도구](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html)
- [ORT Graph Optimizations](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html)
- [ONNX 4-bit 데이터 타입](https://onnx.ai/onnx/technical/int4.html)
- [CoreML Execution Provider](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)
- [XNNPACK Execution Provider](https://onnxruntime.ai/docs/execution-providers/Xnnpack-ExecutionProvider.html)
- [ONNX Operators (Opset 변경사항)](https://onnx.ai/onnx/operators/)
- [ORT Custom Operators](https://onnxruntime.ai/docs/reference/operators/add-custom-op.html)
