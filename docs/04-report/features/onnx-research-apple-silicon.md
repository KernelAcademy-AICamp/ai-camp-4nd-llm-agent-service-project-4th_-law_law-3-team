# Apple Silicon 전용 ML 가속 기법 리서치

> 조사 일자: 2026-02-24
> 대상 모델: nlpai-lab/KURE-v1 (XLM-RoBERTa Large, 24 Transformer layers, 1024 dim)
> 현재 환경: Apple Silicon M-series (arm64), Darwin, 8 CPU (4P+4E), 16GB RAM
> 현재 최선: ORT QDQ INT8 = 772ms (1.94x vs PyTorch CPU 1501ms)

---

## 1. CoreML Execution Provider (ORT 내장)

### 1.1 개요

ONNX Runtime에 내장된 CoreML EP는 ONNX 모델을 Apple의 CoreML 형식으로 변환하여
CPU, GPU, ANE(Apple Neural Engine)에 자동 디스패치한다.
현재 프로젝트에서는 `CPUExecutionProvider`만 사용 중이므로, CoreML EP로 전환하면
GPU/ANE 하드웨어 가속을 활용할 수 있다.

### 1.2 Python 사용법

```python
import onnxruntime as ort

providers = [
    ('CoreMLExecutionProvider', {
        'ModelFormat': 'MLProgram',        # MLProgram (최신) or NeuralNetwork
        'MLComputeUnits': 'ALL',           # ALL | CPUAndNeuralEngine | CPUAndGPU | CPUOnly
        'RequireStaticInputShapes': '1',   # 동적 shape 비활성화 (성능 향상)
        'EnableOnSubgraphs': '0',
    }),
    'CPUExecutionProvider',  # fallback
]

session = ort.InferenceSession('model.onnx', providers=providers)
```

### 1.3 모델 포맷 비교

| 항목 | NeuralNetwork | MLProgram |
|------|---------------|-----------|
| 요구 OS | macOS 10.15+ | macOS 12+ |
| CoreML 버전 | 3+ | 5+ |
| FP16 변환 | 자동(MPS 사용 시) | FP32 유지 가능 |
| 지원 연산자 | ~40개 | 확장됨 (Gelu, LayerNorm, Erf 등) |
| **Transformer 적합성** | 낮음 | **높음** (LayerNormalization, Gelu 지원) |

### 1.4 주요 설정 옵션

| 옵션 | 값 | 설명 |
|------|-----|------|
| `ModelFormat` | `MLProgram` | Transformer 연산자 지원을 위해 필수 |
| `MLComputeUnits` | `ALL` | ANE+GPU+CPU 자동 디스패치 |
| `RequireStaticInputShapes` | `1` | 동적 shape → 성능 저하 방지 |
| `ModelCacheDirectory` | 경로 | 컴파일된 CoreML 모델 캐싱 (재시작 시 속도 향상) |
| `SpecializationStrategy` | `FastPrediction` | 빠른 예측 최적화 |
| `ProfileComputePlan` | `1` | 하드웨어 디스패치 진단 로깅 |

### 1.5 Transformer 모델 적용 시 제약사항

- **부분 지원 문제**: CoreML EP가 모든 연산을 지원하지 않으면 지원되지 않는 노드가 CPU EP로 폴백된다. XLM-RoBERTa Large(24 layers)처럼 복잡한 모델은 상당수 노드가 CPU로 폴백될 수 있다.
- **데이터 전송 오버헤드**: CPU↔ANE 간 데이터 전송이 빈번하면 오히려 CPU 단독 실행보다 느려질 수 있다. 실제 사례에서 CoreML EP(163ms) > CPU EP(78ms)로 역전 현상 보고됨.
- **동적 shape 성능 저하**: Transformer의 가변 시퀀스 길이가 성능에 부정적 영향. `RequireStaticInputShapes=1` + 고정 길이 패딩 권장.
- **첫 실행 컴파일 오버헤드**: CoreML 모델 컴파일에 수 초~수십 초 소요. `ModelCacheDirectory` 설정 필수.

### 1.6 적용 가능성 평가

| 항목 | 평가 |
|------|------|
| 구현 난이도 | **낮음** (providers 배열 변경만) |
| 기대 효과 | **불확실** (부분 폴백 시 역효과 가능) |
| 리스크 | 중간 (fallback으로 CPU EP 유지 가능) |
| 권장 | **실험적 벤치마크 필요** |

---

## 2. Apple Neural Engine (ANE)

### 2.1 하드웨어 스펙

| 칩 | ANE 코어 | 연산 성능 (TOPS) | 비고 |
|----|---------|-----------------|------|
| M1 | 16 | 11 (FP16) | 초기 Apple Silicon |
| M2 | 16 | 15.8 (FP16) | 1.44x vs M1 |
| M3 | 16 | 18 (FP16) | |
| M4 | 16 | 38 (INT4) / ~19 (INT8) | INT4 기준 2x, INT8은 M3과 유사 |
| M5 (최신) | - | - | 시간-첫-토큰 4x 개선 (Neural Accelerators) |

### 2.2 ANE 최적화 Transformer (Apple 공식 연구)

Apple이 공개한 ANE 최적화 Transformer 아키텍처는 다음 원칙을 따른다:

**원칙 1: Chunked Attention**
- Q/K/V 텐서를 단일 헤드 단위로 분할하여 L2 캐시 활용 극대화
- 멀티코어 ANE에서 병렬 처리 효율 향상

**원칙 2: 데이터 레이아웃 변환**
- 표준 `(B, S, C)` → ANE 최적 `(B, C, 1, S)` 4D 포맷 변환
- `nn.Linear` → `nn.Conv2d`로 대체 (가중치 자동 변환)
- ANE 버퍼의 마지막 축 64바이트 정렬 필수 (미정렬 시 FP16 32배, INT8 64배 메모리 페널티)

**원칙 3: 메모리 복사 최소화**
- `einsum("bchq,bkhc→bkhq")` 패턴으로 중간 transpose/reshape 제거
- 메모리 복사가 대역폭 병목의 주 원인

### 2.3 ANE 벤치마크 (Apple 공식)

**DistilBERT, iPhone 13 (A15), seq_len=128, batch=1:**
- 레이턴시: **3.47ms** @ 0.454W
- 기본 대비: **10x 빠르고 14x 적은 메모리**
- AWS Inferentia(서버 사이드) 5-6ms보다 빠름

### 2.4 우리 모델(XLM-RoBERTa Large)에 적용 시 한계

| 문제 | 상세 |
|------|------|
| 아키텍처 수정 필요 | Apple의 ANE 최적화는 `nn.Linear→nn.Conv2d` 변환, 레이아웃 변경 등 모델 아키텍처 자체를 수정해야 함. 사전학습 가중치 호환은 유지되지만 코드 수준 변경 필요. |
| 대역폭 제한 | 시퀀스 길이가 짧을 때 대형 파라미터 텐서 페치가 병목. XLM-RoBERTa Large(1.2GB)는 이 문제가 심각할 수 있음. |
| ONNX 경로 비호환 | ANE 최적화는 PyTorch → CoreML 직접 변환 경로를 사용. 현재 ONNX 파이프라인과 양립 불가. |
| 양자화 필요성 | 대역폭 문제 완화를 위해 INT8/INT4 양자화가 사실상 필수. |

### 2.5 적용 가능성 평가

| 항목 | 평가 |
|------|------|
| 구현 난이도 | **높음** (모델 아키텍처 변경 + CoreML 변환 파이프라인 구축) |
| 기대 효과 | **높음** (DistilBERT 기준 10x, 대형 모델은 2-5x 예상) |
| 리스크 | 높음 (정확도 검증, 파이프라인 복잡성) |
| 권장 | **장기 과제로 분류** (현재 ONNX 파이프라인 유지 우선) |

---

## 3. CoreML 직접 변환 (coremltools)

### 3.1 개요

`coremltools`를 사용하여 PyTorch 모델을 CoreML 형식(.mlpackage)으로 직접 변환하고,
ANE/GPU를 활용하는 방식. ORT를 거치지 않는 별도 추론 파이프라인.

### 3.2 변환 방법

```python
import coremltools as ct
import torch

# PyTorch 모델 로드
model = AutoModel.from_pretrained("nlpai-lab/KURE-v1")
model.eval()

# 트레이싱
dummy_input = {
    'input_ids': torch.randint(0, 30000, (1, 128)),
    'attention_mask': torch.ones(1, 128, dtype=torch.long),
}
traced = torch.jit.trace(model, (dummy_input['input_ids'], dummy_input['attention_mask']))

# CoreML 변환
mlmodel = ct.convert(
    traced,
    inputs=[
        ct.TensorType(name="input_ids", shape=(1, 128), dtype=np.int32),
        ct.TensorType(name="attention_mask", shape=(1, 128), dtype=np.int32),
    ],
    compute_units=ct.ComputeUnit.ALL,  # ANE+GPU+CPU
    convert_to="mlprogram",
)
mlmodel.save("kure_v1.mlpackage")
```

### 3.3 제약사항

| 문제 | 상세 |
|------|------|
| ONNX 경로 비호환 | ONNX→CoreML 변환기는 deprecated. PyTorch→CoreML 직접 변환 권장. |
| 2GB 제한 | XLM-RoBERTa Large(~1.2GB)는 범위 내이나, 외부 데이터 모델은 추가 처리 필요. |
| 동적 shape | CoreML은 고정 shape에서 최적 성능. 가변 시퀀스 길이 처리 비효율적. |
| 추론 API 변경 | `ort.InferenceSession` 대신 `coremltools.models.MLModel.predict()` 사용 필요. |
| macOS 전용 | Linux/Windows에서는 사용 불가. 크로스 플랫폼 배포 시 분기 필요. |

### 3.4 적용 가능성 평가

| 항목 | 평가 |
|------|------|
| 구현 난이도 | **중간** (변환 스크립트 + 추론 코드 분기) |
| 기대 효과 | **중~높음** (ANE 활용 시 2-5x 가능) |
| 리스크 | 중간 (macOS 전용, 정확도 검증 필요) |
| 권장 | **CoreML EP 실험 후 차선책으로 고려** |

---

## 4. MLX (Apple ML 프레임워크)

### 4.1 개요

Apple이 개발한 Apple Silicon 전용 ML 프레임워크. NumPy-like API로 GPU(Metal) 연산을 활용한다.
Lazy evaluation + unified memory 아키텍처를 최적 활용.

### 4.2 Transformer 벤치마크 (학술 논문)

**출처: "Benchmarking On-Device Machine Learning on Apple Silicon with MLX" (arXiv:2510.18921)**

| 모델 | CUDA A10 | M1 (8GB) | M2 Max (32GB) | M2 Max/CUDA 비율 |
|------|----------|----------|---------------|------------------|
| BERT-base | 23.46ms | 179.35ms | 38.23ms | 1.63x 느림 |
| BERT-large | 62.55ms | 531.27ms | 94.06ms | 1.50x 느림 |
| RoBERTa-base | 39.08ms | 401.27ms | 74.32ms | 1.90x 느림 |
| **XLM-RoBERTa-base** | **16.09ms** | **142.42ms** | **27.37ms** | **1.70x 느림** |

**배치 효율 (M2 Max, BERT-base):**
- batch=1: 8.02ms
- batch=32: 70.48ms (9x 증가, 32x 아님 → 효율적 병렬 처리)

### 4.3 우리 모델 적용 시 예상 성능

XLM-RoBERTa-**base**가 M2 Max에서 27.37ms이므로,
XLM-RoBERTa-**Large**(3배 파라미터)는 대략 80-100ms 예상.
단, 이는 M2 Max(32GB)이며 M1/M3 8GB에서는 더 느릴 수 있다.

현재 ORT QDQ INT8(772ms)과 비교하면 MLX가 상당히 빠를 가능성이 있으나:
- MLX 수치는 GPU(Metal) 활용, 우리 ORT는 CPU 전용
- MLX는 FP32, ORT QDQ는 INT8 양자화 적용
- 실제 비교에는 동일 조건 벤치마크 필요

### 4.4 MLX 적용 시 고려사항

| 항목 | 상세 |
|------|------|
| 모델 변환 | HuggingFace 모델을 MLX 포맷으로 변환 필요 (`mlx-lm` 또는 수동) |
| API 변경 | ORT 세션 대신 MLX 추론 코드 작성 필요 |
| macOS 전용 | Linux 서버 배포 불가 (Apple Silicon 전용) |
| 양자화 | MLX 자체 양자화(4-bit, 8-bit) 지원 |
| 생태계 | 주로 LLM 추론에 최적화, 임베딩 모델 지원은 제한적 |

### 4.5 적용 가능성 평가

| 항목 | 평가 |
|------|------|
| 구현 난이도 | **중~높음** (모델 변환 + 추론 파이프라인 재작성) |
| 기대 효과 | **높음** (GPU 활용으로 CPU 대비 2-5x 가능) |
| 리스크 | 높음 (macOS 전용, ONNX 파이프라인 대체) |
| 권장 | **실험적 벤치마크 후 결정** |

---

## 5. Metal Performance Shaders (MPS) via PyTorch

### 5.1 개요

PyTorch의 MPS 백엔드를 통해 Apple GPU(Metal)로 추론하는 방식.
`device="mps"`로 간단히 전환 가능.

### 5.2 벤치마크 (Explosion / spaCy)

| 디바이스 | AMX CPU (WPS) | GPU MPS (WPS) | GPU/CPU 배수 |
|---------|---------------|---------------|-------------|
| Mac Mini M1 | 1,180 | 2,202 | 1.9x |
| MacBook Air M2 | 1,242 | 3,362 | 2.7x |
| MacBook Pro M1 Pro | 1,631 | 4,661 | 2.9x |
| MacBook Pro M1 Max | 1,821 | 8,648 | **4.7x** |
| Mac Studio M1 Ultra | 2,197 | 12,073 | **5.5x** |

### 5.3 현재 프로젝트에서의 MPS 결과

> **이미 테스트 완료**: PyTorch MPS = 1530ms (CPU 1501ms와 동등, 효과 없음)

이유 분석:
- M-series 기본 모델(8-core GPU)에서는 GPU 코어가 부족하여 MPS 이점 미미
- M1 Max/Ultra (24-64 GPU 코어) 이상에서만 유의미한 가속
- 단일 쿼리(batch=1) 추론에서는 GPU 오버헤드가 연산 이득을 상쇄
- FlashAttention, xFormers SDPA 커널 미지원 (Apple Silicon 한계)

### 5.4 MPS 한계 (2025 현재)

- `torch.compile` MPS 지원: 초기 단계
- FlashAttention / xFormers: 미지원
- bitsandbytes (8/4-bit 양자화): 미지원
- `scaled_dot_product_attention`: macOS에서 크래시 보고 있음

### 5.5 적용 가능성 평가

| 항목 | 평가 |
|------|------|
| 구현 난이도 | 낮음 (device 변경만) |
| 기대 효과 | **없음** (이미 테스트 완료, 효과 없음) |
| 리스크 | 낮음 |
| 권장 | **불채택** (M1 Max 이상에서만 의미) |

---

## 6. 기술별 비교 총괄

### 6.1 적용 우선순위

| 순위 | 기술 | 기대 효과 | 구현 난이도 | 리스크 | 권장 |
|------|------|----------|------------|--------|------|
| 1 | **CoreML EP (ORT 내장)** | 중 (1.5-3x) | **낮음** | 중 | **1순위 실험** |
| 2 | **MLX 직접 추론** | 높음 (2-5x) | 높음 | 높음 | 장기 실험 |
| 3 | **CoreML 직접 변환** | 중~높음 (2-5x) | 중간 | 중간 | CoreML EP 실패 시 |
| 4 | **ANE 최적화 아키텍처** | 높음 (5-10x) | 매우 높음 | 높음 | 장기 과제 |
| 5 | ~~MPS (PyTorch)~~ | 없음 | 낮음 | 낮음 | **불채택** |

### 6.2 단기 실행 계획 (CoreML EP)

현재 코드 변경점 (`onnx_session.py:_create_session`):

```python
# 현재 (CPU 전용)
providers=["CPUExecutionProvider"]

# 변경 (CoreML EP 추가)
providers = [
    ('CoreMLExecutionProvider', {
        'ModelFormat': 'MLProgram',
        'MLComputeUnits': 'ALL',
        'RequireStaticInputShapes': '1',
        'ModelCacheDirectory': str(model_dir / '.coreml_cache'),
    }),
    'CPUExecutionProvider',  # fallback
]
```

추가 고려사항:
- 첫 실행 시 CoreML 컴파일 오버헤드 → `ModelCacheDirectory` 필수
- `RequireStaticInputShapes=1` + static variant(static128) 조합 권장
- 노드 폴백 비율 모니터링: `ProfileComputePlan=1`로 디스패치 확인
- 폴백 비율 > 50%이면 CPU 단독이 더 나을 수 있음

### 6.3 중장기 실험 계획

1. **CoreML EP 벤치마크** (1-2일): providers 변경 + 성능 측정
2. **CoreML EP 노드 커버리지 분석**: 몇 %가 ANE/GPU로 가는지 확인
3. **CoreML EP 결과에 따라 분기**:
   - 효과 있음 → 프로덕션 적용 (Mac 전용 분기)
   - 효과 없음 → MLX 또는 CoreML 직접 변환 실험

---

## 7. 참고 자료

- [ONNX Runtime CoreML EP 공식 문서](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)
- [Apple - Deploying Transformers on the Apple Neural Engine](https://machinelearning.apple.com/research/neural-engine-transformers)
- [apple/ml-ane-transformers (GitHub)](https://github.com/apple/ml-ane-transformers)
- [Benchmarking On-Device ML on Apple Silicon with MLX (arXiv:2510.18921)](https://arxiv.org/abs/2510.18921)
- [Fast transformer inference with Metal Performance Shaders (Explosion)](https://explosion.ai/blog/metal-performance-shaders)
- [CoreML EP CPU Fallback 이슈 (GitHub #16934)](https://github.com/microsoft/onnxruntime/issues/16934)
- [ONNX Runtime & CoreML FP16 Silent Conversion](https://ym2132.github.io/ONNX_MLProgram_NN_exploration)
- [Apple Silicon vs NVIDIA CUDA: AI Comparison 2025](https://scalastic.io/en/apple-silicon-vs-nvidia-cuda-ai-2025/)
- [Exploring LLMs with MLX and M5 Neural Accelerators (Apple)](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
