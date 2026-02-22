# ARM ONNX 최적화 벤치마크 보고서

## 개요

Mac M1/M3 + ARM 서버(AWS Graviton) 환경에서 ONNX 트랜스포머 최적화를 통한 CPU 레이턴시 최소화.
기존 x86 벤치마크에서 발견된 `optimum` 2GB+ 외부 데이터 버그를 우회하여
`onnxruntime.transformers.optimizer`를 직접 적용.

## 대상 모델

| 모델 | 용도 | 아키텍처 | 크기 |
|------|------|---------|------|
| `nlpai-lab/KURE-v1` | 임베딩 (1024차원) | XLM-RoBERTa Large (16H, 1024D) | ~2.3GB |
| `dragonkue/bge-reranker-v2-m3-ko` | 리랭킹 | XLM-RoBERTa Large (16H, 1024D) | ~2.3GB |

## 최적화 Variant

| Variant | 설명 | 예상 크기 | 대상 환경 |
|---------|------|----------|----------|
| `ort-opt` | Attention/LayerNorm/GELU Fusion (FP32) | ~2.2GB | 정확도 최우선 |
| `ort-opt-fp16` | Fusion + FP16 mixed precision | ~1.1GB | **Mac Air 8GB 권장** |

### Fusion 효과
- Attention Fusion: 24+ 개의 Multi-Head Attention 블록 통합
- LayerNorm Fusion: element-wise 연산 커널 호출 감소
- GELU Fusion: 활성화 함수 최적화
- FP32 수학적 동등 변환 → 정확도 손실 0%

## 플랫폼별 최적 설정

### Mac M1/M3 MacBook Air

| 설정 | 값 | 이유 |
|------|-----|------|
| Variant | `ort-opt-fp16` | 메모리 50% 절약 (8GB Air 안정) |
| `intra_op_threads` | 4 (P코어만) | E코어 배제로 일관된 성능 |
| `inter_op_threads` | 1 | 단일 쿼리 최적화 |
| `execution_mode` | `ORT_SEQUENTIAL` | 트랜스포머 순차 구조 |

### AWS Graviton3

| 설정 | 값 | 이유 |
|------|-----|------|
| Variant | `ort-opt` (FP32) | BF16 자동 가속 (ORT 1.17+) |
| `intra_op_threads` | 물리코어수 | SMT 없음, 전체 활용 |
| `inter_op_threads` | 1 | 단일 쿼리 최적화 |
| BF16 | 자동 (MMLA 하드웨어) | FP32 대비 최대 65% 향상 |

### x86 (기존 WSL2 환경)

| 설정 | 값 | 이유 |
|------|-----|------|
| Variant | `ort-opt` 또는 `ort-opt-fp16` | 메모리 여유에 따라 |
| `intra_op_threads` | 전체코어 | 가용 코어 전부 활용 |

## 벤치마크 결과

> 실제 수치는 `uv run python scripts/benchmark_arm_optimization.py` 실행 후 채워집니다.

### 임베딩 (단일 쿼리 레이턴시)

| Variant | Mac M1 | Mac M3 | Graviton3 | x86 (WSL2) | Cosine vs PT |
|---------|--------|--------|-----------|------------|-------------|
| PyTorch FP32 (baseline) | TBD ms | TBD ms | TBD ms | ~1500 ms | 1.0000 |
| ONNX 원본 FP32 | TBD ms | TBD ms | TBD ms | TBD ms | TBD |
| ORT-최적화 FP32 | TBD ms | TBD ms | TBD ms | TBD ms | >= 0.9999 |
| ORT-최적화 FP16 | TBD ms | TBD ms | TBD ms | TBD ms | >= 0.995 |
| MPS (Mac GPU) | TBD ms | TBD ms | N/A | N/A | 1.0000 |

### 리랭커 (15문서 배치 레이턴시)

| Variant | Mac M1 | Mac M3 | Graviton3 | x86 (WSL2) | Pearson vs PT |
|---------|--------|--------|-----------|------------|--------------|
| PyTorch FP32 (baseline) | TBD ms | TBD ms | TBD ms | TBD ms | 1.0000 |
| ORT-최적화 FP32 | TBD ms | TBD ms | TBD ms | TBD ms | >= 0.999 |
| ORT-최적화 FP16 | TBD ms | TBD ms | TBD ms | TBD ms | >= 0.990 |

### 메모리 사용량 (RSS)

| Variant | 모델 크기 | Mac Air 8GB 적합 |
|---------|----------|-----------------|
| PyTorch FP32 | ~2.3GB | 주의 (스왑 가능) |
| ORT-최적화 FP32 | ~2.2GB | 주의 |
| ORT-최적화 FP16 | ~1.1GB | **적합** |

## 프로덕션 통합

### Feature Flag

```bash
# backend/.env
USE_ONNX_EMBEDDING=true
ONNX_EMBEDDING_VARIANT=ort-opt-fp16    # ort-opt | ort-opt-fp16
USE_ONNX_RERANKER=true
ONNX_RERANKER_VARIANT=ort-opt-fp16
ONNX_QUALITY_GATE_ENABLED=true         # 시작 시 품질 자동 검증
ONNX_QUALITY_GATE_FALLBACK=true        # 품질 미달 시 PyTorch 자동 폴백
ONNX_INTRA_OP_THREADS=0               # 0 = 자동 (P코어 감지)
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

## 파일 목록

| 파일 | 설명 |
|------|------|
| `backend/scripts/build_optimized_onnx.py` | 최적화 모델 빌드 스크립트 |
| `backend/scripts/benchmark_arm_optimization.py` | 플랫폼별 벤치마크 |
| `backend/app/core/config.py` | Feature Flag (7개 추가) |
| `backend/app/services/rag/onnx_session.py` | ONNX 세션 싱글턴 + 플랫폼 감지 |
| `backend/app/services/rag/onnx_quality_gate.py` | 품질 자동 검증 |
| `backend/app/services/rag/embedding.py` | ONNX dispatch 분기 |
| `backend/app/services/rag/rerank.py` | ONNX dispatch 분기 |
| `backend/app/main.py` | lifespan ONNX 로드/검증 |

## 잠재 리스크

| 리스크 | 가능성 | 대응 |
|--------|--------|------|
| ONNX export cosine 0.76 재현 | 높음 | `torch.onnx.export()` 직접 사용, opset=17 |
| FP16 cosine < 0.995 | 중간 | `keep_io_types=True`로 LayerNorm FP32 유지 |
| Attention fusion 실패 | 중간 | `get_fused_operator_statistics()` 즉시 감지 |
| Mac Air 8GB 메모리 부족 (FP32) | 높음 | FP16 variant 사용 (1.1GB) |
| ARM NEON fusion 커널 미최적화 | 중간 | 그래프 최적화만으로도 커널 호출 수 감소 |
