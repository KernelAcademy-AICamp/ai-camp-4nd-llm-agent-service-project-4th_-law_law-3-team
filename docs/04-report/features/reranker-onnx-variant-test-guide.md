# 리랭커 ONNX Variant 테스트 가이드

> 작성일: 2026-02-24
> 대상: 팀원 테스트용
> 모델: `dragonkue/bge-reranker-v2-m3-ko` (XLM-RoBERTa Large, 568M params, 24 layers)

## 1. 개요

리랭커 ONNX 최적화 3개 variant를 빌드하여 Google Drive에 백업했습니다.
팀원이 각 variant를 개별적으로 테스트하여 품질과 속도를 검증할 수 있도록 안내합니다.

### Variant 요약

| Variant | 설명 | 크기 | Pearson | Spearman | Latency | vs PyTorch |
|---------|------|-----:|--------:|---------:|--------:|-----------:|
| `ort-opt` | FP32 전체 (무손실) | ~2.1GB | 1.000000 | 1.000000 | 1815ms | 1.78x |
| `ort-opt-qdq` | INT8 (4 FP32 레이어) | 688MB | 0.999970 | 0.967857 | 920ms | 3.52x |
| `ort-opt-qdq-6fp32` | INT8 (6 FP32 레이어) | 760MB | 0.999392 | 0.992857 | 981ms | 3.30x |

- **Pearson**: PyTorch 대비 점수 상관관계 (1.0 = 동일)
- **Spearman**: PyTorch 대비 순위 상관관계 (1.0 = 동일 순위)
- 테스트 환경: Mac ARM (Apple Silicon), ORT 1.23.2, CPU EP, 쿼리 1개 x 문서 15건

### 어떤 variant를 선택해야 할까?

| 상황 | 권장 variant | 이유 |
|------|-------------|------|
| **프로덕션 (속도+품질)** | `ort-opt-qdq` | 3.52x 가속, Pearson 0.9999 |
| **무손실 필요** | `ort-opt` | PyTorch와 동일한 출력 |
| **순위 보존 중요** | `ort-opt-qdq-6fp32` | Spearman 0.993으로 순위 보존력 높음 |

## 2. 사전 준비

### 2.1 의존성

```bash
cd backend
uv sync --dev
```

### 2.2 모델 다운로드 (Google Drive)

```bash
# 프로젝트 루트에서 실행 (rclone.conf 위치)
cd /path/to/law-3-team

# 전체 variant 다운로드 (~3.5GB)
rclone --config rclone.conf copy gdrive:data/models/ backend/data/models/ --progress

# 또는 개별 variant만 다운로드
rclone --config rclone.conf copy gdrive:data/models/reranker-ort-opt/ backend/data/models/reranker-ort-opt/ --progress
rclone --config rclone.conf copy gdrive:data/models/reranker-ort-opt-qdq/ backend/data/models/reranker-ort-opt-qdq/ --progress
rclone --config rclone.conf copy gdrive:data/models/reranker-ort-opt-qdq-6fp32/ backend/data/models/reranker-ort-opt-qdq-6fp32/ --progress
```

### 2.3 다운로드 확인

```bash
ls -lh backend/data/models/reranker-ort-opt/
# model_optimized.onnx (346KB) + model_optimized.onnx.data (2.1GB) + config.json + tokenizer.*

ls -lh backend/data/models/reranker-ort-opt-qdq/
# model_optimized.onnx (688MB) + config.json + tokenizer.*

ls -lh backend/data/models/reranker-ort-opt-qdq-6fp32/
# model_optimized.onnx (760MB) + config.json + tokenizer.*
```

## 3. Variant별 테스트 방법

### 3.1 .env 설정으로 variant 전환

```bash
# backend/.env 수정
USE_ONNX_RERANKER=true
ONNX_RERANKER_VARIANT=ort-opt       # FP32 전체
# ONNX_RERANKER_VARIANT=ort-opt-qdq       # INT8 (4 FP32) ← 프로덕션 권장
# ONNX_RERANKER_VARIANT=ort-opt-qdq-6fp32 # INT8 (6 FP32)
```

variant를 바꿀 때마다 서버 재시작이 필요합니다.

### 3.2 서버 기동 테스트

```bash
cd backend
uv run uvicorn app.main:app --reload
```

로그에서 다음을 확인:
```
INFO - ONNX 리랭커 세션 로드 완료: variant=ort-opt-qdq, dir=.../reranker-ort-opt-qdq
INFO - ONNX 리랭커 warmup 완료
```

### 3.3 API를 통한 검색 테스트

서버 기동 후 검색 API를 호출하여 리랭킹 결과를 확인합니다.

```bash
# 판례 검색 (리랭킹 포함)
curl -s -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "교통사고 손해배상 판례 알려줘", "session_id": "test-reranker"}' \
  | python -m json.tool
```

### 3.4 벤치마크 스크립트로 정량 비교

각 variant의 정확한 latency와 품질 지표를 측정하려면:

```bash
cd backend

# Session Config + 프로파일링 벤치마크
uv run --no-sync python scripts/benchmark_reranker_optimizations.py
# → benchmark_reranker_opt_results.json 생성

# 민감 레이어 sweep (QDQ variant 비교)
uv run --no-sync python scripts/sweep_reranker_sensitive_layers.py
# → sweep_reranker_sensitive_results.json 생성
```

## 4. 수동 추론 테스트 (Python)

variant를 직접 로드하여 추론 결과를 확인하는 스크립트입니다.

```python
"""
리랭커 ONNX variant 수동 테스트
사용법: cd backend && uv run --no-sync python -c "$(cat <<'EOF'
... (아래 코드 복사)
EOF
)"
"""
import time
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

# ---- 설정 ----
VARIANT = "reranker-ort-opt-qdq"  # 테스트할 variant 이름
# VARIANT = "reranker-ort-opt"
# VARIANT = "reranker-ort-opt-qdq-6fp32"

MODEL_DIR = Path("data/models") / VARIANT
QUERY = "교통사고 손해배상 판례"
DOCUMENTS = [
    "피고는 원고에게 교통사고로 인한 손해배상금 5000만원을 지급하라",
    "임대차계약 해지 통보는 6개월 전에 해야 한다",
    "교통사고 피해자의 과실비율이 30%인 경우 손해배상액 산정 방법",
    "근로기준법 제23조에 따른 부당해고 구제신청 절차",
    "자동차손해배상보장법에 의한 보험금 청구 요건",
]

# ---- 세션 로드 ----
print(f"\n{'='*60}")
print(f"Variant: {VARIANT}")
print(f"모델 경로: {MODEL_DIR}")

sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = 4
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

model_file = MODEL_DIR / "model_optimized.onnx"
session = ort.InferenceSession(str(model_file), sess_opts, providers=["CPUExecutionProvider"])
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

print(f"입력: {[inp.name for inp in session.get_inputs()]}")
print(f"출력: {[out.name for out in session.get_outputs()]}")

# ---- 추론 ----
queries = [QUERY] * len(DOCUMENTS)
inputs = tokenizer(queries, DOCUMENTS, return_tensors="np", padding=True, truncation=True, max_length=512)

feed = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
if "token_type_ids" in [inp.name for inp in session.get_inputs()]:
    feed["token_type_ids"] = inputs.get("token_type_ids", np.zeros_like(inputs["input_ids"]))

# Warmup
session.run(None, feed)
session.run(None, feed)

# 측정 (10회 중앙값)
times = []
for _ in range(10):
    t0 = time.perf_counter()
    outputs = session.run(None, feed)
    times.append((time.perf_counter() - t0) * 1000)

logits = outputs[0]
if logits.ndim == 2:
    logits = logits[:, 0]
scores = 1.0 / (1.0 + np.exp(-logits))

# ---- 결과 출력 ----
median_ms = np.median(times)
print(f"\n{'='*60}")
print(f"Latency: {median_ms:.1f}ms (중앙값, 10회)")
print(f"\n순위별 결과:")
ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
for rank, (idx, score) in enumerate(ranked, 1):
    print(f"  {rank}. [{score:.6f}] {DOCUMENTS[idx][:60]}")
print(f"{'='*60}\n")
```

### 4.1 PyTorch 기준선과 비교

```python
"""
PyTorch 기준선 생성 + ONNX variant 비교
"""
import torch
from sentence_transformers import CrossEncoder

# PyTorch 기준선
model = CrossEncoder("dragonkue/bge-reranker-v2-m3-ko", activation_fn=torch.nn.Sigmoid())
pairs = [(QUERY, doc) for doc in DOCUMENTS]
pytorch_scores = model.predict(pairs)

# 비교
onnx_scores = scores  # 위 스크립트의 결과
pearson_corr = pearsonr(pytorch_scores, onnx_scores)[0]
spearman_corr = spearmanr(pytorch_scores, onnx_scores)[0]

print(f"Pearson:  {pearson_corr:.6f}")
print(f"Spearman: {spearman_corr:.6f}")
```

## 5. Variant 세부 사양

### 5.1 `ort-opt` (FP32 전체)

| 항목 | 값 |
|------|-----|
| 양자화 | 없음 (FP32) |
| 그래프 최적화 | ORT_ENABLE_ALL |
| Fusion | BiasGelu=24, SkipLayerNormalization=48 |
| 파일 구조 | `model_optimized.onnx` (346KB) + `.onnx.data` (2.12GB) |
| 품질 | PyTorch와 동일 (Pearson=1.0, Spearman=1.0) |
| 용도 | 무손실 필요, 품질 기준선 |

### 5.2 `ort-opt-qdq` (INT8, 4 FP32 레이어)

| 항목 | 값 |
|------|-----|
| 양자화 | QDQ INT8 (20개 레이어 INT8, 4개 FP32) |
| FP32 유지 레이어 | [2, 6, 7, 15] (민감도 상위 4개) |
| 파일 구조 | `model_optimized.onnx` (688MB, 단일 파일) |
| 품질 | Pearson=0.999970, Spearman=0.967857 |
| 용도 | **프로덕션 권장** (속도+품질 최적 밸런스) |

### 5.3 `ort-opt-qdq-6fp32` (INT8, 6 FP32 레이어)

| 항목 | 값 |
|------|-----|
| 양자화 | QDQ INT8 (18개 레이어 INT8, 6개 FP32) |
| FP32 유지 레이어 | [2, 6, 7, 10, 15, 16] (민감도 상위 6개) |
| 파일 구조 | `model_optimized.onnx` (760MB, 단일 파일) |
| 품질 | Pearson=0.999392, Spearman=0.992857 |
| 용도 | 순위 보존력 중시 (Spearman 높음) |

### 5.4 Variant 간 트레이드오프

```
품질(Pearson)  ort-opt(1.000) > ort-opt-qdq(0.9999) > ort-opt-qdq-6fp32(0.9994)
순위(Spearman) ort-opt(1.000) > ort-opt-qdq-6fp32(0.993) > ort-opt-qdq(0.968)
속도           ort-opt-qdq(920ms) > ort-opt-qdq-6fp32(981ms) > ort-opt(1815ms)
크기           ort-opt-qdq(688MB) < ort-opt-qdq-6fp32(760MB) < ort-opt(2.1GB)
```

> **참고**: `ort-opt-qdq`는 Pearson이 높지만 Spearman이 낮고,
> `ort-opt-qdq-6fp32`는 Pearson이 약간 낮지만 Spearman이 높습니다.
> Pearson은 점수 값의 상관관계, Spearman은 순위의 상관관계입니다.
> 리랭킹에서는 **순위가 중요**하므로 Spearman도 함께 확인하세요.

## 6. 환경 변수 참조

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `USE_ONNX_RERANKER` | ONNX 리랭커 활성화 | `false` |
| `ONNX_RERANKER_VARIANT` | variant 선택 | `ort-opt` |
| `ONNX_INTRA_OP_THREADS` | ORT 스레드 수 (0=자동) | `0` |
| `ONNX_QUALITY_GATE_ENABLED` | 품질 게이트 (PyTorch 비교) | `true` |
| `ONNX_QUALITY_GATE_FALLBACK` | 품질 미달 시 PyTorch 폴백 | `true` |
| `ONNX_INFERENCE_TIMEOUT_SECONDS` | 추론 타임아웃 | `30.0` |

## 7. 트러블슈팅

### 모델 파일 없음

```
ERROR - ONNX 모델 디렉토리 없음: .../data/models/reranker-ort-opt-qdq
```

**해결**: Google Drive에서 모델 다운로드 (섹션 2.2 참조)

### external data 로드 실패 (ort-opt variant)

```
onnxruntime.OrtException: Load model from ... failed: external data file not found
```

**해결**: `model_optimized.onnx`와 `model_optimized.onnx.data`가 같은 디렉토리에 있어야 합니다.

### 메모리 부족

`ort-opt` variant는 ~2.1GB 메모리가 필요합니다. RAM이 부족하면 `ort-opt-qdq` (688MB)를 사용하세요.

### ONNX 비활성화하고 PyTorch로 돌아가기

```bash
# backend/.env
USE_ONNX_RERANKER=false
```

## 8. 관련 문서

| 문서 | 설명 |
|------|------|
| [`reranker-onnx-optimization-benchmark.md`](./reranker-onnx-optimization-benchmark.md) | 벤치마크 보고서 (프로파일링, 민감 레이어 sweep 결과) |
| [`arm-onnx-optimization-benchmark.md`](./arm-onnx-optimization-benchmark.md) | 임베딩 모델 ONNX 벤치마크 보고서 |
| `backend/scripts/build_optimized_onnx.py` | ONNX 모델 빌드 스크립트 |
| `backend/app/services/rag/onnx_session.py` | ONNX 세션 로딩 코드 |
| `backend/app/services/rag/rerank.py` | 리랭킹 서비스 (ONNX/PyTorch 분기) |

## 9. Google Drive 디렉토리 구조

```
gdrive:data/models/
├── reranker-ort-opt/              # FP32 전체 (~2.1GB)
│   ├── config.json
│   ├── model_optimized.onnx       # 346KB (그래프만)
│   ├── model_optimized.onnx.data  # 2.12GB (가중치)
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── reranker-ort-opt-qdq/          # INT8, 4 FP32 (688MB)
│   ├── config.json
│   ├── model_optimized.onnx       # 688MB (단일 파일)
│   ├── tokenizer.json
│   └── tokenizer_config.json
└── reranker-ort-opt-qdq-6fp32/    # INT8, 6 FP32 (760MB)
    ├── config.json
    ├── model_optimized.onnx       # 760MB (단일 파일)
    ├── tokenizer.json
    └── tokenizer_config.json
```
