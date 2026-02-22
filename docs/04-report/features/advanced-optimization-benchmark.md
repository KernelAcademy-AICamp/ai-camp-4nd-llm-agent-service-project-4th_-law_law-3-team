# ONNX 고급 최적화 벤치마크 보고서

생성일: 2026-02-23 03:50

## 배경

이전 벤치마크에서 ONNX 그래프 최적화(O2/O3) 단독으로는 효과가 없었고,
O3+INT8만 1.3x 개선이 확인됨. 본 벤치마크는 4가지 추가 최적화를 실험.

## 환경

- OS: Windows 10
- 물리 코어: 6, 논리 코어: 12
- 임베딩 모델: nlpai-lab/KURE-v1
- 리랭커 모델: dragonkue/bge-reranker-v2-m3-ko

## 실험 결과

### Phase 1: Baseline (PyTorch FP32 vs ONNX O3+INT8)

- 임베딩 PyTorch: 2460.1ms
- 임베딩 O3+INT8: 1871.7ms
- 리랭커 PyTorch: 3575.2ms
- 리랭커 O3+INT8: 1877.5ms

### Phase 2: intra_op_num_threads 튜닝

- 임베딩 최적: threads=6 (1682.8ms)
- 리랭커 최적: threads=6 (1811.8ms)

### Phase 3: torch.compile


### Phase 4: ORT transformer optimizer

- 임베딩 ORT opt: 3433.5ms
- 임베딩 ORT opt+INT8: 1772.8ms
- 리랭커 ORT opt: 3817.9ms
- 리랭커 ORT opt+INT8: 1894.0ms

### 품질 검증

- emb_ONNX O3+INT8: cosine=0.985664 [PASS]
- emb_O3+INT8 threads=6: cosine=0.985664 [PASS]
- emb_ORT optimizer: cosine=1.000000 [PASS]
- emb_ORT opt+INT8: cosine=0.985669 [PASS]
- rr_ONNX O3+INT8: pearson=0.999872 [PASS]
- rr_O3+INT8 threads=6: pearson=0.999872 [PASS]
- rr_ORT optimizer: pearson=1.000000 [PASS]
- rr_ORT opt+INT8: pearson=0.999918 [PASS]

## 종합 비교

```

============================================================
  종합 비교 결과
============================================================

  === 임베딩 비교 테이블 ===
  방법                                  |    시간 (ms) |      상대 속도 |         품질
  ----------------------------------- | ---------- | ---------- | ----------
  PyTorch FP32 (baseline)             |   2460.1ms |       1.0x |   1.000000
  ONNX O3+INT8 (기존 optimum)           |   1871.7ms |     1.31x |   0.985664
  O3+INT8 + threads=6                 |   1682.8ms |     1.46x |   0.985664
  ORT optimizer (직접)                  |   3433.5ms |     0.72x |   1.000000
  ORT optimizer + INT8                |   1772.8ms |     1.39x |   0.985669

  === 리랭커 비교 테이블 ===
  방법                                  |    시간 (ms) |      상대 속도 |         품질
  ----------------------------------- | ---------- | ---------- | ----------
  PyTorch FP32 (baseline)             |   3575.2ms |       1.0x |   1.000000
  ONNX O3+INT8 (기존)                   |   1877.5ms |     1.90x |   0.999872
  O3+INT8 + threads=6                 |   1811.8ms |     1.97x |   0.999872
  ORT optimizer (직접)                  |   3817.9ms |     0.94x |   1.000000
  ORT optimizer + INT8                |   1894.0ms |     1.89x |   0.999918
```

## 권장사항

- **임베딩**: O3+INT8 threads=6 (1.46x, 1682.8ms)
- **리랭커**: O3+INT8 threads=6 (1.97x, 1811.8ms)
