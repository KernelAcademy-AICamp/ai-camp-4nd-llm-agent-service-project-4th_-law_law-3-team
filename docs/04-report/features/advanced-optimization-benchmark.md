# ONNX 고급 최적화 벤치마크 (Phase 2-4)

> 생성일: 2026-02-23
> 이 문서는 `onnx-graph-optimization-benchmark.md`의 **보조 문서**입니다.
> 종합 분석은 메인 보고서의 6.5절을 참조하세요.

## 개요

이전 벤치마크(Phase 1: ONNX 그래프 최적화)에서 확인된 한계를 넘기 위해 4가지 추가 최적화를 실험:

1. **ONNX O3+INT8 baseline 재확인** — Windows(CPU) → WSL2(GPU) 환경 변경
2. **intra_op_num_threads 튜닝** — ONNX 세션 스레드 최적화
3. **torch.compile** — PyTorch 2.x 네이티브 JIT 컴파일
4. **onnxruntime.transformers.optimizer** — optimum 버그 우회 직접 최적화

## 환경

- OS: WSL2 Linux 6.6.87.2-microsoft-standard-WSL2
- GPU: NVIDIA GeForce RTX 5060 Ti
- CPU: 6 물리 코어, 12 논리 코어
- PyTorch: 2.10.0+cu128
- ONNX Runtime: 1.23.2

## 실험 결과 요약

### torch.compile (Phase 3)

Windows에서 Triton 미지원으로 스킵되었던 torch.compile을 WSL2 Linux에서 실행.

**초기 측정 (모델 재생성 방식):**

| 모델 | 방법 | 시간 (ms) | 상대 속도 |
|------|------|----------:|----------:|
| 임베딩 | PyTorch FP32 | 1,445 | 1.0x |
| 임베딩 | torch.compile (max-autotune) | 1,702 | 0.85x |
| 리랭커 | PyTorch FP32 | 2,826 | 1.0x |
| 리랭커 | torch.compile (default) | 2,980 | 0.95x |

**정정된 측정 (영속 모델, 진단 스크립트):**

| 방법 | 추론 시간 (ms) | 상대 속도 |
|------|-------------:|----------:|
| PyTorch FP32 | 63.7 | 1.0x |
| torch.compile (default) | 62.9 | 1.01x |
| torch.compile (reduce-overhead) | 62.6 | 1.02x |
| torch.compile (max-autotune) | 63.4 | 1.01x |

초기 측정의 0.85x는 벤치마크 설계 결함(모델 재생성 포함)에 의한 왜곡.
정정 측정에서 torch.compile은 실질적 개선 없음 (1.01-1.02x).

**상세 원인 분석**: 메인 보고서 6.5절 참조.

### Phase 2, 4 (스레드 튜닝, ORT optimizer)

WSL2 환경에서 ONNX 모델 디렉토리 미존재로 스킵됨.
Windows 환경의 기존 결과는 메인 보고서 1-5절에 포함.

## 결론

torch.compile은 KURE-v1 / bge-reranker 모델의 Transformer MatMul(cuBLAS GEMM) 지배 구조에서
실질적 개선 불가. **PyTorch FP32 유지**가 종합 최적.

## 관련 파일

| 파일 | 설명 |
|------|------|
| `backend/scripts/benchmark_advanced_optimization.py` | 고급 최적화 벤치마크 (Phase 1-4) |
| `backend/scripts/benchmark_compile_diagnosis.py` | torch.compile 원인 진단 (5단계) |
| `docs/04-report/features/onnx-graph-optimization-benchmark.md` | **메인 보고서** |
