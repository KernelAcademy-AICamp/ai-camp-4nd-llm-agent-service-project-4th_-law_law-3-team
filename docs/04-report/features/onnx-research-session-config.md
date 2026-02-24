# ORT 세션 설정 + 메모리/런타임 최적화 리서치

> 작성일: 2026-02-24
> 대상 모델: nlpai-lab/KURE-v1 (XLM-RoBERTa Large, 24 layers, 1024 dim)
> 현재 코드: `backend/app/services/rag/onnx_session.py`
> ORT 버전: 1.23.2 | 플랫폼: Mac ARM (M-series) / Graviton3

---

## 목차

1. [현재 설정 분석](#1-현재-설정-분석)
2. [SessionOptions 최적화](#2-sessionoptions-최적화)
3. [메모리 최적화](#3-메모리-최적화)
4. [스레딩 최적화](#4-스레딩-최적화)
5. [토크나이저 최적화](#5-토크나이저-최적화)
6. [배치 추론 최적화](#6-배치-추론-최적화)
7. [Warmup 최적화](#7-warmup-최적화)
8. [Session Config Entry 전체 목록](#8-session-config-entry-전체-목록)
9. [적용 권장 사항 (현재 코드 기준)](#9-적용-권장-사항-현재-코드-기준)
10. [참고 자료](#10-참고-자료)

---

## 1. 현재 설정 분석

### 현재 `_create_session()` 설정

```python
session_options.intra_op_num_threads = platform_config.intra_op_threads  # Mac ARM: 4 (P코어)
session_options.inter_op_num_threads = 1  # 고정
session_options.execution_mode = ORT_SEQUENTIAL
session_options.graph_optimization_level = ORT_ENABLE_ALL
providers = ["CPUExecutionProvider"]
```

### 현재 설정의 문제점

| 항목 | 현재 값 | 문제 | 영향 |
|------|---------|------|------|
| `graph_optimization_level` | `ORT_ENABLE_ALL` | 이미 오프라인 최적화된 모델에 재최적화 시도 | 세션 로드 시간 증가, 잠재적 성능 저하 |
| `enable_mem_pattern` | 기본값 (`True`) | 설정은 합리적이나 명시적 제어 부재 | - |
| `enable_cpu_mem_arena` | 기본값 (`True`) | 임베딩+리랭커 2개 세션이 각각 독립 arena 사용 | 메모리 낭비 |
| Thread spinning | 기본값 (`True`) | 유휴 시에도 CPU 점유 | 서버 환경에서 불필요한 CPU 소비 |
| `set_denormal_as_zero` | 미설정 | denormal 값 처리 오버헤드 | 잠재적 성능 손실 |

---

## 2. SessionOptions 최적화

### 2.1 그래프 최적화 레벨: `ORT_DISABLE_ALL` 권장

**핵심 발견**: 오프라인으로 이미 최적화된 모델(`model_optimized.onnx`)을 로드할 때, `ORT_ENABLE_ALL`은 불필요한 재최적화를 시도하여 세션 초기화 시간이 늘어나고, 경우에 따라 성능이 오히려 저하될 수 있다.

```python
# 변경 전
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# 변경 후 (오프라인 최적화 모델 전용)
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
```

**근거**:
- ORT 공식 문서: "offline mode에서 최적화 후 저장한 모델은 로드 시 `ORT_DISABLE_ALL`을 사용하여 시작 시간을 단축할 수 있다"
- GitHub Issue #15743: `ORT_ENABLE_ALL`이 오히려 느려진 실제 사례 보고
- 현재 빌드 스크립트(`build_optimized_onnx.py`)가 이미 Attention Fusion, SkipLayerNorm Fusion 등을 적용하므로, 런타임에서 재최적화할 필요 없음

**주의**: 원본 ONNX 모델(`model.onnx`)을 직접 로드하는 경우에는 `ORT_ENABLE_ALL` 유지 필요. variant 기반으로 분기 권장.

**예상 효과**: 세션 로드 시간 단축 (모델 크기에 따라 수백 ms~수 초)

### 2.2 Denormal-as-Zero 활성화

Denormal(비정규) 부동소수점 값은 CPU에서 처리 시 일반 값 대비 10~100배 느릴 수 있다. Transformer 모델의 attention 레이어에서 매우 작은 값이 빈번하게 발생하므로, flush-to-zero를 활성화하면 성능 향상 가능.

```python
session_options.add_session_config_entry("session.set_denormal_as_zero", "1")
```

**주의**: 정밀도에 미세한 영향이 있을 수 있으므로, 품질 게이트(cosine similarity 검증)와 함께 사용 권장.

**예상 효과**: 모델/데이터 특성에 따라 0~5% 레이턴시 개선

### 2.3 Gelu 근사 활성화

Transformer 모델의 Gelu 활성화 함수를 근사값으로 대체하여 연산량 감소.

```python
session_options.add_session_config_entry("optimization.enable_gelu_approximation", "1")
```

**주의**: 정밀도 영향 있을 수 있음. QDQ INT8 모델에서는 이미 양자화로 인한 오차가 있으므로 추가 영향은 미미.

**예상 효과**: Gelu 노드 수 x 근사 효율에 따라 1~3% 개선

### 2.4 Prepacking 유지 (기본값)

`session.disable_prepacking`은 기본값(`"0"`, 활성화)을 유지한다. Prepacking은 가중치를 연산에 최적화된 형태로 미리 변환하는 기능으로, 로드 시간이 약간 늘어나지만 추론 성능이 향상된다.

---

## 3. 메모리 최적화

### 3.1 Memory Arena 전략

현재 임베딩 세션과 리랭커 세션이 각각 독립적인 CPU memory arena를 생성한다. 두 세션이 동시에 활성화되면 메모리 사용량이 거의 2배가 된다.

#### 옵션 A: Arena 비활성화 (메모리 우선)

```python
session_options.enable_cpu_mem_arena = False
```

- 장점: 메모리 사용량 대폭 감소 (수백 MB 절약 가능)
- 단점: 매 추론마다 시스템 allocator 호출 → 레이턴시 증가
- 적합: 메모리가 제한된 환경 (예: 16GB Mac에서 다른 서비스와 공존)

#### 옵션 B: Arena 유지 + `kSameAsRequested` 전략 (균형)

```python
session_options.enable_cpu_mem_arena = True
# Python API에서는 arena_extend_strategy 직접 설정 불가
# C API에서 가능: OrtArenaCfg with kSameAsRequested
```

- 장점: arena 유지하면서 불필요한 power-of-2 확장 방지
- 단점: Python API에서 직접 설정 불가 (C API 전용)
- 적합: 메모리 사용량이 예측 가능한 프로덕션 환경

#### 옵션 C: 공유 Allocator (다중 세션 최적화)

```python
# C API에서만 가능:
# CreateAndRegisterAllocator → 환경에 공유 allocator 등록
# session_options.add_session_config_entry("session.use_env_allocators", "1")
```

- 장점: 두 세션이 하나의 arena를 공유하여 메모리 절약
- 단점: Python API에서 직접 설정 불가
- 적합: 다중 모델 서빙 환경

**현재 코드 권장**: 옵션 A를 환경 변수로 제어 가능하게 만들되, 기본값은 `True`(arena 활성화) 유지.

### 3.2 Memory Pattern 최적화

```python
session_options.enable_mem_pattern = True  # 기본값, 명시적 설정 권장
```

Memory pattern은 동일한 입력 shape에 대해 내부 메모리 할당 패턴을 추적하고, 이후 요청에서 하나의 큰 덩어리로 일괄 할당한다. 임베딩 쿼리는 대부분 비슷한 길이이므로 패턴 재사용률이 높다.

### 3.3 Memory Reuse

ORT는 기본적으로 메모리 재사용을 활성화한다. 한 연산자의 출력 버퍼를 다른 연산자의 입력으로 재사용하여 peak memory를 줄인다. 이 설정은 건드리지 않는 것이 좋다.

### 3.4 ORT 모델 바이트 직접 사용 (고급)

```python
session_options.add_session_config_entry(
    "session.use_ort_model_bytes_for_initializers", "1"
)
```

모델 로드 시 initializer(가중치) 데이터를 복사하지 않고 원본 버퍼를 직접 참조한다. Peak memory를 줄일 수 있지만, 모델 바이트의 수명 관리가 필요하다.

**현재 코드에서는 파일 경로로 세션을 생성하므로 이 옵션은 적용 불가** (바이트 버퍼로 로드하는 방식으로 변경해야 함).

---

## 4. 스레딩 최적화

### 4.1 현재 설정 검토

| 설정 | 현재 값 | 평가 |
|------|---------|------|
| `intra_op_num_threads` | Mac ARM: 4 (P코어) | 적절 |
| `inter_op_num_threads` | 1 | 적절 (Sequential 모드에서는 무의미) |
| `execution_mode` | `ORT_SEQUENTIAL` | Transformer 모델에 적합 |

**평가**: 현재 스레딩 설정은 합리적이다. Sequential 모드에서 intra-op 스레드만 사용하므로, P코어 수에 맞춘 현재 값이 최적에 가깝다.

### 4.2 Thread Spinning 제어

```python
# 기본값: spinning 활성화 (낮은 레이턴시, 높은 CPU 사용)
# 서버 환경에서 CPU 사용률을 줄이려면:
session_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
session_options.add_session_config_entry("session.inter_op.allow_spinning", "0")
```

**트레이드오프**:

| 설정 | 레이턴시 | CPU 사용률 | 적합 환경 |
|------|---------|-----------|----------|
| Spinning ON (기본값) | 낮음 | 높음 (유휴 시에도 CPU 점유) | 전용 추론 서버, 레이턴시 민감 |
| Spinning OFF | 약간 높음 (+1~2ms) | 낮음 | 다른 서비스와 공존, 배치 처리 |

**현재 프로젝트 권장**: 법률 서비스 플랫폼은 검색 레이턴시가 수백 ms 단위이므로, 1~2ms 증가는 무시 가능. **Spinning OFF 권장** (CPU 절약).

### 4.3 Thread Affinity (고급)

```python
# Mac ARM: P코어(0-3)에 스레드 고정
session_options.add_session_config_entry(
    "session.intra_op_thread_affinities", "0;1;2;3"
)
```

P코어에 명시적으로 스레드를 고정하여 E코어로의 마이그레이션을 방지한다. 현재 `_detect_mac_p_cores()`로 P코어 수를 감지하고 있으므로, affinity까지 설정하면 더 안정적인 성능을 보장할 수 있다.

**주의**: Mac ARM에서는 OS 스케줄러가 이미 QoS 기반으로 P코어를 우선 배정하므로, 실질적 효과는 제한적일 수 있다. 벤치마크 후 결정 권장.

### 4.4 Force Spinning Stop

```python
session_options.add_session_config_entry("session.force_spinning_stop", "1")
```

마지막 동시 `Run()` 호출이 완료되면 즉시 thread pool spinning을 중지한다. 간헐적 추론 패턴(법률 검색)에 적합.

---

## 5. 토크나이저 최적화

### 5.1 현재 토크나이저 사용 패턴

```python
# encode_embedding_onnx() 내부
inputs = holder.tokenizer(
    query,
    return_tensors="np",
    padding="max_length" if static_length else True,  # True = padding='longest'
    truncation=True,
    max_length=static_length if static_length else 512,
)
```

**분석**: 단일 쿼리(batch=1) 추론에서는 `padding=True`가 `padding='longest'`와 동일하게 동작하므로, 현재 패딩 전략은 적절하다.

### 5.2 Fast Tokenizer 확인

XLM-RoBERTa 모델은 기본적으로 Rust 기반 Fast Tokenizer를 사용한다. `AutoTokenizer.from_pretrained()`가 자동으로 Fast 버전을 로드하므로, 현재 코드는 이미 최적.

확인 방법:
```python
print(type(tokenizer))  # <class 'transformers.models.xlm_roberta.tokenization_xlm_roberta_fast.XLMRobertaTokenizerFast'>
print(tokenizer.is_fast)  # True
```

### 5.3 토크나이저 캐싱/재사용

현재 `OnnxSessionHolder`에 토크나이저가 싱글턴으로 보관되므로, 매 요청마다 재로드하지 않는다. 이 구조는 적절하다.

### 5.4 배치 토크나이징 시 `padding='longest'` 사용

리랭커(`predict_reranker_onnx`)에서 여러 문서를 한 번에 토크나이징할 때:

```python
# 현재 코드 (적절)
inputs = holder.tokenizer(
    queries, documents,
    return_tensors="np",
    padding=True,  # = 'longest' (배치 내 최장 길이에 맞춤)
    truncation=True,
    max_length=512,
)
```

**핵심**: `padding='longest'`는 `padding='max_length'`보다 ONNX 추론에서 4~6배 빠를 수 있다 (패딩 토큰 수에 비례). 현재 코드는 이미 이 최적화를 적용하고 있다.

### 5.5 토크나이저 사전 로드 분리 (개선 가능)

현재 `_load_tokenizer()`가 세션 생성과 동기적으로 실행된다. 토크나이저 로드는 ~100ms 소요되므로 세션 초기화 병목은 아니지만, 향후 별도 스레드에서 사전 로드하는 것도 고려 가능.

---

## 6. 배치 추론 최적화

### 6.1 현재 상태

현재 임베딩 함수(`encode_embedding_onnx`)는 **단일 쿼리만** 처리한다:

```python
def encode_embedding_onnx(query: str) -> list[float]:
```

RAG 파이프라인에서 여러 쿼리를 순차 처리하는 경우, 배치 추론으로 전환하면 throughput이 크게 향상된다.

### 6.2 배치 임베딩 함수 추가 제안

```python
def encode_embedding_onnx_batch(queries: list[str]) -> list[list[float]]:
    """여러 쿼리를 배치로 임베딩한다."""
    holder = _embedding_holder
    if not holder.is_loaded:
        raise RuntimeError("ONNX 임베딩 세션이 로드되지 않았습니다")

    static_length = _parse_static_length(holder.variant)
    inputs = holder.tokenizer(
        queries,
        return_tensors="np",
        padding="max_length" if static_length else "longest",
        truncation=True,
        max_length=static_length if static_length else 512,
    )

    feed = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
    }
    if "token_type_ids" in holder.input_names:
        feed["token_type_ids"] = inputs.get(
            "token_type_ids",
            np.zeros_like(inputs["input_ids"]),
        )

    raw_outputs = holder.session.run(None, feed)

    # CLS pooling + L2 정규화 (배치)
    emb = raw_outputs[0][:, 0, :]
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.clip(norm, a_min=1e-9, a_max=None)

    return emb.tolist()
```

**예상 효과**: 배치 크기 8 기준, 단일 쿼리 8회 대비 2~4배 throughput 향상 (CPU 병렬 GEMM 활용).

### 6.3 동적 배치 크기 결정

| 쿼리 수 | 권장 배치 크기 | 근거 |
|---------|-------------|------|
| 1 | 1 | 현재 단일 쿼리 함수 사용 |
| 2~8 | 전체 배치 | CPU GEMM 병렬화 이점 |
| 9~32 | 8 | 메모리 제한 고려 (seq_len=512 x 1024dim x 4bytes) |
| 33+ | 16 | 패딩 오버헤드 vs 배치 이점 균형 |

### 6.4 리랭커 배치 최적화

현재 리랭커는 이미 배치 추론을 지원한다(`predict_reranker_onnx(query, documents)`). 하지만 문서 수가 많을 경우 청크 단위로 분할하는 것이 메모리 안정성에 좋다:

```python
def predict_reranker_onnx_chunked(
    query: str, documents: list[str], chunk_size: int = 16
) -> list[float]:
    """큰 문서 리스트를 chunk_size 단위로 분할하여 리랭킹."""
    all_scores = []
    for i in range(0, len(documents), chunk_size):
        chunk = documents[i:i + chunk_size]
        scores = predict_reranker_onnx(query, chunk)
        all_scores.extend(scores)
    return all_scores
```

---

## 7. Warmup 최적화

### 7.1 현재 Warmup

```python
def warmup_embedding() -> None:
    encode_embedding_onnx("warm-up")  # 단일 쿼리 1회
```

### 7.2 개선 제안: 다중 Warmup

첫 번째 추론은 메모리 할당, JIT 컴파일(있는 경우), memory pattern 수집 등으로 느리다. 2~3회 warmup으로 안정적인 성능을 확보할 수 있다:

```python
def warmup_embedding() -> None:
    if not _embedding_holder.is_loaded:
        return
    try:
        # 3회 warmup: memory pattern 안정화 + arena pre-allocation
        for i in range(3):
            encode_embedding_onnx(f"warmup-{i}")
        logger.info("ONNX 임베딩 warmup 완료 (3회)")
    except Exception:
        logger.exception("ONNX 임베딩 warmup 실패")
```

### 7.3 대표적 입력 길이로 Warmup

실제 법률 쿼리의 평균 길이에 가까운 텍스트로 warmup하면, memory pattern이 실제 사용 패턴에 더 잘 맞는다:

```python
WARMUP_TEXT = "손해배상 청구 소송에서 불법행위로 인한 민사상 손해배상 책임의 성립 요건과 범위에 관한 판례"
```

---

## 8. Session Config Entry 전체 목록

ORT 1.23.x에서 사용 가능한 주요 session config entry:

### 메모리 관련

| 키 | 값 | 설명 |
|----|-----|------|
| `session.use_env_allocators` | `"1"` | 환경 등록 allocator 사용 (다중 세션 공유) |
| `session.use_ort_model_bytes_for_initializers` | `"1"` | initializer 복사 방지 (peak memory 감소) |
| `session.collect_node_memory_stats_to_file` | 파일 경로 | 노드별 메모리 통계 CSV 출력 |

### 스레딩 관련

| 키 | 값 | 설명 |
|----|-----|------|
| `session.intra_op.allow_spinning` | `"0"` | intra-op spinning 비활성화 (CPU 절약) |
| `session.inter_op.allow_spinning` | `"0"` | inter-op spinning 비활성화 |
| `session.force_spinning_stop` | `"1"` | Run() 완료 시 즉시 spinning 중지 |
| `session.intra_op_thread_affinities` | `"0;1;2;3"` | 스레드-코어 고정 |
| `session.dynamic_block_base` | `"0"` | 동적 블록 사이징 비활성화 |

### 최적화 관련

| 키 | 값 | 설명 |
|----|-----|------|
| `optimization.enable_gelu_approximation` | `"1"` | Gelu 근사 활성화 |
| `optimization.disable_specified_optimizers` | 최적화기 이름 | 특정 최적화기 비활성화 |
| `session.set_denormal_as_zero` | `"1"` | denormal flush-to-zero |
| `session.disable_prepacking` | `"0"` | prepacking 유지 (기본값) |

### ARM 전용

| 키 | 값 | 설명 |
|----|-----|------|
| `mlas.enable_gemm_fastmath_arm64_bfloat16` | `"1"` | BF16 MMLA fastmath (Graviton3+) |
| `session.qdqisint8allowed` | `"1"` | QDQ에서 INT8 허용 (ARM 플랫폼) |

---

## 9. 적용 권장 사항 (현재 코드 기준)

### 즉시 적용 가능 (코드 변경 최소)

| 순위 | 변경 사항 | 예상 효과 | 위험도 |
|------|----------|----------|--------|
| 1 | `graph_optimization_level` → `ORT_DISABLE_ALL` (최적화 모델) | 세션 로드 시간 단축 | 낮음 |
| 2 | `session.set_denormal_as_zero` → `"1"` | 0~5% 레이턴시 개선 | 낮음 (품질 게이트 있음) |
| 3 | `session.intra_op.allow_spinning` → `"0"` | CPU 사용률 절감 | 낮음 |
| 4 | `session.force_spinning_stop` → `"1"` | 유휴 시 CPU 절감 | 없음 |
| 5 | Warmup 3회 + 대표 텍스트 | 첫 추론 레이턴시 안정화 | 없음 |

### 중기 적용 (코드 변경 필요)

| 순위 | 변경 사항 | 예상 효과 | 위험도 |
|------|----------|----------|--------|
| 6 | 배치 임베딩 함수 추가 | throughput 2~4x | 낮음 |
| 7 | 리랭커 chunked 배치 | 메모리 안정성 | 낮음 |
| 8 | `enable_cpu_mem_arena` 환경변수 제어 | 메모리 유연성 | 낮음 |
| 9 | `optimization.enable_gelu_approximation` → `"1"` | 1~3% 개선 | 중간 (정밀도 영향) |

### `_create_session()` 개선 제안 코드

```python
def _create_session(
    model_dir: Path,
    platform_config: PlatformConfig,
) -> Any:
    import onnxruntime as ort

    model_file = _find_model_file(model_dir)
    if model_file is None:
        raise FileNotFoundError(f"ONNX 모델 파일 없음: {model_dir}")

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = platform_config.intra_op_threads
    session_options.inter_op_num_threads = platform_config.inter_op_threads

    # 실행 모드
    if platform_config.execution_mode == "sequential":
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    else:
        session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

    # [개선 1] 오프라인 최적화 모델은 재최적화 건너뛰기
    is_optimized = model_file.name == "model_optimized.onnx"
    if is_optimized:
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    else:
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # [개선 2] Denormal flush-to-zero (성능 개선, 미세 정밀도 영향)
    session_options.add_session_config_entry(
        "session.set_denormal_as_zero", "1",
    )

    # [개선 3] Thread spinning 비활성화 (CPU 절약, 서버 환경 최적화)
    session_options.add_session_config_entry(
        "session.intra_op.allow_spinning", "0",
    )
    session_options.add_session_config_entry(
        "session.inter_op.allow_spinning", "0",
    )

    # [개선 4] Run 완료 후 즉시 spinning 중지
    session_options.add_session_config_entry(
        "session.force_spinning_stop", "1",
    )

    # [기존] Memory pattern 명시적 활성화
    session_options.enable_mem_pattern = True

    # [기존] Graviton3 BF16 fastmath
    if platform_config.enable_bf16_fastmath:
        session_options.add_session_config_entry(
            "mlas.enable_gemm_fastmath_arm64_bfloat16", "1",
        )
        logger.info("BF16 fastmath 활성화 (Graviton3 MMLA)")

    session = ort.InferenceSession(
        str(model_file),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    logger.info(
        "ONNX 세션 생성: %s (threads=%d, mode=%s, opt_level=%s)",
        model_file.name,
        platform_config.intra_op_threads,
        platform_config.execution_mode,
        "DISABLED" if is_optimized else "ALL",
    )
    return session
```

---

## 10. 참고 자료

- [ORT Thread Management](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)
- [ORT Memory Consumption](https://onnxruntime.ai/docs/performance/tune-performance/memory.html)
- [ORT Graph Optimizations](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html)
- [ORT Python API](https://onnxruntime.ai/docs/api/python/api_summary.html)
- [ORT Session Config Keys (GitHub)](https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h)
- [ORT Run Options Config Keys (GitHub)](https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_run_options_config_keys.h)
- [Memory Arena Discussion #13409](https://github.com/microsoft/onnxruntime/discussions/13409)
- [enable_cpu_mem_arena Issue #11627](https://github.com/microsoft/onnxruntime/issues/11627)
- [ORT_ENABLE_ALL slower Issue #15743](https://github.com/microsoft/onnxruntime/issues/15743)
- [Inworld: Reducing CPU Usage with ORT](https://inworld.ai/blog/reducing-cpu-usage-in-machine-learning-model-inference-with-onnx-runtime)
- [SetFit Padding Performance Issue #360](https://github.com/huggingface/setfit/issues/360)
- [Microsoft: Scaling PyTorch Inference with ORT](https://cloudblogs.microsoft.com/opensource/2022/04/19/scaling-up-pytorch-inference-serving-billions-of-daily-nlp-inferences-with-onnx-runtime/)
- [Microsoft: Journey to Optimize Transformer Inference](https://opensource.microsoft.com/blog/2021/06/30/journey-to-optimize-large-scale-transformer-model-inference-with-onnx-runtime)
