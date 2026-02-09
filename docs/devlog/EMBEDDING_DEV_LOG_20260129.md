# 임베딩 시스템 개발 로그 (2026-01-29)

## 개요

판례/법령 데이터 임베딩 시스템 개선 및 로컬 GPU (RTX 5060 Ti) 환경 설정 작업.

---

## 1. 문제점 및 해결

### 1.1 청킹 무한 루프 (MemoryError)

**문제**
```
File "runpod_lancedb_embeddings.py", line 971, in chunk_precedent_text
    chunks.append((chunk_index, chunk_content))
MemoryError
```

`chunk_precedent_text` 함수에서 텍스트가 `chunk_size`보다 짧을 때 무한 루프 발생.

**원인**
```python
# 기존 코드
start = end - config.chunk_overlap
if start >= len(text) - config.min_chunk_size:
    break
```
- 텍스트 길이 500, chunk_size=1250, chunk_overlap=125
- 1회차: start=0, end=500 → 다음 start=375
- 2회차: start=375, end=500 → 다음 start=375 (같은 값!)
- `start`가 진행하지 않아 무한 루프

**해결**
```python
# 수정 코드
new_start = end - config.chunk_overlap

# 무한 루프 방지: start가 진행하지 않으면 종료
if new_start <= start:
    break

start = new_start
```

---

### 1.2 PyTorch CUDA 버전 문제

**문제**
- `torch 2.9.1+cpu` 버전이 설치되어 GPU 미인식
- `uv run` 실행 시 lockfile 동기화로 CPU 버전으로 덮어씀

**해결**
```bash
# CUDA 버전 설치 후 --no-sync 옵션으로 실행
uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu128
uv run --no-sync python scripts/test_precedent_embedding.py
```

**참고**: `pyproject.toml`에 torch를 명시하지 않음 (환경별로 다른 버전 필요)

---

### 1.3 RTX 5060 Ti (Blackwell) 지원 문제

**문제**
```
NVIDIA GeForce RTX 5060 Ti with CUDA capability sm_120 is not compatible
with the current PyTorch installation.
```
- RTX 5060 Ti는 Blackwell 아키텍처 (sm_120)
- PyTorch 2.6.0+cu124까지는 sm_90까지만 지원

**해결**
- **PyTorch 2.10 Stable (2026-01-21 릴리즈)** 에서 CUDA 12.8 공식 지원
```bash
uv pip install --reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**결과**
```
CUDA: True
Device: NVIDIA GeForce RTX 5060 Ti
VRAM: 15.9GB
Torch: 2.10.0+cu128
```

---

### 1.4 법령/판례 임베딩 코드 중복

**문제**
- `run_law_embedding`과 `run_precedent_embedding` 함수가 거의 동일한 로직
- 법령은 전체 개수를 미리 로드, 판례는 스트리밍 (일관성 없음)
- tqdm 진행률 표시 방식 불일치

**해결: 통합 클래스 설계**

```python
# 추상 베이스 클래스
class StreamingEmbeddingProcessor(ABC):
    """스트리밍 방식 임베딩 프로세서"""

    def __init__(self, data_type: str):
        self.data_type = data_type
        self.device_info = get_device_info()
        # ...

    def load_streaming(self, source_path: str) -> tuple:
        """개수 세기 스킵, 즉시 시작"""
        # ...

    def run(self, source_path: str, reset: bool, batch_size: int) -> dict:
        """통합 실행 로직"""
        # ...

    @abstractmethod
    def get_chunk_config(self) -> Any: ...
    @abstractmethod
    def extract_source_id(self, item: dict, idx: int) -> str: ...
    @abstractmethod
    def extract_text_for_embedding(self, item: dict) -> str: ...
    # ...

# 서브클래스
class LawEmbeddingProcessor(StreamingEmbeddingProcessor): ...
class PrecedentEmbeddingProcessor(StreamingEmbeddingProcessor): ...
```

**통일된 동작**
- 개수 세기 스킵 → 즉시 시작
- tqdm에서 속도(it/s)만 표시 (진행률 % 미표시)
- 스트리밍으로 메모리 효율적 처리

---

## 2. 최종 임베딩 결과

### 데이터 통계

| 데이터 타입 | 원본 건수 | 임베딩 청크 수 |
|-------------|-----------|----------------|
| 판례 | 65,107건 | 134,846개 |
| 법령 | 5,841건 | 118,922개 |
| **총합** | **70,948건** | **253,768개** |

### 검색 테스트

```python
# 쿼리: "임대차 보증금 반환"
# 결과 (코사인 유사도)
[1] 0.6833 | 판례 | 건물명도 | 2002다52657
[2] 0.6811 | 판례 | 건물철거등 | 98다15545
[3] 0.6760 | 판례 | 보증금반환 | 76다1032
```

```python
# 쿼리: "주택임대차보호법"
# 결과 (법령 + 판례 혼합)
[1] 0.6388 | 법령 | 부동산 거래신고 등에 관한 법률
[2] 0.6358 | 법령 | 주택임대차분쟁조정위원회 확인서류...
[3] 0.6329 | 판례 | 약정금청구
```

---

## 3. 환경 설정 요약

### 로컬 개발 환경 (RTX 5060 Ti)

```bash
# PyTorch CUDA 12.8 설치
uv pip install --reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 임베딩 실행 (--no-sync 필수)
cd backend
uv run --no-sync python scripts/runpod_lancedb_embeddings.py \
  --type precedent \
  --precedent-source "../data/[cleaned]precedents_partial_done.json" \
  --batch-size 100
```

### 자동 감지 설정

| GPU VRAM | batch_size | num_workers | gc_interval |
|----------|------------|-------------|-------------|
| 20GB+ (3090/4090) | 128 | 4 | 25 |
| 14GB+ (4080/5060Ti) | 100 | 4 | 20 |
| 8GB+ (3070/4060) | 70 | 2 | 15 |
| 8GB 미만 | 50 | 2 | 10 |

---

## 4. 파일 변경 내역

| 파일 | 변경 내용 |
|------|----------|
| `runpod_lancedb_embeddings.py` | 청킹 버그 수정, 통합 클래스 추가 |
| `pyproject.toml` | 변경 없음 (torch는 환경별 수동 설치) |

---

## 5. 알려진 제한사항

### LanceDB compact 경고
```
DeprecatedWarning: compact_files is deprecated as of 0.21.0
[WARN] Compact failed (non-critical): The lance library is required
```
- 파일 압축 최적화 기능, 동작에는 영향 없음
- `pylance` 설치하면 경고 제거 가능

### psutil 미설치
```
[Memory] psutil not installed
```
- 메모리 모니터링용, 없어도 정상 동작

---

## 6. 다음 단계

- [ ] 검색 API 연동 (LanceDB → 프론트엔드)
- [ ] 하이브리드 검색 (벡터 + 키워드) 구현
- [ ] 검색 결과 캐싱
