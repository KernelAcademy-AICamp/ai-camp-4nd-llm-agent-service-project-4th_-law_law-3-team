# CLAUDE.md - 벡터 DB 임베딩 스크립트

이 폴더는 LanceDB 벡터 데이터베이스 생성 및 관리 스크립트를 포함합니다.

## 핵심 스크립트

| 스크립트 | 용도 |
|----------|------|
| `runpod_lancedb_embeddings.py` | 메인 임베딩 스크립트 (로컬/RunPod/Colab) |
| `colab_lancedb_embeddings.py` | Google Colab 전용 |
| `test_precedent_embedding.py` | 임베딩 테스트 |

## 빠른 시작

### 로컬 실행 (GPU)

```bash
cd backend

# PyTorch CUDA 설치 (환경에 맞게)
uv pip install --reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 판례 임베딩
uv run --no-sync python scripts/runpod_lancedb_embeddings.py \
  --type precedent \
  --precedent-source "../data/[cleaned]precedents_partial_done.json"

# 법령 임베딩
uv run --no-sync python scripts/runpod_lancedb_embeddings.py \
  --type law \
  --law-source "../data/law_cleaned.json"

# 전체 (법령 + 판례)
uv run --no-sync python scripts/runpod_lancedb_embeddings.py --type all

# 리셋 후 재생성
uv run --no-sync python scripts/runpod_lancedb_embeddings.py --type all --reset

# 통계 확인
uv run --no-sync python scripts/runpod_lancedb_embeddings.py --stats
```

### Python API

```python
from runpod_lancedb_embeddings import (
    LawEmbeddingProcessor,
    PrecedentEmbeddingProcessor,
    run_law_embedding,
    run_precedent_embedding,
)

# 방법 1: 클래스 직접 사용
processor = PrecedentEmbeddingProcessor()
stats = processor.run("precedents.json", reset=True, batch_size=100)

# 방법 2: 함수 사용 (하위 호환)
stats = run_precedent_embedding("precedents.json", reset=True)
```

## 클래스 구조

```
StreamingEmbeddingProcessor (ABC)
├── LawEmbeddingProcessor      # 법령 임베딩
└── PrecedentEmbeddingProcessor # 판례 임베딩
```

### 주요 메서드

| 메서드 | 설명 |
|--------|------|
| `run(source_path, reset, batch_size)` | 임베딩 실행 |
| `load_streaming(source_path)` | 스트리밍 로드 (개수 세기 스킵) |
| `get_chunk_config()` | 청킹 설정 반환 |
| `extract_text_for_embedding(item)` | 임베딩 텍스트 추출 |

## 설정값

### 청킹 설정 (CONFIG)

```python
# 판례
PRECEDENT_CHUNK_SIZE = 1250      # 최대 글자 수
PRECEDENT_CHUNK_OVERLAP = 125   # 오버랩 (10%)
PRECEDENT_MIN_CHUNK_SIZE = 100  # 최소 글자 수

# 법령
LAW_MAX_TOKENS = 800            # 최대 토큰 수
LAW_MIN_TOKENS = 100            # 최소 토큰 수
```

### 자동 감지 설정

GPU VRAM에 따라 자동 설정:

| VRAM | batch_size | gc_interval |
|------|------------|-------------|
| 20GB+ | 128 | 25 |
| 14GB+ | 100 | 20 |
| 8GB+ | 70 | 15 |
| <8GB | 50 | 10 |

## 데이터 흐름

```
JSON 파일
    ↓ (ijson 스트리밍)
청킹 (LawChunkConfig / PrecedentChunkConfig)
    ↓
배치 수집 (batch_size개)
    ↓
임베딩 생성 (KURE-v1, 1024차원)
    ↓
LanceDB 저장 (./lancedb_data)
    ↓
GC + 메모리 정리
```

## LanceDB 저장 위치

```
backend/lancedb_data/
└── legal_chunks.lance/   # 테이블 데이터
```

## 임베딩 모델

- **모델**: `nlpai-lab/KURE-v1`
- **차원**: 1024
- **특징**: 한국어 법률 도메인 최적화

## 유틸리티 함수

```python
# 디바이스 정보
print_device_info()

# 메모리 상태
print_memory_status()

# 메모리 정리 (GC + CUDA cache)
clear_memory()

# 모델 캐시 정리
clear_model_cache()

# 랜덤 시드 고정 (재현성)
set_seed(42, deterministic=False)

# LanceDB 통계
show_stats()
```

## 임베딩 캐싱

동일 텍스트 재임베딩 방지를 위한 해시 기반 디스크 캐시.

```python
from runpod_lancedb_embeddings import EmbeddingCache, create_embeddings

# 캐시 초기화
cache = EmbeddingCache("./embedding_cache")

# 캐시 조회 후 없으면 계산
embedding = cache.get_or_compute("법률 텍스트", create_embeddings)

# 수동 조회/저장
cached = cache.get("텍스트")
if cached is None:
    emb = create_embeddings(["텍스트"])[0]
    cache.set("텍스트", emb)

# 캐시 통계
stats = cache.get_stats()
# {'hits': 150, 'misses': 50, 'hit_rate': '75.0%', 'memory_cache_size': 200}

# 캐시 정리
cache.clear_memory_cache()  # 메모리만
cache.clear_all()           # 전체 (디스크 포함)
```

### 캐시 구조

```
embedding_cache/
├── a1/
│   ├── a1b2c3d4...json
│   └── a1e5f6g7...json
├── b2/
│   └── b2c3d4e5...json
└── ...
```

## 임베딩 품질 검증

유사/비유사 문서 쌍으로 임베딩 품질 평가.

```python
from runpod_lancedb_embeddings import EmbeddingQualityChecker

checker = EmbeddingQualityChecker()

# 빠른 테스트 (법률 도메인 기본 쌍)
report = checker.quick_test()
# Similar pairs avg:    0.8542
# Dissimilar pairs avg: 0.3215
# Separation:           0.5327
# Quality:              GOOD

# 커스텀 테스트
similar_pairs = [
    ("손해배상 청구권", "손해배상 청구"),
    ("민법 제750조", "민법상 불법행위"),
]
dissimilar_pairs = [
    ("민법 제750조", "형법 제250조"),
    ("손해배상 청구", "회사 설립 절차"),
]
report = checker.evaluate(similar_pairs, dissimilar_pairs)

# 두 텍스트 유사도 직접 계산
sim = checker.compute_similarity("텍스트1", "텍스트2")
```

### 품질 기준

| Separation | Quality | 의미 |
|------------|---------|------|
| > 0.2 | GOOD | 유사/비유사 잘 구분 |
| 0.1 ~ 0.2 | FAIR | 구분 가능 |
| < 0.1 | POOR | 구분 어려움 |

## 분할 처리 (대용량)

```python
# 판례 분할 (5000건씩)
split_precedents('precedents.json', chunk_size=5000)
run_all_precedent_parts('precedents_part_*.json', batch_size=64)

# 법령 분할 (2000건씩)
split_laws('laws.json', chunk_size=2000)
run_all_law_parts('laws_part_*.json', batch_size=64)
```

## 검색 테스트

```python
import lancedb
from runpod_lancedb_embeddings import get_embedding_model

model = get_embedding_model('cuda')
query_vector = model.encode('임대차 보증금 반환')

db = lancedb.connect('./lancedb_data')
table = db.open_table('legal_chunks')

# 코사인 유사도 검색
results = table.search(query_vector).metric('cosine').limit(10).to_pandas()
for _, row in results.iterrows():
    sim = 1 - row['_distance']
    print(f"{sim:.4f} | {row['data_type']} | {row['title']}")
```

## 주의사항

1. **--no-sync 필수**: `uv run --no-sync` 사용 (torch 버전 유지)
2. **torch 환경별 설치**: pyproject.toml에 torch 없음, 수동 설치 필요
3. **compact 경고**: `pylance` 미설치 시 경고 발생, 동작에 영향 없음
4. **메모리 모니터링**: `psutil` 설치 시 메모리 상태 출력

## PyTorch 최적화 패턴

스크립트에 적용된 최적화 패턴:

| 패턴 | 함수/클래스 | 설명 |
|------|------------|------|
| 디바이스 자동 선택 | `get_device_info()` | CUDA > MPS > CPU 우선순위 |
| 멀티 GPU 지원 | `get_optimal_cuda_device()` | VRAM 최대 GPU 선택 |
| 메모리 정리 | `clear_memory()` | GC + CUDA cache 통합 |
| 재현성 | `set_seed()` | 랜덤 시드 고정 |
| VRAM 기반 설정 | `get_optimal_config()` | 배치 크기 자동 조정 |

## 관련 문서

- `docs/vectordb_design.md` - 전체 설계 문서
- `docs/EMBEDDING_DEV_LOG_20260129.md` - 개발 로그
