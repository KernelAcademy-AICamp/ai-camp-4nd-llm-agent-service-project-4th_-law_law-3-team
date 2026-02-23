# CLAUDE.md - 벡터 DB 임베딩 스크립트

이 폴더는 LanceDB 벡터 데이터베이스 생성 및 관리 스크립트를 포함합니다.

## 핵심 스크립트

| 스크립트 | 용도 |
|----------|------|
| `ingest/cli.py` | **메인** 인제스트 파이프라인 CLI (`python -m scripts.ingest.cli`) |
| `runpod_lancedb_embeddings.py` | RunPod 노트북용 thin wrapper (ingest 파이프라인 호출) |
| `colab_lancedb_embeddings.py` | Google Colab 노트북용 thin wrapper (runpod wrapper re-export) |
| `check_environment.py` | 데이터 로드 전 환경 검증 (Python, MeCab, Docker, Alembic 등) |

### 범용 공통 모듈 (`common/`)

스크립트 간 공유되는 JSON 로딩, DB 세션, 로깅, 배치 처리, 인용 추출 유틸리티입니다.

| 모듈 | 설명 |
|------|------|
| `paths.py` | 경로 상수 (SCRIPTS_DIR, BACKEND_DIR, PROJECT_ROOT, DATA_DIR) + `setup_sys_path()` |
| `logging_config.py` | `setup_logging(name, level, fmt, datefmt) -> Logger` |
| `json_loader.py` | `load_json_file`, `load_json_directory`, `load_items`, `stream_json`, `smart_load`, `resolve_source_path` |
| `db.py` | `create_sync_engine(echo)`, `create_sync_session_factory(echo)` |
| `batch.py` | `batch_iterate(items, batch_size)`, `process_in_batches(items, fn, batch_size)` |
| `citation.py` | `extract_citations`, `extract_law_names`, `extract_case_numbers`, `extract_statute_names_plain` |

사용 예시:
```python
from scripts.common import load_items, setup_logging, create_sync_session_factory
```

### 임베딩 공통 모듈 (`embedding_common/`)

| 모듈 | 설명 |
|------|------|
| `device.py` | GPU/CPU/MPS 디바이스 감지, DeviceInfo |
| `config.py` | 하드웨어 프로필, 배치 크기 최적 설정 |
| `model.py` | 임베딩 모델 로딩 (KURE-v1) + ONNX 배치 분기 |
| `store.py` | LanceDB 테이블 생성/연결 |
| `chunking.py` | 텍스트 청킹 (법령/판례) |
| `schema.py` | 스키마 v2 re-export + 검증 유틸 |
| `cache.py` | MD5 기반 임베딩 캐시 |
| `quality.py` | 임베딩 품질 검증 (유사/비유사 쌍 평가) |
| `temperature.py` | GPU 온도 모니터링 (nvidia-smi) |
| `memory.py` | GPU/시스템 메모리 모니터링 |

### Jupyter Notebook (`../notebooks/`)

| 노트북 | 환경 | 설명 |
|--------|------|------|
| `runpod_lancedb_embeddings.ipynb` | RunPod (A100/H100) | 클라우드 GPU 임베딩 |
| `colab_lancedb_embeddings.ipynb` | Google Colab (T4) | Drive 저장, 분할 처리 |

## 빠른 시작

### 메인: ingest 파이프라인 (권장)

```bash
cd backend

# PyTorch 설치 (환경에 맞게)
uv pip install --reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128  # CUDA
# uv pip install --reinstall torch torchvision torchaudio  # CPU/MPS

# 벡터 임베딩 생성 (--no-sync 필수: torch 버전 유지)
uv run --no-sync python -m scripts.ingest.cli --type all --step vector --reset

# 특정 타입만
uv run --no-sync python -m scripts.ingest.cli --type precedent --step vector
uv run --no-sync python -m scripts.ingest.cli --type law --step vector

# 통계 확인
uv run --no-sync python -m scripts.ingest.cli --type all --stats
```

### RunPod/Colab 노트북용 thin wrapper

RunPod/Colab 노트북에서는 하위 호환 API를 사용합니다.
내부적으로 ingest 파이프라인의 `run_vector_ingest()`를 호출합니다.

```bash
# RunPod CLI (노트북 외부에서 사용 시)
uv run --no-sync python scripts/runpod_lancedb_embeddings.py --type all --reset
uv run --no-sync python scripts/runpod_lancedb_embeddings.py --stats
```

### Python API

```python
# 방법 1: ingest 파이프라인 직접 사용 (권장)
from scripts.ingest.vector_writer import run_vector_ingest
from scripts.ingest.config import get_config

config = get_config("precedent")
stats = run_vector_ingest(config, reset=True)

# 방법 2: thin wrapper (노트북 하위 호환)
from scripts.runpod_lancedb_embeddings import (
    run_law_embedding,
    run_precedent_embedding,
    show_stats,
)

stats = run_precedent_embedding("precedents.json", reset=True)
```

## 설정값

### 청킹 설정 (`embedding_common/chunking.py`)

```python
# 판례
PRECEDENT_CHUNK_SIZE = 1250      # 최대 글자 수
PRECEDENT_CHUNK_OVERLAP = 125   # 오버랩 (10%)
PRECEDENT_MIN_CHUNK_SIZE = 100  # 최소 글자 수

# 법령
LAW_MAX_TOKENS = 800            # 최대 토큰 수
LAW_MIN_TOKENS = 100            # 최소 토큰 수
```

### 하드웨어 자동 감지 (`embedding_common/config.py`)

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
청킹 (embedding_common/chunking.py)
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
├── legal_chunks.lance/              # 19개 타입 통합 테이블
└── local_ordinance_chunks.lance/   # 자치법규 전용 (전체요약 + 조문요약)
```

## 임베딩 모델

- **모델**: `nlpai-lab/KURE-v1`
- **차원**: 1024
- **특징**: 한국어 법률 도메인 최적화

## 유틸리티 함수

```python
from scripts.embedding_common.device import print_device_info
from scripts.embedding_common.memory import print_memory_status
from scripts.embedding_common.model import clear_memory, clear_model_cache, set_seed
from scripts.embedding_common.store import EmbeddingStore

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
store = EmbeddingStore()
print(f"Total rows: {store.count()}")
```

## 임베딩 캐싱

동일 텍스트 재임베딩 방지를 위한 해시 기반 디스크 캐시.

```python
from scripts.embedding_common.cache import EmbeddingCache
from scripts.embedding_common.model import create_embeddings

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
from scripts.embedding_common.quality import EmbeddingQualityChecker

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
from scripts.embedding_common.model import get_embedding_model

model = get_embedding_model()
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

`embedding_common/` 모듈에 적용된 최적화 패턴:

| 패턴 | 모듈/함수 | 설명 |
|------|----------|------|
| 디바이스 자동 선택 | `device.get_device_info()` | CUDA > MPS > CPU 우선순위 |
| 멀티 GPU 지원 | `device.get_optimal_cuda_device()` | VRAM 최대 GPU 선택 |
| 메모리 정리 | `model.clear_memory()` | GC + CUDA cache 통합 |
| 재현성 | `model.set_seed()` | 랜덤 시드 고정 |
| VRAM 기반 설정 | `config.get_optimal_config()` | 배치 크기 자동 조정 |

## 관련 문서

- `docs/architecture/vectordb_design.md` - 전체 설계 문서
- `docs/devlog/EMBEDDING_DEV_LOG_20260129.md` - 개발 로그
- `notebooks/runpod_lancedb_embeddings.ipynb` - RunPod 노트북
- `notebooks/colab_lancedb_embeddings.ipynb` - Colab 노트북

---

## ONNX 최적화 모델 빌드 + RAG 테스트 환경 구축

다른 컴퓨터에서 ONNX 최적화 모델을 빌드하고, 두 variant로 임베딩하여 RAG 검색 품질을 비교하는 가이드입니다.

### 비교 대상 모델

| Variant | 설명 | Latency | Cosine | 비고 |
|---------|------|---------|--------|------|
| `ort-opt` | ORT 그래프 최적화 FP32 | 218ms | 1.0000 | 무손실 |
| `ort-opt-qdq` | QDQ INT8 (16 FP32 레이어) | 167ms | 0.9990 | 품질 우선 권장 |

### Step 1: 환경 설정

```bash
# 1-1. 저장소 클론 + 브랜치 전환
git clone <repo-url>
cd law-3-team/backend
git checkout feature/onnx-graph-optimization-benchmark

# 1-2. Python 의존성 설치
uv sync --dev

# 1-3. PyTorch 설치 (환경에 맞게 선택)
# CUDA (RunPod/서버)
uv pip install --reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
# CPU/MPS (Mac)
# uv pip install --reinstall torch torchvision torchaudio

# 1-4. 추가 의존성 (ONNX 빌드용)
uv pip install optimum onnxruntime onnx
```

### Step 2: ONNX 모델 빌드

`build_optimized_onnx.py`가 HuggingFace에서 원본 모델을 다운로드하고, ONNX 변환 → 그래프 최적화 → 양자화를 순서대로 수행합니다.

```bash
cd backend

# 2-1. 임베딩 모델 (KURE-v1) + 리랭커 모두 빌드 (모든 variant)
uv run --no-sync python scripts/build_optimized_onnx.py

# 2-2. 임베딩 모델만 빌드
uv run --no-sync python scripts/build_optimized_onnx.py --embedding-only

# 2-3. 리랭커만 빌드
uv run --no-sync python scripts/build_optimized_onnx.py --reranker-only

# 2-4. 기존 모델 덮어쓰기
uv run --no-sync python scripts/build_optimized_onnx.py --overwrite

# 2-5. 빌드 후 수치 검증만 실행
uv run --no-sync python scripts/build_optimized_onnx.py --verify
```

**빌드 출력 디렉토리** (`data/models/`):

| 디렉토리 | 설명 | 크기 |
|----------|------|------|
| `kure-v1-ort-opt/` | 임베딩 FP32 최적화 | ~1.1GB |
| `kure-v1-ort-opt-qdq/` | 임베딩 QDQ INT8 (16 FP32) | ~700MB |
| `reranker-ort-opt/` | 리랭커 FP32 최적화 | ~1.1GB |
| `reranker-ort-opt-qdq/` | 리랭커 QDQ INT8 | ~700MB |

> **디스크 요구량**: 빌드 중간 파일 포함 최소 **10GB** 여유 필요.
> 빌드 중 임시 파일(onnx-export-tmp)은 자동 삭제됩니다.

### Step 3: 임베딩 생성 (두 variant 비교)

두 variant로 **각각 별도 LanceDB 테이블**에 임베딩하여 비교합니다.

```bash
cd backend

# 3-1. data/ 폴더에 법령/판례 JSON 준비
# data/law_v3.json, data/precedents_v2.json 필요
# Google Drive에서 복원: rclone copy --config rclone.conf gdrive:data/ data/ --progress

# 3-2. Variant A: ORT-opt FP32 (무손실)로 임베딩
USE_ONNX_EMBEDDING=true \
ONNX_EMBEDDING_VARIANT=ort-opt \
LANCEDB_URI=./lancedb_data_ort_opt \
uv run --no-sync python -m scripts.ingest.cli --type all --step vector --reset

# 3-3. Variant B: QDQ INT8 (cosine 0.999)로 임베딩
USE_ONNX_EMBEDDING=true \
ONNX_EMBEDDING_VARIANT=ort-opt-qdq \
LANCEDB_URI=./lancedb_data_qdq \
uv run --no-sync python -m scripts.ingest.cli --type all --step vector --reset

# 3-4. 기존 PyTorch FP32 임베딩도 비교하려면
USE_ONNX_EMBEDDING=false \
LANCEDB_URI=./lancedb_data_pytorch \
uv run --no-sync python -m scripts.ingest.cli --type all --step vector --reset
```

> **주의**: `LANCEDB_URI`를 variant별로 다르게 설정하여 데이터가 섞이지 않도록 합니다.
> **CUDA 자동 감지**: `onnxruntime-gpu` 설치 시 CUDA EP를 자동 사용합니다. GPU 서버에서 ONNX+CUDA로 인제스트하면 PyTorch CUDA 대비 그래프 최적화+fusion 효과를 볼 수 있습니다. CPU 전용 환경에서는 자동 fallback.

### Step 4: RAG 검색 품질 비교

```bash
cd backend

# 4-1. Variant A (ORT-opt FP32)로 RAG 평가
USE_ONNX_EMBEDDING=true \
ONNX_EMBEDDING_VARIANT=ort-opt \
LANCEDB_URI=./lancedb_data_ort_opt \
uv run --no-sync python -m evaluation.run

# 4-2. Variant B (QDQ INT8)로 RAG 평가
USE_ONNX_EMBEDDING=true \
ONNX_EMBEDDING_VARIANT=ort-opt-qdq \
LANCEDB_URI=./lancedb_data_qdq \
uv run --no-sync python -m evaluation.run

# 4-3. PyTorch 기준선
USE_ONNX_EMBEDDING=false \
LANCEDB_URI=./lancedb_data_pytorch \
uv run --no-sync python -m evaluation.run
```

**평가 지표 목표**: Recall@10 ≥ 0.8, MRR ≥ 0.7, Hit Rate ≥ 0.9

### Step 5: 리랭커 variant 비교 (선택)

리랭커도 ONNX variant를 비교하려면:

```bash
# .env 또는 환경변수로 설정
USE_ONNX_RERANKER=true
ONNX_RERANKER_VARIANT=ort-opt       # 또는 ort-opt-qdq
```

### ONNX 환경 변수 요약

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `USE_ONNX_EMBEDDING` | `false` | ONNX 임베딩 사용 여부 |
| `ONNX_EMBEDDING_VARIANT` | `ort-opt` | `ort-opt` (FP32) 또는 `ort-opt-qdq` (INT8) |
| `USE_ONNX_RERANKER` | `false` | ONNX 리랭커 사용 여부 |
| `ONNX_RERANKER_VARIANT` | `ort-opt` | `ort-opt` (FP32) 또는 `ort-opt-qdq` (INT8) |
| `ONNX_INTRA_OP_THREADS` | `0` | 0=자동, 4=Mac ARM P코어만 권장 |
| `ONNX_ENABLE_IO_BINDING` | `false` | CUDA EP에서 유효 (CPU EP에서 무효) |
| `ONNX_ENABLE_BF16_FASTMATH` | `false` | Graviton3+ 전용 (Mac ARM 미지원) |
| `ONNX_QDQ_SENSITIVE_LAYERS` | `""` | 커스텀 민감 레이어 (빈 문자열=기본 16개) |
| `ONNX_QUALITY_GATE_ENABLED` | `true` | 품질 게이트 활성화 |
| `ONNX_QUALITY_GATE_FALLBACK` | `true` | 품질 미달 시 PyTorch 폴백 |

### 관련 스크립트

| 스크립트 | 용도 |
|----------|------|
| `build_optimized_onnx.py` | ONNX 모델 빌드 (변환+최적화+양자화+검증) |
| `benchmark_arm_optimization.py` | ARM 최적화 벤치마크 (latency, cosine) |
| `sweep_sensitive_layers.py` | QDQ INT8 민감 레이어 탐색 (최적 FP32 레이어 결정) |
| `benchmark_new_optimizations.py` | Session Config / CoreML / Dynamic INT8 벤치마크 |
| `download_models.py` | HuggingFace 모델 다운로드 (PyTorch 원본) |

### 벤치마크 보고서

- `docs/04-report/features/arm-onnx-optimization-benchmark.md` — 전체 최적화 벤치마크 결과

---

## 법률 용어 PostgreSQL 로드 (load_legal_terms_data.py)

`lawterms_v1.json` (81,488건 → ~72,700 고유 용어)을 PostgreSQL `legal_terms` 테이블로 로드합니다.
fallback: `data/law_data/lawterms_full.json` (37,169건)
MeCab 토크나이저에서 법률 복합명사를 보강하기 위한 용어 사전 데이터입니다.

### 사전 조건

```bash
# 1. 마이그레이션 실행 (legal_terms 테이블 생성)
cd backend
uv run alembic upgrade head
```

### 사용법

```bash
cd backend

# 데이터 로드
uv run python scripts/load_legal_terms_data.py

# 기존 데이터 삭제 후 재로드
uv run python scripts/load_legal_terms_data.py --reset

# 검증만 (로드 없이)
uv run python scripts/load_legal_terms_data.py --verify

# 통계만 확인
uv run python scripts/load_legal_terms_data.py --stats
```

### 주요 동작

1. `lawterms_v1.json` 로드 (fallback: `lawterms_full.json`)
2. 리스트 타입 레코드 평탄화 (flatten)
3. 법령한영사전 역방향 한글 용어 추출 (reverse extraction)
4. 우선순위 기반 중복 제거 + `source_count` 집계
5. `ON CONFLICT (term) DO UPDATE`로 멱등성 보장
6. 1,000건 단위 배치 insert
7. 로드 후 통계 출력 (총 건수, 한글 전용 비율, 길이 분포, 제외 통계)

### 환경 변수

```bash
# backend/.env
DATABASE_URL=postgresql://lawuser:lawpassword@localhost:5432/lawdb
USE_LEGAL_TERM_DICT=true  # 앱에서 사전 사용 활성화
```

### 데이터 현황

| 항목 | 수치 |
|------|------|
| 원본 레코드 | 81,488건 (lawterms_v1.json) |
| 고유 용어 | ~72,700개 (평탄화+역추출 포함) |
| 한글 전용 (2-10자) | ~35,200개 (MeCab 로드 대상) |
| 사전유형 | 법령정의사전, 생활용어사전, 법령한영사전, 법령용어사전, 한영역추출 |

---

## 변호사 데이터 PostgreSQL 로드 (load_lawyers_data.py)

`data/lawyers.json` (17,326건)을 PostgreSQL `lawyers` 테이블로 로드합니다.

### 사전 조건

```bash
# 1. 마이그레이션 실행 (lawyers 테이블 생성)
cd backend
uv run alembic upgrade head
```

### 사용법

```bash
cd backend

# 데이터 로드
uv run python scripts/load_lawyers_data.py

# 기존 데이터 삭제 후 재로드
uv run python scripts/load_lawyers_data.py --reset

# 검증만 (로드 없이)
uv run python scripts/load_lawyers_data.py --verify
```

### 주요 동작

1. `data/lawyers.json` 읽기
2. 각 레코드에 `extract_region()` 적용 → province, district, region 계산
3. `ON CONFLICT (detail_id) DO UPDATE`로 멱등성 보장
4. 1,000건 단위 배치 insert
5. 로드 후 통계 출력 (총 건수, 좌표/전문분야 비율, 상위 지역)

### 환경 변수

```bash
# backend/.env
DATABASE_URL=postgresql://lawuser:lawpassword@localhost:5432/lawdb
USE_DB_LAWYERS=true  # DB 모드 활성화
```

---

## 변호사 지오코딩 (geocode_lawyers.py)

변호사 주소를 카카오 API로 좌표 변환합니다.

> **데이터 경계:** 변호사 데이터 수집(크롤링, 전문분야)은 별도 저장소에서 관리합니다.
> 이 프로젝트에서는 좌표 변환(지오코딩)만 수행합니다.

### 데이터 흐름

```
별도 저장소 → all_lawyers.json → geocode_lawyers.py → data/lawyers.json
```

### 사용법

```bash
cd backend

# 기본 실행 (all_lawyers.json → data/lawyers.json)
uv run python scripts/geocode_lawyers.py

# API 키 직접 전달
uv run python scripts/geocode_lawyers.py --api-key YOUR_KAKAO_REST_API_KEY

# 입출력 경로 지정
uv run python scripts/geocode_lawyers.py --input path/to/input.json --output path/to/output.json

# 실패 항목만 재시도 (기존 출력 파일에서 좌표 없는 항목)
uv run python scripts/geocode_lawyers.py --retry-failed

# 현재 데이터 상태 확인 (지오코딩 실행 안 함)
uv run python scripts/geocode_lawyers.py --stats
```

### CLI 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--api-key` | 카카오 REST API 키 | `KAKAO_REST_API_KEY` 환경변수 |
| `--input` | 입력 파일 경로 | `all_lawyers.json` |
| `--output` | 출력 파일 경로 | `data/lawyers.json` |
| `--retry-failed` | 좌표 없는 항목만 재시도 | - |
| `--stats` | 데이터 상태만 출력 | - |

### 필수 환경 변수

```bash
# backend/.env
KAKAO_REST_API_KEY=your_kakao_rest_api_key
```

### 출력 파일

| 파일 | 설명 |
|------|------|
| `data/lawyers.json` | 좌표가 추가된 변호사 데이터 |
| `data/geocode_failures.json` | 지오코딩 실패 목록 |

---

## 인제스트 파이프라인 (scripts/ingest/)

config-driven 인제스트 파이프라인. 데이터 타입별 설정을 `types/` 하위에 정의하면 벡터 DB + PostgreSQL + FTS를 일괄 처리합니다.

### 1. DB 적재 데이터 소스 위치

모든 소스는 프로젝트 루트 `data/` 하위에 위치합니다. (`config.py`의 `DATA_DIR`)

```
data/
├── law_v3.json                    # 법령
├── precedents_v2.json             # 판례
├── admin_rule_v2.json             # 행정규칙
├── constitutional_v2.json         # 헌재결정례
├── administration_v2.json         # 행정심판례
├── legislation_v2.json            # 법령해석례
├── treaty_v2.json                 # 조약
├── interpretation_ministry/       # 부처해석례 (28개 부처별 JSON)
│   ├── intp_min_경찰청_v2.json
│   ├── intp_min_고용노동부_v2.json
│   └── ...
├── special_admin_appeal/          # 특별행정심판례 (2개 기관별 JSON)
│   ├── sadm_case_조세심판원_v2.json
│   └── sadm_case_해양안전심판원_v2.json
├── local_rules_v1.json            # 자치법규 (160,276건)
└── decisions_committee/           # 위원회 결정문 (10개 위원회별 JSON)
    ├── dec_comm_개인정보보호위원회_v2.json
    ├── dec_comm_고용보험심사위원회_v2.json
    ├── dec_comm_공정거래위원회_v3.json
    ├── dec_comm_국가인권위원회_v2.json
    ├── dec_comm_국민권익위원회_v2.json
    ├── dec_comm_금융위원회_v2.json
    ├── dec_comm_노동위원회_v2.json
    ├── dec_comm_산업재해보상위험재심사위원회_v2.json
    ├── dec_comm_중앙환경분쟁조정위원회_v2.json
    └── dec_comm_증권선물위원회_v2.json
```

### 2. 데이터 타입 구성 (20개)

| `--type` 타입명 | 데이터 | 건수 | 소스 형식 |
|-----------------|--------|------|----------|
| `law` | 법령 | 5,548 | 단일 JSON |
| `precedent` | 판례 | 92,055 | 단일 JSON |
| `admin_rule` | 행정규칙 | 17,332 | 단일 JSON |
| `constitutional` | 헌재결정례 | 31,718 | 단일 JSON |
| `administration` | 행정심판례 | 34,254 | 단일 JSON |
| `legislation` | 법령해석례 | 8,597 | 단일 JSON |
| `treaty` | 조약 | 3,589 | 단일 JSON |
| `interpretation_ministry` | 부처해석례 (28개 부처) | 37,325 | 디렉토리 |
| `special_admin_appeal` | 특별행정심판례 (2개 기관) | 148,778 | 디렉토리 |
| `dec_privacy` | 개인정보보호위원회 결정문 | 1,448 | 개별 JSON |
| `dec_employment` | 고용보험심사위원회 결정문 | 118 | 개별 JSON |
| `dec_fair_trade` | 공정거래위원회 결정문 | ~7,728 | 개별 JSON |
| `dec_human_rights` | 국가인권위원회 결정문 | 3,721 | 개별 JSON |
| `dec_civil_rights` | 국민권익위원회 결정문 | 635 | 개별 JSON |
| `dec_financial` | 금융위원회 결정문 | 662 | 개별 JSON |
| `dec_labor` | 노동위원회 결정문 | 40,714 | 개별 JSON |
| `dec_industrial` | 산업재해보상보험재심사위원회 결정문 | 782 | 개별 JSON |
| `dec_environment` | 중앙환경분쟁조정위원회 결정문 | 358 | 개별 JSON |
| `dec_securities` | 증권선물위원회 결정문 | 636 | 개별 JSON |
| `local_ordinance` | 자치법규 | 160,276 | 단일 JSON |
| **합계** | **20개 타입** | **~584,200** | |

### 3. 사전 조건

인제스트 파이프라인 실행 전 아래 환경이 준비되어야 합니다.

| 조건 | 필요 단계 | 확인 방법 |
|------|----------|----------|
| PostgreSQL 실행 | `db`, `fts` | `docker compose up -d postgres` → `docker logs law-platform-db` |
| Alembic 마이그레이션 | `db`, `fts` | `uv run alembic upgrade head` → `uv run alembic current` |
| MeCab 시스템 패키지 | `db` (FTS 동시 생성), `fts` | `mecab --version` (미설치 시 에러 발생, fallback 없음) |
| 임베딩 모델 (2.3GB) | `vector` | `uv run python scripts/download_models.py --check` |
| PyTorch | `vector` | `uv pip install torch` (환경별 수동 설치, `--no-sync` 필수) |
| `DATABASE_URL` 환경변수 | `db`, `fts` | `backend/.env`에 `DATABASE_URL=postgresql://lawuser:lawpassword@localhost:5432/lawdb` |
| JSON 소스 파일 | 전체 | `data/` 하위에 배치 (위 섹션 1 참조) |

### 4. CLI 전체 옵션

```bash
cd backend
uv run python -m scripts.ingest.cli --type <타입명|all> [옵션]
```

| 옵션 | 값 | 기본값 | 설명 |
|------|-----|--------|------|
| `--type` | `all` 또는 20개 타입명 | (필수) | 인제스트 대상 (`all`: 전체 20개, 또는 `precedent`, `dec_fair_trade` 등 개별 타입) |
| `--step` | `all`, `db`, `vector`, `fts`, `index` | `all` | 실행 단계 |
| `--reset` | - | `false` | 기존 데이터 삭제 후 재실행 |
| `--source` | 파일 경로 | 타입별 기본 경로 | 커스텀 JSON 소스 경로 (`--type all`과 함께 사용 불가) |
| `--batch-size` | 정수 | DB: 1000, 벡터: 하드웨어 자동 | 배치 크기 |
| `--stats` | - | - | 통계만 출력 (적재 안 함) |
| `--verify` | - | - | 검증만 실행 (적재 안 함) |
| `--device` | `cuda`, `mps`, `cpu` | 자동 감지 | 임베딩 디바이스 (`vector` 단계용) |
| `--profile` | `desktop`, `laptop`, `mac`, `cpu` | 자동 감지 | 하드웨어 프로필 (`vector` 단계용) |
| `--no-cache` | - | `false` | 임베딩 캐시 비활성화 (`vector` 단계용) |

### 5. 단계별 실행 가이드

| 단계 | 설명 | 의존성 | 사용 시점 |
|------|------|--------|----------|
| `db` | JSON → PostgreSQL ORM + FTS tsvector 동시 적재 | PostgreSQL, Alembic, (MeCab) | 최초 적재, 데이터 갱신 |
| `vector` | JSON → LanceDB 벡터 임베딩 (1문서=1벡터) | 임베딩 모델, PyTorch | 최초 적재, 데이터 갱신 |
| `fts` | PostgreSQL ORM에서 읽어 tsvector만 재빌드 | PostgreSQL, (MeCab), `db` 완료 | 토크나이저/userdic 변경 후 |
| `index` | LanceDB ANN 인덱스 재빌드 (IVF_FLAT) | `vector` 완료 | 벡터 데이터 변경 후 |
| `all` | `db` → `vector` → `index` 순차 실행 | 전체 | 최초 적재 |

**ai_summary만 업데이트** (JSON 요약 필드 변경 후):
```bash
# 전체 타입 ai_summary 갱신 (FTS 재빌드 없음, MeCab 불필요)
uv run python -m scripts.ingest.summary_updater

# 특정 타입 제외
uv run python -m scripts.ingest.summary_updater --exclude admin_rule

# 특정 타입만
uv run python -m scripts.ingest.summary_updater --type precedent
```

**FTS 재빌드 워크플로우** (토크나이저/userdic 변경 후):
```bash
# MeCab userdic 재빌드
uv run python scripts/build_mecab_userdic.py

# FTS만 재빌드 (ORM 재적재 없이 tsvector만 갱신)
uv run python -m scripts.ingest.cli --type precedent --step fts --reset
```

### 6. 사용 예시

```bash
cd backend

# 전체 타입 × 전체 파이프라인 (최초 적재 시)
uv run python -m scripts.ingest.cli --type all --step all --reset

# 특정 타입 전체 파이프라인
uv run python -m scripts.ingest.cli --type precedent --step all --reset

# DB+FTS만 적재
uv run python -m scripts.ingest.cli --type precedent --step db --reset

# 벡터만 적재 (GPU 환경)
uv run python -m scripts.ingest.cli --type precedent --step vector --device cuda

# FTS만 재빌드 (토크나이저 변경 후)
uv run python -m scripts.ingest.cli --type precedent --step fts --reset

# ANN 인덱스만 재빌드
uv run python -m scripts.ingest.cli --type precedent --step index

# 전체 타입 통계 / 검증
uv run python -m scripts.ingest.cli --type all --stats
uv run python -m scripts.ingest.cli --type all --verify

# 특정 타입 통계 / 검증
uv run python -m scripts.ingest.cli --type precedent --stats
uv run python -m scripts.ingest.cli --type precedent --verify

# 저장 구조 상세 → scripts/ingest/ingest.md 참조
```

### 참고: 코드 구조

```
scripts/ingest/
├── cli.py              # CLI 진입점 (python -m scripts.ingest.cli)
├── config.py           # IngestConfig dataclass + 레지스트리 + get_source_path()
├── sources.yaml        # 20개 타입 데이터 소스 경로 (YAML 중앙 관리)
├── db_writer.py        # PostgreSQL + FTS 적재
├── local_ordinance_vector_writer.py  # 자치법규 전용 벡터 라이터 (1문서→다중벡터)
├── summary_updater.py  # ai_summary 컬럼만 일괄 업데이트 (FTS/MeCab 불필요)
├── shared.py           # 공유 유틸 (토크나이저, FTS 배치)
├── ingest.md           # 20개 타입 저장 구조 상세 문서
└── types/              # 데이터 타입별 설정 (20개 타입)
    ├── __init__.py     # 타입 자동 등록
    ├── _template.py    # 신규 타입 템플릿
    ├── _dec_comm_common.py  # 위원회 결정례 공통 (벡터/FTS 메타)
    ├── law.py          # 법령
    ├── precedent.py    # 판례
    ├── admin_rule.py   # 행정규칙
    ├── constitutional.py    # 헌재결정례
    ├── administration.py    # 행정심판례
    ├── legislation.py       # 법령해석례
    ├── treaty.py            # 조약
    ├── interpretation_ministry.py  # 부처해석례 (디렉토리 소스)
    ├── special_admin_appeal.py     # 특별행정심판례 (디렉토리 소스)
    ├── dec_privacy.py       # 개인정보보호위원회 결정문
    ├── dec_employment.py    # 고용보험심사위원회 결정문
    ├── dec_fair_trade.py    # 공정거래위원회 결정문
    ├── dec_human_rights.py  # 국가인권위원회 결정문
    ├── dec_civil_rights.py  # 국민권익위원회 결정문
    ├── dec_financial.py     # 금융위원회 결정문
    ├── dec_labor.py         # 노동위원회 결정문
    ├── dec_industrial.py    # 산업재해보상보험재심사위원회 결정문
    ├── dec_environment.py   # 중앙환경분쟁조정위원회 결정문
    ├── dec_securities.py    # 증권선물위원회 결정문
    └── local_ordinance.py   # 자치법규 (별도 벡터 테이블: local_ordinance_chunks)
```

### 참고: 비문자열 필드 처리

대부분의 타입은 JSON 필드가 모두 `str`이지만, 아래 2개 타입은 `list` 필드를 포함하므로 `"\n".join()` 처리:

| 타입 | 필드 | JSON 타입 | 처리 |
|------|------|----------|------|
| admin_rule | 조문내용 | `list[str]` | `"\n".join()` → Text 칼럼 |
| dec_fair_trade | 각주목록 | `list[str]` | `"\n".join()` → Text 칼럼 |

### 참고: 새 타입 추가 패턴

아래 순서대로 진행합니다. (기존 타입 예: `dec_fair_trade` 참고)

| # | 파일 | 작업 | 참고 |
|---|------|------|------|
| 1 | `app/models/ingest/new_type_document.py` | **생성** — ORM 테이블 정의 | `id`(PK) + `serial_number`(unique) + 데이터 칼럼 + `ai_summary` + `created_at`/`updated_at` |
| 2 | `app/models/ingest/__init__.py` | **수정** — import + `__all__` 추가 | |
| 3 | `app/models/__init__.py` | **수정** — import + `__all__` 추가 | |
| 4 | `alembic/env.py` | **수정** — import 추가 (autogenerate 감지용) | |
| 5 | `alembic/versions/NNN_*.py` | **생성** — `op.create_table()` 마이그레이션 | |
| 6 | `scripts/ingest/sources.yaml` | **수정** — 소스 경로 등록 | `타입명: 상대경로` 형식 |
| 7 | `scripts/ingest/types/new_type.py` | **생성** — `_template.py` 복사 후 TODO 수정 | 자동 등록 (`__init__.py` 수정 불필요) |
| 8 | `data/` | JSON 소스 파일 배치 | |

---

## 탐색적 데이터 분석 (EDA)

법률 데이터 48개 JSON 파일(~4.9GB)을 분석하는 EDA 프레임워크입니다.

### 구조

```
backend/
├── scripts/common/             # 범용 공통 모듈 (JSON, DB, 로깅, 배치, 인용)
│   ├── json_loader.py
│   ├── db.py
│   ├── logging_config.py
│   ├── batch.py
│   ├── citation.py
│   └── paths.py
├── scripts/eda/
│   ├── common.py           # EDA 전용 유틸 + common/ re-export (하위 호환)
│   └── data_registry.py    # 데이터 카테고리 레지스트리 (11개)
├── notebooks/eda/
│   ├── 01_inventory_schema.ipynb       # 데이터 인벤토리 + 스키마 분석
│   ├── 02_quality_text.ipynb           # 텍스트 품질 분석
│   ├── 03_temporal_relationships.ipynb # 시계열 + 참조 관계
│   ├── 04_projections_summary.ipynb    # 임베딩/그래프 규모 추정
│   ├── 05_lancedb_embedding_strategy.ipynb  # LanceDB 임베딩 전략
│   ├── 06_neo4j_citation_analysis.ipynb     # Neo4j 인용 네트워크
│   └── 07_citation_recovery.ipynb           # 인용 복구 잠재력
└── eda_output/             # 분석 결과 JSON 저장
    ├── phase1_inventory.json
    ├── phase2_schema.json
    ├── phase3_quality.json
    ├── phase4_temporal.json
    ├── phase5_lancedb_strategy.json
    ├── phase6_neo4j_citation.json
    └── phase7_citation_recovery.json
```

### 공유 모듈 (`eda/common.py`)

I/O와 인용 추출 함수는 `scripts/common/`에서 re-export됩니다. EDA 전용 함수만 이 모듈에 직접 구현되어 있습니다.

| 카테고리 | 함수 | 설명 | 출처 |
|----------|------|------|------|
| **I/O** | `smart_load(path)` | 파일 크기 기반 자동 로드 (200MB 기준) | `common/json_loader` re-export |
| | `load_all(path)` | 전체 레코드 리스트 로드 | EDA 자체 |
| | `stream_json(path)` | ijson 스트리밍 로드 | `common/json_loader` re-export |
| | `load_json(path)` | 전체 JSON 로드 | `common/json_loader` re-export |
| | `save_result(name, data)` | `eda_output/`에 JSON 저장 | EDA 자체 |
| | `load_result(name)` | `eda_output/`에서 JSON 로드 | EDA 자체 |
| **샘플링** | `get_sample(path, n, fast)` | 통합 샘플링 (fast=True: head, False: reservoir) | EDA 자체 |
| | `head_sample(path, n)` | 처음 n개 추출 (빠름, 편향 가능) | EDA 자체 |
| | `cached_sample(path, n, seed)` | 디스크 캐시 기반 reservoir sampling | EDA 자체 |
| | `reservoir_sample(iterable, k, seed)` | 무작위 reservoir sampling | EDA 자체 |
| **메타** | `count_records(path)` | 레코드 수 카운트 (스트리밍) | EDA 자체 |
| | `count_records_fast(path)` | 레코드 수 추정 (정규식, 빠름) | EDA 자체 |
| | `discover_done_files()` | `data/` 폴더 `[DONE]*.json` 파일 탐색 | EDA 자체 |
| | `detect_root_type(path)` | JSON 루트 타입 감지 (array/object) | EDA 자체 |
| **분석** | `infer_field_types(sample)` | 필드별 타입/분포 추론 | EDA 자체 |
| **인용 추출** | `extract_citations(text)` | `「법령명」 제N조` 패턴 추출 | `common/citation` re-export |
| | `extract_law_names(text)` | `「법령명」` 패턴 추출 | `common/citation` re-export |
| | `extract_case_numbers(text)` | 사건번호 (`2022다12345`) 추출 | `common/citation` re-export |
| | `extract_statute_names_plain(text)` | 꺾쇠 없는 법령명 추출 | `common/citation` re-export |

### 데이터 레지스트리 (`eda/data_registry.py`)

11개 카테고리, 48개 JSON 파일을 관리합니다.

| 카테고리 | label | 파일 수 | 주요 필드 |
|----------|-------|---------|-----------|
| `precedent` | 판례 | 1 | 판례내용, 판결요지, 판시사항, 이유 |
| `law` | 법령 | 1 | 조문 |
| `constitutional` | 헌재결정례 | 1 | 판시사항, 결정요지, 이유 |
| `administration` | 행정심판례 | 1 | 주문, 이유 |
| `special_tribunal` | 특별행정심판 | 2 | 주문, 이유, 청구취지 |
| `legislation` | 법령해석례 | 1 | 질의요지, 회답, 이유 |
| `committee` | 위원회 결정문 | 10 | 이유, 결정요지, 주문 |
| `cgm_expc` | 부처 해석례 | 27 | 질의요지, 회답 |
| `law_term` | 법률용어사전 | 1 | 법령용어정의 |
| `treaty` | 조약 | 1 | 조약내용 |
| `school` | 행정규칙 | 1 | 조문내용 |

### 노트북 사용법

```python
# 노트북 공통 패턴
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent.parent))

from scripts.eda.common import get_sample, head_sample, load_all, save_result, DATA_DIR
from scripts.eda.data_registry import CATEGORIES

# USE_FULL_DATA 토글 (모든 노트북 공통)
USE_FULL_DATA = False  # True: load_all(), False: head_sample(n=1000)

path = DATA_DIR / "precedents_v2.json"
if USE_FULL_DATA:
    records = load_all(path)
else:
    records = head_sample(path, n=1000)
```

### 관련 문서

- `docs/architecture/EDA_DB_TRANSITION_DESIGN.md` - EDA→DB 전환 설계
- `docs/01-plan/features/data-analysis.plan.md` - PDCA Plan
- `docs/02-design/features/data-analysis.design.md` - PDCA Design

---

## 요약 품질 감사 (audit_summary_quality.py)

JSON 데이터의 LLM 요약 필드를 탐색적으로 감사합니다.
7개 검사: 기본 통계, 마크다운, LLM 아티팩트, 포맷 일관성, 이상치, 중복, 교차 비교.

기존 도구와의 관계:
- `validate_summaries.py` — 사전 정의 규칙 기반 검증 (길이, 프롬프트 누출)
- `clean_summaries.py` — 규칙 기반 클리닝 (마크다운 제거, 불완전 문장 보정)
- **이 스크립트** — 탐색적(EDA) 품질 감사

### 사용법

```bash
cd backend

# Mode 1: IngestConfig 등록 타입 (자동 필드 해석)
uv run python scripts/audit_summary_quality.py --type special_admin_appeal --data-dir ../data

# Mode 2: 임의 JSON 파일 (필드 직접 지정)
uv run python scripts/audit_summary_quality.py \
    --file ../data/special_admin_appeal/sadm_case_조세심판원_v2.json \
    --summary-field 심판례요약 \
    --id-field 특별행정심판재결례일련번호

# 교차 비교 + JSON 보고서
uv run python scripts/audit_summary_quality.py --type special_admin_appeal \
    --data-dir ../data \
    --compare-field 재결요지 \
    --output eda_output/audit_special_admin_appeal.json \
    --samples 10
```

### CLI 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--type` | 인제스트 타입명 (자동 필드 해석) | - |
| `--file` | 임의 JSON 파일 경로 | - |
| `--summary-field` | 요약 필드명 (`--file` 모드 필수) | - |
| `--id-field` | ID 필드명 (`--file` 모드, 미지정 시 인덱스) | - |
| `--data-dir` | 데이터 디렉토리 재매핑 (`--type` 전용) | - |
| `--compare-field` | 교차 비교 대상 필드명 | - |
| `--output` | JSON 보고서 저장 경로 | - |
| `--samples` | 이상치 샘플 수 | 5 |

### 관련 스킬

- `.claude/skills/summary-quality-audit/SKILL.md`

---

## 환경 검증 (check_environment.py)

데이터 로드 전 필수 조건을 자동 검증하는 스크립트입니다. 새 기기에서 환경 세팅 후 실행하면 누락 항목을 한눈에 확인할 수 있습니다.

### 사용법

```bash
cd backend

# 전체 검증
uv run python scripts/check_environment.py

# 특정 범위만 검증
uv run python scripts/check_environment.py --step db      # PostgreSQL 관련
uv run python scripts/check_environment.py --step vector   # LanceDB/임베딩 관련
uv run python scripts/check_environment.py --step neo4j    # Neo4j 관련
```

### 검증 항목

| 범위 | 검증 항목 |
|------|----------|
| 공통 | Python 3.11+, backend/.env 존재, 필수 환경변수, 디스크 공간 |
| db | PostgreSQL 컨테이너, Alembic 마이그레이션, 핵심 JSON 파일, MeCab 시스템 패키지 + Python 바인딩 |
| vector | 임베딩 모델 캐시, PyTorch 설치 |
| neo4j | Neo4j 컨테이너 |

### 출력 예시

```
[공통]
  ✅ Python 3.11.14
  ✅ backend/.env 존재
  ❌ DATABASE_URL 미설정 → backend/.env에 DATABASE_URL 설정 필요
  ⚠️  디스크 여유: 8.2GB (50GB 이상 권장) → 불필요한 파일 정리

총: 9/12 통과, 2 실패, 1 경고
```
