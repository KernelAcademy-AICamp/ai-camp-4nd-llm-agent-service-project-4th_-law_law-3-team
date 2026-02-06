# CLAUDE.md - 벡터 DB 임베딩 스크립트

이 폴더는 LanceDB 벡터 데이터베이스 생성 및 관리 스크립트를 포함합니다.

## 핵심 스크립트

| 스크립트 | 용도 |
|----------|------|
| `runpod_lancedb_embeddings.py` | 메인 임베딩 스크립트 (RunPod/클라우드 GPU) |
| `local_lancedb_embeddings.py` | 로컬 임베딩 스크립트 (멀티 하드웨어 지원) |
| `colab_lancedb_embeddings.py` | Google Colab 전용 |
| `test_precedent_embedding.py` | 임베딩 테스트 |

### 공통 모듈 (`embedding_common/`)

| 모듈 | 설명 |
|------|------|
| `device.py` | GPU/CPU/MPS 디바이스 감지, DeviceInfo |
| `config.py` | 하드웨어 프로필, 배치 크기 최적 설정 |
| `model.py` | 임베딩 모델 로딩 (KURE-v1) |
| `store.py` | LanceDB 테이블 생성/연결 |
| `chunking.py` | 텍스트 청킹 (법령/판례) |
| `schema.py` | 스키마 v2 re-export + 검증 유틸 |
| `cache.py` | MD5 기반 임베딩 캐시 |
| `temperature.py` | GPU 온도 모니터링 (nvidia-smi) |
| `memory.py` | GPU/시스템 메모리 모니터링 |

### Jupyter Notebook (`../notebooks/`)

| 노트북 | 환경 | 설명 |
|--------|------|------|
| `runpod_lancedb_embeddings.ipynb` | RunPod (A100/H100) | 클라우드 GPU 임베딩 |
| `colab_lancedb_embeddings.ipynb` | Google Colab (T4) | Drive 저장, 분할 처리 |

## 빠른 시작

### 로컬 실행 (권장: `local_lancedb_embeddings.py`)

```bash
cd backend

# PyTorch 설치 (환경에 맞게)
uv pip install --reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128  # CUDA
# uv pip install --reinstall torch torchvision torchaudio  # CPU/MPS

# 전체 임베딩 (하드웨어 자동 감지)
uv run --no-sync python scripts/local_lancedb_embeddings.py --type all --reset

# 판례만
uv run --no-sync python scripts/local_lancedb_embeddings.py --type precedent

# 법령만
uv run --no-sync python scripts/local_lancedb_embeddings.py --type law

# 프로필 수동 지정
uv run --no-sync python scripts/local_lancedb_embeddings.py --type all --profile laptop  # 발열 보호
uv run --no-sync python scripts/local_lancedb_embeddings.py --type all --profile mac     # MPS 백엔드

# 통계 / 검증
uv run --no-sync python scripts/local_lancedb_embeddings.py --stats
uv run --no-sync python scripts/local_lancedb_embeddings.py --verify
```

### 하드웨어 프로필

| 프로필 | batch_size | 온도 모니터링 | 비고 |
|--------|-----------|--------------|------|
| desktop | 128 | OFF | 5060Ti 등 데스크톱 GPU |
| laptop | 50 | ON (85°C) | 3060 Laptop 등 발열 보호 |
| mac | 50 | OFF | Apple Silicon MPS |
| cpu | 20 | OFF | CPU 전용 |

### 체크포인트/재개

중단 시 자동으로 체크포인트를 저장합니다. 재실행 시 이어서 처리합니다.

```bash
# 재개 (기본 동작)
uv run --no-sync python scripts/local_lancedb_embeddings.py --type precedent

# 처음부터 다시
uv run --no-sync python scripts/local_lancedb_embeddings.py --type precedent --no-resume
```

### RunPod/클라우드 실행

```bash
cd backend

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
- `notebooks/runpod_lancedb_embeddings.ipynb` - RunPod 노트북
- `notebooks/colab_lancedb_embeddings.ipynb` - Colab 노트북

---

## 법률 용어 PostgreSQL 로드 (load_legal_terms_data.py)

`data/law_data/lawterms_full.json` (36,797건)을 PostgreSQL `legal_terms` 테이블로 로드합니다.
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

1. `data/law_data/lawterms_full.json` 읽기
2. 각 레코드에서 `term_length`, `is_korean_only` 자동 계산
3. `ON CONFLICT (term) DO UPDATE`로 멱등성 보장
4. 1,000건 단위 배치 insert
5. 로드 후 통계 출력 (총 건수, 한글 전용 비율, 길이 분포)

### 환경 변수

```bash
# backend/.env
DATABASE_URL=postgresql://lawuser:lawpassword@localhost:5432/lawdb
USE_LEGAL_TERM_DICT=true  # 앱에서 사전 사용 활성화
```

### 데이터 현황

| 항목 | 수치 |
|------|------|
| 총 엔트리 | 36,797개 |
| 고유 용어 | 36,797개 |
| 한글 전용 (2-10자) | 33,430개 (MeCab 로드 대상) |
| 추후 확대 예정 | ~73,000개+ |

---

## 변호사 데이터 PostgreSQL 로드 (load_lawyers_data.py)

`data/lawyers_with_coords.json` (17,326건)을 PostgreSQL `lawyers` 테이블로 로드합니다.

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

1. `data/lawyers_with_coords.json` 읽기
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
별도 저장소 → all_lawyers.json → geocode_lawyers.py → data/lawyers_with_coords.json
```

### 사용법

```bash
cd backend

# 기본 실행 (all_lawyers.json → data/lawyers_with_coords.json)
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
| `--output` | 출력 파일 경로 | `data/lawyers_with_coords.json` |
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
| `data/lawyers_with_coords.json` | 좌표가 추가된 변호사 데이터 |
| `data/geocode_failed.json` | 지오코딩 실패 목록 |
