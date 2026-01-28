# Backend - 법률 서비스 플랫폼

법률 서비스 플랫폼의 백엔드 API 서버입니다.

## 목차

- [요구사항](#요구사항)
- [설치](#설치)
  - [1. PyTorch 설치 (GPU 환경별)](#1-pytorch-설치-gpu-환경별)
  - [2. 의존성 설치](#2-의존성-설치)
- [LanceDB 벡터 데이터베이스](#lancedb-벡터-데이터베이스)
  - [임베딩 생성](#임베딩-생성)
  - [Colab에서 실행](#colab에서-실행)
- [개발 서버 실행](#개발-서버-실행)
- [테스트](#테스트)

---

## 요구사항

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (패키지 관리자)
- CUDA 11.8+ (GPU 사용 시) 또는 Mac M1/M2/M3

### CUDA 버전 확인

```bash
# CUDA Toolkit 버전 확인
nvcc --version

# NVIDIA 드라이버 확인
nvidia-smi
```

---

## 설치

### 1. PyTorch 설치 (GPU 환경별)

> ⚠️ **중요**: PyTorch를 먼저 설치한 후 `uv sync`를 실행해야 합니다.
> sentence-transformers가 torch를 의존성으로 가지고 있어, 순서가 중요합니다.

#### NVIDIA GPU (CUDA)

```bash
cd backend

# CUDA 버전 확인
nvcc --version
```

**CUDA 12.4 (최신)**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

**CUDA 12.1**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**CUDA 11.8**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

#### Mac (Apple Silicon - M1/M2/M3)

```bash
# MPS (Metal Performance Shaders) 자동 지원
pip install torch
```

#### CPU only

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

#### 설치 확인

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'MPS: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')"
```

예상 출력:
```
PyTorch: 2.7.1+cu118
CUDA: True
MPS: False
```

### 2. 의존성 설치

```bash
# uv로 의존성 설치
uv sync

# 개발 의존성 포함 (pytest, ruff, mypy)
uv sync --dev
```

---

## LanceDB 벡터 데이터베이스

법령/판례 데이터를 KURE-v1 모델로 임베딩하여 LanceDB에 저장합니다.

### 환경 변수 설정

`.env` 파일에 추가:

```bash
VECTOR_DB=lancedb
LANCEDB_URI=./data/lancedb
LANCEDB_TABLE_NAME=legal_chunks
```

### Hugging Face 로그인

KURE 모델 접근을 위해 Hugging Face 로그인이 필요합니다:

```bash
huggingface-cli login
```

### 임베딩 생성

#### 디바이스 정보 확인

```bash
uv run python scripts/create_lancedb_embeddings.py --device-info
```

출력 예시:
```
============================================================
Device Information
============================================================
  Device: CUDA
  Name: NVIDIA GeForce RTX 3060
  Memory: 6.0 GB
  Type: Laptop/Mobile

Recommended Settings:
  batch_size: 50
  num_workers: 2
============================================================
```

#### 법령 임베딩

```bash
uv run python scripts/create_lancedb_embeddings.py \
  --type law \
  --source ../data/law_cleaned.json
```

#### 판례 임베딩

```bash
uv run python scripts/create_lancedb_embeddings.py \
  --type precedent \
  --source ../data/precedents.json
```

#### 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--type` | 문서 유형 (`law`, `precedent`) | 필수 |
| `--source` | JSON 파일 경로 | 필수 |
| `--batch-size` | 배치 크기 | 자동 (디바이스에 따라) |
| `--num-workers` | DataLoader 워커 수 | 자동 |
| `--reset` | 기존 데이터 삭제 후 재생성 | False |
| `--stats` | 통계만 출력 | False |
| `--device-info` | 디바이스 정보 출력 | False |

#### 통계 확인

```bash
uv run python scripts/create_lancedb_embeddings.py --stats
```

출력 예시:
```
============================================================
LanceDB Statistics (v2 Schema)
============================================================
Total chunks: 125,432

By data_type:
  - 법령: 45,123
  - 판례: 80,309
```

#### 전체 재생성

```bash
# 법령 재생성
uv run python scripts/create_lancedb_embeddings.py \
  --type law \
  --source ../data/law_cleaned.json \
  --reset

# 판례 재생성
uv run python scripts/create_lancedb_embeddings.py \
  --type precedent \
  --source ../data/precedents.json \
  --reset
```

### Colab에서 실행

Google Colab에서 실행할 때는 `scripts/colab_lancedb_embeddings.py`를 사용합니다.

#### 1. 패키지 설치

```python
!pip install lancedb sentence-transformers pyarrow ijson psutil -q
```

#### 2. 스크립트 업로드 및 실행

```python
# 스크립트 파일과 데이터 파일 업로드 후

# 디바이스 정보 확인
print_device_info()

# 법령 임베딩 (자동 설정)
run_law_embedding("law_cleaned.json", reset=True)

# 판례 임베딩 (자동 설정)
run_precedent_embedding("precedents.json", reset=True)

# 통계 확인
show_stats()

# 메모리 정리
clear_model_cache()
```

#### 3. 수동 설정 (필요시)

```python
# 배치 크기 조정 (GPU 메모리 부족 시)
run_precedent_embedding("precedents.json", batch_size=30, auto_config=False)
```

#### 환경별 자동 설정값

| 환경 | batch_size | num_workers | gc_interval |
|------|------------|-------------|-------------|
| CUDA 16GB+ (5060 Ti 등) | 100 | 4 | 20 |
| CUDA 8GB+ (3070 등) | 70 | 2 | 15 |
| CUDA 6GB (3060 등) | 50 | 2 | 10 |
| Mac M3 16GB (MPS) | 50 | 0 | 10 |
| Mac M1/M2 8GB | 30 | 0 | 5 |
| Colab T4 | 100 | 0 | 10 |
| CPU only | 20 | 2 | 5 |

---

## 개발 서버 실행

```bash
uv run uvicorn app.main:app --reload
```

서버가 `http://localhost:8000`에서 실행됩니다.

- API 문서: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## 테스트

```bash
# 전체 테스트
uv run pytest

# 특정 테스트
uv run pytest tests/test_file.py::test_name

# 커버리지
uv run pytest --cov=app
```

---

## 린트 및 타입 체크

```bash
# Ruff 린트
uv run ruff check .

# 자동 수정
uv run ruff check . --fix

# MyPy 타입 체크
uv run mypy .
```

---

## 문제 해결

### PyTorch CUDA 버전 불일치

```bash
# 현재 설치된 torch 버전 확인
python -c "import torch; print(torch.__version__)"

# CUDA 사용 가능 여부 확인
python -c "import torch; print(torch.cuda.is_available())"
```

CUDA가 `False`로 나오면:
1. NVIDIA 드라이버 업데이트
2. 올바른 CUDA 버전의 PyTorch 재설치

### 메모리 부족 (OOM)

```bash
# batch_size 줄이기
uv run python scripts/create_lancedb_embeddings.py \
  --type precedent \
  --source ../data/precedents.json \
  --batch-size 30
```

### Hugging Face 모델 다운로드 실패

```bash
# 로그인 확인
huggingface-cli whoami

# 재로그인
huggingface-cli login
```

---

## 관련 문서

- [벡터 DB 설계 문서](../docs/vectordb_design.md)
- [프로젝트 가이드](../CLAUDE.md)
