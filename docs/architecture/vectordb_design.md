# 법률 벡터 DB 구축 설계 문서

## 데이터 소스

### JSON 원본
| 데이터 | 파일 경로 | 건수 |
|--------|-----------|------|
| 법령 | `data/law_cleaned.json` | 5,841건 |
| 판례 | `data/precedents_cleaned.json` | 65,107건 |

### PostgreSQL 테이블 (LanceDB 전용)
| 테이블 | 설명 | 비고 |
|--------|------|------|
| `law_documents` | 법령 원본 데이터 | LanceDB 검색 후 원본 조회용 |
| `precedent_documents` | 판례 원본 데이터 | ruling, claim, reasoning 등 전체 텍스트 저장 |
| `legal_documents` | 기존 테이블 | ChromaDB 호환 (유지) |

---

## 1. 스키마 설계 (v2 - 단일 테이블 + NULL)

### 설계 원칙
- 모든 필드를 개별 컬럼으로 정의 (JSON metadata 사용 안 함)
- 해당하지 않는 필드는 NULL
- `data_type` 컬럼으로 문서 유형 구분 ("법령" | "판례")

### PyArrow 스키마
```python
LEGAL_CHUNKS_SCHEMA = pa.schema([
    # ========== 공통 필드 (10개) ==========
    pa.field("id", pa.utf8()),              # 청크 고유 ID (예: "010719_0")
    pa.field("source_id", pa.utf8()),       # 원본 문서 ID (예: "010719")
    pa.field("data_type", pa.utf8()),       # "법령" | "판례"
    pa.field("title", pa.utf8()),           # 제목 (법령명 / 사건명)
    pa.field("content", pa.utf8()),         # 청크 텍스트 (prefix 포함)
    pa.field("vector", pa.list_(pa.float32(), 1024)),  # 임베딩 벡터 (KURE 1024차원)
    pa.field("date", pa.utf8()),            # 날짜 (법령: 시행일, 판례: 선고일)
    pa.field("source_name", pa.utf8()),     # 출처 (법령: 소관부처, 판례: 법원명)
    pa.field("chunk_index", pa.int32()),    # 청크 인덱스
    pa.field("total_chunks", pa.int32()),   # 해당 문서의 총 청크 수

    # ========== 법령 전용 (판례는 NULL) ==========
    pa.field("promulgation_date", pa.utf8()),   # 공포일자 (예: "20230808")
    pa.field("promulgation_no", pa.utf8()),     # 공포번호 (예: "19592")
    pa.field("law_type", pa.utf8()),            # 법령 유형 (법률/시행령/시행규칙)
    pa.field("article_no", pa.utf8()),          # 조문 번호 (예: "제750조")

    # ========== 판례 전용 (법령은 NULL) ==========
    # NOTE: ruling, claim, reasoning은 PostgreSQL precedent_documents 테이블에서 조회
    pa.field("case_number", pa.utf8()),         # 사건번호 (예: "84나3990")
    pa.field("case_type", pa.utf8()),           # 사건 유형 (민사/형사/행정)
    pa.field("judgment_type", pa.utf8()),       # 판결 법원부 (예: "제11민사부판결")
    pa.field("judgment_status", pa.utf8()),     # 판결 상태 (확정/미확정)
    pa.field("reference_provisions", pa.utf8()),# 참조 조문 (예: "민법 제750조, 제756조")
    pa.field("reference_cases", pa.utf8()),     # 참조 판례
])
# 총 20개 컬럼 (기존 23개에서 ruling, claim, reasoning 3개 제거)
```

### 컬럼 그룹 (총 20개)
```python
COMMON_COLUMNS = [  # 10개
    "id", "source_id", "data_type", "title", "content",
    "vector", "date", "source_name", "chunk_index", "total_chunks"
]

LAW_COLUMNS = [  # 4개
    "promulgation_date", "promulgation_no", "law_type", "article_no"
]

PRECEDENT_COLUMNS = [  # 6개 (ruling, claim, reasoning 제거)
    "case_number", "case_type", "judgment_type",
    "judgment_status", "reference_provisions", "reference_cases"
]
```

---

## 2. 필드 매핑

| 스키마 필드 | 법령 원본 필드 | 판례 원본 필드 |
|-------------|----------------|----------------|
| `id` | `{law_id}_{chunk_idx}` | `{판례정보일련번호}_{chunk_idx}` |
| `source_id` | `law_id` | `판례정보일련번호` |
| `data_type` | "법령" | "판례" |
| `title` | `law_name` | `사건명` |
| `content` | 조문번호 + 조문내용 | prefix + 판시사항 + 판결요지 |
| `date` | `enforcement_date` | `선고일자` |
| `source_name` | `ministry` | `법원명` |
| `chunk_index` | 청크 순번 | 청크 순번 |
| `total_chunks` | 해당 문서 총 청크 수 | 해당 문서 총 청크 수 |
| `promulgation_date` | `공포일자` | NULL |
| `promulgation_no` | `공포번호` | NULL |
| `law_type` | `법령유형` | NULL |
| `article_no` | `조문번호` | NULL |
| `case_number` | NULL | `사건번호` |
| `case_type` | NULL | `사건종류명` |
| `judgment_type` | NULL | `판결유형` |
| `judgment_status` | NULL | `판결상태` |
| `reference_provisions` | NULL | `참조조문` |
| `reference_cases` | NULL | `참조판례` |

> **NOTE**: `ruling`, `claim`, `reasoning`은 LanceDB에 저장하지 않음 (메모리 효율화).
> 검색 후 PostgreSQL `precedent_documents` 테이블에서 조회하여 접근.

---

## 2.5. 검색 흐름 (PostgreSQL 연동)

```
┌─────────────────┐
│ 사용자 쿼리      │
└────────┬────────┘
         ▼
┌─────────────────┐
│ LanceDB 벡터 검색 │ ← 임베딩 벡터 유사도 기반
└────────┬────────┘
         ▼
┌─────────────────┐
│ source_id 추출   │ ← 법령: law_id, 판례: serial_number
└────────┬────────┘
         ▼
┌─────────────────────────────────────────────────────┐
│ PostgreSQL 원본 조회                                  │
│   - 법령: law_documents (law_id로 조회)              │
│   - 판례: precedent_documents (serial_number로 조회) │
│     → ruling, claim, full_reason 등 전체 텍스트 접근 │
└─────────────────────────────────────────────────────┘
```

### 코드 예시
```python
# 1. LanceDB 벡터 검색
results = store.search(query_embedding, n_results=10, where={"data_type": "판례"})

# 2. source_id 추출
source_ids = [meta["source_id"] for meta in results.metadatas[0]]

# 3. PostgreSQL에서 원본 조회
from app.models.precedent_document import PrecedentDocument
from sqlalchemy import select

async with async_session_factory() as session:
    query = select(PrecedentDocument).where(
        PrecedentDocument.serial_number.in_(source_ids)
    )
    result = await session.execute(query)
    originals = {doc.serial_number: doc for doc in result.scalars()}

# 4. ruling, claim, reasoning 접근
for source_id in source_ids:
    doc = originals.get(source_id)
    if doc:
        print(f"주문: {doc.ruling}")
        print(f"청구취지: {doc.claim}")
        print(f"이유: {doc.full_reason}")
```

---

## 3. 청킹 전략

### 법령 청킹
- **split_mode**: 조문(\n\n) 분리 → 토큰 초과 시 항(①②③) 단위로 추가 분리
- **max_tokens**: 800
- **min_tokens**: 100
- **prefix_mode**: 제3조 ① 형태

**동작 방식:**
1. 조문 단위(`\n\n`)로 1차 분리
2. 조문이 800 토큰 초과 시 → 항(①②③) 단위로 2차 분리
3. 100 토큰 미만 청크 → 인접 청크와 병합
4. prefix 형식: `제N조 ①` (법령명 제외, 조문번호만)

### 판례 청킹
- **최대 길이**: 1250자
- **오버랩**: 10% (125자)

**동작 방식:**
1. **판시사항 + 판결요지만 사용** (이유, 주문 등 제외)
2. 1250자 초과 시 오버랩 청킹 적용
3. prefix 형식: `[법원명 사건번호]`

---

## 4. ID 구조 및 활용

### ID 형식
```
{source_id}_{chunk_index}
```

### 활용 방법
```python
# ID 파싱
id = "010719_2"
source_id = id.rsplit("_", 1)[0]  # "010719"
chunk_index = int(id.rsplit("_", 1)[1])  # 2

# 같은 문서의 모든 청크 조회
all_chunks = store.get_by_source_id(source_id)

# 원본 순서대로 정렬 (자동 정렬됨)
sorted_chunks = all_chunks["documents"]

# 전체 문서 복원
full_content = "\n".join(sorted_chunks)
```

---

## 5. 디렉토리 구조

```
backend/
├── app/
│   └── common/
│       └── vectorstore/
│           ├── __init__.py          # 팩토리 및 export
│           ├── base.py              # VectorStoreBase 인터페이스
│           ├── schema_v2.py         # LanceDB 스키마 v2 (단일 테이블 + NULL)
│           ├── lancedb.py           # LanceDBStore 구현체
│           ├── chroma.py            # ChromaDB 구현체 (기존)
│           └── qdrant.py            # Qdrant 구현체 (기존)
├── scripts/
│   └── create_lancedb_embeddings.py # LanceDB 임베딩 생성 스크립트
└── data/
    └── lancedb/                     # LanceDB 데이터 저장소
```

---

## 6. 구현 상태

### 완료
- [x] 스키마 정의 (`schema_v2.py`) - 단일 테이블 + NULL 방식 (20개 컬럼)
- [x] LanceDBStore 구현 (`lancedb.py`) - v2 스키마 기반
- [x] PostgreSQL 모델 (`law_document.py`, `precedent_document.py`)
- [x] Alembic 마이그레이션 (`003_add_lancedb_tables.py`)
- [x] 데이터 로드 스크립트 (`load_lancedb_data.py`) - JSON → PostgreSQL
- [x] 임베딩 스크립트 (`create_lancedb_embeddings.py`) - PostgreSQL → LanceDB
- [x] ruling, claim, reasoning 제거 (메모리 효율화)
- [x] **판례 데이터 전체 임베딩** (65,107건 → 134,846 청크)
- [x] **법령 데이터 전체 임베딩** (5,841건 → 118,922 청크)
- [x] **통합 임베딩 프로세서 클래스** (`StreamingEmbeddingProcessor`)
- [x] **청킹 무한루프 버그 수정** (2026-01-29)

### 진행 예정
- [ ] 검색 API 연동 (LanceDB → 프론트엔드)
- [ ] 하이브리드 검색 (벡터 + 키워드) 구현 로직 완성
- [ ] 검색 결과 캐싱

---

## 6.5. 하이브리드 검색 전략 (Vector + FTS)

법률 도메인의 특성상 의미 검색(Vector)뿐만 아니라 정확한 키워드 매칭(FTS)이 필수적입니다.

### FTS (Full-Text Search) 설정
- **대상 컬럼**: `content`
- **엔진**: Tantivy (LanceDB 내장)
- **생성 시점**: 대량 임베딩 작업 완료 후 `table.create_fts_index("content")` 호출

### 검색 흐름 (Hybrid)
1. **Vector Search**: 질문의 의도와 의미가 유사한 문서 추출
2. **FTS Search**: 특정 조문번호, 판례번호, 고유 법률 용어 매칭 문서 추출
3. **Reciprocal Rank Fusion (RRF)**: 두 검색 결과를 결합하여 최종 순위 산정

---

## 7. 실행 명령

```bash
# 의존성 설치
uv sync

# Hugging Face 로그인 (KURE 모델 접근용)
huggingface-cli login

# 환경변수 설정 (.env)
VECTOR_DB=lancedb
LANCEDB_URI=./data/lancedb
LANCEDB_TABLE_NAME=legal_chunks

# ========== Step 1: PostgreSQL 마이그레이션 ==========
cd backend
alembic upgrade head

# ========== Step 2: JSON → PostgreSQL 데이터 로드 ==========
# 법령 데이터 로드
uv run python scripts/load_lancedb_data.py --type law

# 판례 데이터 로드
uv run python scripts/load_lancedb_data.py --type precedent

# 전체 로드 (법령 + 판례)
uv run python scripts/load_lancedb_data.py --type all

# 기존 데이터 삭제 후 재로드
uv run python scripts/load_lancedb_data.py --type all --reset

# 통계 확인
uv run python scripts/load_lancedb_data.py --stats

# ========== Step 3: PostgreSQL → LanceDB 임베딩 생성 ==========
# 판례 임베딩 생성 (precedent_documents 테이블에서)
uv run python scripts/create_lancedb_embeddings.py --type precedent

# 법령 임베딩 생성 (law_documents 테이블에서)
uv run python scripts/create_lancedb_embeddings.py --type law

# 전체 (판례 + 법령)
uv run python scripts/create_lancedb_embeddings.py --type all

# 전체 재생성 (기존 데이터 삭제)
uv run python scripts/create_lancedb_embeddings.py --type all --reset

# 통계 확인
uv run python scripts/create_lancedb_embeddings.py --stats

# ========== 옵션 ==========
# 판례 옵션
--batch-size 100      # 배치 크기 (GPU 메모리에 따라 조정)
--chunk-size 1250     # 판례 청크 크기 (기본값)
--chunk-overlap 125   # 판례 오버랩 (기본값 10%)

# 법령 옵션
--max-tokens 800      # 법령 청크 최대 토큰 (기본값)
--min-tokens 100      # 법령 청크 최소 토큰 (기본값)
```

---

## 8. 검증 계획

### 청킹 검증
- [ ] 법령: 800 토큰 이하, 100 토큰 이상인지 확인
- [ ] 판례: 1250자 이하, 오버랩 정상 적용 확인
- [ ] prefix가 올바르게 포함되었는지 확인

### 스키마 검증
- [ ] 필수 필드 누락 없는지 확인
- [ ] data_type별 NULL 필드 정상 여부 확인
- [ ] id 형식 일관성 확인

### 검색 검증
- [ ] 동일 쿼리로 법령/판례 모두 검색되는지 확인
- [ ] data_type 필터링 정상 작동 확인
- [ ] source_id로 원본 문서 복원 가능 여부 확인

---

## 9. 설계 결정 요약

| 항목 | 결정 | 이유 |
|------|------|------|
| DB | LanceDB (단일 테이블) | 통합 검색 용이, 디스크 기반으로 메모리 효율적 |
| 메타데이터 | **개별 컬럼 + NULL (20개)** | JSON보다 필터링 5.8x 빠름 |
| 임베딩 모델 | KURE-v1 (1024차원) | 한국어 최적화, 최고 성능 |
| 필터 컬럼 | date, source_name 등 | 자주 사용하는 필터 최적화 |
| 법령 청킹 | hybrid (800/100 토큰) | Test B 조건 적용, 조문-항 구조 보존 |
| 판례 청킹 | 1250자 + 10% 오버랩 | 임베딩 모델 제한 고려 |
| 판례 임베딩 대상 | 판시사항 + 판결요지만 | 핵심 법리 중심 검색 |
| ID 구조 | source_id_chunkIdx | 원본 추적 + 청크 식별 |
| ruling/claim/reasoning | **PostgreSQL 저장** | 메모리 효율화, 검색 후 원본 조회 |
| 데이터 흐름 | JSON → PostgreSQL → LanceDB | 호환성 + 확장성 |

---

## 10. RunPod/Colab 임베딩 (GPU 환경)

대용량 데이터 임베딩을 위한 GPU 환경 스크립트입니다.

### 스크립트 위치
```
backend/scripts/
├── runpod_lancedb_embeddings.py  # RunPod GPU용 (분할 처리 포함)
├── runpod_split_embeddings.py    # 분할 전용 (간소화 버전)
└── colab_lancedb_embeddings.py   # Google Colab용
```

### 분할 처리 방식 (권장)

대용량 판례 데이터(6만건+)는 메모리 문제로 분할 처리가 필요합니다.

**두 가지 분할의 차이:**
| 분할 유형 | 단위 | 용도 |
|-----------|------|------|
| `split_precedents(chunk_size=5000)` | 판례 건수 | 파일 분할 (메모리 절약) |
| `PRECEDENT_CHUNK_SIZE=1250` | 글자 수 | 임베딩 텍스트 분할 |

### 사용법 (RunPod Jupyter)

```python
# 1. 패키지 설치
!pip install lancedb sentence-transformers pyarrow ijson psutil tqdm gdown -q

# 2. 데이터 다운로드 (Google Drive에서)
!gdown --id YOUR_FILE_ID -O precedents_cleaned.json
!gdown --id YOUR_FILE_ID -O law_cleaned.json

# 3. 스크립트 실행
exec(open('runpod_lancedb_embeddings.py').read())

# 4. 디바이스 확인
print_device_info()

# 5. 판례 분할 처리 (권장)
split_precedents('precedents_cleaned.json', chunk_size=5000)
run_all_precedent_parts('precedents_part_*.json', batch_size=64)

# 6. 법령 분할 처리
split_laws('law_cleaned.json', chunk_size=2000)
run_all_law_parts('laws_part_*.json', batch_size=64)

# 7. 결과 확인
show_stats()

# 8. 결과 다운로드
!zip -r lancedb_data.zip ./lancedb_data
```

### 함수 목록

| 함수 | 설명 |
|------|------|
| `print_device_info()` | GPU/디바이스 정보 출력 |
| `print_memory_status()` | 메모리 사용량 확인 |
| `split_precedents(path, chunk_size)` | 판례 JSON 파일 분할 |
| `split_laws(path, chunk_size)` | 법령 JSON 파일 분할 |
| `run_precedent_embedding_part(path, reset, batch_size)` | 분할된 판례 파일 처리 |
| `run_all_precedent_parts(pattern, batch_size)` | 모든 분할 판례 파일 처리 |
| `run_law_embedding_part(path, reset, batch_size)` | 분할된 법령 파일 처리 |
| `run_all_law_parts(pattern, batch_size)` | 모든 분할 법령 파일 처리 |
| `show_stats()` | LanceDB 통계 출력 |
| `clear_model_cache()` | 모델 메모리 정리 |

### 권장 설정

| GPU | batch_size | chunk_size (파일) |
|-----|------------|-------------------|
| RTX 3090 (24GB) | 64~128 | 5000 |
| RTX 5060 Ti (16GB) | 100 | 5000 |
| RTX 4080 (16GB) | 64 | 5000 |
| RTX 3070 (8GB) | 32 | 3000 |
| T4 (16GB) | 64 | 5000 |

### 자동 감지 설정 (get_optimal_config)

스크립트가 GPU VRAM을 감지하여 자동으로 최적 설정을 적용합니다:

| GPU VRAM | batch_size | num_workers | gc_interval |
|----------|------------|-------------|-------------|
| 20GB+ | 128 | 4 | 25 |
| 14GB+ | 100 | 4 | 20 |
| 8GB+ | 70 | 2 | 15 |
| 8GB 미만 | 50 | 2 | 10 |

---

## 10.5. 통합 임베딩 프로세서 (v2)

### 클래스 구조

2026-01-29 리팩토링으로 법령/판례 임베딩 로직이 통합되었습니다.

```python
# 추상 베이스 클래스
class StreamingEmbeddingProcessor(ABC):
    """스트리밍 방식 임베딩 프로세서"""

    def __init__(self, data_type: str):
        self.data_type = data_type  # "법령" | "판례"
        self.device_info = get_device_info()
        self.optimal_config = get_optimal_config(self.device_info)
        self.store = LanceDBStore()

    def load_streaming(self, source_path: str) -> tuple:
        """개수 세기 스킵, 즉시 시작"""

    def run(self, source_path: str, reset: bool, batch_size: int) -> dict:
        """통합 실행 로직"""

    # 추상 메서드 (서브클래스에서 구현)
    @abstractmethod
    def get_chunk_config(self) -> Any: ...
    @abstractmethod
    def extract_source_id(self, item: dict, idx: int) -> str: ...
    @abstractmethod
    def extract_text_for_embedding(self, item: dict) -> str: ...
    @abstractmethod
    def chunk_text(self, text: str, config: Any) -> List[tuple]: ...
    @abstractmethod
    def extract_metadata(self, item: dict) -> dict: ...
    @abstractmethod
    def create_batch_data(self) -> dict: ...
    @abstractmethod
    def add_to_batch(self, batch_data, source_id, chunk_idx, ...): ...
    @abstractmethod
    def save_batch(self, batch_data, embeddings) -> int: ...

# 구현 클래스
class LawEmbeddingProcessor(StreamingEmbeddingProcessor):
    """법령 임베딩 프로세서"""

class PrecedentEmbeddingProcessor(StreamingEmbeddingProcessor):
    """판례 임베딩 프로세서"""
```

### 통일된 동작

- **개수 세기 스킵**: 대용량 파일에서 즉시 시작
- **tqdm**: 속도(it/s)만 표시 (진행률 % 미표시)
- **스트리밍**: ijson으로 메모리 효율적 처리
- **GC**: 매 배치마다 가비지 컬렉션
- **압축**: 50배치마다 LanceDB compact

### 사용 예시

```python
# 새 클래스 직접 사용
processor = PrecedentEmbeddingProcessor()
stats = processor.run(
    source_path="precedents.json",
    reset=True,
    batch_size=100
)

# 기존 함수 (래퍼) 사용 - 하위 호환
stats = run_precedent_embedding("precedents.json", reset=True, batch_size=100)
stats = run_law_embedding("laws.json", reset=True, batch_size=100)
```

---

## 11. API 사용 예시

```python
from app.common.vectorstore import get_vector_store

# 환경변수 VECTOR_DB=lancedb 설정 필요
store = get_vector_store()

# 법령 문서 추가
store.add_law_documents(
    source_ids=["010719"],
    chunk_indices=[0],
    embeddings=[[0.1] * 1024],
    titles=["민법"],
    contents=["[법령] 민법 제750조: 불법행위 책임..."],
    enforcement_dates=["2023-08-08"],
    departments=["법무부"],
    law_types=["법률"],
)

# 판례 문서 추가
store.add_precedent_documents(
    source_ids=["84나3990"],
    chunk_indices=[0],
    embeddings=[[0.2] * 1024],
    titles=["손해배상청구사건"],
    contents=["[판례] 수련의 의료사고 책임..."],
    decision_dates=["1986-01-15"],
    court_names=["서울고법"],
    case_numbers=["84나3990"],
    case_types=["민사"],
)

# 벡터 검색
results = store.search(query_embedding, n_results=10)

# 유형별 검색
results = store.search_by_type(query_embedding, "판례", n_results=10)

# 필터 검색
results = store.search(
    query_embedding,
    n_results=10,
    where={"data_type": "판례", "case_type": "민사"}
)

# source_id로 전체 청크 조회
chunks = store.get_by_source_id("010719")

# 통계
total = store.count()
law_count = store.count_by_type("법령")
precedent_count = store.count_by_type("판례")
```

---

## 12. 임베딩 캐싱 (Embedding Pipeline)

동일 텍스트 재임베딩 방지를 위한 해시 기반 디스크 캐시.

### 사용법

```python
from scripts.runpod_lancedb_embeddings import EmbeddingCache, create_embeddings

# 캐시 초기화
cache = EmbeddingCache("./embedding_cache")

# 캐시 조회 후 없으면 계산
embedding = cache.get_or_compute("법률 텍스트", create_embeddings)

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
├── a1/                    # 해시 앞 2글자로 분산
│   ├── a1b2c3d4...json
│   └── a1e5f6g7...json
├── b2/
│   └── b2c3d4e5...json
└── ...
```

---

## 13. 임베딩 품질 검증

유사/비유사 문서 쌍으로 임베딩 품질 평가.

### 사용법

```python
from scripts.runpod_lancedb_embeddings import EmbeddingQualityChecker

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
| > 0.2 | GOOD | 유사/비유사 명확히 구분 |
| 0.1 ~ 0.2 | FAIR | 구분 가능하나 개선 필요 |
| < 0.1 | POOR | 구분 어려움, 모델/청킹 점검 필요 |

---

## 14. PyTorch 최적화 패턴

스크립트에 적용된 최적화 패턴 요약.

### 적용된 패턴

| 패턴 | 함수/클래스 | 설명 |
|------|------------|------|
| 디바이스 자동 선택 | `get_device_info()` | CUDA > MPS > CPU 우선순위 |
| 멀티 GPU 지원 | `get_optimal_cuda_device()` | VRAM 최대 GPU 선택 |
| 메모리 정리 | `clear_memory()` | GC + CUDA cache 통합 |
| 재현성 | `set_seed()` | 랜덤 시드 고정 |
| VRAM 기반 설정 | `get_optimal_config()` | 배치 크기 자동 조정 |

### 유틸리티 함수

```python
from scripts.runpod_lancedb_embeddings import (
    clear_memory,       # GC + CUDA cache 정리
    set_seed,           # 랜덤 시드 고정
    print_memory_status,# 메모리 상태 출력
    get_optimal_cuda_device,  # 멀티 GPU 시 최적 디바이스
)

# 재현성 확보
set_seed(42, deterministic=False)

# 메모리 정리
clear_memory()

# 현재 메모리 상태
print_memory_status()
# [Memory] RAM: 8.2GB / 32.0GB (25.6%)
# [Memory] GPU: 2.1GB allocated, 4.0GB reserved, 3.5GB max
```

---

## 15. 구현 완료 현황

| 항목 | 상태 | 비고 |
|------|------|------|
| 단일 테이블 스키마 | ✅ 완료 | 20개 컬럼 |
| JSON → PostgreSQL 로드 | ✅ 완료 | load_lancedb_data.py |
| PostgreSQL → LanceDB 임베딩 | ✅ 완료 | create_lancedb_embeddings.py |
| RunPod 스크립트 | ✅ 완료 | runpod_lancedb_embeddings.py |
| 분할 처리 (대용량) | ✅ 완료 | split_precedents, split_laws |
| 통합 프로세서 클래스 | ✅ 완료 | StreamingEmbeddingProcessor |
| 임베딩 캐싱 | ✅ 완료 | EmbeddingCache |
| 품질 검증 | ✅ 완료 | EmbeddingQualityChecker |
| PyTorch 최적화 | ✅ 완료 | clear_memory, set_seed 등 |
| 검색 API 통합 | 🔄 진행중 | VectorStoreBase 인터페이스 |

---

## 16. TODO / Known Issues

### 판례 메타데이터 누락 필드 (해결됨)

2026-02-05 업데이트: `runpod_lancedb_embeddings.py` 및 `create_lancedb_embeddings.py` 수정으로 해결되었습니다.

| 필드 | JSON 원본 | 상태 | 비고 |
|------|-----------|------|------|
| `judgment_type` | 판결유형 | ✅ 완료 | `extract_metadata`에 추가됨 |
| `judgment_status` | 판결상태 | ✅ 완료 | `extract_metadata`에 추가됨 |

### 판례 JSON 필드 중 미사용 필드

| 필드 | 용도 | 저장 여부 |
|------|------|----------|
| 선고 | "선고" 고정값 | 불필요 |
| 법원종류코드 | 코드값 | 법원명으로 대체 |
| 사건종류코드 | 코드값 | 사건종류명으로 대체 |
| 판례내용 | 전문 텍스트 | PostgreSQL 저장 권장 |
| 주문 | 판결 주문 | PostgreSQL 저장 권장 |
| 청구취지 | 청구 내용 | PostgreSQL 저장 권장 |
| 이유 | 판결 이유 | PostgreSQL 저장 권장 |
