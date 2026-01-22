# 법률 데이터 DB 아키텍처

법률 서비스 플랫폼의 데이터 저장소 구조, 청킹 전략, RAG 검색 흐름에 대한 기술 문서입니다.

---

## 1. 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────┐
│                      데이터 저장 구조                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   PostgreSQL (정본 저장소)        ChromaDB (검색 인덱스)      │
│   ┌─────────────────────┐        ┌─────────────────────┐    │
│   │ legal_documents     │        │ 임베딩 벡터 (384D)   │    │
│   │ laws               │   ←──→  │ 메타데이터           │    │
│   │ legal_references   │        │ doc_id 포인터        │    │
│   └─────────────────────┘        └─────────────────────┘    │
│                                                             │
│   역할: 원문 + 메타데이터           역할: 유사도 검색         │
│   용도: 정확한 원문 제공            용도: RAG 검색            │
└─────────────────────────────────────────────────────────────┘
```

### 1.1 저장소 역할 분담

| 저장소 | 역할 | 저장 내용 |
|--------|------|----------|
| **PostgreSQL** | 정본(원문) 저장소 | 모든 법률 문서 원문 + 메타데이터 |
| **ChromaDB** | 검색용 인덱스 | 임베딩 벡터 + 메타데이터 (텍스트 미저장) |

### 1.2 용량 최적화

ChromaDB에는 **텍스트를 저장하지 않습니다**. 검색 시 `doc_id` + `chunk_start` + `chunk_end`를 이용해 PostgreSQL에서 원문을 조회합니다.

- 최적화 전: ~4.9GB
- 최적화 후: ~1.7GB (65% 절감)

---

## 2. PostgreSQL 스키마

### 2.1 legal_documents (법률 문서 통합 테이블)

판례, 결정례, 위원회 결정문 등 "사건/결정" 성격의 문서를 저장합니다.

```sql
CREATE TABLE legal_documents (
    id SERIAL PRIMARY KEY,

    -- 문서 식별
    doc_type VARCHAR(20) NOT NULL,      -- precedent, constitutional, administration, legislation, committee
    serial_number VARCHAR(100) NOT NULL, -- 원본 일련번호
    source VARCHAR(50) NOT NULL,         -- 데이터 출처 (precedents, ftc, nhrck 등)

    -- 사건 정보
    case_name TEXT,                      -- 사건명/안건명
    case_number TEXT,                    -- 사건번호
    decision_date DATE,                  -- 선고일/의결일

    -- 기관 정보
    court_name TEXT,                     -- 법원/위원회/기관명
    court_type TEXT,                     -- 법원종류/기관유형
    case_type TEXT,                      -- 사건종류 (민사/형사/헌마 등)

    -- 본문 (RAG 청킹 대상)
    summary TEXT,                        -- 판시사항/결정요지/주문
    reasoning TEXT,                      -- 판결요지/이유
    full_text TEXT,                      -- 전문
    claim TEXT,                          -- 청구취지

    -- 참조
    reference_articles TEXT,             -- 참조조문
    reference_cases TEXT,                -- 참조판례

    -- 메타
    raw_data JSONB NOT NULL,             -- 원본 JSON 전체
    created_at TIMESTAMP,
    updated_at TIMESTAMP,

    UNIQUE(doc_type, serial_number, source)
);
```

#### 문서 유형 (doc_type)

| 값 | 설명 | 데이터 출처 |
|----|------|------------|
| `precedent` | 일반 판례 | precedents_full*.json |
| `constitutional` | 헌법재판소 결정례 | constitutional_full.json |
| `administration` | 행정심판례 | administration_full.json |
| `legislation` | 법령해석례 | legislation_full.json |
| `committee` | 위원회 결정문 | ftc, nhrck, ppc 등 11개 |

#### 위원회 소스 매핑

| source 코드 | 위원회명 |
|------------|---------|
| ftc | 공정거래위원회 |
| nhrck | 국가인권위원회 |
| ppc | 개인정보보호위원회 |
| kcc | 방송통신위원회 |
| fsc | 금융위원회 |
| acrc | 국민권익위원회 |
| sfc | 증권선물위원회 |
| ecc | 선거관리위원회 |
| iaciac | 국제입양심사위원회 |
| eiac | 환경영향평가협의회 |
| oclt | 원산지조사위원회 |

### 2.2 laws (법령 테이블)

법률, 대통령령, 부령 등 법령 조문 정보를 저장합니다.

```sql
CREATE TABLE laws (
    id SERIAL PRIMARY KEY,

    -- 법령 식별
    law_id VARCHAR(20) NOT NULL UNIQUE,  -- 법령 ID
    law_name TEXT NOT NULL,              -- 법령명

    -- 법령 정보
    law_type VARCHAR(20),                -- 법률, 대통령령, 부령 등
    ministry TEXT,                       -- 소관부처
    promulgation_date DATE,              -- 공포일
    promulgation_no VARCHAR(20),         -- 공포번호
    enforcement_date DATE,               -- 시행일

    -- 본문
    content TEXT,                        -- 전체 조문 내용
    supplementary TEXT,                  -- 부칙

    -- 메타
    raw_data JSONB NOT NULL,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

### 2.3 legal_references (참조 데이터 테이블)

조약, 행정규칙, 법률용어 등 참조용 데이터를 저장합니다.

```sql
CREATE TABLE legal_references (
    id SERIAL PRIMARY KEY,

    -- 식별
    ref_type VARCHAR(20) NOT NULL,       -- treaty, admin_rule, law_term
    serial_number VARCHAR(100) NOT NULL,

    -- 공통 필드
    title TEXT,                          -- 명칭
    content TEXT,                        -- 내용/정의

    -- 기관/출처
    organization TEXT,                   -- 소관기관/체결국가
    category TEXT,                       -- 분류/종류

    -- 날짜
    effective_date DATE,                 -- 발효일/시행일

    -- 메타
    raw_data JSONB NOT NULL,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,

    UNIQUE(ref_type, serial_number)
);
```

---

## 3. ChromaDB (Vector Store)

### 3.1 컬렉션 구조

```python
{
    # ChromaDB 필수
    "id": "precedent_12345_chunk_0",      # {source}_{doc_id}_chunk_{index}
    "embedding": [0.1, 0.2, ...],         # 384 dimensions (로컬 모델)

    # 메타데이터 (필터링 + PostgreSQL 포인터)
    "metadata": {
        "doc_id": 12345,                  # PostgreSQL PK (포인터)
        "source": "precedents",           # 데이터 출처
        "doc_type": "precedent",          # 문서 유형
        "chunk_index": 0,                 # 청크 순서
        "chunk_start": 0,                 # 원문 내 시작 위치
        "chunk_end": 500,                 # 원문 내 종료 위치
        "case_number": "2020다12345",     # 사건번호 (필터용)
        "court_name": "대법원",            # 기관명 (필터용)
        "decision_date": "2021-05-01",    # 날짜 (필터용)
    }

    # documents: None (텍스트 미저장 - 용량 최적화)
}
```

### 3.2 저장 위치

```
backend/data/chroma/
├── chroma.sqlite3          # 메타데이터 DB
└── [collection_id]/        # 임베딩 데이터
    ├── data_level0.bin
    ├── header.bin
    ├── index_metadata.pickle
    └── length.bin
```

---

## 4. 청킹 전략

### 4.1 청킹 설정

```python
CHUNK_CONFIG = {
    "chunk_size": 500,      # 청크 크기 (문자 수)
    "chunk_overlap": 50,    # 오버랩 (문자 수)
    "min_chunk_size": 100,  # 최소 청크 크기
}
```

### 4.2 청킹 대상 텍스트

`LegalDocument.embedding_text` 프로퍼티가 청킹 대상 텍스트를 생성합니다:

```python
@property
def embedding_text(self) -> str:
    parts = []
    if self.case_name:
        parts.append(f"사건명: {self.case_name}")
    if self.summary:
        parts.append(f"요지: {self.summary}")
    if self.reasoning:
        reasoning = self.reasoning[:3000]  # 토큰 제한
        parts.append(f"내용: {reasoning}")
    return "\n".join(parts)
```

### 4.3 청킹 알고리즘

1. 텍스트를 `chunk_size` 단위로 분할
2. 문장 경계(`. `, `.\n`, `\n\n` 등)에서 자르기 시도
3. `min_chunk_size` 미만 청크는 제외
4. 다음 청크 시작 시 `chunk_overlap` 만큼 이전 텍스트 포함

```
[원문 텍스트]
사건명: 손해배상청구의 소
요지: 피고가 원고에게 ...
내용: 1. 사건 개요 ...

[청크 분할]
┌─────────────────────┐
│ Chunk 0 (0-500)     │
│ 사건명: 손해배상...  │
├─────────────────────┤
│ Chunk 1 (450-950)   │  ← 50자 오버랩
│ ...피고가 원고에게...│
├─────────────────────┤
│ Chunk 2 (900-1400)  │
│ ...1. 사건 개요...   │
└─────────────────────┘
```

---

## 5. 임베딩 생성

### 5.1 임베딩 모델

| 모델 | 차원 | 용도 |
|------|-----|------|
| `jhgan/ko-sbert-nli` | 384 | 로컬 임베딩 (기본) |
| `text-embedding-3-small` | 1536 | OpenAI API (선택) |

### 5.2 임베딩 생성 흐름

```
1. PostgreSQL에서 문서 조회 (배치 500건)
2. 각 문서를 청크로 분할
3. 청크 텍스트로 임베딩 벡터 생성 (배치 100건)
4. ChromaDB에 저장 (임베딩 + 메타데이터, 텍스트 제외)
```

### 5.3 생성 명령어

```bash
# 전체 임베딩 생성 (리셋 포함)
uv run python scripts/create_embeddings.py --type all --reset

# 특정 유형만
uv run python scripts/create_embeddings.py --type precedent

# 청킹 설정 변경
uv run python scripts/create_embeddings.py --chunk-size 500 --chunk-overlap 50

# OpenAI API 사용
uv run python scripts/create_embeddings.py --use-openai

# 통계 확인
uv run python scripts/create_embeddings.py --stats
```

---

## 6. RAG 검색 흐름

### 6.1 검색 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                      RAG 검색 흐름                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 사용자 질문 → 쿼리 임베딩 생성                           │
│              ↓                                              │
│  2. ChromaDB에서 유사 청크 검색 (ANN)                        │
│     - 메타데이터 + distances 반환                            │
│     - documents는 None (저장 안 함)                          │
│              ↓                                              │
│  3. 메타데이터에서 doc_id, chunk_start, chunk_end 추출       │
│              ↓                                              │
│  4. PostgreSQL에서 원문 조회                                 │
│     - LegalDocument.embedding_text[start:end]               │
│              ↓                                              │
│  5. 컨텍스트 구성 → LLM 응답 생성                            │
│              ↓                                              │
│  6. 응답 + 출처 정보 반환                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 검색 코드 흐름

```python
def search_relevant_documents(query: str, n_results: int = 5) -> List[dict]:
    """관련 법률 문서 검색"""

    # 1. 쿼리 임베딩 생성
    query_embedding = create_query_embedding(query)

    # 2. ChromaDB 검색 (메타데이터만)
    results = store.search(
        query_embedding=query_embedding,
        n_results=n_results,
        include=["metadatas", "distances"],  # documents 제외
    )

    # 3. doc_id와 청크 위치 수집
    doc_ids = set()
    chunk_positions = {}  # {doc_id: [(start, end), ...]}

    for metadata in results["metadatas"][0]:
        doc_id = metadata.get("doc_id")
        doc_ids.add(doc_id)
        chunk_positions[doc_id].append(
            (metadata["chunk_start"], metadata["chunk_end"])
        )

    # 4. PostgreSQL에서 원문 조회
    chunk_texts = _fetch_document_texts(list(doc_ids), chunk_positions)

    # 5. 결과 조합
    documents = []
    for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):
        chunk_key = f"{doc_id}_{chunk_start}_{chunk_end}"
        documents.append({
            "content": chunk_texts.get(chunk_key, ""),
            "metadata": metadata,
            "similarity": 1 - distance,
        })

    return documents
```

---

## 7. 데이터 현황

### 7.1 PostgreSQL

| 테이블 | 데이터 유형 | 건수 |
|--------|------------|------|
| **legal_documents** | | **191,728** |
| | precedent | 92,264 |
| | constitutional | 36,781 |
| | administration | 34,258 |
| | legislation | 8,597 |
| | committee | 19,828 |
| **laws** | | **5,572** |
| **legal_references** | | **62,008** |
| | law_term | 36,797 |
| | admin_rule | 21,622 |
| | treaty | 3,589 |
| **총계** | | **259,308** |

### 7.2 ChromaDB (Vector Store)

| 문서 유형 | 청크 수 | 원본 문서 수 |
|-----------|---------|-------------|
| precedent | 166,212 | 92,167 |
| constitutional | 36,015 | 36,015 |
| administration | 260,173 | 34,258 |
| legislation | 81,899 | 8,597 |
| committee | 98,796 | 13,978 |
| **총계** | **643,095** | **185,015** |

---

## 8. 데이터 관리 명령어

### 8.1 데이터 로드

```bash
# PostgreSQL에 데이터 로드
uv run python scripts/load_legal_data.py --type precedent
uv run python scripts/load_legal_data.py --type constitutional
uv run python scripts/load_legal_data.py --type administration
uv run python scripts/load_legal_data.py --type legislation
uv run python scripts/load_legal_data.py --type committee
uv run python scripts/load_legal_data.py --type law
uv run python scripts/load_legal_data.py --type treaty
uv run python scripts/load_legal_data.py --type admin_rule
uv run python scripts/load_legal_data.py --type law_term

# 전체 로드
uv run python scripts/load_legal_data.py --type all

# 통계 확인
uv run python scripts/load_legal_data.py --stats
```

### 8.2 임베딩 생성

```bash
# 전체 생성 (기존 삭제 후)
uv run python scripts/create_embeddings.py --type all --reset

# 개별 타입
uv run python scripts/create_embeddings.py --type precedent

# 통계 확인
uv run python scripts/create_embeddings.py --stats
```

### 8.3 데이터 검증

```bash
# 전체 검증
uv run python scripts/validate_data.py

# PostgreSQL만
uv run python scripts/validate_data.py --pg-only

# ChromaDB만
uv run python scripts/validate_data.py --chroma-only

# 불일치 자동 수정
uv run python scripts/validate_data.py --fix
```

### 8.4 마이그레이션

```bash
# 마이그레이션 적용
uv run alembic upgrade head

# 롤백
uv run alembic downgrade -1
```

---

## 9. 팀 공유 방법

### 9.1 PostgreSQL

Docker를 사용하여 PostgreSQL 데이터 공유:

```bash
# 데이터 덤프
pg_dump -h localhost -U postgres law_db > law_db_dump.sql

# 데이터 복원
psql -h localhost -U postgres law_db < law_db_dump.sql
```

### 9.2 ChromaDB

ChromaDB 디렉토리를 통째로 복사:

```bash
# 위치
backend/data/chroma/

# 압축 (약 1.7GB → 압축 시 ~500MB)
tar -czvf chroma_backup.tar.gz backend/data/chroma/

# 압축 해제
tar -xzvf chroma_backup.tar.gz
```

---

## 10. 설정 파일

### 10.1 환경 변수 (.env)

```bash
# PostgreSQL
DATABASE_URL=postgresql://postgres:password@localhost:5432/law_db
DATABASE_URL_ASYNC=postgresql+asyncpg://postgres:password@localhost:5432/law_db

# ChromaDB
CHROMA_PERSIST_DIR=./data/chroma
CHROMA_COLLECTION_NAME=legal_documents

# 임베딩
USE_LOCAL_EMBEDDING=true
LOCAL_EMBEDDING_MODEL=jhgan/ko-sbert-nli
EMBEDDING_MODEL=text-embedding-3-small

# OpenAI (선택)
OPENAI_API_KEY=sk-...
```

---

## 11. 파일 구조

```
backend/
├── app/
│   ├── common/
│   │   ├── chat_service.py      # RAG 검색 + 응답 생성
│   │   ├── database.py          # DB 연결 (sync + async)
│   │   └── vectorstore.py       # ChromaDB 래퍼
│   ├── models/
│   │   ├── legal_document.py    # 법률 문서 모델
│   │   ├── law.py               # 법령 모델
│   │   └── legal_reference.py   # 참조 데이터 모델
│   └── core/
│       └── config.py            # 설정
├── scripts/
│   ├── load_legal_data.py       # PostgreSQL 데이터 로드
│   ├── create_embeddings.py     # 임베딩 생성
│   ├── validate_data.py         # 데이터 검증
│   └── backup_data.py           # 백업
├── data/
│   ├── chroma/                  # ChromaDB 저장소
│   └── models/                  # 임베딩 모델 캐시
└── alembic/
    └── versions/                # DB 마이그레이션
```

---

*최종 업데이트: 2026-01-20*
