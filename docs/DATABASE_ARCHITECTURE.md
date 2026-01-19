# 데이터베이스 아키텍처

## 개요

본 프로젝트는 **PostgreSQL (RDBMS)**과 **ChromaDB (Vector DB)**를 함께 사용하는 하이브리드 데이터베이스 아키텍처를 채택했습니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                        법률 데이터 플로우                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   [원본 JSON 데이터]                                              │
│         │                                                       │
│         ▼                                                       │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                  PostgreSQL (RDBMS)                      │   │
│   │  ─────────────────────────────────────────────────────  │   │
│   │  • 정형 데이터 저장 (사건번호, 날짜, 법원명 등)              │   │
│   │  • 원본 JSON 보관 (raw_data)                              │   │
│   │  • 트랜잭션 관리                                           │   │
│   │  • 171,900건 법률 문서                                     │   │
│   └─────────────────────────────────────────────────────────┘   │
│         │                                                       │
│         │ embedding_text 추출                                   │
│         ▼                                                       │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │            Sentence-Transformers (Local)                 │   │
│   │  ─────────────────────────────────────────────────────  │   │
│   │  • 한국어 특화 모델                                        │   │
│   │  • paraphrase-multilingual-MiniLM-L12-v2                 │   │
│   │  • 384차원 벡터 생성                                       │   │
│   └─────────────────────────────────────────────────────────┘   │
│         │                                                       │
│         ▼                                                       │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                  ChromaDB (Vector DB)                    │   │
│   │  ─────────────────────────────────────────────────────  │   │
│   │  • 임베딩 벡터 저장                                        │   │
│   │  • 코사인 유사도 검색                                      │   │
│   │  • 메타데이터 필터링                                       │   │
│   │  • 171,134개 임베딩                                       │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. PostgreSQL (RDBMS)

### 역할
- **원본 데이터 저장소** (Single Source of Truth)
- 정형화된 법률 문서 메타데이터 관리
- 복잡한 필터링/정렬 쿼리 지원
- 데이터 무결성 및 트랜잭션 보장

### 테이블 구조

```sql
-- legal_documents 테이블
CREATE TABLE legal_documents (
    id              SERIAL PRIMARY KEY,

    -- 문서 식별
    doc_type        VARCHAR(20) NOT NULL,    -- precedent, constitutional, administration, legislation
    serial_number   VARCHAR(50) NOT NULL,    -- 원본 일련번호

    -- 사건 기본 정보
    case_name       TEXT,                    -- 사건명
    case_number     TEXT,                    -- 사건번호
    decision_date   DATE,                    -- 선고일/의결일

    -- 기관 정보
    court_name      TEXT,                    -- 법원명/재결청
    court_type      TEXT,                    -- 법원종류
    case_type       TEXT,                    -- 사건종류

    -- 주요 내용 (임베딩 대상)
    summary         TEXT,                    -- 판시사항/요지
    reasoning       TEXT,                    -- 판결요지/이유
    full_text       TEXT,                    -- 전문

    -- 원본 보관
    raw_data        JSONB NOT NULL,          -- 원본 JSON 전체

    -- 타임스탬프
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW(),

    UNIQUE (doc_type, serial_number)
);

-- 인덱스
CREATE INDEX idx_doc_type ON legal_documents(doc_type);
CREATE INDEX idx_case_number ON legal_documents(case_number);
CREATE INDEX idx_decision_date ON legal_documents(decision_date);
CREATE INDEX idx_court_name ON legal_documents(court_name);
```

### 저장 데이터

| 문서 유형 | 설명 | 건수 |
|----------|------|------|
| precedent | 일반 판례 | ~160,000 |
| constitutional | 헌법재판소 결정례 | ~7,000 |
| administration | 행정심판례 | ~3,000 |
| legislation | 법령해석례 | ~1,900 |
| **합계** | | **171,900** |

### 사용 목적

```python
# 1. 상세 조회 - ID로 전체 데이터 조회
document = await session.get(LegalDocument, document_id)

# 2. 필터링 검색 - 특정 조건의 문서 목록
query = select(LegalDocument).where(
    LegalDocument.doc_type == "precedent",
    LegalDocument.court_name == "대법원",
    LegalDocument.decision_date >= "2020-01-01"
)

# 3. 원본 데이터 접근
raw_json = document.raw_data  # 원본 JSON 전체
```

---

## 2. ChromaDB (Vector DB)

### 역할
- **의미 기반 유사도 검색** (Semantic Search)
- RAG (Retrieval-Augmented Generation) 지원
- 빠른 근사 최근접 이웃 검색 (ANN)

### 컬렉션 구조

```python
# 컬렉션 설정
collection = client.get_or_create_collection(
    name="legal_documents",
    metadata={
        "hnsw:space": "cosine",  # 코사인 유사도
    }
)

# 저장 데이터 구조
{
    "ids": ["precedent_12345", ...],           # PostgreSQL 연결 ID
    "embeddings": [[0.1, 0.2, ...], ...],      # 384차원 벡터
    "documents": ["사건명: ... 요지: ...", ...], # 검색용 텍스트
    "metadatas": [{                             # 필터링용 메타데이터
        "doc_type": "precedent",
        "case_number": "2023다12345",
        "case_name": "손해배상",
        "court_name": "대법원",
        "db_id": 12345                          # PostgreSQL ID
    }, ...]
}
```

### ID 연결 규칙

ChromaDB의 ID는 PostgreSQL과 연결됩니다:

```
ChromaDB ID: "{doc_type}_{serial_number}"
예: "precedent_12345" → PostgreSQL의 doc_type='precedent', serial_number='12345'
```

### 임베딩 텍스트 생성

```python
# LegalDocument.embedding_text 속성
@property
def embedding_text(self) -> str:
    parts = []
    if self.case_name:
        parts.append(f"사건명: {self.case_name}")
    if self.summary:
        parts.append(f"요지: {self.summary}")
    if self.reasoning:
        parts.append(f"내용: {self.reasoning[:3000]}")
    return "\n".join(parts)
```

### 사용 목적

```python
# 의미 기반 검색
results = store.search(
    query_embedding=user_query_embedding,
    n_results=5,
    where={"doc_type": "precedent"}  # 선택적 필터
)

# 결과 예시
# {
#     "ids": [["precedent_123", "precedent_456", ...]],
#     "distances": [[0.12, 0.15, ...]],  # 코사인 거리 (1 - 유사도)
#     "metadatas": [[{"case_name": "...", "case_number": "..."}, ...]]
# }
```

---

## 3. 분리한 이유

### 3.1 각 DB의 강점 활용

| 기능 | PostgreSQL | ChromaDB |
|------|------------|----------|
| 정확한 키워드 검색 | O | X |
| 의미 기반 유사도 검색 | X | O |
| 복잡한 필터링/조인 | O | 제한적 |
| 트랜잭션/ACID | O | X |
| 벡터 검색 성능 | 느림 | 빠름 (ANN) |
| 데이터 무결성 | O | X |

### 3.2 RAG 파이프라인 요구사항

```
사용자 질문: "교통사고 손해배상 청구 방법"
     │
     ▼
┌─────────────────────────────────────────┐
│ ChromaDB: 의미적으로 유사한 판례 검색      │
│ → "손해배상", "교통사고" 관련 판례 5건     │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│ PostgreSQL: 상세 정보 조회 (필요 시)       │
│ → 전문, 참조조문, 참조판례 등              │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│ LLM: 컨텍스트 기반 응답 생성              │
│ → GPT-4o-mini                           │
└─────────────────────────────────────────┘
```

### 3.3 성능 및 비용 최적화

1. **검색 성능**
   - PostgreSQL LIKE 검색: O(n) - 전체 스캔
   - ChromaDB ANN 검색: O(log n) - HNSW 인덱스

2. **저장 효율**
   - 원본 데이터는 PostgreSQL에만 저장 (중복 방지)
   - ChromaDB는 임베딩 + 메타데이터만 저장

3. **유연한 임베딩 교체**
   - 임베딩 모델 변경 시 ChromaDB만 재생성
   - PostgreSQL 데이터는 그대로 유지

### 3.4 확장성

```
현재 구조:
PostgreSQL (Docker) ←→ ChromaDB (Local File)

프로덕션 확장:
PostgreSQL (RDS) ←→ ChromaDB (서버 모드 또는 Pinecone/Weaviate)
```

---

## 4. 데이터 동기화

### 초기 로드 프로세스

```bash
# 1. JSON → PostgreSQL
uv run python scripts/load_legal_data.py

# 2. PostgreSQL → ChromaDB (임베딩 생성)
uv run python scripts/create_embeddings.py
```

### 동기화 전략

| 상황 | 처리 방법 |
|------|----------|
| 새 문서 추가 | PostgreSQL INSERT → ChromaDB ADD |
| 문서 수정 | PostgreSQL UPDATE → ChromaDB UPDATE (재임베딩) |
| 문서 삭제 | PostgreSQL DELETE → ChromaDB DELETE |
| 임베딩 모델 변경 | ChromaDB 전체 재생성 |

현재는 배치 처리 방식이며, 실시간 동기화가 필요하면 이벤트 기반 아키텍처(Change Data Capture)로 확장 가능합니다.

---

## 5. 파일 구조

```
backend/
├── app/
│   ├── common/
│   │   ├── database.py       # PostgreSQL 연결 설정
│   │   ├── vectorstore.py    # ChromaDB 연결 설정
│   │   └── chat_service.py   # RAG 서비스 (양쪽 DB 사용)
│   └── models/
│       └── legal_document.py # SQLAlchemy 모델
├── scripts/
│   ├── load_legal_data.py    # JSON → PostgreSQL
│   └── create_embeddings.py  # PostgreSQL → ChromaDB
└── data/
    ├── chroma/               # ChromaDB 저장 디렉토리
    └── models/               # 임베딩 모델 캐시
```

---

## 6. 설정

### 환경 변수 (.env)

```env
# PostgreSQL
DATABASE_URL=postgresql://lawuser:lawpassword@localhost:5432/lawdb

# ChromaDB
CHROMA_PERSIST_DIR=./data/chroma
CHROMA_COLLECTION_NAME=legal_documents

# 임베딩
USE_LOCAL_EMBEDDING=True
LOCAL_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

---

## 7. 요약

| 구분 | PostgreSQL | ChromaDB |
|------|------------|----------|
| **용도** | 원본 데이터 저장 | 유사도 검색 |
| **데이터** | 전체 문서 (171,900건) | 임베딩 벡터 (171,134개) |
| **검색 방식** | SQL 쿼리 | 벡터 유사도 |
| **연결** | SQLAlchemy Async | chromadb Python |
| **저장 위치** | Docker Container | Local File |

**핵심 원칙**: PostgreSQL이 Single Source of Truth, ChromaDB는 검색 최적화용 인덱스 역할
