> **NOTE (2026-01-29)**: 이 문서는 초기 구현 계획입니다.
> 현재 구현 상태는 `docs/vectordb_design.md`를 참고하세요.
> 임베딩 스크립트 사용법은 `backend/scripts/CLAUDE.md`를 참고하세요.

---

S3 없이 **로컬 디스크**를 활용하여 LanceDB를 도입하는 구체적인 구현 계획입니다.
기존 코드(`create_embeddings.py`)의 "청킹 로직"을 그대로 살리면서, LanceDB의 장점인 **"디스크 기반 데이터 저장"** 기능을 100% 활용할 수 있도록 구성했습니다.

---

### 📅 구현 로드맵

1. **설정(Config):** LanceDB 경로 및 설정 추가
2. **스키마(Schema):** Pydantic을 이용한 데이터 구조 정의 (법률 데이터 최적화)
3. **구현(Implementation):** `VectorStoreBase`를 상속받는 `LanceDBStore` 클래스 개발
4. **통합(Integration):** 벡터 스토어 팩토리(`__init__.py`) 연결
5. **데이터 주입 수정:** `create_embeddings.py`에서 텍스트를 함께 저장하도록 수정

---

### 1단계: 패키지 설치 및 설정 (`backend/app/core/config.py`)

먼저 LanceDB를 설치하고 설정을 추가합니다.

**터미널:**

```bash
uv pip install lancedb  # 또는 pip install lancedb

```

**`backend/app/core/config.py` 수정:**

```python
class Settings(BaseSettings):
    # ... 기존 설정 ...

    # Vector DB 선택 (chroma, qdrant, lancedb 추가)
    VECTOR_DB: str = "lancedb"  # 변경

    # LanceDB 설정 (추가)
    LANCEDB_URI: str = "./data/lancedb"      # 로컬 데이터 저장 경로
    LANCEDB_TABLE_NAME: str = "legal_chunks" # 테이블 이름

```

---

### 2단계: 스키마 정의 (`backend/app/common/vectorstore/schema.py`)

LanceDB는 명시적인 스키마가 있을 때 가장 성능이 좋습니다. 새로 파일을 생성합니다.

**생성: `backend/app/common/vectorstore/schema.py**`

```python
from lancedb.pydantic import LanceModel, Vector
from typing import Optional

# OpenAI 임베딩 차원 (1536), 로컬 모델 사용 시 모델에 맞춰 변경 필요 (예: 768)
# create_embeddings.py 로그에서 확인 가능
VECTOR_DIM = 1536 

class LegalChunkSchema(LanceModel):
    """
    법률 문서 청크 스키마
    
    기존 create_embeddings.py의 Chunk 데이터클래스와 호환되도록 설계
    """
    # 1. 벡터 데이터
    vector: Vector(VECTOR_DIM)

    # 2. 식별자
    id: str                 # Chunk ID (source_doc_id_chunk_idx)
    doc_id: int             # 원본 문서 ID (PostgreSQL FK)
    
    # 3. 텍스트 데이터 (LanceDB는 디스크 기반이라 원문 저장에 부담이 없음)
    text: str               # 실제 청크 텍스트

    # 4. 메타데이터 (필터링용)
    source: str             # precedent, constitutional 등
    doc_type: str
    chunk_index: int
    case_number: Optional[str]
    court_name: Optional[str]
    decision_date: Optional[str]
    
    # 5. 구조 정보 (나중에 정밀 검색 시 활용)
    chunk_start: int
    chunk_end: int

```

---

### 3단계: LanceDB 스토어 구현 (`backend/app/common/vectorstore/lancedb.py`)

기존 인터페이스(`VectorStoreBase`)를 준수하는 구현체를 만듭니다.

**생성: `backend/app/common/vectorstore/lancedb.py**`

```python
import lancedb
from typing import List, Optional, Dict, Any
from pathlib import Path

from app.core.config import settings
from app.common.vectorstore.base import VectorStoreBase, SearchResult
from app.common.vectorstore.schema import LegalChunkSchema

class LanceDBStore(VectorStoreBase):
    def __init__(self):
        # 데이터 디렉토리 생성
        db_path = Path(settings.LANCEDB_URI)
        db_path.mkdir(parents=True, exist_ok=True)
        
        # DB 연결
        self.db = lancedb.connect(settings.LANCEDB_URI)
        self.table_name = settings.LANCEDB_TABLE_NAME
        
        # 테이블 초기화 (스키마 적용)
        try:
            self.table = self.db.create_table(
                self.table_name,
                schema=LegalChunkSchema,
                exist_ok=True
            )
        except Exception:
            # 이미 존재하면 엽니다
            self.table = self.db.open_table(self.table_name)

    def add_documents(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[str]] = None,
    ) -> None:
        """
        데이터 대량 삽입
        """
        data = []
        for i, doc_id in enumerate(ids):
            meta = metadatas[i] if metadatas else {}
            text = documents[i] if documents else ""
            
            # Pydantic 스키마에 맞춰 데이터 매핑
            record = LegalChunkSchema(
                vector=embeddings[i],
                id=doc_id,
                doc_id=int(meta.get("doc_id", 0)),
                text=text,
                source=meta.get("source", "unknown"),
                doc_type=meta.get("doc_type", "unknown"),
                chunk_index=int(meta.get("chunk_index", 0)),
                case_number=meta.get("case_number", ""),
                court_name=meta.get("court_name", ""),
                decision_date=str(meta.get("decision_date", "")),
                chunk_start=int(meta.get("chunk_start", 0)),
                chunk_end=int(meta.get("chunk_end", 0)),
            )
            data.append(record)
            
        # LanceDB에 추가 (Batch Insert)
        if data:
            self.table.add(data)

    def search(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
    ) -> SearchResult:
        # 쿼리 빌더 시작
        query = self.table.search(query_embedding).limit(n_results)
        
        # 필터 적용 (SQL 스타일 문자열로 변환 필요)
        # 예: where={"doc_type": "precedent"} -> "doc_type = 'precedent'"
        if where:
            filter_conditions = []
            for key, value in where.items():
                if isinstance(value, str):
                    filter_conditions.append(f"{key} = '{value}'")
                else:
                    filter_conditions.append(f"{key} = {value}")
            
            if filter_conditions:
                query = query.where(" AND ".join(filter_conditions))
                
        # 검색 실행 (Pandas DataFrame으로 반환받음)
        df = query.to_pandas()
        
        if df.empty:
            return SearchResult(ids=[], distances=[], metadatas=[], documents=[])

        # 결과 변환
        return SearchResult(
            ids=[df["id"].tolist()], # 2중 리스트 구조 유지 (Base 호환)
            distances=[df["_distance"].tolist()],
            documents=[df["text"].tolist()], # 저장해둔 텍스트 반환
            metadatas=[df[[
                "doc_id", "source", "doc_type", "case_number", "decision_date"
            ]].to_dict(orient="records")]
        )

    def get_by_ids(self, ids: List[str]) -> Dict[str, Any]:
        """ID로 조회 (LanceDB는 SQL 쿼리 사용)"""
        formatted_ids = ", ".join([f"'{id}'" for id in ids])
        results = self.table.search().where(f"id IN ({formatted_ids})").to_pandas()
        # 변환 로직 (생략 - 필요 시 구현)
        return {} 

    def delete_by_ids(self, ids: List[str]) -> None:
        formatted_ids = ", ".join([f"'{id}'" for id in ids])
        self.table.delete(f"id IN ({formatted_ids})")

    def count(self) -> int:
        return len(self.table)

    def reset(self) -> None:
        self.db.drop_table(self.table_name)
        self.table = self.db.create_table(
            self.table_name, 
            schema=LegalChunkSchema
        )

```

---

### 4단계: 팩토리 연결 (`backend/app/common/vectorstore/__init__.py`)

이제 `VectorStore`를 호출할 때 `LanceDBStore`가 나오도록 연결합니다.

**수정: `backend/app/common/vectorstore/__init__.py**`

```python
from app.core.config import settings
from .base import VectorStoreBase

def VectorStore() -> VectorStoreBase:
    """설정에 따라 적절한 VectorStore 구현체 반환"""
    
    if settings.VECTOR_DB == "lancedb":
        from .lancedb import LanceDBStore
        return LanceDBStore()
        
    elif settings.VECTOR_DB == "qdrant":
        from .qdrant import QdrantVectorStore
        return QdrantVectorStore()
        
    else:  # 기본값 chroma
        from .chroma import ChromaVectorStore
        return ChromaVectorStore()

```

---

### 5단계: 데이터 주입 스크립트 수정 (`backend/scripts/create_embeddings.py`)

**가장 중요한 부분입니다.**
기존 코드는 ChromaDB 용량 문제로 `documents=None`을 보내 텍스트 저장을 안 했지만, LanceDB는 디스크를 쓰므로 **반드시 텍스트를 함께 저장**해야 검색 결과로 원문을 바로 볼 수 있습니다.

`_store_chunk_batch` 함수만 수정하면 됩니다.

**수정 전:**

```python
store.add_documents(
    ids=ids,
    documents=None,  # 텍스트 저장 안 함 (용량 최적화)
    metadatas=metadatas,
    embeddings=embeddings,
)

```

**수정 후:**

```python
def _store_chunk_batch(store: VectorStore, chunks: List[Chunk], use_local: bool) -> int:
    # ... (상단 동일) ...
    texts = [c.chunk_text for c in chunks]
    embeddings = create_embeddings_batch(texts, use_local)

    # ... (중단 동일) ...
    
    # LanceDB는 텍스트를 저장해도 효율적이므로 texts를 넘깁니다.
    # 기존 코드와의 호환성을 위해 settings.VECTOR_DB 체크
    documents_to_save = texts if settings.VECTOR_DB == "lancedb" else None

    store.add_documents(
        ids=ids,
        documents=documents_to_save,  # LanceDB일 경우 텍스트 저장!
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return len(chunks)

```

---

### 🚀 실행 방법

1. **초기화 및 생성:** 기존 임베딩이 있다면 리셋하고 다시 만듭니다.
```bash
uv run python backend/scripts/create_embeddings.py --reset --type all

```


2. **결과 확인:** 생성 후 `./data/lancedb` 폴더에 `.lance` 파일들이 생성되었는지 확인합니다.

이제 S3 없이도 수 기가바이트의 법률 데이터를 로컬 디스크에서 빠르고 효율적으로 검색할 수 있습니다. 메모리 사용량도 확연히 줄어들 것입니다.