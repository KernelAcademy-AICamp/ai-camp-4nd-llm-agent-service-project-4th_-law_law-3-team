# Embedding Pipeline Skill (법률 RAG 특화)

## Overview
대규모 문서 임베딩 생성을 위한 최적화된 파이프라인.
법률 RAG 시스템에 특화되어 있으나, 범용적으로 사용 가능.

## When to Use
- 대규모 문서 임베딩 생성 (10만+ 문서)
- 법률 문서 청킹 및 처리
- 벡터 DB (LanceDB, Qdrant) 저장
- 임베딩 캐싱 및 증분 업데이트

## Templates Location
- `templates/chunking.py` - 문서 청킹 전략
- `templates/embedding_generator.py` - 임베딩 생성기
- `templates/vector_db_lancedb.py` - LanceDB 연동
- `templates/vector_db_qdrant.py` - Qdrant 연동
- `templates/cache_manager.py` - 임베딩 캐싱
- `templates/document_parser.py` - 문서 파서 (JSON, XML, HTML)
- `templates/tokenizer_utils.py` - 토크나이저 유틸리티

## Target Model
- **Primary**: `nlpai-lab/KURE-v1` (Korean Legal Embedding)
- 차원: 1536
- Max Length: 512 tokens

---

## 1. 문서 청킹 전략

### 규칙
- 법률 문서는 조문/항/호 단위로 자연스럽게 분리
- 오버랩으로 문맥 유지 (기본 50 tokens)
- 최대 길이는 모델 한계보다 여유 있게 (480 tokens 권장)

### 청킹 옵션
```python
from templates.chunking import (
    TokenChunker,      # 토큰 기반 청킹
    SentenceChunker,   # 문장 기반 청킹
    ParagraphChunker,  # 문단 기반 청킹
    LegalChunker,      # 법률 문서 특화 (조문 단위)
)

# 토큰 기반 (권장)
chunker = TokenChunker(
    model_name="nlpai-lab/KURE-v1",
    max_tokens=480,
    overlap_tokens=50
)

# 법률 문서 특화
chunker = LegalChunker(
    max_tokens=480,
    preserve_article_structure=True  # 조문 구조 유지
)
```

### 청킹 패턴
```python
# 단일 문서
chunks = chunker.chunk(document_text)
# Returns: List[Chunk] where Chunk has text, metadata, start_idx, end_idx

# 배치 처리
all_chunks = []
for doc in documents:
    chunks = chunker.chunk(doc['content'])
    for chunk in chunks:
        chunk.metadata['doc_id'] = doc['id']
        all_chunks.append(chunk)
```

---

## 2. 임베딩 생성

### 규칙
- 항상 배치 처리로 throughput 최대화
- 문서 길이에 따른 동적 배칭 (긴 문서 = 작은 배치)
- Mixed Precision 사용
- 진행 상황 저장으로 중단 시 재개 가능

### 기본 패턴
```python
from templates.embedding_generator import EmbeddingGenerator

generator = EmbeddingGenerator(
    model_name="nlpai-lab/KURE-v1",
    device="auto",  # 자동 디바이스 선택
    batch_size=32,
    use_amp=True,
    normalize=True,  # L2 정규화 (코사인 유사도용)
)

# 단일 텍스트
embedding = generator.encode("법률 텍스트")

# 배치 처리 with progress
embeddings = generator.encode_batch(
    texts,
    show_progress=True,
    checkpoint_path="embeddings_checkpoint.pt"  # 중단 시 재개
)
```

### 대규모 처리 패턴 (76만 문서)
```python
from templates.embedding_generator import BatchEmbeddingProcessor

processor = BatchEmbeddingProcessor(
    model_name="nlpai-lab/KURE-v1",
    output_dir="embeddings/",
    batch_size=32,
    checkpoint_interval=10000,  # 10000개마다 저장
)

# 스트리밍 처리
processor.process_streaming(
    data_iterator=document_generator(),  # 제너레이터
    id_field='doc_id',
    text_field='content'
)

# 또는 파일에서
processor.process_file(
    input_path="legal_documents.jsonl",
    id_field='법령ID',
    text_field='내용'
)
```

---

## 3. 벡터 정규화

### 규칙
- 코사인 유사도 검색 시 L2 정규화 필수
- 정규화는 임베딩 생성 시 또는 저장 전 한 번만

### 패턴
```python
import torch
import torch.nn.functional as F

def normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    """L2 정규화"""
    return F.normalize(embeddings, p=2, dim=-1)

# 코사인 유사도 계산 (정규화된 벡터)
similarity = torch.mm(query_embedding, document_embeddings.T)
```

---

## 4. 벡터 DB 저장

### LanceDB (권장 - 로컬/저비용)
```python
from templates.vector_db_lancedb import LanceDBManager

db = LanceDBManager(
    db_path="./legal_vectors",
    table_name="documents",
    embedding_dim=1536
)

# 저장
db.add_documents(
    ids=doc_ids,
    embeddings=embeddings,
    metadata=metadata_list  # [{"title": "...", "법령ID": "..."}, ...]
)

# 검색
results = db.search(
    query_embedding=query_emb,
    top_k=10,
    filter={"법령종류": "법률"}  # 메타데이터 필터
)
```

### Qdrant (분산/고성능)
```python
from templates.vector_db_qdrant import QdrantManager

db = QdrantManager(
    host="localhost",
    port=6333,
    collection_name="legal_documents",
    embedding_dim=1536
)

# 저장
db.add_documents(
    ids=doc_ids,
    embeddings=embeddings,
    metadata=metadata_list
)

# 검색
results = db.search(
    query_embedding=query_emb,
    top_k=10,
    filter_conditions={"must": [{"key": "법령종류", "match": {"value": "법률"}}]}
)
```

---

## 5. 임베딩 캐싱

### 규칙
- 동일 텍스트 재임베딩 방지
- 해시 기반 캐시 키
- 디스크 캐시로 영속성 확보

### 패턴
```python
from templates.cache_manager import EmbeddingCache

cache = EmbeddingCache(
    cache_dir="./embedding_cache",
    model_name="nlpai-lab/KURE-v1"
)

# 캐시 확인 후 임베딩
text = "법률 텍스트"
cached = cache.get(text)

if cached is not None:
    embedding = cached
else:
    embedding = generator.encode(text)
    cache.set(text, embedding)

# 배치 처리 (캐시 통합)
embeddings = cache.get_or_compute_batch(
    texts=text_list,
    compute_fn=generator.encode_batch
)
```

---

## 6. 문서 파싱

### JSON (법제처 API 기본 형식)
```python
from templates.document_parser import JSONParser

parser = JSONParser()

# 단일 파일
documents = parser.parse_file("legal_data.json")

# JSONL (스트리밍)
for doc in parser.parse_jsonl_streaming("legal_data.jsonl"):
    process(doc)
```

### XML (법제처 XML 형식)
```python
from templates.document_parser import XMLParser

parser = XMLParser(
    text_xpath="//조문내용",
    id_xpath="//법령ID"
)

documents = parser.parse_file("legal_data.xml")
```

### HTML
```python
from templates.document_parser import HTMLParser

parser = HTMLParser(
    content_selector="article.content",
    remove_selectors=["script", "style", "nav"]
)

documents = parser.parse_file("legal_page.html")
```

---

## 7. 토크나이저 유틸리티

### HuggingFace Tokenizer
```python
from templates.tokenizer_utils import get_tokenizer, count_tokens

tokenizer = get_tokenizer("nlpai-lab/KURE-v1")

# 토큰 수 계산
num_tokens = count_tokens(text, tokenizer)

# 길이 제한 자르기
truncated = truncate_to_tokens(text, tokenizer, max_tokens=480)
```

### KoNLPy (형태소 분석)
```python
from templates.tokenizer_utils import KoreanTokenizer

# Mecab 기반 (빠름)
tokenizer = KoreanTokenizer(backend="mecab")

# Okt 기반 (설치 쉬움)
tokenizer = KoreanTokenizer(backend="okt")

tokens = tokenizer.tokenize("법률 텍스트")
nouns = tokenizer.nouns("법률 텍스트")
```

---

## 8. 진행 상황 저장/복구

### 규칙
- 대규모 작업은 반드시 체크포인트
- 중단 시 마지막 체크포인트에서 재개
- 완료된 문서 ID 추적

### 패턴
```python
from templates.embedding_generator import ResumableProcessor

processor = ResumableProcessor(
    checkpoint_path="progress.json",
    model_name="nlpai-lab/KURE-v1"
)

# 중단된 지점에서 재개
processor.process_with_resume(
    documents=all_documents,
    id_field='doc_id',
    output_path="embeddings.npy"
)
```

---

## 9. 품질 검증

### 패턴
```python
from templates.embedding_generator import EmbeddingQualityChecker

checker = EmbeddingQualityChecker(generator)

# 유사 문서 쌍 테스트
similar_pairs = [
    ("민법 제1조", "민법 제1조의2"),
    ("상법 제1조", "상법 제2조"),
]

dissimilar_pairs = [
    ("민법 제1조", "형법 제250조"),
]

report = checker.evaluate(similar_pairs, dissimilar_pairs)
print(f"Similar avg: {report['similar_avg']:.4f}")
print(f"Dissimilar avg: {report['dissimilar_avg']:.4f}")
print(f"Separation: {report['separation']:.4f}")
```

---

## 10. 메타데이터 관리

### 법률 문서 메타데이터 스키마
```python
metadata_schema = {
    "법령ID": str,           # 고유 식별자
    "법령명": str,           # 법률 이름
    "법령종류": str,         # 법률, 시행령, 시행규칙 등
    "시행일자": str,         # YYYY-MM-DD
    "소관부처": str,         # 담당 부처
    "조문번호": str,         # 제1조, 제2조 등
    "chunk_index": int,      # 청크 인덱스
    "total_chunks": int,     # 전체 청크 수
}
```

### 메타데이터 포함 저장
```python
db.add_documents(
    ids=[f"{doc['법령ID']}_{i}" for i, chunk in enumerate(chunks)],
    embeddings=embeddings,
    metadata=[
        {
            "법령ID": doc['법령ID'],
            "법령명": doc['법령명'],
            "법령종류": doc['법령종류'],
            "시행일자": doc['시행일자'],
            "chunk_index": i,
            "total_chunks": len(chunks),
        }
        for i, chunk in enumerate(chunks)
    ]
)
```

---

## 11. 증분 업데이트

### 규칙
- 신규/변경된 문서만 임베딩
- 해시 기반 변경 감지
- 이전 버전 삭제 후 새 버전 추가

### 패턴
```python
from templates.cache_manager import IncrementalUpdater

updater = IncrementalUpdater(
    db=vector_db,
    cache=embedding_cache,
    hash_store_path="document_hashes.json"
)

# 변경된 문서만 처리
updated_docs = updater.get_changed_documents(new_documents)

if updated_docs:
    # 이전 임베딩 삭제
    updater.delete_old_embeddings(updated_docs)
    
    # 새 임베딩 생성 및 저장
    updater.process_updates(updated_docs, generator)
```

---

## 12. 중복 문서 처리

### 패턴
```python
from templates.cache_manager import DuplicateDetector

detector = DuplicateDetector()

# 해시 기반 정확 중복
duplicates = detector.find_exact_duplicates(documents, text_field='content')

# 임베딩 기반 유사 중복
similar = detector.find_similar_documents(
    embeddings=embeddings,
    threshold=0.95  # 95% 이상 유사도
)

# 중복 제거
unique_documents = detector.remove_duplicates(documents)
```

---

## Quick Reference

| 작업 | 해결책 |
|------|--------|
| 문서 청킹 | `TokenChunker` 또는 `LegalChunker` |
| 대규모 임베딩 | `BatchEmbeddingProcessor` |
| 캐싱 | `EmbeddingCache` |
| LanceDB 저장 | `LanceDBManager` |
| Qdrant 저장 | `QdrantManager` |
| 중단 재개 | `ResumableProcessor` |
| 증분 업데이트 | `IncrementalUpdater` |
| 품질 검증 | `EmbeddingQualityChecker` |

---

## Full Pipeline Example

```python
from templates.document_parser import JSONParser
from templates.chunking import LegalChunker
from templates.embedding_generator import BatchEmbeddingProcessor
from templates.vector_db_lancedb import LanceDBManager
from templates.cache_manager import EmbeddingCache

# 1. 문서 로드
parser = JSONParser()
documents = parser.parse_file("legal_documents.json")

# 2. 청킹
chunker = LegalChunker(max_tokens=480, overlap_tokens=50)
all_chunks = []
for doc in documents:
    chunks = chunker.chunk(doc['content'])
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "doc_id": doc['법령ID'],
            "title": doc['법령명'],
            "chunk_index": i,
        })
        all_chunks.append(chunk)

# 3. 임베딩 생성 (with caching)
cache = EmbeddingCache("./cache", "nlpai-lab/KURE-v1")
processor = BatchEmbeddingProcessor(
    model_name="nlpai-lab/KURE-v1",
    batch_size=32,
    cache=cache
)
embeddings = processor.encode_chunks(all_chunks)

# 4. 벡터 DB 저장
db = LanceDBManager("./legal_vectors", "documents", 1536)
db.add_documents(
    ids=[c.id for c in all_chunks],
    embeddings=embeddings,
    metadata=[c.metadata for c in all_chunks]
)

# 5. 검색 테스트
query = "손해배상 청구권 소멸시효"
query_emb = processor.encode_single(query)
results = db.search(query_emb, top_k=5)

for r in results:
    print(f"Score: {r['score']:.4f} - {r['metadata']['title']}")
```
