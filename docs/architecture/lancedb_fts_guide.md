# LanceDB Vector Search & Hybrid FTS 아키텍처 가이드

이 문서는 LanceDB를 활용한 법률 문서 벡터 검색 및 FTS(Full-Text Search) 구현 내용, 그리고 그 배경이 되는 기술적 의사결정을 정리한 문서입니다.

---

## 1. 아키텍처 개요: 이원화 저장 전략 (Dual Storage)

데이터의 성격과 용도에 따라 **저장소를 철저히 분리**하여 운영합니다.

| 저장소 | 역할 | 저장 데이터 | 비고 |
| :--- | :--- | :--- | :--- |
| **PostgreSQL** | **정본 데이터 (Cold Storage)** | 법령/판례 원문 전체, 메타데이터, 원본 JSON | `law_documents`, `precedent_documents` |
| **LanceDB** | **검색 인덱스 (Hot Storage)** | 임베딩 벡터, **토큰화된 텍스트**, 연결 ID | `legal_chunks` (단일 테이블 v2) |

### 핵심 설계 원칙
*   **LanceDB는 가볍게**: 원본 텍스트는 저장하지 않거나 최소화합니다. 오직 "검색"에 필요한 데이터(`content_tokenized`)와 "연결"에 필요한 ID(`source_id`)만 저장합니다.
*   **검색은 LanceDB, 조회는 PG**: 검색 결과로 ID를 얻으면, 실제 사용자에게 보여줄 상세 내용은 PostgreSQL에서 가져옵니다.

---

## 2. LanceDB 스키마 (Schema V2)

모든 데이터는 `legal_chunks`라는 **단일 테이블**에 통합 저장되며, `data_type`으로 구분됩니다. (`backend/app/tools/vectorstore/schema_v2.py`)

### 주요 컬럼 구조
```python
LEGAL_CHUNKS_SCHEMA = pa.schema([
    # [식별자]
    pa.field("id", pa.utf8()),              # 청크 ID ("문서ID_순번")
    pa.field("source_id", pa.utf8()),       # 원본 문서 ID (PostgreSQL 연결용 FK)
    
    # [검색용 데이터]
    pa.field("vector", ...),                # 임베딩 벡터 (1024차원)
    pa.field("content_tokenized", ...),     # [핵심] FTS 검색용 토큰 문자열
    
    # [메타데이터]
    pa.field("data_type", pa.utf8()),       # "법령" | "판례"
    pa.field("title", pa.utf8()),           # 제목 (법령명/사건명)
    # ... (그 외 필터링용 메타데이터)
])
```

*   **`content_tokenized`**: FTS를 위해 추가된 유일한 컬럼입니다. 이곳에 형태소 분석된 텍스트가 공백으로 구분되어 저장됩니다.
*   **`source_id`**: 검색 결과(청크)가 어떤 법령/판례에 속하는지 식별하는 핵심 연결고리입니다.

---

## 3. FTS (Full-Text Search) 구현 원리

단순한 키워드 매칭이 아니라, **MeCab 형태소 분석**과 **Tantivy 검색 엔진**을 결합하여 고품질 검색을 구현했습니다.

### 3.1 토큰화 파이프라인 (Tokenization Pipeline)
입력 텍스트는 3단계 변환을 거쳐 저장됩니다. (`backend/app/tools/vectorstore/mecab_tokenizer.py`)

1.  **전처리**: 특수문자(`ㆍ`, `·`)를 공백으로 치환 (오분석 방지)
2.  **형태소 분석 (MeCab)**: "손해배상청구권" -> `["손해", "배상", "청구", "권"]`
3.  **복합명사 보강**: 분해된 결과 뒤에 "복합어"를 추가 -> `... + ["손해배상", "손해배상청구권"]`

> **저장 예시**:
> "피고 의 손해 배상 청구 권 은 ... (중략) ... **손해배상 배상청구 손해배상청구권**"

### 3.2 저장 및 인덱싱 (Inverted Index)
*   **저장**: `content_tokenized` 컬럼에 위 문자열이 저장됩니다.
*   **인덱싱**: `create_fts_index()` 실행 시, 내부적으로 **역색인(Inverted Index)** 파일이 생성됩니다.
    *   구조: `단어(Term) -> 문서ID 목록(Postings)`
    *   위치: `data/lancedb/legal_chunks.lance/_indices/tantivy/`

### 3.3 상세 저장 및 검색 매커니즘 (Deep Dive)

**"보이지 않는 연결"**의 실체는 **역색인(Inverted Index)** 입니다. LanceDB 내부에서는 다음과 같이 데이터가 관리됩니다.

#### 1) `content_tokenized` 컬럼의 실제 모습
실제 DB 테이블에는 **공백으로 구분된 긴 문자열** 형태로 저장됩니다.

| Row ID | content (원본) | content_tokenized (검색용) |
| :--- | :--- | :--- |
| **0** | "피고는 원고에게..." | "피고 원고 피고원고 손해 배상 손해배상 ..." |
| **1** | "손해배상 청구..." | "손해 배상 손해배상 청구 손해배상청구 ..." |

#### 2) 역색인 (Inverted Index) 생성
`create_fts_index()`를 실행하면, Tantivy 엔진이 `content_tokenized`를 읽어서 **[단어 -> Row ID]** 매핑 장부를 만듭니다.

| 단어 (Term) | 포함된 Row IDs (Postings) |
| :--- | :--- |
| **"피고"** | `[0, 5, 12, ...]` |
| **"손해배상"** | `[0, 1, 8, ...]` |
| **"청구"** | `[1, 15, ...]` |

#### 3) 검색 과정 (Search Flow)
사용자가 **"손해배상"**을 검색하면:
1.  **쿼리 토큰화**: "손해배상" (MeCab 처리)
2.  **인덱스 조회**: 역색인 장부에서 `"손해배상"`을 찾음.
3.  **ID 추출**: `[0, 1, 8]`이라는 ID 목록을 즉시 획득. (원본 텍스트를 스캔하지 않음!)
4.  **결과 반환**: 해당 ID의 데이터를 사용자에게 반환.

### 3.4 검색 프로세스 요약 (User Mental Model)

사용자가 이해해야 할 **검색의 핵심 흐름**은 다음과 같습니다.

1.  **PostgreSQL (저장소)**: 원문(Full Text)만 저장합니다. FTS 인덱스나 토큰 데이터는 전혀 저장하지 않습니다.
2.  **LanceDB (검색기)**: `content_tokenized` (토큰 문자열)를 가지고 검색을 수행합니다.
3.  **조회 흐름**:
    *   **Step 1 (검색)**: LanceDB에서 `content_tokenized`를 뒤져 ID를 찾습니다.
    *   **Step 2 (조회)**: 찾은 ID로 PostgreSQL에서 원문을 가져와 보여줍니다.

---

## 4. 커스텀 사전 (MeCab UserDic) 운영

법률 용어의 정확한 인식을 위해 정교한 **사전 빌드 시스템**을 구축했습니다. (`backend/scripts/build_mecab_userdic.py`)

### 4.1 핵심 로직
1.  **회귀 방지 (Anti-Regression)**: 사전에 단어를 추가했을 때 기존 문장 분석이 깨지는지 시뮬레이션하고, 문제가 생기면 해당 단어의 **비용(Cost)을 -3000**으로 낮춰 강제로 인식시킵니다.
2.  **분해맵 (Decomposition Map)**: "손해배상"을 한 단어로 인식시키면서도, 검색 시 "손해", "배상"으로도 걸리게 하기 위해 **[복합어 -> 구성단어]** 매핑 정보를 생성합니다 (`decomposition_map.json`).
3.  **괄호 처리**: `전자(세금)계산서` 같은 표기를 `전자계산서`, `세금계산서`로 자동 확장하여 등록합니다.

---

## 5. 왜 PostgreSQL FTS 대신 LanceDB를 썼는가?

PostgreSQL도 훌륭한 FTS(tsvector)를 지원하지만, 다음과 같은 이유로 LanceDB(Tantivy)를 선택했습니다.

1.  **하이브리드 검색 (Hybrid Search) 최적화**:
    *   벡터 검색(Semantic)과 키워드 검색(Lexical)을 **동시에** 수행하고 합쳐야(RRF) 합니다.
    *   LanceDB 내에서 이 두 가지가 동시에 이루어지므로, 데이터를 애플리케이션으로 가져와서 합치는 비용이 없습니다.
2.  **관리 편의성 (Python-centric)**:
    *   PG의 한국어 형태소 분석기(플러그인)를 DB 서버에 설치하고 관리하는 것은 복잡합니다.
    *   LanceDB 방식은 Python 코드(`mecab_tokenizer.py`)에서 로직을 수정하면 바로 반영되므로, **법률 용어 사전 업데이트**나 **분석 알고리즘 개선**이 훨씬 빠르고 유연합니다.

---

## 6. 운영 가이드 (Troubleshooting)

### 6.1 검색 결과가 이상할 때 (Debug Checklist)

1.  **토큰화 결과 확인**:
    ```python
    from app.tools.vectorstore.mecab_tokenizer import MeCabTokenizer
    t = MeCabTokenizer()
    print(t.tokenize("이상하게 검색되는 문장"))
    ```
    (원하는 단어로 분리되지 않는다면 MeCab userdic에 추가 필요)

2.  **FTS 인덱스 재생성**:
    데이터가 대량으로 추가되거나 스키마가 변경되었다면 인덱스를 다시 만들어야 할 수 있습니다.
    ```python
    store.create_fts_index("content_tokenized")
    ```

3.  **RRF 파라미터 조정**:
    `hybrid_search()` 호출 시 `rrf_k` 값(기본 60)을 조절하여 벡터 검색과 키워드 검색의 비중을 튜닝할 수 있습니다.

### 6.2 데이터 동기화 (Sync Strategy)
현재는 **배치(Batch)** 방식으로 동기화를 수행합니다.

*   **Law/Precedent Update**: PostgreSQL에 데이터 적재 완료 후, 별도 스크립트(`create_lancedb_embeddings.py`)를 실행하여 LanceDB를 갱신합니다.
*   **주의**: LanceDB는 `upsert`를 지원하지 않으므로, 중복 방지를 위해 기존 데이터를 지우고 다시 넣거나(`overwrite`), `merge` 로직을 구현해야 합니다.

---

## 7. 벡터 인덱스 (Vector Index)

FTS 인덱스와 별도로, **벡터 검색 속도**를 높이기 위한 ANN(Approximate Nearest Neighbor) 인덱스를 지원합니다.

### 7.1 인덱스 타입별 성능 비교 (~253K 청크 기준)

| Index Type | Mean (ms) | Recall@10 | 빌드 시간 | 비고 |
|:-----------|----------:|----------:|----------:|------|
| Brute-force | 90.86 | 100% | - | 기본값 (인덱스 없음) |
| IVF_PQ | 3.41 | 60% | 26s | 벡터 압축, recall 손실 큼 |
| **IVF_FLAT** | **6.37** | **100%** | **36s** | **권장** — recall 유지, ~14x 빠름 |
| IVF_HNSW_SQ | 3.68 | 90% | 46s | 가장 빠르나 recall 10% 손실 |

### 7.2 활성화 방법

```bash
# backend/.env
LANCEDB_INDEX_TYPE=IVF_FLAT
```

*   빈 문자열(기본값)이면 인덱스를 생성하지 않고 brute-force로 동작합니다.
*   앱 시작(lifespan) 시 `create_vector_index()`가 호출되며, 이미 존재하면 스킵합니다.
*   인덱스 생성 실패 시 앱 시작은 차단되지 않고 brute-force로 fallback됩니다.

### 7.3 인덱스 파라미터

*   `num_partitions`: `max(16, sqrt(row_count))` 자동 계산
*   `metric`: cosine
*   `vector_column_name`: "vector" (1024차원, KURE-v1)

### 7.4 벤치마크 재현

```bash
cd backend
uv run python scripts/benchmark_lancedb_search.py
```

상세 결과: `docs/devlog/LANCEDB_VECTOR_INDEX_20260211.md`
