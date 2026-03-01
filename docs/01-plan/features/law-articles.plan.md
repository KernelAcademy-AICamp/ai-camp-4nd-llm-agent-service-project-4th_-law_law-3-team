# law_articles 테이블 구현 계획

> PDCA Plan | Feature: law-articles | Created: 2026-03-01

## Context

법령 문서(`law_documents`)의 `content` 컬럼은 모든 조문+항+호를 `\n`으로 연결한 전체 텍스트를 저장한다.
LLM 컨텍스트 구성 시 이 전체 텍스트가 그대로 주입되어 **비효율적으로 길어지는 문제**가 있다.

LanceDB에는 이미 조문 단위 벡터가 존재한다:
- `summary_type="Basic"` + `article_number=None` → 법령 전체 요약
- `summary_type="Specific"` + `article_number="1"` → 조문별 요약

그러나 검색 파이프라인에서 `article_number`와 `summary_type` 메타데이터가 전파되지 않아 활용되지 못하고 있다.

**목표**: `law_articles` 테이블을 만들어, 벡터 검색에서 매칭된 조문만 선별적으로 LLM 컨텍스트에 포함시킨다.

### 데이터 공존 전략

`law_documents.content`는 모든 조문을 `\n`으로 연결한 전체 텍스트이고, `law_articles`는 동일 텍스트를 조문 단위로 분리 저장한다.

**`law_documents.content`는 그대로 유지한다.**
- `pg_graph_service.py`의 체계도 원문 보기 등 다른 팀원 담당 영역에서 `content`를 직접 참조하고 있음
- 해당 영역의 전환은 담당자가 별도로 진행

이 작업의 범위:
1. `law_articles` 테이블 생성 + 조문 데이터 적재
2. RAG 파이프라인의 LLM 컨텍스트 조회를 `law_articles` 조문 단위로 전환
3. `law_documents` 테이블은 **변경하지 않음**

---

## 핵심 설계 원칙: LanceDB ↔ law_articles 키 정합성

LanceDB 벡터와 PostgreSQL 조문 테이블이 **동일한 키 형식**을 사용해야 조문 단위 조회가 가능하다.

### LanceDB 저장 형식 (현재)

`law_article_vector_writer.py` L88-93:
```python
# article_number = str(article.get("조문번호", ""))
# → 원본 JSON의 조문번호 값 그대로 저장 (예: "1", "2", "3")
{
    "summary_type": "Specific",
    "article_number": "1",      # ← JSON 조문번호 원본값
    "chunk_index": 1,
}
```

### law_articles 테이블 키 형식 (맞춰야 할 것)

```
law_articles.law_id = LanceDB source_id       (예: "100001")
law_articles.article_number = LanceDB article_number  (예: "1", "2")
```

**정합성 보장**: 적재 스크립트에서 동일 소스(`조문번호`)에서 값을 추출하여 양쪽 형식 일치.

### 하이브리드 검색 전략: 검색 소스별 content 분기

키워드 검색(BM25)은 `fts_index`에서 **법령 전체 단위**(1법령=1행)로 검색하므로 `article_number`를 반환하지 않는다.
조문 특정은 **벡터 검색의 역할**이고, 키워드 검색은 **법령 특정** 역할에 집중한다.

| 검색 소스 | article_number | content 전략 |
|-----------|---------------|-------------|
| 벡터+키워드 (both) | 벡터의 article_number 사용 | `law_articles`에서 해당 조문 조회 |
| 벡터 only | article_number 있음 | `law_articles`에서 해당 조문 조회 |
| 키워드 only | article_number 없음 | **content 조회 안 함** — 메타데이터의 `law_name`만 참조 |

키워드 only일 때 `law_documents.content` 전체를 넣지 않는 이유:
- 이 기능의 목적 자체가 **컨텍스트 길이 절감**
- 조문을 특정할 수 없는 법령의 전체 텍스트를 넣으면 목적에 반함
- 법령명만으로도 "이 법령이 관련있다"는 참조 역할은 충분

### 조회 흐름

```
[벡터 검색 경로]
LanceDB 조문별 벡터 검색
  → source_id="100001", article_number="750"
    ↓
metadata 전파 (_extract_metadatas → _search_vector_ids)
  → metadata["article_number"] = "750"
    ↓
PostgreSQL 조회
  → SELECT * FROM law_articles WHERE law_id='100001' AND article_number='750'
    ↓
LLM 컨텍스트: [제750조] 조문 본문 포함

[키워드 only 경로]
fts_index BM25 검색 (법령 전체 단위)
  → source_id="100001", article_number 없음
    ↓
content 조회 스킵 — metadata의 title(법령명)만 사용
    ↓
LLM 컨텍스트: 법령명만 참조 (본문 없음)
```

LLM 컨텍스트 출력 예시:
```
## 관련 법령
[법령 1] 민법 (id: law_001)                    ← 벡터 매칭 (조문 포함)
  [제750조] 불법행위의 내용
  고의 또는 과실로 인한 위법행위로 타인에게 손해를 가한 자는...

[법령 2] 국가배상법 (id: law_002)               ← 키워드 only (법령명만)
```

---

## Layer 1: DB 스키마 (ORM + Alembic 021)

### 1-1. ORM 모델 생성

**파일**: `backend/app/models/law_article.py` (신규)

```python
class LawArticle(Base):
    __tablename__ = "law_articles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    law_id = Column(String(50), nullable=False, comment="법령 ID (law_documents.law_id 대응)")
    article_number = Column(String(50), nullable=False, comment="조문번호 (LanceDB article_number과 동일 형식)")
    article_title = Column(String(500), nullable=True, comment="조문제목")
    article_content = Column(Text, nullable=False, comment="조문 본문 (항+호 포함)")
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("law_id", "article_number", name="uq_law_articles_law_article"),
        Index("idx_law_articles_law_id", "law_id"),
    )
```

- `article_number` 형식: JSON `조문번호` 값 그대로 (`"1"`, `"2"`, ...) → LanceDB와 동일
- FK 제약조건 없음 (로드 순서 유연성)
- `(law_id, article_number)` UNIQUE → 멱등 upsert 가능

### 1-2. models/__init__.py에 등록

**파일**: `backend/app/models/__init__.py`
- `from app.models.law_article import LawArticle` import + `__all__` 추가

### 1-3. alembic/env.py에 import 추가

**파일**: `backend/alembic/env.py`

### 1-4. Alembic 마이그레이션

**파일**: `backend/alembic/versions/021_add_law_articles_table.py` (신규)

마이그레이션 작업:
1. `law_articles` 테이블 생성 (인덱스 포함)

---

## Layer 2: 데이터 적재 (JSON → law_articles)

### 2-1. 적재 스크립트

**파일**: `backend/scripts/load_law_articles_data.py` (신규)

기존 파싱 함수 재사용 (`backend/scripts/ingest/types/law.py`):
- `_extract_article_body(article)` (L115-139) → 항+호 텍스트 추출
- `_extract_ho(ho_data)` (L89-112) → 호 데이터 추출

로직:
1. `data/ingest_source/law_v3.json` 읽기 (sources.yaml 경로)
2. 각 법령의 `조문` 배열 순회
3. `article_number = str(article.get("조문번호", ""))` ← **LanceDB와 동일 추출 로직**
4. `article_title = str(article.get("조문제목", ""))`
5. `article_content = "\n".join(_extract_article_body(article))`
6. `ON CONFLICT (law_id, article_number) DO UPDATE` upsert
7. 1,000건 배치, 진행률 로깅
8. `--verify` 옵션: 총 건수, law_id별 조문 수 상위 5개

---

## Layer 3: 벡터 메타데이터 전파 (article_number, summary_type)

현재 `_extract_metadatas()`에서 drop되는 두 컬럼을 파이프라인에 전파.

### 3-1. _extract_metadatas 수정

**파일**: `backend/app/tools/vectorstore/lancedb.py` (L440-454)

현재 (drop):
```python
meta: Dict[str, Any] = {
    "source_id": row.get("source_id"),
    "data_type": row.get("data_type"),
    "title": row.get("title"),
    "date": row.get("date"),
    "source_name": row.get("source_name"),
    "chunk_index": row.get("chunk_index"),
    "total_chunks": row.get("total_chunks"),
}
```

변경 (추가):
```python
meta: Dict[str, Any] = {
    ...기존 7개...,
    "summary_type": row.get("summary_type"),
    "article_number": row.get("article_number"),
}
```

### 3-2. _search_vector_ids 메타데이터 확장

**파일**: `backend/app/services/rag/retrieval.py` (L320-335)

현재 metadata dict에 `article_number` 없음. 추가:
```python
best[source_id] = {
    ...기존...,
    "metadata": {
        ...기존...,
        "article_number": raw_meta.get("article_number", ""),
        "summary_type": raw_meta.get("summary_type", ""),
    },
}
```

**변경**: 기존 `best` dict는 `source_id`를 키로 사용하여 법령당 1건만 유지했으나,
같은 법령이라도 관련 조문이 다를 수 있으므로 `(source_id, article_number)` 복합키로 변경한다.
이를 통해 동일 법령의 서로 다른 조문이 모두 LLM 컨텍스트에 포함된다.

---

## Layer 4: 원문 조회 (조문 단위) + 레지스트리 전환

### 4-1. DOCUMENT_TABLE_REGISTRY 유지 + 조문 분기

**파일**: `backend/app/services/rag/retrieval.py` (L55-57)

기존 레지스트리는 **변경하지 않는다** (`law_documents.content` 유지):
```python
# 그대로 유지:
"법령": [
    TableConfig("law_documents", "law_id", ("content",)),
],
```

대신 `_fetch_contents_for_type` 내부에서 `article_number`가 있으면 `law_articles` 조문 조회로 분기한다.
`article_number`가 없으면 **content 조회를 스킵**하고 metadata의 `law_name`만 참조한다.

### 4-2. 조문 단위 content fetch 함수

**파일**: `backend/app/services/rag/retrieval.py`

새 함수:
```python
def _fetch_law_article_contents(
    source_ids_with_articles: dict[str, str],  # {source_id: article_number}
) -> dict[str, dict[str, str]]:
```

로직:
1. `source_ids_with_articles` → `law_articles` WHERE `law_id=:sid AND article_number=:an`
2. `{source_id: {"content": article_content}}` 반환
3. `article_number`가 없는 source_id는 **content 조회를 하지 않음** (법령명만 참조)

### 4-3. _fetch_contents_for_type 분기

`data_type == "법령"`일 때:
- metadata에 `article_number`가 있으면 → `_fetch_law_article_contents()` 호출
- `article_number`가 없으면 → **content 조회 스킵** (metadata의 `law_name`만 참조, `law_documents.content` 사용 안 함)

---

## Layer 5: LLM 컨텍스트 포맷 개선

### 5-1. format_law_context 수정

**파일**: `backend/app/services/rag/format_utils.py` (L64-77)

변경:
- `article_number`가 있으면 → 조문 번호 + 조문 본문 표시
- `article_number`가 없으면 → 법령명만 표시 (content 없음)
- 같은 법령의 여러 조문을 그룹핑

예시 출력:
```
## 관련 법령
[법령 1] 민법 (id: law_001)                    ← 벡터 매칭 (조문 포함)
  [제750조] 불법행위의 내용
  고의 또는 과실로 인한 위법행위로 타인에게 손해를 가한 자는...

[법령 2] 국가배상법 (id: law_002)               ← 키워드 only (법령명만)
```

---

## ~~Layer 6: BM25 FTS 조문 레벨~~ (불필요)

키워드 검색은 **법령 특정** 역할로 한정하고, 조문 특정은 벡터 검색에 위임하는 전략을 채택했으므로
fts_index의 조문 단위 확장은 불필요하다.

---

## 수정 대상 파일 요약

| # | 파일 | 작업 |
|---|------|------|
| 1 | `backend/app/models/law_article.py` | **신규** — ORM 모델 |
| 2 | `backend/app/models/__init__.py` | import/export 추가 |
| 3 | `backend/alembic/env.py` | import 추가 |
| 4 | `backend/alembic/versions/021_add_law_articles_table.py` | **신규** — 테이블 생성 |
| 5 | `backend/scripts/load_law_articles_data.py` | **신규** — 데이터 적재 |
| 6 | `backend/app/tools/vectorstore/lancedb.py` | `_extract_metadatas()` 수정 |
| 7 | `backend/app/services/rag/retrieval.py` | `_search_vector_ids()` + 조문 조회 함수 추가 |
| 8 | `backend/app/services/rag/format_utils.py` | `format_law_context()` 수정 |

재사용 기존 함수:
- `backend/scripts/ingest/types/law.py`: `_extract_article_body()` (L115), `_extract_ho()` (L89)

---

## 검증 계획

1. **마이그레이션**: `uv run alembic upgrade head` → `law_articles` 테이블 생성 확인
2. **데이터 적재**: `uv run python scripts/load_law_articles_data.py --verify` → 건수 + 키 형식 확인
3. **키 정합성**: LanceDB `article_number` 샘플 vs `law_articles.article_number` 일치 확인
4. **기존 데이터 무변경**: `law_documents.content`가 그대로 유지되는지 확인
5. **정적 검증**: `uv run ruff check backend/app/` + `uv run mypy backend/app/`
6. **메타데이터 전파**: LanceDB 검색 결과에 `article_number` 포함 확인
7. **조문 조회**: article_number 있는 법령 → `law_articles`에서 해당 조문만 반환
8. **키워드 only**: article_number 없는 법령 → content 조회 스킵, `law_name`만 참조
9. **LLM 컨텍스트**: `format_law_context()` 출력에서 조문 단위 표시
10. **롤백**: `alembic downgrade -1` → `law_articles` 테이블 삭제 확인

---

## 버전 이력

| 버전 | 날짜 | 변경 |
|------|------|------|
| v1 | 2026-03-01 | 초안 작성 |
| v2 | 2026-03-01 | LanceDB ↔ law_articles 키 정합성 원칙 추가, article_number 형식 명시 |
| v3 | 2026-03-01 | 데이터 중복 해소 (B안): law_documents.content NULL 처리, DOCUMENT_TABLE_REGISTRY 전환 |
| v4 | 2026-03-01 | pg_graph_service.py 체계도 원문 보기 전환 추가 (Layer 4-4) |
| v5 | 2026-03-01 | law_documents.content 유지로 전환 (pg_graph 영역 제외), 파일 수 10→8개 |
