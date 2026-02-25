# PostgreSQL + FTS 코드 분석 및 개선 보고서

> 일시: 2026-02-26
> 대상: `retrieval.py`, `keyword_search.py`, `database.py`, `db_writer.py`
> 범위: 전문가 패널 분석 → 버그 수정 + 성능 개선

---

## 1. 분석 개요

PostgreSQL FTS 기반 하이브리드 검색(벡터 + 키워드) 코드를 전문가 패널 방식으로 분석하여 **10개 이슈**를 식별하고, 그 중 **실질적인 5건을 수정** 적용함.

> **추가 발견 (2차 검증)**: sync 경로(`fetch_document_contents`, `fetch_ai_summaries`)에 접두사 라우팅이 미적용되어 있었음. async 경로에만 적용된 상태를 발견하여 sync 함수도 동일하게 `_for_type` 헬퍼를 사용하도록 수정 완료.

### 분석 대상 파일

| 파일 | 역할 |
|------|------|
| `app/services/rag/retrieval.py` | 하이브리드 검색 + 원문/요약문 배치 조회 |
| `app/services/rag/keyword_search.py` | PostgreSQL tsvector 기반 키워드 검색 |
| `app/core/database.py` | SQLAlchemy 엔진 + 세션 팩토리 |
| `scripts/ingest/db_writer.py` | JSON → PostgreSQL ORM + FTS 동시 적재 |

---

## 2. 수정 사항 요약

| # | 심각도 | 파일 | 이슈 | 수정 내용 |
|---|--------|------|------|----------|
| 1 | **버그** | `retrieval.py` | DOCUMENT_TABLE_REGISTRY 누락 | `dec_media` + `자치법규` 2개 타입 추가 |
| 2 | **성능** | `keyword_search.py` | `is_fts_available_sync()` 매 요청 COUNT(*) | 모듈 레벨 캐시 적용 |
| 3 | **성능+버그** | `retrieval.py` | 위원회결정례 11테이블 순차 스캔 + 접두사 불일치 | 접두사 기반 직접 라우팅 |
| 4 | **운영** | `database.py` | sync pool_size=5 부족 | pool_size=10으로 증가 |
| 5 | **운영** | `db_writer.py` | ORM↔FTS 건수 불일치 무감지 | verify_db() 경고 로깅 추가 |

---

## 3. 상세 설명

### 3.1 DOCUMENT_TABLE_REGISTRY 누락 (버그)

**문제**: `retrieval.py`의 `DOCUMENT_TABLE_REGISTRY`에 `dec_media_documents`(방송미디어통신위원회)와 `자치법규`(local_ordinance_documents) 2개 타입이 누락되어 있었음.

**영향**: FTS/벡터 검색에서 해당 타입의 문서가 검색되더라도, 원문(`fetch_document_contents`) 및 요약문(`fetch_ai_summaries`) 조회 시 매핑을 찾지 못해 **content가 빈 문자열로 반환**됨. 검색 결과는 나오지만 내용이 없는 상태.

**수정**:
```python
DOCUMENT_TABLE_REGISTRY = {
    # ... 기존 타입들 ...
    "위원회결정례": [
        # ... 10개 기존 테이블 ...
        TableConfig("dec_media_documents", "serial_number", ("ruling",)),  # 추가
    ],
    "자치법규": [  # 추가
        TableConfig(
            "local_ordinance_documents", "ordinance_id", ("content", "overall_summary")
        ),
    ],
}
```

---

### 3.2 is_fts_available_sync() COUNT(*) 캐싱 (성능)

**문제**: 매 검색 요청마다 `fts_index` 테이블(57만 행)에 `SELECT COUNT(*) FROM fts_index LIMIT 1`을 실행하여 FTS 가용 여부를 확인.

**왜 느린가**: PostgreSQL은 MVCC(Multi-Version Concurrency Control) 아키텍처로 인해 `COUNT(*)`가 테이블 전체를 스캔(heap scan)해야 함. MySQL의 메타데이터 기반 즉시 반환과 다름.

**영향 규모**:
- 57만 행 테이블 → 매 요청마다 수십~수백 ms 소요
- RAG 파이프라인의 `asyncio.gather` 병렬 검색에서 각 브랜치마다 호출 → 요청당 2~4회 반복

**수정**: 모듈 레벨 캐시 변수 `_fts_available_cache`를 도입하여 서버 프로세스당 1회만 DB 조회.

```python
# keyword_search.py
_fts_available_cache: bool | None = None

def is_fts_available_sync() -> bool:
    global _fts_available_cache
    if _fts_available_cache is not None:
        return _fts_available_cache  # 캐시 히트 → DB 조회 없음
    try:
        with sync_session_factory() as session:
            result = session.execute(
                select(func.count()).select_from(FtsIndex).limit(1)
            )
            count = result.scalar_one()
            _fts_available_cache = count > 0
            return _fts_available_cache
    except Exception:
        return False
```

**트레이드오프**: 인제스트 후 서버 재시작 없이는 캐시가 갱신되지 않음. 하지만 인제스트는 서버 외부에서 실행되고, 이후 서버를 재시작하는 것이 일반적 워크플로우이므로 문제 없음.

---

### 3.3 위원회결정례 11테이블 순차 스캔 → 접두사 라우팅 (성능)

**문제**: `위원회결정례` 11개 타입은 `data_type='위원회결정례'`를 공유하지만, 실제로는 각각 다른 ORM 테이블(dec_labor_documents, dec_privacy_documents 등)에 저장됨. 기존 코드는 11개 테이블을 **순차적으로 `ANY(:ids)` 쿼리**하여 어느 테이블에 있는지 찾음.

**기존 흐름** (위원회결정례 10건 조회 시):
```
dec_privacy_documents → 0건 발견
dec_labor_documents → 3건 발견
dec_human_rights_documents → 0건 발견
... (11테이블 모두 스캔)
```

**숨겨진 정확성 버그**: FTS의 source_id는 `dec_labor:12345` (접두사 포함) 형식이지만, ORM 테이블의 `serial_number`는 `12345` (접두사 없음). 기존 코드는 접두사를 strip하지 않고 그대로 `WHERE serial_number = ANY(...)` 쿼리에 전달하여 **FTS 결과의 원문이 항상 매칭 실패**함.

**수정**: 접두사 기반 직접 라우팅 + 접두사 strip 처리.

```python
# 접두사 → TableConfig 직접 매핑 (서버 시작 시 1회 생성)
_DEC_TABLE_BY_PREFIX: dict[str, TableConfig] = {
    tc.table_name.removesuffix("_documents"): tc
    for tc in DOCUMENT_TABLE_REGISTRY.get("위원회결정례", [])
}
# {"dec_labor": TableConfig("dec_labor_documents", ...), ...}

def _group_dec_source_ids(source_ids):
    """dec_labor:12345 → dec_labor_documents 직접 라우팅"""
    routed = {}   # {TableConfig: {original_sid: serial_number}}
    unrouted = []  # 접두사 없는 sid (벡터 검색 결과)
    for sid in source_ids:
        if ":" in sid:
            prefix, serial = sid.split(":", 1)
            tc = _DEC_TABLE_BY_PREFIX.get(prefix)
            if tc:
                routed.setdefault(tc, {})[sid] = serial
            else:
                unrouted.append(sid)
        else:
            unrouted.append(sid)
    return routed, unrouted
```

**적용 범위**: sync + async 양쪽 모든 경로에 적용.

| 함수 | 경로 | 라우팅 |
|------|------|--------|
| `fetch_document_contents` (sync) | `_fetch_contents_for_type` | 적용 |
| `fetch_ai_summaries` (sync) | `_fetch_summaries_for_type` | 적용 |
| `fetch_document_contents_async` | `_fetch_contents_for_type` | 적용 |
| `fetch_ai_summaries_async` | `_fetch_summaries_for_type` | 적용 |

> 초기 수정 시 async 경로(`_for_type` 헬퍼)에만 적용하고 sync 함수는 기존 인라인 로직을 유지한 채 남겨두었음. 2차 검증에서 발견하여 sync 함수도 `_for_type` 헬퍼를 호출하도록 리팩토링 완료.

**개선 효과**:

| 항목 | Before | After |
|------|--------|-------|
| 쿼리 수 (FTS 결과) | 11개 테이블 순차 | 1~3개 테이블 직접 |
| 정확성 (FTS) | 접두사 불일치로 매칭 실패 | 접두사 strip 후 정상 매칭 |
| 쿼리 수 (벡터 결과) | 11개 (변동 없음) | 11개 (접두사 없으므로 기존 방식 유지) |

---

### 3.4 sync_engine pool_size 증가 (운영)

**문제**: `database.py`의 `sync_engine`이 `pool_size=5, max_overflow=10`으로 설정되어 있었음. RAG 파이프라인은 `asyncio.gather`로 벡터+FTS 검색을 병렬 실행하는데, 각 검색 브랜치가 `sync_session_factory()`로 동기 세션을 사용.

**영향 시나리오**:
```
사용자 요청 → asyncio.gather(
    벡터 검색 (sync 세션 1개),
    FTS 검색 (sync 세션 1개),
    보충 검색 벡터 (sync 세션 1개),
    보충 검색 FTS (sync 세션 1개),
)
→ 동시에 4개 sync 세션 사용
```

단일 사용자는 문제없지만, **동시 2~3명 요청** 시 pool_size=5가 부족할 수 있음.

**수정**: `pool_size=10, max_overflow=10` (최대 20개 동시 연결 허용)

```python
sync_engine = create_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_size=10,       # 5 → 10
    max_overflow=10,
    pool_pre_ping=True,
)
```

---

### 3.5 verify_db() ORM↔FTS 건수 불일치 경고 (운영)

**문제**: `db_writer.py`의 `verify_db()` 함수가 ORM 건수와 FTS 건수를 각각 출력하지만, 두 값이 다를 때 경고를 발생시키지 않음. 인제스트 실패로 인한 불일치를 놓칠 수 있음.

**수정**:
```python
if result["orm_count"] != result["fts_count"]:
    diff = result["orm_count"] - result["fts_count"]
    logger.warning(
        "  ⚠️  ORM↔FTS 건수 불일치: ORM %d건, FTS %d건 (차이: %+d)",
        result["orm_count"], result["fts_count"], diff,
    )
```

---

## 4. 미적용 개선 사항 (향후 검토)

| # | 심각도 | 이슈 | 설명 |
|---|--------|------|------|
| 1 | 중간 | OR fallback 광범위 스캔 | 개념 AND 결과 <5건 시 전체 OR fallback → 고빈도 토큰으로 느린 스캔 가능 |
| 2 | 중간 | focus 모드 sync 세션 누적 | `asyncio.gather`에서 4+ 동시 `to_thread` → 세션 풀 압박 |
| 3 | 낮음 | fts_builder OFFSET 페이지네이션 | 대량 데이터에서 OFFSET이 커지면 느려짐 → keyset 방식 권장 |
| 4 | 낮음 | `datetime.utcnow` deprecated | Python 3.12+ 권고: `datetime.now(timezone.utc)` (20+ 파일 해당) |
| 5 | 낮음 | dec_* reset 로직 중복 | db_writer의 dec_* 삭제 보호 로직을 shared 헬퍼로 추출 가능 |

---

## 5. 아키텍처 요약: 인제스트 파이프라인

### Strategy Pattern 기반 Config-driven 설계

`db_writer.py`는 컬럼 구조를 전혀 모름. `IngestConfig`의 **함수 포인터**를 호출할 뿐:

```
IngestConfig {
    orm_factory_fn(item) → ORM 인스턴스    # 각 타입이 자체 정의
    fulltext_fn(item) → 검색용 텍스트      # 각 타입이 자체 정의
    fts_metadata_fn(item) → FTS 메타데이터  # _dec_comm_common.py 자동 생성
}
```

21개 타입이 각자 `scripts/ingest/types/` 하위에 설정을 정의하면, `db_writer`가 일괄 처리:

```
JSON → orm_factory_fn() → PostgreSQL ORM upsert
     → fulltext_fn() + MeCab → tsvector → fts_index upsert
```

### dec_* 11개 위원회결정례의 source_id 이중 형식

| 저장소 | source_id 형식 | 예시 |
|--------|---------------|------|
| `fts_index` | `{타입접두사}:{serial_number}` | `dec_labor:12345` |
| `LanceDB` (벡터) | `{serial_number}` | `12345` |
| `ORM 테이블` | `serial_number` 컬럼 | `12345` |

이 이중 형식은 fts_index의 composite PK `(source_id, data_type)` 충돌 방지를 위해 도입됨 (11개 타입이 `data_type='위원회결정례'`를 공유하므로 serial_number만으로는 PK 충돌 발생).

---

## 6. 검색 흐름 요약

```
사용자 쿼리
    │
    ├── 벡터 검색 (LanceDB) ──────────┐
    │   └── source_id + similarity    │
    │                                  ├── RRF 병합
    ├── FTS 검색 (PostgreSQL) ────────┘
    │   └── source_id + fts_rank
    │
    ▼
source_id 리스트
    │
    ├── ai_summary 조회 (PostgreSQL) → 리랭킹
    │
    ├── 원문 조회 (PostgreSQL) ← DOCUMENT_TABLE_REGISTRY
    │
    ▼
최종 결과 (content + metadata + score)
```
