# Plan: Vector DB + PostgreSQL 최적화

## 1. 개요

### 배경
법률 서비스 플랫폼의 데이터 인프라를 최적화하여 검색 품질을 향상하고 운영 비용을 절감한다.

### 현재 상태 (AS-IS)

| 항목 | 현재 상태 | 문제점 |
|------|----------|--------|
| **LanceDB** | 253,768 청크 (법령 118,922 + 판례 134,846) | 판례 full_reason 미포함, 하이브리드 검색 미구현 |
| **PostgreSQL** | 변호사만 DB 활용 (USE_DB_LAWYERS=true) | 법령/판례는 JSON→LanceDB만, PostgreSQL 미활용 |
| **임베딩 모델** | KURE-v1 (1024d, 2.3GB) 로컬 | 품질 검증 부족, 도메인 적합성 미평가 |
| **검색 파이프라인** | LanceDB 벡터 검색만 | FTS 미적용, 리랭킹은 구현됨 |
| **비용** | Solar API 무료 (~2026-03-02), Oracle Cloud Free 가능 | 무료 기간 종료 후 비용 급증 |

### 목표 상태 (TO-BE)

| 항목 | 목표 | 기대 효과 |
|------|------|----------|
| **LanceDB** | 판례 full_reason 포함 재임베딩, 하이브리드 검색 적용 | Recall@10 ≥ 0.8, Hit Rate ≥ 0.9 |
| **PostgreSQL** | 법령/판례 원본 조회를 PostgreSQL 기반으로 통합 | 일관된 데이터 접근 계층, JSON 파일 의존 제거 |
| **임베딩** | 기존 KURE-v1 유지, 청킹 전략 개선 | 검색 정확도 향상 |
| **검색** | Vector + FTS 하이브리드 검색 | 키워드 매칭 + 의미 검색 동시 지원 |
| **비용** | PostgreSQL + LanceDB 로컬 구성 유지 | 월 5,000원 이하 (Solar API 비용만) |

---

## 2. 현재 데이터 현황

### 2.1 원본 데이터

| 데이터 | 파일 | 크기 | 건수 |
|--------|------|------|------|
| 법령 | data/law_cleaned.json | 214 MB | 5,841건 |
| 판례 | data/precedents_cleaned.json | 591 MB | 65,107건 |
| 변호사 | data/lawyers_with_coords.json | 9.7 MB | 17,326건 |
| 법령 약칭 | data/lsAbrv.json | 591 KB | - |
| 인구 데이터 | data/population.json | 26 KB | - |

### 2.2 PostgreSQL 테이블 (현재)

| 테이블 | 건수 | 용도 | 활용 상태 |
|--------|------|------|----------|
| law_documents | 5,841 | 법령 원본 저장 | 데이터 적재됨, 서비스에서 미사용 |
| precedent_documents | 65,107 | 판례 원본 저장 | 데이터 적재됨, 서비스에서 미사용 |
| lawyers | 17,326 | 변호사 정보 | **활발히 사용 중** |
| legal_documents | - | 레거시 (ChromaDB 호환) | 미사용 |

### 2.3 LanceDB (현재)

| 항목 | 값 |
|------|-----|
| 테이블 | legal_chunks (통합) |
| 법령 청크 | 118,922개 (평균 20.37 청크/문서) |
| 판례 청크 | 134,846개 (평균 2.07 청크/문서) |
| 벡터 차원 | 1024 (KURE-v1) |
| 디스크 크기 | 1.6 GB |

---

## 3. 작업 범위

### Phase 1: PostgreSQL 통합 (비용 절감)

**목표**: 법령/판례 원본 조회를 PostgreSQL 기반으로 전환

| 작업 | 설명 | 난이도 |
|------|------|--------|
| 1-1. Feature Flag 추가 | `USE_DB_LAWS`, `USE_DB_PRECEDENTS` config에 추가 | 낮음 |
| 1-2. DB 서비스 함수 작성 | `law_db_service.py`, `precedent_db_service.py` 작성 | 중간 |
| 1-3. 라우터 분기 처리 | 모듈 라우터에서 feature flag 기반 분기 | 낮음 |
| 1-4. RAG 파이프라인 연동 | LanceDB source_id → PostgreSQL 원본 조회 통합 | 중간 |
| 1-5. JSON 서비스 유지 | 기존 JSON 기반 서비스 삭제하지 않음 (롤백 보장) | - |

**결과**: JSON 파일 직접 읽기 제거 → PostgreSQL 단일 소스로 통합

### Phase 2: 임베딩 품질 개선

**목표**: 검색 성능 지표 달성 (Recall@10 ≥ 0.8)

| 작업 | 설명 | 난이도 |
|------|------|--------|
| 2-1. 판례 full_reason 포함 여부 결정 | full_reason 포함 시 청크 수 대폭 증가 (예상 3~5배) → 비용/성능 트레이드오프 분석 | 중간 |
| 2-2. 청킹 전략 개선 | 법령: 현행 조문 단위 유지, 판례: 오버랩 비율 조정 (10%→15%) | 낮음 |
| 2-3. 메타데이터 강화 | 법령 유형별 가중치, 판례 법원 레벨별 가중치 추가 | 중간 |
| 2-4. 평가 데이터셋으로 검증 | evaluation/ 모듈의 기존 평가 시스템 활용 | 낮음 |

### Phase 3: 하이브리드 검색 구현

**목표**: 벡터 검색 + FTS(Full-Text Search) 결합

| 작업 | 설명 | 난이도 |
|------|------|--------|
| 3-1. PostgreSQL FTS 인덱스 생성 | law_documents, precedent_documents에 tsvector 컬럼 추가 | 중간 |
| 3-2. FTS 검색 서비스 작성 | pg_trgm 또는 mecab 기반 한국어 FTS | 높음 |
| 3-3. Reciprocal Rank Fusion | 벡터 검색 + FTS 결과 병합 (RRF 알고리즘) | 중간 |
| 3-4. 가중치 튜닝 | 벡터:FTS 비율 최적화 (기본 0.7:0.3) | 낮음 |

### Phase 4: 레거시 정리

| 작업 | 설명 | 난이도 |
|------|------|--------|
| 4-1. 레거시 테이블 정리 | legal_documents, legal_references 마이그레이션 제거 검토 | 낮음 |
| 4-2. ChromaDB/Qdrant 코드 제거 | VECTOR_DB config에서 lancedb 외 옵션 제거 | 낮음 |
| 4-3. 문서 업데이트 | CLAUDE.md, vectordb_design.md 등 갱신 | 낮음 |

---

## 4. 우선순위 및 의존관계

```
Phase 1 (PostgreSQL 통합)
  ├── 1-1 Feature Flag ──→ 1-2 DB 서비스 ──→ 1-3 라우터 분기
  │                                           │
  │                                           ▼
  │                                      1-4 RAG 연동
  │
  ▼
Phase 2 (임베딩 개선) ── Phase 1 완료 후 시작
  ├── 2-1 full_reason 분석 ──→ 2-2 청킹 개선
  │                              │
  │                              ▼
  └── 2-3 메타데이터 ──→ 2-4 평가 검증

  ▼
Phase 3 (하이브리드 검색) ── Phase 1, 2 완료 후 시작
  ├── 3-1 FTS 인덱스 ──→ 3-2 FTS 서비스
  │                         │
  │                         ▼
  └──────────────────→ 3-3 RRF 병합 ──→ 3-4 가중치 튜닝

Phase 4 (레거시 정리) ── 언제든 가능
```

**권장 실행 순서**: Phase 1 → Phase 2 → Phase 3 → Phase 4

---

## 5. 수정 대상 파일

### Phase 1

| 파일 | 변경 유형 |
|------|----------|
| `backend/app/core/config.py` | 수정 - USE_DB_LAWS, USE_DB_PRECEDENTS 추가 |
| `backend/app/services/service_function/law_db_service.py` | 신규 |
| `backend/app/services/service_function/precedent_db_service.py` | 신규 |
| `backend/app/modules/case_precedent/router/__init__.py` | 수정 - feature flag 분기 |
| `backend/app/services/rag/retrieval.py` | 수정 - PostgreSQL 조회 통합 |

### Phase 2

| 파일 | 변경 유형 |
|------|----------|
| `backend/scripts/runpod_lancedb_embeddings.py` | 수정 - 청킹 전략 개선 |
| `backend/evaluation/datasets/` | 수정 - 평가 데이터셋 보완 |

### Phase 3

| 파일 | 변경 유형 |
|------|----------|
| `backend/alembic/versions/005_add_fts_indices.py` | 신규 - FTS 마이그레이션 |
| `backend/app/services/rag/fts_search.py` | 신규 - FTS 검색 서비스 |
| `backend/app/services/rag/hybrid_search.py` | 신규 - RRF 병합 |
| `backend/app/services/rag/pipeline.py` | 수정 - 하이브리드 파이프라인 |

---

## 6. 리스크 및 제약사항

| 리스크 | 영향 | 대응 |
|--------|------|------|
| full_reason 포함 시 임베딩 크기 3~5배 증가 | 디스크 5~8GB, 처리 시간 증가 | GPU 환경(RunPod) 필요, 선택적 포함 |
| 한국어 FTS 토큰화 품질 | 검색 정확도 저하 | pg_trgm 먼저 적용, 이후 mecab 고려 |
| PostgreSQL 전환 시 성능 차이 | 응답 시간 변화 | Feature flag로 점진 전환, 벤치마크 |
| Solar API 무료 종료 (2026-03-02) | 비용 급증 | 로컬 LLM 대안 또는 OpenAI gpt-4o-mini 전환 |
| 재임베딩 시 서비스 중단 | 검색 불가 기간 | 별도 테이블에 생성 후 스왑 |

---

## 7. 성공 지표

| 지표 | 현재 | 목표 | 측정 방법 |
|------|------|------|----------|
| Recall@10 | 미측정 | ≥ 0.8 | evaluation 모듈 |
| MRR | 미측정 | ≥ 0.7 | evaluation 모듈 |
| Hit Rate | 미측정 | ≥ 0.9 | evaluation 모듈 |
| 검색 응답 시간 | 미측정 | ≤ 2초 | curl 벤치마크 |
| JSON 파일 의존 | 3개 모듈 | 0개 모듈 | 코드 검색 |
| 월 운영 비용 | ~5,000원 | ≤ 5,000원 | API 사용량 추적 |

---

## 8. 참고 문서

- `docs/vectordb_design.md` - 벡터 DB 설계
- `docs/DB_ARCHITECTURE.md` - DB 아키텍처
- `docs/DEPLOYMENT_COST_ESTIMATION.md` - 배포 비용 추정
- `docs/EMBEDDING_DEV_LOG_20260129.md` - 임베딩 개발 로그
- `backend/evaluation/CLAUDE.md` - RAG 평가 시스템
