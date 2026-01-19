# 법률 데이터 DB 구현 계획서

## 1. 개요

법률 데이터를 PostgreSQL과 ChromaDB에 저장하여 챗봇(RAG)과 로스쿨 학습 기능을 지원합니다.

### 1.1 데이터 현황

| 데이터 | 파일 | 레코드 수 | 크기 |
|--------|------|-----------|------|
| 판례 | `precedents_full*.json` | 29,120개 | ~1GB |
| 헌재결정례 | `constitutional_full.json` | 36,781개 | 24MB |
| 행정심판례 | `administation_full.json` | 34,258개 | 444MB |
| 법령해석례 | `legislation_full.json` | 8,597개 | 80MB |
| **총합** | | **108,756개** | **~1.5GB** |

### 1.2 데이터 품질

| 데이터 | 핵심 필드 채워진 비율 | RAG 사용 가능 여부 |
|--------|----------------------|-------------------|
| 판례 | 판례내용 100%, 판결요지 ~90% | ✅ 우수 |
| 헌재결정례 | 판시사항 97.9%, 결정요지 97.9%, **전문 0%** | ⚠️ 제한적 |
| 행정심판례 | 이유 100%, 주문 100% | ✅ 우수 |
| 법령해석례 | 질의요지 100%, 회답 100%, 이유 100% | ✅ 우수 |

---

## 2. 스키마 설계

### 2.1 PostgreSQL 스키마

#### 스키마 검증 결과

| 검증 항목 | 결과 | 비고 |
|----------|------|------|
| 필드 매핑 완전성 | ⚠️ 수정 필요 | 행정심판례 `주문`, `청구취지` 필드 추가 필요 |
| 통합 테이블 vs 분리 | ✅ 통합 적합 | 필드 70% 이상 공통, 조인 불필요 |
| 인덱스 전략 | ✅ 적절 | doc_type, case_number, decision_date |
| 전문 검색 | ⚠️ 개선 필요 | 한글 형태소 분석기 고려 |
| 원본 보존 | ✅ JSONB로 보존 | 필드 누락 방지 |

#### 수정된 스키마

```sql
-- 법률 문서 통합 테이블
CREATE TABLE legal_documents (
    id SERIAL PRIMARY KEY,
    doc_type VARCHAR(20) NOT NULL,  -- 'precedent', 'constitutional', 'administration', 'legislation'
    serial_number VARCHAR(50) NOT NULL,  -- 원본 일련번호

    -- 공통 필드
    case_name TEXT,                 -- 사건명/안건명
    case_number VARCHAR(100),       -- 사건번호/안건번호
    decision_date DATE,             -- 선고일/의결일/종국일

    -- 기관 정보
    court_name VARCHAR(100),        -- 법원명/재결청/해석기관
    court_type VARCHAR(50),         -- 법원종류/재결례유형
    case_type VARCHAR(50),          -- 사건종류 (민사/형사/헌마 등)

    -- 주요 내용 (RAG 임베딩 대상)
    summary TEXT,                   -- 판시사항/결정요지/질의요지/주문
    reasoning TEXT,                 -- 판결요지/이유/회답
    full_text TEXT,                 -- 판례내용/전문

    -- 추가 필드 (행정심판례)
    claim TEXT,                     -- 청구취지

    -- 참조 정보
    reference_articles TEXT,        -- 참조조문
    reference_cases TEXT,           -- 참조판례

    -- 메타데이터
    raw_data JSONB NOT NULL,        -- 원본 데이터 전체 보존
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    -- 제약조건
    CONSTRAINT unique_doc UNIQUE (doc_type, serial_number)
);

-- 인덱스
CREATE INDEX idx_legal_docs_type ON legal_documents(doc_type);
CREATE INDEX idx_legal_docs_case_number ON legal_documents(case_number);
CREATE INDEX idx_legal_docs_decision_date ON legal_documents(decision_date);
CREATE INDEX idx_legal_docs_court ON legal_documents(court_name);
CREATE INDEX idx_legal_docs_case_type ON legal_documents(case_type);

-- GIN 인덱스 (JSONB 검색용)
CREATE INDEX idx_legal_docs_raw ON legal_documents USING gin(raw_data);

-- 전문 검색 인덱스 (한글)
CREATE INDEX idx_legal_docs_summary_search ON legal_documents
    USING gin(to_tsvector('simple', coalesce(summary, '')));
CREATE INDEX idx_legal_docs_reasoning_search ON legal_documents
    USING gin(to_tsvector('simple', coalesce(reasoning, '')));
```

### 2.2 ChromaDB 스키마

```python
# 컬렉션 설정
COLLECTION_NAME = "legal_documents"

# 임베딩 대상 텍스트 생성 전략
def get_embedding_text(doc: dict) -> str:
    """RAG 검색용 텍스트 생성"""
    parts = []

    if doc.get('case_name'):
        parts.append(f"사건명: {doc['case_name']}")
    if doc.get('summary'):
        parts.append(f"요지: {doc['summary']}")
    if doc.get('reasoning'):
        # 너무 길면 앞부분만 (토큰 제한)
        reasoning = doc['reasoning'][:2000] if len(doc['reasoning']) > 2000 else doc['reasoning']
        parts.append(f"내용: {reasoning}")

    return "\n".join(parts)

# 메타데이터 (필터링용)
metadata = {
    "doc_type": "precedent",
    "case_number": "84나3990",
    "court_name": "서울고법",
    "case_type": "민사",
    "decision_date": "1986-01-15",
    "pg_id": 123  # PostgreSQL 연결
}
```

### 2.3 필드 매핑 테이블

| 통합 필드 | 판례 | 헌재결정례 | 행정심판례 | 법령해석례 |
|----------|------|-----------|-----------|-----------|
| serial_number | 판례정보일련번호 | 헌재결정례일련번호 | 행정심판례일련번호 | 법령해석례일련번호 |
| case_name | 사건명 | 사건명 | 사건명 | 안건명 |
| case_number | 사건번호 | 사건번호 | 사건번호 | 안건번호 |
| decision_date | 선고일자 | 종국일자 | 의결일자 | 등록일시 |
| court_name | 법원명 | - | 재결청 | 해석기관명 |
| case_type | 사건종류명 | 판시사항* | 재결례유형명 | - |
| summary | 판시사항 | 판시사항 | 주문 | 질의요지 |
| reasoning | 판결요지 | 결정요지 | 이유 | 회답+이유 |
| full_text | 판례내용 | 전문 | - | - |
| claim | - | - | 청구취지 | - |
| reference_articles | 참조조문 | 참조조문+심판대상조문 | - | - |
| reference_cases | 참조판례 | 참조판례 | - | - |

---

## 3. 구현 계획

### 3.1 단계별 작업

#### Phase 1: 환경 설정 (Day 1)

| 작업 | 파일 | 설명 |
|------|------|------|
| 1.1 | `backend/app/common/database.py` | SQLAlchemy 설정, 세션 관리 |
| 1.2 | `backend/app/common/chromadb.py` | ChromaDB 클라이언트 설정 |
| 1.3 | `backend/app/models/legal_document.py` | SQLAlchemy 모델 정의 |
| 1.4 | `backend/alembic/` | DB 마이그레이션 설정 |

#### Phase 2: 데이터 로드 스크립트 (Day 2-3)

| 작업 | 파일 | 설명 |
|------|------|------|
| 2.1 | `backend/scripts/load_legal_data.py` | JSON → PostgreSQL 로드 |
| 2.2 | `backend/scripts/create_embeddings.py` | PostgreSQL → ChromaDB 임베딩 |
| 2.3 | `backend/scripts/validate_data.py` | 데이터 무결성 검증 |

#### Phase 3: API 구현 (Day 4-5)

| 작업 | 파일 | 설명 |
|------|------|------|
| 3.1 | `backend/app/modules/case_precedent/` | 판례 검색 API (PostgreSQL) |
| 3.2 | `backend/app/common/rag.py` | RAG 검색 서비스 (ChromaDB) |
| 3.3 | `backend/app/modules/law_study/` | 학습 기능 API |

### 3.2 의존성 추가

```toml
# backend/pyproject.toml에 추가
[project.dependencies]
sqlalchemy = "^2.0"
alembic = "^1.13"
asyncpg = "^0.29"          # PostgreSQL async 드라이버
chromadb = "^0.4"
openai = "^1.0"            # 임베딩 생성용
tiktoken = "^0.5"          # 토큰 카운팅
```

### 3.3 환경 변수 추가

```env
# backend/.env에 추가
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/lawdb

# ChromaDB
CHROMA_PERSIST_DIR=./data/chroma
CHROMA_COLLECTION_NAME=legal_documents

# 임베딩 설정
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_BATCH_SIZE=100
```

---

## 4. 데이터 로드 스크립트 설계

### 4.1 load_legal_data.py

```python
"""
법률 데이터 PostgreSQL 로드 스크립트

사용법:
    uv run python scripts/load_legal_data.py --type all
    uv run python scripts/load_legal_data.py --type precedent
    uv run python scripts/load_legal_data.py --type constitutional
"""

# 주요 기능:
# 1. JSON 파일 스트리밍 로드 (메모리 효율)
# 2. 배치 INSERT (1000건씩)
# 3. 중복 체크 (UPSERT)
# 4. 진행률 표시
# 5. 에러 로깅

# 예상 소요 시간:
# - 판례 29,120건: ~5분
# - 헌재 36,781건: ~5분
# - 행정 34,258건: ~5분
# - 법령 8,597건: ~1분
# - 총합: ~20분
```

### 4.2 create_embeddings.py

```python
"""
ChromaDB 임베딩 생성 스크립트

사용법:
    uv run python scripts/create_embeddings.py --batch-size 100
    uv run python scripts/create_embeddings.py --doc-type precedent
"""

# 주요 기능:
# 1. PostgreSQL에서 문서 조회
# 2. 임베딩 텍스트 생성 (summary + reasoning)
# 3. OpenAI 임베딩 API 호출 (배치)
# 4. ChromaDB 저장
# 5. 진행률 및 비용 추정 표시

# 비용 추정 (text-embedding-3-small):
# - 평균 문서 길이: ~2000자 (~500 토큰)
# - 108,756건 × 500토큰 = 54M 토큰
# - 비용: $0.02/1M 토큰 = ~$1.1

# 예상 소요 시간:
# - 배치 100건, 초당 ~50건
# - 108,756건 / 50 = ~36분
```

---

## 5. 리스크 및 대응

| 리스크 | 영향 | 대응 방안 |
|--------|------|----------|
| 헌재결정례 `전문` 없음 | RAG 품질 저하 | `판시사항`+`결정요지` 조합 사용 |
| 대용량 파일 메모리 | OOM 에러 | ijson 스트리밍 파서 사용 |
| OpenAI API 비용 | 예산 초과 | 로컬 임베딩 모델 대안 (sentence-transformers) |
| 임베딩 토큰 초과 | 잘림 발생 | 8191 토큰 이내로 truncate |
| PostgreSQL 부하 | 느린 INSERT | COPY 명령 또는 배치 처리 |

---

## 6. 검증 체크리스트

### 6.1 데이터 로드 후

- [ ] 총 레코드 수 일치 확인 (108,756개)
- [ ] 각 doc_type별 카운트 확인
- [ ] NULL 필드 비율 확인
- [ ] 샘플 데이터 조회 테스트

### 6.2 임베딩 생성 후

- [ ] ChromaDB 컬렉션 문서 수 확인
- [ ] 샘플 쿼리 테스트 (유사 판례 검색)
- [ ] PostgreSQL ID ↔ ChromaDB ID 매핑 확인

### 6.3 API 테스트

- [ ] 키워드 검색 동작 확인
- [ ] RAG 검색 동작 확인
- [ ] 필터 조합 테스트 (doc_type, court_name, date range)

---

## 7. 다음 단계

1. **Phase 1 시작**: 환경 설정 및 모델 정의
2. **데이터 로드 스크립트 구현**
3. **임베딩 전략 결정**: OpenAI vs 로컬 모델
4. **API 구현 및 테스트**
