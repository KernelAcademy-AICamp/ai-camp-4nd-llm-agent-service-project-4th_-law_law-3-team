# 법률 데이터 DB 마이그레이션 작업 보고서

**작업일시**: 2026-01-20
**작업시간**: 약 2시간 30분 (임베딩 생성 포함)

---

## 1. 작업 목적

기존 단일 테이블 구조에서 3-테이블 구조로 데이터베이스 아키텍처를 개선하고, 청크 기반 RAG 검색을 위한 Vector DB를 재구성.

---

## 2. 아키텍처 변경 사항

### 2.1 기존 구조
```
PostgreSQL: legal_documents (단일 테이블)
ChromaDB: 문서 단위 임베딩 (doc_id 없음)
```

### 2.2 새로운 구조
```
PostgreSQL:
├── legal_documents (통합 문서 테이블) - source 컬럼 추가
├── laws (법령 테이블) - 신규
└── legal_references (참조 데이터 테이블) - 신규

ChromaDB:
└── 청크 기반 임베딩 + doc_id 포인터
```

---

## 3. 수정/생성된 파일

### 3.1 모델 파일
| 파일 | 변경 내용 |
|------|----------|
| `app/models/legal_document.py` | `source` 컬럼 추가, `COMMITTEE` DocType 추가, `from_committee()` 메서드 추가 |
| `app/models/law.py` | 신규 생성 - 법령 데이터 모델 |
| `app/models/legal_reference.py` | 신규 생성 - 조약/행정규칙/법률용어 모델 |
| `app/models/__init__.py` | 새 모델 export 추가 |

### 3.2 마이그레이션 파일
| 파일 | 내용 |
|------|------|
| `alembic/versions/002_add_laws_and_references_tables.py` | laws, legal_references 테이블 생성, legal_documents에 source 컬럼 추가 |

### 3.3 스크립트 파일
| 파일 | 변경 내용 |
|------|----------|
| `scripts/load_legal_data.py` | 9개 데이터 타입 지원, 11개 위원회 소스 처리 |
| `scripts/create_embeddings.py` | 청크 기반 임베딩 생성 (500자 청크, 50자 오버랩) |
| `scripts/validate_data.py` | 3개 테이블 검증, 청크 기반 일관성 검증 |
| `scripts/backup_data.py` | 신규 생성 - 백업 스크립트 |

---

## 4. 데이터 현황

### 4.1 PostgreSQL 최종 상태

| 테이블 | 데이터 유형 | 건수 |
|--------|------------|------|
| **legal_documents** | | **191,728** |
| | precedent/precedents | 92,264 |
| | constitutional/constitutional | 36,781 |
| | administration/administration | 34,258 |
| | legislation/legislation | 8,597 |
| | committee/ftc | 8,029 |
| | committee/ppc | 3,889 |
| | committee/nhrck | 3,732 |
| | committee/iaciac | 934 |
| | committee/kcc | 811 |
| | committee/fsc | 663 |
| | committee/sfc | 636 |
| | committee/acrc | 635 |
| | committee/ecc | 358 |
| | committee/eiac | 118 |
| | committee/oclt | 23 |
| **laws** | | **5,572** |
| | 법률 | 1,723 |
| | 대통령령 | 1,967 |
| | 기타 (부령, 규칙 등) | 1,882 |
| **legal_references** | | **62,008** |
| | law_term (법률용어) | 36,797 |
| | admin_rule (행정규칙) | 21,622 |
| | treaty (조약) | 3,589 |
| **총계** | | **259,308** |

### 4.2 ChromaDB (Vector DB) 최종 상태

| 문서 유형 | 청크 수 | 문서 수 |
|-----------|---------|---------|
| precedent | 166,212 | 92,167 |
| constitutional | 36,015 | 36,015 |
| administration | 260,173 | 34,258 |
| legislation | 81,899 | 8,597 |
| committee | 98,796 | 13,978 |
| **총계** | **643,095** | **185,015** |

### 4.3 청킹 설정
```python
CHUNK_CONFIG = {
    "chunk_size": 500,      # 문자 수
    "chunk_overlap": 50,    # 오버랩
    "min_chunk_size": 100,  # 최소 청크 크기
}
```

---

## 5. 백업 정보

### 5.1 백업 위치
```
/Users/gimjuhyeong/dev/law-3-team/data/backup_20260120/
├── legal_documents_backup.json  (34.9 MB) - PostgreSQL 데이터
└── chroma_backup/               (1.6 GB)  - ChromaDB 데이터
```

### 5.2 백업 내용
- **PostgreSQL**: 기존 171,900건의 legal_documents 레코드 (JSON 형식)
- **ChromaDB**: 기존 171,134개의 임베딩 (디렉토리 복사)

---

## 6. 검색 흐름

```
┌─────────────────────────────────────────────────────────────┐
│                      RAG 검색 흐름                           │
├─────────────────────────────────────────────────────────────┤
│  1. 사용자 질문                                              │
│  2. Vector DB에서 유사 청크 검색 (ANN)                       │
│  3. 청크의 doc_id로 PostgreSQL 원문 조회                     │
│  4. 원문 + 메타데이터로 답변 생성                            │
│  5. 출처 표기                                                │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. 명령어 참조

### 7.1 데이터 로드
```bash
# 개별 타입 로드
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

### 7.2 임베딩 생성
```bash
# 전체 생성 (리셋 포함)
uv run python scripts/create_embeddings.py --type all --reset

# 개별 타입
uv run python scripts/create_embeddings.py --type precedent

# 통계 확인
uv run python scripts/create_embeddings.py --stats
```

### 7.3 데이터 검증
```bash
# 전체 검증
uv run python scripts/validate_data.py

# PostgreSQL만
uv run python scripts/validate_data.py --pg-only

# ChromaDB만
uv run python scripts/validate_data.py --chroma-only

# 불일치 수정
uv run python scripts/validate_data.py --fix
```

### 7.4 마이그레이션
```bash
# 마이그레이션 실행
uv run alembic upgrade head

# 롤백
uv run alembic downgrade -1
```

---

## 8. 참고 사항

1. **임베딩 스킵**: summary와 reasoning이 모두 비어있는 6,713개 문서는 임베딩이 생성되지 않음 (정상 동작)

2. **위원회 소스 매핑**:
   - ftc: 공정거래위원회
   - nhrck: 국가인권위원회
   - ppc: 개인정보보호위원회
   - kcc: 방송통신위원회
   - fsc: 금융위원회
   - acrc: 국민권익위원회
   - sfc: 증권선물위원회
   - ecc: 선거관리위원회
   - iaciac: 국제입양심사위원회
   - eiac: 환경영향평가협의회
   - oclt: 원산지조사위원회

3. **청크 메타데이터**:
   - `doc_id`: PostgreSQL PK (포인터)
   - `source`: 데이터 출처
   - `doc_type`: 문서 유형
   - `chunk_index`: 청크 순서
   - `chunk_start`, `chunk_end`: 원문 내 위치

---

## 9. 용량 최적화 (추가 작업)

### 9.1 최적화 내용

ChromaDB에 청크 텍스트를 저장하지 않고, 검색 시 PostgreSQL에서 원문을 조회하는 방식으로 변경.

| 항목 | 최적화 전 | 최적화 후 | 절감률 |
|------|----------|----------|--------|
| ChromaDB 용량 | 4.9GB | 1.7GB | **65%** |

### 9.2 변경된 파일

| 파일 | 변경 내용 |
|------|----------|
| `app/common/vectorstore.py` | `documents` 파라미터를 Optional로 변경 |
| `scripts/create_embeddings.py` | `documents=None`으로 텍스트 미저장 |
| `app/common/chat_service.py` | PostgreSQL에서 청크 텍스트 조회 로직 추가 |
| `app/common/database.py` | 동기 세션 팩토리 추가 (`sync_session_factory`) |

### 9.3 검색 흐름 (최적화 후)

```
1. 사용자 질문 → 쿼리 임베딩 생성
2. ChromaDB에서 유사 청크 검색 (메타데이터 + distances만)
3. 메타데이터에서 doc_id, chunk_start, chunk_end 추출
4. PostgreSQL에서 원문 조회 → 청크 텍스트 추출
5. 컨텍스트 구성 → LLM 응답 생성
```

---

## 10. 팀 공유용 백업

### 10.1 백업 파일 위치

```
/Users/gimjuhyeong/dev/law-3-team/data/backup_share/
├── law_db_dump.sql.gz    # PostgreSQL (724MB)
└── chroma_backup.tar.gz  # ChromaDB (1.0GB)
```

### 10.2 팀원 복원 방법

**PostgreSQL 복원:**
```bash
gunzip law_db_dump.sql.gz
docker exec -i law-platform-db psql -U lawuser lawdb < law_db_dump.sql
```

**ChromaDB 복원:**
```bash
cd backend
tar -xzvf chroma_backup.tar.gz
```

---

## 11. 완료된 작업

- [x] 3-테이블 구조로 마이그레이션
- [x] 청크 기반 임베딩 생성 (643,095 청크)
- [x] ChromaDB 용량 최적화 (4.9GB → 1.7GB)
- [x] 검색 API에 청크 기반 검색 로직 적용
- [x] 팀 공유용 백업 파일 생성

## 12. 향후 작업

- [ ] 법령(laws) 테이블 임베딩 생성 (선택적)
- [ ] 필터링 기능 구현 (doc_type, source, date 등)
