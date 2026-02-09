---
name: alembic-migration-safety
description: Alembic DB 마이그레이션 안전성 자동 검증. 롤백 가능성, 데이터 손실 위험, 인덱스 영향, Feature Flag 연동을 검증. 마이그레이션 작성, 실행, 리뷰 시 사용.
---

# Alembic Migration Safety Skill

Alembic 마이그레이션 작성 시 안전성을 자동 검증하는 가이드라인.

> **참조**: `database-operations.md` (마이그레이션 필수 절차), `postgresql-migration/SKILL.md` (ORM/마이그레이션 패턴)

## 1. 마이그레이션 안전성 체크리스트

마이그레이션 파일 작성 후 반드시 아래 항목을 검증합니다.

### 필수 검증 항목

| # | 검증 항목 | 위험도 | 검증 방법 |
|---|----------|--------|----------|
| 1 | `downgrade()` 함수 존재 및 동작 | Critical | 코드 리뷰 |
| 2 | 데이터 손실 위험 (DROP, ALTER TYPE) | Critical | 패턴 매칭 |
| 3 | 장시간 락 (대량 테이블 ALTER) | High | 테이블 크기 확인 |
| 4 | Feature Flag 연동 여부 | High | config.py 확인 |
| 5 | 인덱스 동시 생성 여부 | Medium | CONCURRENTLY 확인 |
| 6 | NULL 허용 → NOT NULL 변경 | Medium | 기존 데이터 확인 |
| 7 | revision chain 연속성 | Low | alembic heads 확인 |

## 2. 위험 패턴 탐지

### Critical - 즉시 중단

```python
# 데이터 손실 위험
op.drop_table("...")          # 테이블 삭제
op.drop_column("...", "...")  # 컬럼 삭제
op.alter_column("...", "...", type_=...)  # 타입 변경 (데이터 유실 가능)

# 롤백 불가
def downgrade():
    pass  # 빈 downgrade - 금지!
```

### High - 사용자 확인 필요

```python
# 장시간 락 위험 (10만건+ 테이블)
op.add_column("large_table", Column("new_col", String, nullable=False))
# → nullable=True로 먼저 추가 → 데이터 채우기 → NOT NULL 변경

# 인덱스 생성 시 테이블 락
op.create_index("idx_name", "table", ["col"])
# → op.create_index("idx_name", "table", ["col"], postgresql_concurrently=True)
```

### Medium - 주의 필요

```python
# NULL → NOT NULL 변경 (기존 NULL 데이터가 있으면 실패)
op.alter_column("table", "col", nullable=False)
# → server_default 설정 또는 데이터 마이그레이션 먼저 실행

# ENUM 타입 변경 (PostgreSQL에서 복잡)
op.alter_column("table", "col", type_=sa.Enum("A", "B", "C"))
# → 새 컬럼 추가 → 데이터 이전 → 구 컬럼 삭제 패턴 사용
```

## 3. 검증 명령어

```bash
# 1. 마이그레이션 체인 검증 (head가 1개인지)
cd backend && uv run alembic heads

# 2. 마이그레이션 SQL 미리보기 (실행 없이 SQL 확인)
uv run alembic upgrade head --sql

# 3. 실행 + 롤백 테스트 (dev 환경)
uv run alembic upgrade head
uv run alembic downgrade -1
uv run alembic upgrade head  # 재적용 확인

# 4. 현재 상태 확인
uv run alembic current
uv run alembic history --verbose
```

## 4. Feature Flag 연동 검증

마이그레이션에 대응하는 Feature Flag가 있는지 확인합니다.

```python
# backend/app/core/config.py 에 존재해야 함
USE_DB_<RESOURCE>: bool = False

# 라우터에서 분기 처리 확인
# backend/app/modules/<module>/router/__init__.py
if settings.USE_DB_<RESOURCE>:
    result = await service_db(db, ...)
else:
    result = service_json(...)
```

### 확인 절차

1. `config.py`에서 `USE_DB_` 로 시작하는 설정 검색
2. 해당 설정이 관련 라우터에서 분기 처리되는지 확인
3. JSON 서비스 파일이 삭제되지 않았는지 확인 (롤백 보장)

## 5. 안전한 마이그레이션 패턴

### 컬럼 추가 (안전)

```python
def upgrade():
    op.add_column("table", sa.Column("new_col", sa.String(100), nullable=True, comment="설명"))

def downgrade():
    op.drop_column("table", "new_col")
```

### 컬럼 삭제 (2단계)

```python
# Step 1: 마이그레이션 - Feature Flag로 사용 중단
# Step 2: 다음 마이그레이션 - 실제 삭제 (데이터 백업 후)
def upgrade():
    # 백업 테이블에 데이터 복사
    op.execute("CREATE TABLE _backup_col AS SELECT id, old_col FROM table")
    op.drop_column("table", "old_col")

def downgrade():
    op.add_column("table", sa.Column("old_col", sa.String(100), nullable=True))
    op.execute("UPDATE table SET old_col = b.old_col FROM _backup_col b WHERE table.id = b.id")
    op.execute("DROP TABLE _backup_col")
```

### 대량 데이터 마이그레이션

```python
def upgrade():
    # 배치 단위로 처리 (ORM 사용 금지, raw SQL 사용)
    conn = op.get_bind()
    total = conn.execute(sa.text("SELECT COUNT(*) FROM large_table")).scalar()
    batch_size = 1000
    for offset in range(0, total, batch_size):
        conn.execute(sa.text("""
            UPDATE large_table
            SET new_col = compute_value(old_col)
            WHERE id IN (
                SELECT id FROM large_table
                ORDER BY id LIMIT :batch OFFSET :offset
            )
        """), {"batch": batch_size, "offset": offset})
```

## 6. 프로젝트 현재 마이그레이션 상태

| 버전 | 설명 | Feature Flag |
|------|------|-------------|
| 001 | 초기 테이블 | - |
| 002 | case_precedent | - |
| 003 | 통합 테이블 정리 | - |
| 004 | lawyers 테이블 | `USE_DB_LAWYERS` |
| 005 | trial_statistics 테이블 | - |
| 006 | legal_terms 테이블 | `USE_LEGAL_TERM_DICT` |

새 마이그레이션 번호는 `007`부터 시작합니다.

