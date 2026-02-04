# PostgreSQL Migration Skill

JSON 파일 데이터를 PostgreSQL로 마이그레이션하는 패턴 가이드.

## 1. ORM 모델 작성 패턴

```python
# backend/app/models/<model>.py
from datetime import datetime
from sqlalchemy import Column, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY
from app.core.database import Base

class ModelName(Base):
    __tablename__ = "table_name"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # ... 컬럼 정의 (comment 필수) ...
    created_at = Column(DateTime, default=datetime.utcnow, comment="레코드 생성일시")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="레코드 수정일시")

    __table_args__ = (
        Index("idx_table_col1_col2", "col1", "col2"),  # 복합 인덱스
        Index("idx_table_array_col", "array_col", postgresql_using="gin"),  # GIN 인덱스
        {"comment": "테이블 설명"},
    )
```

### 컨벤션
- 모든 컬럼에 `comment` 파라미터 추가
- `__table_args__`에 복합/특수 인덱스 정의
- `created_at`, `updated_at` 타임스탬프 필수
- `backend/app/models/__init__.py`에 import/export 추가
- `backend/alembic/env.py`에 import 추가

## 2. Alembic 마이그레이션 패턴

```python
# backend/alembic/versions/NNN_description.py
revision: str = 'NNN'
down_revision: Union[str, None] = 'NNN-1'

def upgrade() -> None:
    op.create_table('table_name', ...)
    op.create_index('idx_...', 'table_name', ['col'], ...)

def downgrade() -> None:
    op.drop_index('idx_...', table_name='table_name')  # 인덱스 먼저
    op.drop_table('table_name')  # 테이블 나중에
```

### 주의사항
- revision ID는 순차 번호 (`001`, `002`, ...)
- downgrade: 인덱스 → 테이블 순서로 삭제
- GIN 인덱스: `postgresql_using='gin'` 명시

## 3. 데이터 로드 스크립트 패턴

```bash
uv run python scripts/load_<data>_data.py           # 로드
uv run python scripts/load_<data>_data.py --reset    # 삭제 후 재로드
uv run python scripts/load_<data>_data.py --verify   # 검증만
```

### 핵심 원칙
- **멱등성**: `ON CONFLICT DO UPDATE` 사용
- **배치 처리**: 1,000건 단위 insert + commit
- **검증**: 로드 후 JSON과 건수/비율 일치 확인
- **Sync Engine**: 스크립트에서는 sync engine 사용 (async 불필요)

## 4. PostgreSQL ARRAY + GIN 인덱스

```python
# 모델
specialties = Column(ARRAY(Text), server_default="{}", nullable=False)

# GIN 인덱스 (contains, overlap 연산 최적화)
Index("idx_specialties", "specialties", postgresql_using="gin")

# 쿼리: 특정 값 포함 (ANY)
query.where(Lawyer.specialties.any("이혼"))

# 쿼리: 교집합 존재 (OVERLAP)
query.where(Lawyer.specialties.overlap(["이혼", "상속"]))
```

## 5. Feature Flag 기반 점진적 마이그레이션

```python
# config.py
USE_DB_<RESOURCE>: bool = False

# 라우터에서 분기
if settings.USE_DB_LAWYERS:
    from app.services.service_function.lawyer_db_service import find_nearby_lawyers_db
    result = await find_nearby_lawyers_db(db=db, ...)
else:
    result = find_nearby_lawyers(...)
```

### 패턴
1. JSON 서비스 (`_service.py`)는 그대로 유지
2. DB 서비스 (`_db_service.py`)를 별도 파일로 생성
3. 라우터에서 `settings.USE_DB_*` 분기
4. DB 서비스 함수의 첫 번째 파라미터: `db: AsyncSession`
5. 롤백: `.env`에서 `USE_DB_*=false`로 즉시 복귀

## 6. DB 서비스 함수 시그니처 규칙

```python
async def find_nearby_lawyers_db(
    db: AsyncSession,           # 첫 번째 파라미터: DB 세션
    latitude: float,            # 이후 파라미터: JSON 서비스와 동일
    longitude: float,
    ...
) -> list[dict[str, Any]]:     # 반환 타입: JSON 서비스와 동일
```

## 7. 비정규화 필드 패턴

통계 쿼리 최적화를 위해 계산된 필드를 미리 저장:

```python
# 데이터 로드 시 계산
province, district, region = extract_region_parts(address)

# DB 컬럼
province = Column(String(20), index=True)   # GROUP BY province
district = Column(String(50), index=True)   # GROUP BY district
region = Column(String(50), index=True)     # "시도 시군구" 결합
```

이렇게 하면 통계 쿼리에서 정규표현식 파싱 없이 바로 GROUP BY 가능.
