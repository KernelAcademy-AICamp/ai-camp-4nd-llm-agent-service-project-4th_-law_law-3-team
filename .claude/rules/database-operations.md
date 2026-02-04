# Database Operations Rules

Claude는 데이터베이스 관련 작업 시 이 규칙들을 **항상(ALWAYS)** 따라야 합니다.

## 1. 마이그레이션 필수 절차

데이터를 DB에 추가할 때 아래 순서를 준수합니다:

1. **ORM 모델 생성** (`backend/app/models/<model>.py`)
2. **models/__init__.py** 에 import/export 추가
3. **alembic/env.py** 에 모델 import 추가
4. **Alembic 마이그레이션 작성** (`backend/alembic/versions/NNN_*.py`)
5. **데이터 로드 스크립트 작성** (`backend/scripts/load_*_data.py`)
6. **DB 서비스 함수 작성** (`backend/app/services/service_function/*_db_service.py`)
7. **라우터 수정** (feature flag 분기 추가)

순서를 건너뛰면 안 됩니다.

## 2. Feature Flag 사용 의무

JSON → DB 전환 시 반드시 feature flag를 사용합니다:

```python
# backend/app/core/config.py
USE_DB_<RESOURCE>: bool = False
```

- 기본값은 `False` (JSON 모드)
- `.env`에서 `USE_DB_<RESOURCE>=true`로 전환
- 라우터에서 `settings.USE_DB_<RESOURCE>` 분기 처리
- JSON 서비스 파일은 삭제하지 않음 (롤백 보장)

## 3. DB 서비스 함수 시그니처

```python
async def function_name_db(
    db: AsyncSession,    # 첫 번째 파라미터: DB 세션
    ...                  # 이후 파라미터: JSON 서비스와 동일
) -> ReturnType:         # 반환 타입: JSON 서비스와 동일
```

- 함수명: JSON 서비스 함수명 + `_db` 접미사
- 파일명: JSON 서비스 파일명 + `_db` 접미사 (예: `lawyer_db_service.py`)

## 4. 배치 처리 규칙

- **배치 크기**: 1,000건 단위
- **멱등성**: `ON CONFLICT DO UPDATE` 사용 (unique 키 기준)
- **진행률 로깅**: 배치마다 진행률 출력
- **트랜잭션**: 배치 단위로 commit

```python
BATCH_SIZE = 1000
for i in range(0, len(records), BATCH_SIZE):
    batch = records[i:i + BATCH_SIZE]
    stmt = insert(Model).values(batch)
    stmt = stmt.on_conflict_do_update(
        index_elements=["unique_key"],
        set_={...},
    )
    db.execute(stmt)
    db.commit()
```

## 5. 인덱스 설계 원칙

| 용도 | 인덱스 타입 | 예시 |
|------|-----------|------|
| 단일 컬럼 조회 | B-tree (기본) | `name`, `status` |
| 범위 검색 (좌표) | B-tree 복합 | `(latitude, longitude)` |
| ARRAY 검색 | GIN | `specialties` |
| GROUP BY 통계 | B-tree | `region`, `province` |
| 텍스트 검색 | B-tree + ILIKE | `name`, `office_name` |

- GIN 인덱스는 `@>` (contains), `&&` (overlap) 연산에 최적화
- 복합 인덱스는 자주 함께 조회되는 컬럼만 포함

## 6. 비정규화 허용 조건

통계 쿼리 최적화를 위한 비정규화 필드는 다음 조건에서만 허용:

1. **원본 데이터에서 계산 가능** (데이터 로드 시 계산)
2. **GROUP BY에서 빈번히 사용** (매 요청마다 파싱하면 비효율)
3. **값 변경 빈도가 낮음** (주소 변경은 드묾)

예시: `address` → `province`, `district`, `region` 추출 저장

## 7. 데이터 무결성 검증

데이터 로드 후 반드시 검증:

```bash
uv run python scripts/load_<data>_data.py --verify
```

검증 항목:
- [ ] 총 건수가 JSON과 일치
- [ ] 좌표 보유 건수/비율 확인
- [ ] 전문분야 보유 건수/비율 확인
- [ ] 지역별 상위 5개 확인
- [ ] 인덱스 동작 확인 (EXPLAIN ANALYZE)

---

**중요**: 이 규칙들은 데이터베이스 작업 시 예외 없이 적용됩니다.
