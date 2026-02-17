# Database Operations Rules

Claude는 데이터베이스 관련 작업 시 이 규칙들을 **항상(ALWAYS)** 따라야 합니다.

> **상세 가이드**: `.claude/skills/postgresql-migration/SKILL.md` 참조

## 1. 마이그레이션 필수 절차

순서를 건너뛰면 안 됩니다:

1. **ORM 모델 생성** (`backend/app/models/<model>.py`)
2. **models/__init__.py** 에 import/export 추가
3. **alembic/env.py** 에 모델 import 추가
4. **Alembic 마이그레이션 작성** (`backend/alembic/versions/NNN_*.py`)
5. **데이터 로드 스크립트 작성** (`backend/scripts/load_*_data.py`)
6. **DB 서비스 함수 작성** (`backend/app/services/service_function/*_db_service.py`)
7. **라우터 수정** (feature flag 분기 추가)

## 2. Feature Flag 사용 의무

- `USE_DB_<RESOURCE>: bool = False` (기본값)
- `.env`에서 `true`로 전환
- JSON 서비스 파일은 삭제하지 않음 (롤백 보장)

## 3. DB 서비스 함수 시그니처

- 함수명: JSON 함수명 + `_db` 접미사
- 첫 번째 파라미터: `db: AsyncSession`
- 반환 타입: JSON 서비스와 동일

## 4. 배치 처리 규칙

- **1,000건** 단위, **멱등성** (`ON CONFLICT DO UPDATE`), 배치마다 commit + 진행률 로깅

## 5. 인덱스 설계

| 용도 | 인덱스 타입 |
|------|-----------|
| 단일 컬럼 조회 | B-tree |
| ARRAY 검색 | GIN (`@>`, `&&`) |
| GROUP BY 통계 | B-tree |
| 범위 검색 (좌표) | B-tree 복합 |

## 6. 비정규화 허용 조건

1. 원본 데이터에서 계산 가능
2. GROUP BY에서 빈번히 사용
3. 값 변경 빈도가 낮음

## 7. 데이터 무결성 검증

로드 후 `--verify` 실행: 총 건수, 좌표/전문분야 비율, 지역별 상위 5개, 인덱스 동작 확인.
