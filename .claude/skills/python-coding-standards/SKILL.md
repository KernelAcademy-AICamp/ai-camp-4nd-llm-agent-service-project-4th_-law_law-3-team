---
name: python-coding-standards
description: Python 코딩 시 AI 친화적이고 안전한 코드 작성을 위한 가이드라인. 코드 수정 요청 시 최소 변경 원칙 적용, AI가 이해하기 쉬운 네이밍 규칙, 디자인 패턴 적용, 방어적 코딩, PEP 8 스타일 준수. Python 코드 작성, 수정, 리뷰, 리팩토링 시 이 스킬 사용.
---

# Python Coding Standards

Python 코드 작성 시 일관성, 가독성, 안전성을 위한 표준 가이드라인.

## 1. 코드 보존 원칙

- 변경 요청 범위 외의 코드는 절대 수정하지 않기
- 기존 코드 스타일(들여쓰기, 따옴표, 줄바꿈) 유지
- 주석, 빈 줄, 포맷팅 그대로 유지
- 삭제보다 주석 처리 후 확인 받기

## 2. AI 친화적 네이밍

**함수/변수**: `snake_case`
- 축약어 금지: `btn` → `button`, `msg` → `message`, `usr` → `user`
- 동작+대상 형식: `get_user_by_id`, `validate_email_format`, `calculate_total_price`

**클래스**: `PascalCase`
- 명확한 역할 표현: `UserRepository`, `EmailValidator`, `PaymentProcessor`

**Boolean**: `is_`, `has_`, `should_`, `can_` 접두사
```python
is_valid = True
has_permission = False
should_retry = True
can_edit = user.role == "admin"
```

**상수**: `UPPER_SNAKE_CASE`
```python
MAX_RETRY_COUNT = 3
DEFAULT_TIMEOUT_SECONDS = 30
API_BASE_URL = "https://api.example.com"
```

## 3. Python 스타일 (PEP 8)

**Import 순서**:
```python
# 1. 표준 라이브러리
import os
import sys
from datetime import datetime

# 2. 서드파티
import requests
from sqlalchemy import Column

# 3. 로컬
from app.models import User
from app.utils import helper
```

**한 줄 길이**: 79자 (또는 프로젝트 설정)

## 4. 타입 힌트

함수 시그니처에 타입 힌트 필수:
```python
from typing import Optional, List, Dict

def get_user_by_id(user_id: int) -> Optional[User]:
    """사용자 ID로 사용자 조회."""
    pass

def process_items(items: List[str]) -> Dict[str, int]:
    """아이템 목록 처리 후 결과 반환."""
    pass
```

## 5. Docstring (Google Style)

```python
def calculate_discount(
    price: float,
    discount_rate: float,
    max_discount: Optional[float] = None
) -> float:
    """할인된 가격 계산.

    Args:
        price: 원래 가격
        discount_rate: 할인율 (0.0 ~ 1.0)
        max_discount: 최대 할인 금액 (None이면 제한 없음)

    Returns:
        할인 적용된 최종 가격

    Raises:
        ValueError: discount_rate가 0~1 범위를 벗어난 경우
    """
    pass
```

## 6. 에러 처리 및 방어적 코딩

**구체적인 예외 처리**:
```python
# Bad
try:
    result = process_data(data)
except:
    pass

# Good
try:
    result = process_data(data)
except ValidationError as e:
    logger.error(f"Validation failed: {e}")
    raise
except DatabaseError as e:
    logger.error(f"Database operation failed: {e}")
    return None
```

**Null 체크 및 기본값**:
```python
def get_user_name(user: Optional[User]) -> str:
    if user is None:
        return "Unknown"
    return user.name or "Anonymous"
```

**구체적인 에러 메시지**:
```python
# Bad
raise ValueError("Error")

# Good
raise ValueError(f"Invalid user_id: {user_id}. Expected positive integer.")
```

## 7. Python 흔한 함정 피하기

**Mutable 기본값 피하기**:
```python
# Bad
def add_item(item: str, items: list = []) -> list:
    items.append(item)
    return items

# Good
def add_item(item: str, items: Optional[list] = None) -> list:
    if items is None:
        items = []
    items.append(item)
    return items
```

**얕은 복사 vs 깊은 복사**:
```python
import copy

original = [[1, 2], [3, 4]]
shallow = original.copy()      # 내부 리스트는 참조 공유
deep = copy.deepcopy(original)  # 완전히 독립적인 복사본
```

**`is None` vs `== None`**:
```python
# Good
if value is None:
    pass

# Bad
if value == None:
    pass
```

## 8. Pythonic 코드

**리스트 컴프리헨션** (단, 복잡하면 for문 사용):
```python
# Good - 간단한 경우
squares = [x ** 2 for x in range(10)]

# Good - 복잡한 경우 for문 사용
results = []
for item in items:
    if item.is_valid:
        processed = complex_process(item)
        if processed.score > threshold:
            results.append(processed)
```

**내장 함수 활용**:
```python
# enumerate
for index, item in enumerate(items):
    print(f"{index}: {item}")

# zip
for name, score in zip(names, scores):
    print(f"{name}: {score}")

# any, all
has_error = any(item.has_error for item in items)
all_valid = all(item.is_valid for item in items)
```

**Context Manager**:
```python
# Good
with open("file.txt", "r") as f:
    content = f.read()

# Good - 여러 리소스
with open("input.txt") as f_in, open("output.txt", "w") as f_out:
    f_out.write(f_in.read())
```

## 9. 디자인 패턴 적용

상황에 맞는 패턴 선택. 자세한 내용은 [references/design-patterns.md](references/design-patterns.md) 참조.

| 상황 | 패턴 |
|------|------|
| 복잡한 객체 생성 | Builder |
| 조건에 따라 다른 객체 | Factory |
| 기존 클래스에 기능 추가 | Decorator (`@decorator`) |
| 알고리즘 교체 가능 | Strategy (함수를 인자로 전달) |
| 상태에 따른 동작 변경 | State |
| 이벤트 기반 처리 | Observer |
| 컬렉션 순회 | Iterator (제너레이터 `yield`) |

## 10. 코드 품질 원칙

**함수는 한 가지 역할만**:
```python
# Bad - 여러 역할
def process_user(user_data: dict) -> User:
    # 검증 + 변환 + 저장 + 알림 전부 수행
    pass

# Good - 역할 분리
def validate_user_data(data: dict) -> bool:
    pass

def create_user_from_data(data: dict) -> User:
    pass

def save_user(user: User) -> None:
    pass
```

**매직 넘버 금지**:
```python
# Bad
if retry_count > 3:
    pass

# Good
MAX_RETRY_COUNT = 3
if retry_count > MAX_RETRY_COUNT:
    pass
```

**복잡한 조건문 분리**:
```python
# Bad
if user.age >= 18 and user.has_verified_email and user.account_status == "active":
    pass

# Good
is_adult = user.age >= 18
is_verified = user.has_verified_email
is_active = user.account_status == "active"

if is_adult and is_verified and is_active:
    pass
```

## 11. 보안 기본

- 사용자 입력 항상 검증/이스케이프
- 민감 정보(API 키, 비밀번호) 하드코딩 금지 → 환경변수 사용
- SQL 파라미터 바인딩 사용 (SQL 인젝션 방지)
- f-string으로 SQL 쿼리 작성 금지

## 12. 커뮤니케이션

- 변경 사항 요약해서 알려주기
- 여러 해석이 가능하면 먼저 확인하기
- 사이드 이펙트 있을 수 있는 변경은 미리 알려주기
- 잠재적 버그나 개선점 발견 시 제안하기
- 기존 테스트가 있다면 깨지지 않도록 주의

## 13. 일관성 유지

- 프로젝트에 이미 있는 패턴 따르기
- 비슷한 기능은 비슷한 방식으로 구현
- 기존 유틸 함수가 있으면 재사용
- 새로운 패턴 도입은 신중하게
