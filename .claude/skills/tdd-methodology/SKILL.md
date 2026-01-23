---
name: tdd-methodology
description: 고품질 소프트웨어 개발을 위한 테스트 주도 개발(TDD) 및 테스트 전략 가이드라인. Red-Green-Refactor 사이클, 계층별 테스트 전략(Unit, Integration, E2E), 스택별(Python/FastAPI, React/Next.js) 테스트 도구 및 모범 사례를 포함함. 기능 구현, 버그 수정, 리팩토링 시 필수 적용.
---

# TDD & Testing Strategy

안정적이고 유지보수 가능한 코드 작성을 위한 TDD 원칙 및 스택별 구체적인 가이드라인.

## 1. TDD 핵심 원칙 (The Cycle)

개발은 항상 다음의 **Red-Green-Refactor** 사이클을 따릅니다.

1.  **Red (실패하는 테스트 작성)**:
    - 구현하려는 기능의 요구사항을 반영한 테스트 코드를 먼저 작성합니다.
    - 테스트를 실행하여 의도한 대로 실패하는지 확인합니다(컴파일 에러 또는 Assertion 실패).
2.  **Green (테스트 통과)**:
    - 테스트를 통과하기 위한 **최소한의 코드**만 작성합니다.
    - 코드의 품질보다 테스트 통과를 최우선으로 합니다.
3.  **Refactor (리팩토링)**:
    - 테스트 통과 상태를 유지하며 코드를 개선합니다.
    - 중복 제거, 가독성 향상, 구조 개선을 수행합니다.

## 2. 테스트 원칙 (FIRST)

모든 테스트 코드는 다음 **FIRST** 원칙을 준수해야 합니다.

-   **Fast (빠르게)**: 테스트는 자주 실행할 수 있도록 빨라야 합니다.
-   **Isolated (독립적으로)**: 각 테스트는 서로 의존하지 않고 독립적으로 실행되어야 합니다.
-   **Repeatable (반복 가능하게)**: 언제 어디서 실행하든(로컬, CI) 동일한 결과가 나와야 합니다.
-   **Self-Validating (자가 검증)**: 테스트 결과는 성공/실패(Bool)로 명확히 나와야 합니다.
-   **Timely (적시에)**: 실제 코드를 작성하기 **직전에** 작성해야 합니다.

## 3. Backend Testing (Python/FastAPI)

**도구**: `pytest`, `pytest-asyncio`, `httpx` (비동기 테스트), `pytest-mock`

### 3.1 파일 및 네이밍 컨벤션
-   **위치**: `tests/` 디렉터리 내에 소스 코드(`app/`)와 동일한 구조 유지
-   **파일명**: `test_*.py` (예: `test_user_service.py`)
-   **함수명**: `test_` 접두사 + `동작` + `상황` + `예상결과`
    -   예: `test_create_user_success`, `test_create_user_fails_with_duplicate_email`

### 3.2 단위 테스트 (Unit Test)
-   **대상**: Service 레이어, 유틸리티 함수, 비즈니스 로직
-   **전략**: 외부 의존성(DB, 외부 API)은 철저히 **Mocking** 처리합니다.

```python
# tests/services/test_user_service.py
from unittest.mock import Mock
import pytest
from app.services.user_service import UserService

def test_create_user_success():
    # Given
    mock_repo = Mock()
    mock_repo.save.return_value = {"id": 1, "name": "Test User"}
    service = UserService(repo=mock_repo)

    # When
    user = service.create_user(name="Test User")

    # Then
    assert user["id"] == 1
    mock_repo.save.assert_called_once()

```

### 3.3 통합 테스트 (Integration Test)

* **대상**: API 엔드포인트(Router), DB 연동 확인
* **전략**: `TestClient` 또는 `AsyncClient`를 사용하여 실제 요청 흐름 테스트. DB는 테스트용 격리 DB 또는 트랜잭션 롤백을 사용합니다.

```python
# tests/api/test_users.py
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_read_users(async_client: AsyncClient):
    # Given: 테스트 데이터 셋업 (Fixture 활용)

    # When
    response = await async_client.get("/users")

    # Then
    assert response.status_code == 200
    assert len(response.json()) >= 0

```

## 4. Frontend Testing (React/Next.js)

**도구**: `Jest`, `React Testing Library`, `MSW` (Mock Service Worker)

### 4.1 테스트 철학

* **구현 세부사항**보다 **사용자 관점의 동작**을 테스트합니다.
* "버튼이 클릭되었을 때 state가 변경된다"(X) -> "버튼을 클릭하면 성공 메시지가 화면에 보인다"(O)

### 4.2 컴포넌트 테스트

* **위치**: 컴포넌트와 동일한 폴더 내 `__tests__` 또는 `*.test.tsx`
* **작성 패턴**:

```tsx
// components/LoginForm.test.tsx
import { render, screen, fireEvent } from '@testing-library/react';
import LoginForm from './LoginForm';

describe('LoginForm', () => {
  it('이메일 형식이 잘못되면 에러 메시지를 표시해야 한다', () => {
    // 1. Render
    render(<LoginForm />);

    // 2. Interaction
    const input = screen.getByLabelText('Email');
    fireEvent.change(input, { target: { value: 'invalid-email' } });
    fireEvent.click(screen.getByRole('button', { name: 'Login' }));

    // 3. Assertion
    expect(screen.getByText('유효하지 않은 이메일입니다')).toBeInTheDocument();
  });
});

```

### 4.3 Hooks 테스트

* 비즈니스 로직이 포함된 Custom Hook은 `renderHook`을 사용하여 독립적으로 테스트합니다.

## 5. 테스트 커버리지 및 CI

* **목표**: 무의미한 100% 달성보다 **핵심 비즈니스 로직**과 **에러 케이스** 커버에 집중합니다.
* **CI 파이프라인**: PR 생성 시 모든 테스트가 자동으로 실행되어야 하며, 테스트 실패 시 머지를 차단합니다.

## 6. 테스트 작성 체크리스트

1. [ ] **Happy Path**: 정상적인 입력에 대해 기대한 결과가 나오는가?
2. [ ] **Edge Cases**:
* 입력이 없거나(Null/Undefined) 빈 값일 때
* 입력값이 최대/최소 범위를 벗어날 때
* 배열이 비어있거나 아주 많을 때


3. [ ] **Error Handling**: 예외 발생 시 적절한 에러 메시지나 처리가 수행되는가?