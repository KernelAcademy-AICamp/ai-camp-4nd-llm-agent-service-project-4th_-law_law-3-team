# Coding Style Rules

Claude는 모든 코드 작성 시 이 규칙들을 **항상(ALWAYS)** 따라야 합니다.

## 1. 일반 원칙

### 명확성 우선
- 영리한 코드보다 **읽기 쉬운 코드** 작성
- 복잡한 로직은 주석으로 의도 설명
- 매직 넘버 금지 - 상수로 정의하고 의미 있는 이름 부여

### 불변성(Immutability) 선호
- 가능한 한 불변 데이터 구조 사용
- Python: tuple, frozenset 우선 고려
- JavaScript: const 기본 사용, let은 필요시만

### 단일 책임 원칙
- 함수는 하나의 일만 수행
- 함수 길이: 30줄 이하 권장
- 클래스는 명확한 단일 목적

## 2. 네이밍 컨벤션

### Python
```python
# 변수, 함수: snake_case
user_name = "Alice"
def calculate_loss(predictions, targets):
    pass

# 클래스: PascalCase
class ModelTrainer:
    pass

# 상수: UPPER_SNAKE_CASE
MAX_EPOCHS = 100
LEARNING_RATE = 0.001

# Private: _leading_underscore
def _internal_helper():
    pass
```

### JavaScript/TypeScript
```typescript
// 변수, 함수: camelCase
const userName = "Alice";
function calculateLoss(predictions: Tensor, targets: Tensor) {}

// 클래스, 컴포넌트: PascalCase
class ModelTrainer {}
function UserProfile() {}

// 상수: UPPER_SNAKE_CASE
const MAX_RETRIES = 3;
const API_BASE_URL = "https://api.example.com";

// Private: #prefix (클래스 필드)
class Example {
  #privateField = 0;
}
```

### 의미 있는 이름
```python
# ❌ Bad
def process(d):
    return d * 2

# ✅ Good
def double_learning_rate(learning_rate: float) -> float:
    return learning_rate * 2
```

## 3. 파일 및 모듈 구조

### Python 파일
```python
"""
모듈 docstring: 파일의 목적 설명
"""

# 1. Standard library imports
import os
import sys
from pathlib import Path

# 2. Third-party imports
import torch
import numpy as np
from langchain.agents import Agent

# 3. Local imports
from .models import CustomModel
from .utils import load_config

# 4. 상수 정의
CONFIG_PATH = Path("config.yaml")
DEFAULT_BATCH_SIZE = 32

# 5. 클래스 및 함수 정의
class Trainer:
    """Trainer class docstring"""
    pass

def main():
    """Main function"""
    pass

# 6. 실행 블록
if __name__ == "__main__":
    main()
```

### TypeScript/React 파일
```typescript
// 1. Type imports
import type { NextPage } from 'next';
import type { User } from '@/types';

// 2. Library imports
import { useState, useEffect } from 'react';
import { useRouter } from 'next/router';

// 3. Local imports
import { Button } from '@/components/ui';
import { fetchUser } from '@/lib/api';

// 4. Constants
const MAX_RETRIES = 3;

// 5. Component
export const UserProfile: React.FC<Props> = ({ userId }) => {
  // ...
};
```

### 파일 크기 제한
- **Python**: 500줄 이하 권장, 800줄 초과 시 분리 고려
- **TypeScript**: 400줄 이하 권장, 600줄 초과 시 분리 고려
- **React 컴포넌트**: 200줄 이하 권장

## 4. 주석 및 문서화

### Docstring (Python)
```python
def train_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    epochs: int = 10
) -> dict[str, float]:
    """
    모델을 학습시킵니다.
    
    Args:
        model: 학습할 PyTorch 모델
        data_loader: 학습 데이터 로더
        epochs: 학습 에포크 수 (기본값: 10)
    
    Returns:
        학습 메트릭을 담은 딕셔너리
        - loss: 최종 손실값
        - accuracy: 최종 정확도
    
    Raises:
        ValueError: epochs가 0 이하일 때
    
    Example:
        >>> metrics = train_model(model, train_loader, epochs=5)
        >>> print(metrics['accuracy'])
        0.95
    """
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    # ...
```

### JSDoc (TypeScript)
```typescript
/**
 * 사용자 데이터를 가져옵니다.
 * 
 * @param userId - 사용자 ID
 * @param options - 옵션 객체
 * @returns 사용자 데이터 Promise
 * @throws {ApiError} API 요청 실패 시
 * 
 * @example
 * const user = await fetchUser('123');
 */
async function fetchUser(
  userId: string,
  options?: FetchOptions
): Promise<User> {
  // ...
}
```

### 인라인 주석
```python
# ❌ Bad: 코드를 그대로 설명
# x를 1 증가시킴
x = x + 1

# ✅ Good: 왜(Why) 설명
# 배치 인덱스는 1부터 시작하므로 보정
batch_idx = batch_idx + 1

# ✅ Good: 복잡한 로직 설명
# Attention mask를 적용하여 패딩 토큰이 loss 계산에 
# 영향을 주지 않도록 함
masked_loss = loss * attention_mask
```

## 5. 금지 사항

### 절대 금지
```python
# ❌ 하드코딩된 비밀번호/API 키
api_key = "sk-1234567890abcdef"

# ❌ 디버그 코드 커밋
print("DEBUG: user =", user)
import pdb; pdb.set_trace()

# ❌ 주석 처리된 코드 방치
# def old_function():
#     pass

# ❌ TODO 없이 방치
# TODO 나중에 수정
```

### 제한적 사용
```python
# ⚠️ 최소화: try-except 과다 사용
try:
    result = complex_operation()
except Exception:  # ❌ 너무 광범위
    pass

# ✅ 구체적 예외 처리
try:
    result = complex_operation()
except ValueError as e:
    logger.error(f"Invalid value: {e}")
    raise
```

## 6. 타입 힌팅

### Python (Type Hints)
```python
# ✅ 항상 타입 힌트 사용
from typing import List, Dict, Optional, Union

def process_data(
    data: List[Dict[str, float]],
    threshold: float = 0.5
) -> Optional[np.ndarray]:
    """데이터 처리"""
    if not data:
        return None
    # ...
```

### TypeScript
```typescript
// ✅ any 금지, unknown 또는 구체적 타입 사용
// ❌ Bad
function process(data: any) {}

// ✅ Good
function process<T extends Record<string, unknown>>(data: T) {}

// ✅ Interface 정의
interface TrainingConfig {
  epochs: number;
  batchSize: number;
  learningRate?: number; // Optional
}
```

## 7. 에러 처리

### Python
```python
# ✅ 구체적 예외 처리
try:
    config = load_config(config_path)
except FileNotFoundError:
    logger.warning(f"Config not found at {config_path}, using defaults")
    config = get_default_config()
except yaml.YAMLError as e:
    logger.error(f"Invalid YAML format: {e}")
    raise

# ✅ 커스텀 예외 정의
class ModelLoadError(Exception):
    """모델 로드 실패 시 발생하는 예외"""
    pass
```

### TypeScript
```typescript
// ✅ Error 타입 명시
try {
  await fetchData();
} catch (error) {
  if (error instanceof ApiError) {
    console.error('API Error:', error.message);
  } else {
    console.error('Unknown error:', error);
  }
}
```

## 8. 코드 포맷팅 도구

### Python
- **Black**: 자동 포맷팅 (line length: 88)
- **isort**: import 정렬
- **mypy**: 타입 체크
```bash
black .
isort .
mypy .
```

### TypeScript/JavaScript
- **Prettier**: 자동 포맷팅
- **ESLint**: 린팅
```bash
prettier --write .
eslint --fix .
```

## 9. 코드 리뷰 체크리스트

작업 완료 전 자가 점검:
- [ ] 타입 힌트가 모든 함수에 있는가?
- [ ] Docstring/JSDoc이 public 함수에 있는가?
- [ ] 하드코딩된 값이 없는가?
- [ ] 디버그 코드(print, console.log)가 없는가?
- [ ] 매직 넘버가 상수로 정의되었는가?
- [ ] 함수가 30줄 이하인가?
- [ ] 변수명이 명확한가?
- [ ] 불필요한 주석이 제거되었는가?

## 10. 스킬/에이전트 유지보수

### 규칙
코드 구조(파일 경로, 클래스명, 함수 시그니처 등)가 변경될 때,
관련된 스킬(`.claude/skills/`)과 에이전트(`.claude/agents/`)도 반드시 함께 업데이트해야 합니다.

### 체크 대상
- **경로 변경** (파일/디렉토리 이동, 이름 변경):
  스킬 내 `import` 예시, 디렉토리 구조 다이어그램 업데이트
- **클래스/함수 변경** (이름 변경, 시그니처 변경, 삭제):
  스킬 내 코드 예시, 패턴 설명 업데이트
- **아키텍처 변경** (새 모듈 추가, 계층 구조 변경):
  관련 스킬의 아키텍처 섹션 업데이트
- **의존성 변경** (DB, 외부 서비스 교체):
  관련 스킬의 설정/연동 섹션 업데이트

### 확인 방법
변경된 파일 경로를 `.claude/skills/` 및 `.claude/agents/` 내에서 검색:
```bash
grep -r "변경전_경로" .claude/skills/ .claude/agents/
```

### 주의
- 스킬에 존재하지 않는 템플릿 파일을 참조하지 않는다
- 스킬의 코드 예시는 실제 동작하는 코드여야 한다
- 스킬이 더 이상 프로젝트와 관련 없으면 삭제한다

## 11. 외부 AI CLI 위임

### 규칙
다음 상황에서는 외부 AI CLI(Gemini CLI, Codex CLI)를 활용하여 작업을 효율화한다:

- **많은 파일 동시 파악** (10개 이상 파일의 패턴/구조 분석) → Gemini CLI
- **프로젝트 전체 분석** (아키텍처 파악, 의존관계 분석) → Gemini CLI
- **대규모 리팩토링 설계** (영향 범위 분석, 변경 계획 수립) → Gemini CLI
- **보안 코드 리뷰** (다수 파일에 걸친 보안 취약점 분석) → Gemini CLI
- **마이그레이션 영향도 분석** (라이브러리/프레임워크 업그레이드 시 영향 파악) → Gemini CLI
- **코드 품질 감사** (프로젝트 전반의 안티패턴, 중복 코드 탐지) → Gemini CLI
- **코드 리뷰** (PR/커밋 diff 분석) → Codex CLI
- **샌드박스 실행** (격리 환경에서 코드 실행 검증) → Codex CLI
- **웹 검색 연동 코딩** (외부 정보 기반 코드 작성) → Codex CLI

### 사용 조건
1. `which gemini`, `which codex`로 설치 여부 먼저 확인
2. 설치되어 있지 않으면 Claude가 직접 수행 (Fallback)
3. 외부 CLI의 출력은 참고 자료로만 활용하며, 최종 판단과 코드 작성은 Claude가 수행
4. **CLI 도구 선택 기준**: `.claude/rules/cli-tool-routing.md` 참조

### 상세 가이드
- **Gemini CLI**: `.claude/skills/gemini-cli-delegation/SKILL.md`
- **Codex CLI**: `.claude/skills/codex-cli-delegation/SKILL.md`
- **CLI 조합**: `.claude/skills/multi-cli-integration/SKILL.md`
- **오케스트레이션**: `.claude/agents/ai-orchestrator.md`

## 12. 정보 검색 시 최신 날짜 기준

### 규칙
웹 검색, 문서 조회, 기술 정보 확인 시 **항상 현재 날짜 기준**으로 최신 정보를 검색한다.

### 적용 대상
- **라이브러리/프레임워크 문서**: 최신 버전 문서를 검색 (예: "React 19 docs 2026")
- **API 레퍼런스**: 현재 날짜 기준 최신 API 문서 확인
- **모범 사례 / 패턴**: 최신 권장 사항 기준으로 코드 작성
- **보안 취약점**: 현재 알려진 최신 CVE 기반으로 보안 검토
- **의존성 버전**: 패키지 최신 버전 및 호환성 확인

### 주의
- 검색 쿼리에 연도를 포함하여 최신 결과를 우선 확보한다
- 오래된 정보(2년 이상)는 현재도 유효한지 반드시 교차 확인한다
- deprecated된 API나 패턴을 사용하지 않도록 주의한다

---

**중요**: 이 규칙들은 예외 없이 항상 적용됩니다. 규칙을 어길 합당한 이유가 있다면 주석으로 명시적으로 설명해야 합니다.