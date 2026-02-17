# Coding Style Rules

Claude는 모든 코드 작성 시 이 규칙들을 **항상(ALWAYS)** 따라야 합니다.

> **상세 가이드**: Python → `.claude/skills/python-coding-standards/SKILL.md`, React/TS → `.claude/skills/react-nextjs-frontend/SKILL.md`

## 1. 일반 원칙

- 영리한 코드보다 **읽기 쉬운 코드** 작성
- 매직 넘버 금지 - 상수로 정의
- 불변성 선호 (Python: tuple/frozenset, JS: const)
- 함수는 단일 책임, 30줄 이하 권장

## 2. 네이밍 컨벤션

| 대상 | Python | TypeScript |
|------|--------|-----------|
| 변수, 함수 | `snake_case` | `camelCase` |
| 클래스, 컴포넌트 | `PascalCase` | `PascalCase` |
| 상수 | `UPPER_SNAKE_CASE` | `UPPER_SNAKE_CASE` |
| Boolean | `is_`, `has_`, `should_` 접두사 | 동일 |
| Private | `_leading_underscore` | `#prefix` |

축약어 금지: `btn`→`button`, `msg`→`message`, `usr`→`user`

## 3. 파일 크기 제한

- **Python**: 500줄 이하 권장, 800줄 초과 시 분리
- **TypeScript**: 400줄 이하 권장, 600줄 초과 시 분리
- **React 컴포넌트**: 200줄 이하 권장

## 4. 금지 사항

- 하드코딩된 비밀번호/API 키
- 디버그 코드 커밋 (`print`, `console.log`, `pdb`)
- 주석 처리된 코드 방치
- `any` 타입 사용 (TypeScript) → `unknown` 또는 구체적 타입
- `except Exception: pass` (너무 광범위) → 구체적 예외 처리

## 5. 타입 힌팅 (필수)

- **Python**: 모든 함수에 타입 힌트 필수 (`-> ReturnType`)
- **TypeScript**: `any` 금지, interface/type 정의 필수
- 주석은 "Why" 설명 (코드를 반복하지 않음)

## 6. 포맷팅 도구

| 언어 | 린트 | 포맷 | 타입 체크 |
|------|------|------|----------|
| Python | `ruff check` | `ruff format` | `mypy` |
| TypeScript | ESLint | Prettier | `tsc --noEmit` |

## 7. 코드 리뷰 체크리스트

- [ ] 타입 힌트가 모든 함수에 있는가?
- [ ] 하드코딩된 값이 없는가?
- [ ] 디버그 코드가 없는가?
- [ ] 매직 넘버가 상수로 정의되었는가?
- [ ] 함수가 30줄 이하인가?
- [ ] 변수명이 명확한가?

## 8. 스킬/에이전트 유지보수

코드 구조(파일 경로, 클래스명, 함수 시그니처)가 변경될 때 관련 스킬/에이전트도 함께 업데이트:

- **경로 변경**: 스킬 내 import 예시, 디렉토리 구조 다이어그램
- **클래스/함수 변경**: 스킬 내 코드 예시, 패턴 설명
- **아키텍처 변경**: 관련 스킬의 아키텍처 섹션

확인: `grep -r "변경전_경로" .claude/skills/ .claude/agents/`

## 9. 정보 검색 시 최신 날짜 기준

- 웹 검색 시 **현재 날짜 기준** 최신 정보 검색
- 검색 쿼리에 연도 포함
- 2년 이상 된 정보는 교차 확인
- deprecated API/패턴 사용 금지

## 10. Python 의존성 동기화 (pyproject.toml)

`pyproject.toml` 의존성 추가/삭제 시 **두 섹션 반드시 동기화**:

| 섹션 | 용도 |
|------|------|
| `[project.optional-dependencies]` dev | PyPI 표준 (PEP 621) |
| `[dependency-groups]` dev | uv 전용 (PEP 735) |

- uv는 `[dependency-groups]`를 우선 사용
- 양쪽 불일치 시 환경별 다른 패키지 설치됨
